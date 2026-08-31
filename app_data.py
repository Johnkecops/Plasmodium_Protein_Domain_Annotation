#!/usr/bin/env python3
"""
Module: app_data
Purpose: Cached data-loading helpers for app.py (UniProt/InterPro fetches
         wrapped in st.cache_data, plus JSON<->DataFrame round-tripping).
         Split out of app.py to keep that file focused on page layout.
Author: Dr. Arli Aditya Parikesit
Date: 2026

Rationale (response to reviewers):
    The manuscript's primary dataset is now the frozen snapshot of all 22 Plasmodium
    reference proteomes (128,121 proteins), not the live Swiss-Prot query the app
    previously offered, which the editor found too sparse to support genus-scale claims.
    load_frozen_snapshot lets the app load that exact dataset, or the frozen copy of the
    submitted query kept for direct comparison, with no network call. Reshaping the
    frozen UniProt-schema snapshot into the app's existing protein schema (organism,
    pfam_ids, domain_names, ...) via _snapshot_to_app_schema means every downstream tab,
    chart and the domain_analysis facade run unchanged regardless of which data source
    populated the table.
"""

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from domain_dataset import load_snapshot
from fetch_proteins import (
    fetch_plasmodium_proteins,
    build_interpro_name_map,
    fetch_fasta_sequences,
    parse_ft_domain,
    parse_semicolon_ids,
    domain_names_from_positions,
)

_REPO_ROOT = Path(__file__).resolve().parent
_FROZEN_SNAPSHOTS = {
    "reference": _REPO_ROOT / "data" / "frozen" / "plasmodium_reference_proteomes.tsv.gz",
    "swissprot": _REPO_ROOT / "data" / "frozen" / "plasmodium_swissprot_reviewed.tsv.gz",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_proteins(taxon_id: str, reviewed: bool, max_results: int) -> str:
    """Fetch protein table and return as JSON string for cache compatibility."""
    df = fetch_plasmodium_proteins(
        taxon_id=taxon_id,
        reviewed=reviewed,
        max_results=max_results,
    )
    return df.to_json(orient="records")


def frozen_snapshot_available(scope: str) -> bool:
    """True if the frozen snapshot for this scope has been built (SCRIPT/fetch_reference_proteomes.py)."""
    path = _FROZEN_SNAPSHOTS.get(scope)
    return path is not None and path.exists()


@st.cache_data(ttl=None, show_spinner=False)
def load_frozen_snapshot(scope: str) -> str:
    """
    Load a frozen manuscript dataset and reshape it into the app's protein schema.

    scope : "reference"  - the 22 UniProt reference proteomes (the manuscript's primary
                            dataset, 128,121 proteins across 20 species).
            "swissprot"   - the frozen copy of the submitted query (taxonomy_id:5820 AND
                             reviewed:true), kept for direct comparison against "reference".

    A frozen file, unlike a live query, does not change between runs, so this is cached
    for the life of the process (ttl=None) rather than the 1 hour used for live fetches.
    """
    path = _FROZEN_SNAPSHOTS.get(scope)
    if path is None:
        raise RuntimeError(f"unknown frozen scope {scope!r}; expected one of {sorted(_FROZEN_SNAPSHOTS)}")
    if not path.exists():
        raise RuntimeError(
            f"Frozen snapshot not found at {path}. Run "
            "`python3 SCRIPT/fetch_reference_proteomes.py` to build it (~12 min, one-off)."
        )
    snapshot = load_snapshot(path)
    df = _snapshot_to_app_schema(snapshot)
    return df.to_json(orient="records")


def _snapshot_to_app_schema(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape a domain_dataset.load_snapshot() frame into the schema fetch_proteins._normalise()
    produces, so the rest of the app (domain_analysis facade, app_tabs, viz_*) is agnostic to
    which loader populated the table.

    The frozen snapshot's UniProt fields (accession, reviewed, length, ft_domain, xref_pfam,
    xref_interpro) are the same fields the live query retrieves, so the same parsers apply
    directly. Signal peptide, GPI anchor, GO terms and keywords were not part of the reference-
    proteome retrieval fields and are set to empty defaults; they remain available in live mode.
    """
    out = pd.DataFrame(
        {
            "accession": snapshot["accession"],
            "protein_name": snapshot.get("protein_name", pd.Series("", index=snapshot.index)).fillna(""),
            "organism": snapshot["species"],
            "taxon_id": snapshot.get("organism_id", pd.Series("", index=snapshot.index)).astype(str),
            "length": pd.to_numeric(snapshot["length"], errors="coerce").fillna(0).astype(int),
            "reviewed": snapshot["reviewed"].astype(bool),
        }
    )
    out["domains"] = snapshot["ft_domain_text"].apply(parse_ft_domain)
    out["domain_names"] = out["domains"].apply(domain_names_from_positions)
    out["n_domains"] = out["domains"].apply(len)
    out["pfam_ids"] = snapshot["xref_pfam"].apply(parse_semicolon_ids)
    out["interpro_ids"] = snapshot["xref_interpro"].apply(parse_semicolon_ids)
    out["n_interpro"] = out["interpro_ids"].apply(len)
    out["has_signal_peptide"] = False
    out["has_gpi_anchor"] = False
    out["go_functions"] = [[] for _ in range(len(out))]
    out["keywords"] = [[] for _ in range(len(out))]
    return out.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fasta(taxon_id: str, reviewed: bool, max_fasta: int) -> str:
    return fetch_fasta_sequences(taxon_id=taxon_id, reviewed=reviewed, max_results=max_fasta)


@st.cache_data(ttl=7200, show_spinner=False)
def get_interpro_names(ipr_ids_json: str) -> str:
    """Fetch InterPro names and return as JSON string."""
    ipr_ids = json.loads(ipr_ids_json)
    name_map = build_interpro_name_map(ipr_ids[:80])  # cap to avoid slow load
    return json.dumps(name_map)


def json_to_df(json_str: str) -> pd.DataFrame:
    """Restore DataFrame from JSON, fixing list columns."""
    if not json_str:
        return pd.DataFrame()
    df = pd.read_json(io.StringIO(json_str), orient="records")
    for col in ["domains", "domain_names", "interpro_ids", "pfam_ids", "go_functions", "keywords"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    return df
