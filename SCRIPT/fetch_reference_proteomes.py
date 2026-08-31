#!/usr/bin/env python3
"""
Module: fetch_reference_proteomes
Purpose: Build a frozen, accession-keyed snapshot of all Plasmodium UniProt reference proteomes.
Author: Dr. Arli Aditya Parikesit, Dr. Arif Nur Muhammad Ansori, and Moch. Royhan Afnani.,M.Sc
Date: 2026

Rationale (response to reviewers):
    All reviewers noted that the Swiss-Prot reviewed set (764 proteins, 16 species)
    reflects curation coverage rather than genus content. This module retrieves the
    22 UniProt reference proteomes for taxon 5820 and writes a single frozen TSV plus a
    SHA-256 manifest, so that every reported number is recomputable from a fixed artefact
    rather than from a live query.

Parameters:
    --outdir      : destination directory for the frozen snapshot (default data/frozen)
    --taxon       : NCBI taxonomy identifier of the clade (default 5820, Plasmodium)
    --timeout     : per-request timeout in seconds (default 300)

References:
    UniProt Consortium (2025) Nucleic Acids Research 53(D1):D609-D617. doi:10.1093/nar/gkae1010
    Manni M. et al. (2021) Molecular Biology and Evolution 38(10):4647-4654. doi:10.1093/molbev/msab199
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROTEOMES_STREAM = "https://rest.uniprot.org/proteomes/stream"
UNIPROTKB_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

DEFAULT_TAXON = "5820"

# One identifier system is used downstream: Pfam accession is the primary key.
# Short names and InterPro accessions travel as attributes, never as keys.
PROTEIN_FIELDS = [
    "accession",
    "id",
    "reviewed",
    "protein_name",
    "organism_name",
    "organism_id",
    "length",
    "ft_domain",
    "xref_pfam",
    "xref_interpro",
    "xref_smart",
    "xref_prosite",
    "xref_panther",
    "xref_gene3d",
    "xref_supfam",
]

PROTEOME_FIELDS = [
    "upid",
    "organism",
    "organism_id",
    "protein_count",
    "busco",
    "cpd",
    "genome_assembly",
]

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 3.0


def _get(url: str, params: dict, timeout: int) -> requests.Response:
    """GET with exponential backoff. Raises on final failure rather than returning partial data."""
    delay = _RETRY_BASE_DELAY
    last: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            last = exc
            logger.warning("attempt %d/%d failed for %s: %s", attempt, _MAX_RETRIES, url, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"exhausted retries for {url}") from last


def fetch_reference_proteome_index(taxon: str = DEFAULT_TAXON, timeout: int = 300) -> pd.DataFrame:
    """
    Retrieve the reference proteome index for a clade.

    Returns one row per reference proteome with BUSCO completeness and CPD
    (Complete Proteome Detector) status, which supply the per-species completeness
    measures the editor requested.
    """
    resp = _get(
        PROTEOMES_STREAM,
        {
            "query": f"(taxonomy_id:{taxon}) AND (reference:true)",
            "format": "tsv",
            "fields": ",".join(PROTEOME_FIELDS),
        },
        timeout,
    )
    df = pd.read_csv(io.StringIO(resp.text), sep="\t", dtype=str)
    df.columns = [
        "proteome_id",
        "organism_full",
        "organism_id",
        "protein_count",
        "busco",
        "cpd",
        "genome_assembly",
    ][: len(df.columns)]
    df["protein_count"] = pd.to_numeric(df["protein_count"], errors="coerce").astype("Int64")
    return df.sort_values("organism_full").reset_index(drop=True)


def fetch_proteome_proteins(proteome_id: str, timeout: int = 300) -> pd.DataFrame:
    """Retrieve every protein of one reference proteome with all domain cross-references."""
    resp = _get(
        UNIPROTKB_STREAM,
        {
            "query": f"(proteome:{proteome_id})",
            "format": "tsv",
            "fields": ",".join(PROTEIN_FIELDS),
        },
        timeout,
    )
    df = pd.read_csv(io.StringIO(resp.text), sep="\t", dtype=str)
    df["proteome_id"] = proteome_id
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_snapshot(outdir: Path, taxon: str = DEFAULT_TAXON, timeout: int = 300) -> dict:
    """
    Download every reference proteome for the clade and freeze it to disk.

    Writes three artefacts:
        proteome_index.tsv           - one row per proteome, with completeness metrics
        plasmodium_reference_proteomes.tsv.gz - one row per protein, accession-keyed
        manifest.json                - retrieval date, release, row counts, SHA-256 digests
    """
    outdir.mkdir(parents=True, exist_ok=True)

    index = fetch_reference_proteome_index(taxon=taxon, timeout=timeout)
    index_path = outdir / "proteome_index.tsv"
    index.to_csv(index_path, sep="\t", index=False)
    logger.info("reference proteomes: %d", len(index))

    # Per-proteome parts are cached so an interrupted download resumes instead of
    # restarting, and so a single failing proteome does not discard the whole snapshot.
    parts_dir = outdir / "parts"
    parts_dir.mkdir(exist_ok=True)

    frames = []
    for row in index.itertuples():
        part = parts_dir / f"{row.proteome_id}.tsv.gz"
        if part.exists() and part.stat().st_size > 0:
            with gzip.open(part, "rt", encoding="utf-8") as fh:
                frames.append(pd.read_csv(fh, sep="\t", dtype=str, low_memory=False))
            logger.info("%-12s cached", row.proteome_id)
            continue
        t0 = time.time()
        frame = fetch_proteome_proteins(row.proteome_id, timeout=timeout)
        with gzip.open(part, "wt", encoding="utf-8") as fh:
            frame.to_csv(fh, sep="\t", index=False)
        frames.append(frame)
        logger.info(
            "%-12s %-45s %6d proteins in %5.1f s",
            row.proteome_id,
            row.organism_full[:45],
            len(frame),
            time.time() - t0,
        )

    proteins = pd.concat(frames, ignore_index=True)

    # A protein can belong to more than one reference proteome (for example the two
    # P. falciparum isolates). Keep every proteome membership but flag duplicates so
    # downstream counts can choose between protein-level and proteome-level denominators.
    proteins["is_duplicate_accession"] = proteins.duplicated(subset=["Entry"], keep="first")

    proteins_path = outdir / "plasmodium_reference_proteomes.tsv.gz"
    with gzip.open(proteins_path, "wt", encoding="utf-8") as fh:
        proteins.to_csv(fh, sep="\t", index=False)

    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taxon": taxon,
        "source": "UniProt REST API (rest.uniprot.org)",
        "n_reference_proteomes": int(len(index)),
        "n_protein_rows": int(len(proteins)),
        "n_unique_accessions": int(proteins["Entry"].nunique()),
        "n_reviewed": int((proteins["Reviewed"] == "reviewed").sum()),
        "fields": PROTEIN_FIELDS,
        "files": {
            index_path.name: {"sha256": _sha256(index_path), "bytes": index_path.stat().st_size},
            proteins_path.name: {
                "sha256": _sha256(proteins_path),
                "bytes": proteins_path.stat().st_size,
            },
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("data/frozen"))
    parser.add_argument("--taxon", default=DEFAULT_TAXON)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = build_snapshot(args.outdir, taxon=args.taxon, timeout=args.timeout)
    print(json.dumps({k: v for k, v in manifest.items() if k != "fields"}, indent=2))


if __name__ == "__main__":
    main()
