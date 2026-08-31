#!/usr/bin/env python3
"""
Module: domain_dataset
Purpose: Normalise a frozen UniProt snapshot into one accession-keyed domain table.
Author: Dr. Arli Aditya Parikesit, Dr. Arif Nur Muhammad Ansori, and Moch. Royhan Afnani.,M.Sc
Date: 2026

Rationale (response to reviewers):
    Reviewer showed that the previous pipeline mixed two identifier systems: the
    occurrence table was built from ft_domain note strings while the network and pan-core
    analyses were built from xref_pfam accessions, so the headline count of 361 belonged
    to a different analysis than the table it labelled. Reviewer 4 also showed that
    UniProt enumerates repeated DOMAIN features by appending an integer to the note,
    so "6-Cys 4" and "6-Cys 7" were counted as two distinct domain families, and that
    note-less DOMAIN features all collapsed into a single "Unknown domain" node.

    This module resolves all three defects. Every downstream analysis consumes the long
    table produced here, whose primary key is (accession, source_db, domain_accession).
    Pfam accession is the canonical family key; short names and InterPro accessions are
    carried as attributes and are never used for grouping.

Parameters:
    snapshot : path to data/frozen/plasmodium_reference_proteomes.tsv.gz
    layer    : "pfam" (default, canonical) or "ft_domain" (curated features, for comparison)

References:
    Blum M. et al. (2025) Nucleic Acids Research 53(D1):D444-D456. doi:10.1093/nar/gkae1082
    UniProt Consortium (2025) Nucleic Acids Research 53(D1):D609-D617. doi:10.1093/nar/gkae1010
    Verified against PubMed PMID 39565202 (InterPro 2025).
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# UniProt appends an incrementing integer to the /note of repeated DOMAIN features
# within one protein ("6-Cys 4", "ABC transporter 2"). The integer is an instance
# counter, not part of the family name, and must be stripped before any set operation.
_INSTANCE_SUFFIX_RE = re.compile(r"\s+\d+$")

# "DOMAIN 25..180; /note="EGF-like"; /evidence="ECO:..."" - note is optional.
_FT_DOMAIN_RE = re.compile(
    r"DOMAIN\s+(?P<start><?\d+)\.\.(?P<end>>?\d+)(?:\s*;\s*/note=\"(?P<note>[^\"]*)\")?",
    re.IGNORECASE,
)

# UniProt TSV headers are human-readable and contain spaces and brackets, which do not
# survive itertuples. Every column is renamed once, on load, so that all downstream code
# refers to one stable set of names and no analysis depends on a UniProt header string.
COLUMN_RENAMES = {
    "Entry": "accession",
    "Entry Name": "entry_name",
    "Reviewed": "reviewed_raw",
    "Protein names": "protein_name",
    "Organism": "organism_full",
    "Organism (ID)": "organism_id",
    "Length": "length_raw",
    "Domain [FT]": "ft_domain_text",
    "Pfam": "xref_pfam",
    "InterPro": "xref_interpro",
    "SMART": "xref_smart",
    "PROSITE": "xref_prosite",
    "PANTHER": "xref_panther",
    "Gene3D": "xref_gene3d",
    "SUPFAM": "xref_supfam",
}

# Member-database cross-reference columns, semicolon-delimited accession lists.
XREF_COLUMNS = {
    "Pfam": "xref_pfam",
    "InterPro": "xref_interpro",
    "SMART": "xref_smart",
    "PROSITE": "xref_prosite",
    "PANTHER": "xref_panther",
    "Gene3D": "xref_gene3d",
    "SUPFAM": "xref_supfam",
}

# Host-defined clades, used for the phylogenetic representativeness statement and as the
# grouping factor of the phylogeny-aware avoidance null. Assignment follows the
# host association used throughout the Plasmodium systematics literature.
CLADE_BY_SPECIES = {
    "Plasmodium falciparum": "Laverania (human)",
    "Plasmodium reichenowi": "Laverania (ape)",
    "Plasmodium gaboni": "Laverania (ape)",
    "Plasmodium vivax": "primate (human)",
    "Plasmodium malariae": "primate (human)",
    "Plasmodium ovale": "primate (human)",
    "Plasmodium ovale wallikeri": "primate (human)",
    "Plasmodium knowlesi": "primate (simian, zoonotic)",
    "Plasmodium cynomolgi": "primate (simian)",
    "Plasmodium inui": "primate (simian)",
    "Plasmodium coatneyi": "primate (simian)",
    "Plasmodium fragile": "primate (simian)",
    "Plasmodium gonderi": "primate (simian)",
    "Plasmodium brasilianum": "primate (simian)",
    "Plasmodium simium": "primate (simian)",
    "Plasmodium berghei": "rodent",
    "Plasmodium yoelii": "rodent",
    "Plasmodium chabaudi": "rodent",
    "Plasmodium vinckei": "rodent",
    "Plasmodium vinckei petteri": "rodent",
    "Plasmodium gallinaceum": "avian",
    "Plasmodium relictum": "avian",
    "Plasmodium lophurae": "avian",
}

_ORGANISM_QUALIFIER_RE = re.compile(r"\s*[\(\[].*", re.DOTALL)
_STRAIN_TOKENS_RE = re.compile(
    r"\s+(?:strain|isolate)\b.*|\s+(?:17X|Salvador\s*I|San\s+Antonio\s*1|ANKA|Anka|3D7|NF54|H|B)\b\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetSummary:
    """Counts recomputed from the frozen snapshot, for the manuscript's dataset section."""

    n_protein_rows: int
    n_unique_accessions: int
    n_species: int
    n_proteomes: int
    n_reviewed: int
    pfam_protein_coverage: float
    interpro_protein_coverage: float
    ft_domain_protein_coverage: float

    def to_dict(self) -> dict:
        return {
            "n_protein_rows": self.n_protein_rows,
            "n_unique_accessions": self.n_unique_accessions,
            "n_species": self.n_species,
            "n_proteomes": self.n_proteomes,
            "n_reviewed": self.n_reviewed,
            "pfam_protein_coverage_pct": round(self.pfam_protein_coverage * 100, 2),
            "interpro_protein_coverage_pct": round(self.interpro_protein_coverage * 100, 2),
            "ft_domain_protein_coverage_pct": round(self.ft_domain_protein_coverage * 100, 2),
        }


def normalise_species(organism_full: str) -> str:
    """
    Reduce a UniProt organism name to a binomial (or trinomial for named subspecies).

    "Plasmodium yoelii yoelii" and "Plasmodium yoelii 17X" both resolve to
    "Plasmodium yoelii", so that the two reference proteomes of one species are not
    counted as two species in the pan-core denominator.
    """
    name = _ORGANISM_QUALIFIER_RE.sub("", str(organism_full)).strip()
    name = _STRAIN_TOKENS_RE.sub("", name).strip()
    tokens = name.split()
    if len(tokens) >= 3 and tokens[2].islower() and tokens[2].isalpha():
        # Trinomials such as "Plasmodium ovale wallikeri" and "Plasmodium vinckei petteri"
        # are retained only where the subspecies epithet differs from the species epithet.
        if tokens[2] != tokens[1]:
            return " ".join(tokens[:3])
    return " ".join(tokens[:2])


def strip_instance_suffix(note: str) -> str:
    """Remove the UniProt repeat-instance counter from a DOMAIN note."""
    return _INSTANCE_SUFFIX_RE.sub("", str(note)).strip()


def _split_xref(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [tok.strip() for tok in value.split(";") if tok.strip()]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename UniProt TSV headers to stable identifiers and derive species and clade.

    Downstream code never touches a UniProt header string, so a header change in a future
    UniProt release breaks here, once, with a clear failure, rather than silently
    producing an empty domain table.
    """
    df = df.rename(columns={k: v for k, v in COLUMN_RENAMES.items() if k in df.columns}).copy()

    missing = {"accession", "organism_full"} - set(df.columns)
    if missing:
        raise KeyError(f"snapshot is missing required columns after renaming: {sorted(missing)}")

    for column in list(XREF_COLUMNS.values()) + ["ft_domain_text"]:
        if column not in df.columns:
            df[column] = pd.NA

    df["length"] = pd.to_numeric(df.get("length_raw"), errors="coerce")
    df["species"] = df["organism_full"].map(normalise_species)
    df["clade"] = df["species"].map(CLADE_BY_SPECIES).fillna("unassigned")
    df["reviewed"] = df.get("reviewed_raw", pd.Series(dtype=str)).eq("reviewed")
    if "proteome_id" not in df.columns:
        df["proteome_id"] = pd.NA
    return df


def load_snapshot(path: Path) -> pd.DataFrame:
    """Load a frozen snapshot and normalise it."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str, low_memory=False)
    return normalise_columns(df)


def build_domain_table(df: pd.DataFrame, layer: str = "pfam") -> pd.DataFrame:
    """
    Expand the protein table into one row per (protein, domain family) with copy number.

    layer="pfam"      : canonical. Keys are Pfam accessions from xref_pfam.
    layer="ft_domain" : UniProt curated DOMAIN features. Keys are instance-stripped notes.
                        Retained only for the layer comparison requested by the reviewers;
                        never mixed with the Pfam layer in a single analysis.

    Copy number is recorded rather than being allowed to inflate the family count, which
    is the defect Reviewer 4 identified in the previous ft_domain co-occurrence path.
    """
    if layer in XREF_COLUMNS:
        return _build_from_xref(df, column=XREF_COLUMNS[layer], source_db=layer)
    if layer == "pfam":
        return _build_from_xref(df, column="xref_pfam", source_db="Pfam")
    if layer == "ft_domain":
        return _build_from_ft_domain(df)
    raise ValueError(f"unknown layer: {layer!r}")


def _base_record(row) -> dict:
    return {
        "accession": row.accession,
        "species": row.species,
        "clade": row.clade,
        "organism_id": getattr(row, "organism_id", None),
        "proteome_id": getattr(row, "proteome_id", None),
        "reviewed": row.reviewed,
        "length": row.length,
    }


def _build_from_xref(df: pd.DataFrame, column: str, source_db: str) -> pd.DataFrame:
    records = []
    for row in df.itertuples():
        accessions = _split_xref(getattr(row, column, None))
        if not accessions:
            continue
        # A cross-reference list carries each family once per protein; copy number is
        # therefore 1 by construction and multi-copy families are handled through the
        # ft_domain layer, where instance counts are actually recorded.
        for acc in dict.fromkeys(accessions):
            records.append(
                _base_record(row)
                | {
                    "source_db": source_db,
                    "domain_accession": acc,
                    "domain_name": None,
                    "copy_number": 1,
                }
            )
    columns = [
        "accession", "species", "clade", "organism_id", "proteome_id", "reviewed",
        "length", "source_db", "domain_accession", "domain_name", "copy_number",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _build_from_ft_domain(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    n_unnamed = 0
    for row in df.itertuples():
        text = getattr(row, "ft_domain_text", None)
        if not isinstance(text, str) or "DOMAIN" not in text:
            continue
        counts: dict[str, int] = {}
        for match in _FT_DOMAIN_RE.finditer(text):
            note = match.group("note")
            if not note:
                # A DOMAIN feature without a /note carries no family identity. Collapsing
                # these into one "Unknown domain" key merged unrelated domains in the
                # previous pipeline, so they are counted separately and excluded from
                # family-level analyses instead.
                n_unnamed += 1
                continue
            key = strip_instance_suffix(note)
            counts[key] = counts.get(key, 0) + 1
        for name, copies in counts.items():
            records.append(
                _base_record(row)
                | {
                    "source_db": "UniProt ft_domain",
                    "domain_accession": None,
                    "domain_name": name,
                    "copy_number": copies,
                }
            )
    if n_unnamed:
        logger.info("ft_domain features without /note excluded from family analysis: %d", n_unnamed)
    columns = [
        "accession", "species", "clade", "organism_id", "proteome_id", "reviewed",
        "length", "source_db", "domain_accession", "domain_name", "copy_number",
    ]
    out = pd.DataFrame.from_records(records, columns=columns)
    out.attrs["n_unnamed_features"] = n_unnamed
    return out


def summarise_dataset(df: pd.DataFrame) -> DatasetSummary:
    """Protein-level annotation coverage, with the denominator stated explicitly."""
    n = len(df)
    has_pfam = df["xref_pfam"].notna() & df["xref_pfam"].astype(str).str.strip().ne("")
    has_ipr = df["xref_interpro"].notna() & df["xref_interpro"].astype(str).str.strip().ne("")
    has_ft = df["ft_domain_text"].astype(str).str.contains("DOMAIN", na=False)
    return DatasetSummary(
        n_protein_rows=n,
        n_unique_accessions=int(df["accession"].nunique()),
        n_species=int(df["species"].nunique()),
        n_proteomes=int(df["proteome_id"].nunique()),
        n_reviewed=int(df["reviewed"].sum()),
        pfam_protein_coverage=float(has_pfam.mean()),
        interpro_protein_coverage=float(has_ipr.mean()),
        ft_domain_protein_coverage=float(has_ft.mean()),
    )


def species_coverage_table(df: pd.DataFrame, domains: pd.DataFrame) -> pd.DataFrame:
    """
    Per-species annotation depth: the table Reviewer 1 asked for.

    Columns: proteins, annotated proteins, annotation coverage, total domain instances,
    unique families, domains per protein, unannotated fraction.
    """
    per_species = df.groupby("species").agg(
        n_proteins=("accession", "nunique"),
        n_reviewed=("reviewed", "sum"),
        mean_length=("length", "mean"),
    )
    if domains.empty:
        per_species["n_annotated_proteins"] = 0
        per_species["n_domain_instances"] = 0
        per_species["n_unique_families"] = 0
    else:
        dom = domains.groupby("species").agg(
            n_annotated_proteins=("accession", "nunique"),
            n_domain_instances=("copy_number", "sum"),
            n_unique_families=("domain_accession", "nunique"),
        )
        if dom["n_unique_families"].sum() == 0:
            dom["n_unique_families"] = domains.groupby("species")["domain_name"].nunique()
        per_species = per_species.join(dom, how="left").fillna(0)

    per_species["annotation_coverage"] = (
        per_species["n_annotated_proteins"] / per_species["n_proteins"]
    ).round(4)
    per_species["unannotated_fraction"] = (1 - per_species["annotation_coverage"]).round(4)
    per_species["domains_per_protein"] = (
        per_species["n_domain_instances"] / per_species["n_proteins"]
    ).round(3)
    # A species with no annotated protein has no defined domains-per-annotated-protein
    # ratio. numpy NaN is used rather than pandas NA so the column stays a float dtype
    # and remains roundable and writable to CSV.
    per_species["domains_per_annotated_protein"] = (
        per_species["n_domain_instances"] / per_species["n_annotated_proteins"].replace(0, np.nan)
    ).astype(float).round(3)
    per_species["mean_length"] = per_species["mean_length"].round(1)
    per_species["clade"] = per_species.index.map(lambda s: CLADE_BY_SPECIES.get(s, "unassigned"))
    for col in ("n_annotated_proteins", "n_domain_instances", "n_unique_families", "n_reviewed"):
        per_species[col] = per_species[col].astype(int)
    return per_species.sort_values("n_proteins", ascending=False).reset_index()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/frozen/plasmodium_reference_proteomes.tsv.gz"))
    parser.add_argument("--outdir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_snapshot(args.snapshot)
    summary = summarise_dataset(df)
    domains = build_domain_table(df, layer="pfam")
    coverage = species_coverage_table(df, domains)

    args.outdir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.outdir / "table1_species_coverage.csv", index=False)
    (args.outdir / "dataset_summary.json").write_text(json.dumps(summary.to_dict(), indent=2))
    print(json.dumps(summary.to_dict(), indent=2))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
