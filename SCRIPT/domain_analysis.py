#!/usr/bin/env python3
"""
Module: Domain occurrence and avoidance analysis for Plasmodium proteins.
Purpose: Public facade over genus_analysis.py for the Streamlit app (app.py, app_tabs.py).
Author: Dr. Arli Aditya Parikesit
Date: 2026

Rationale (response to reviewers):
    Reviewer 4 identified two independent co-occurrence implementations reading two
    identifier systems: this module previously built its occurrence table from ft_domain
    note strings while network_builder.py built its graph from Pfam accessions, so a
    headline count quoted from one path did not match the table computed by the other.
    The avoidance test here was a uniform binomial null that ignored per-species
    annotation depth, and the co-occurrence lift denominator counted only proteins
    retaining one of the top sixty domains rather than the analysed set.

    genus_analysis.py is now the single implementation of every one of these statistics,
    built on Pfam accessions throughout and carrying the corrected avoidance null,
    corrected co-occurrence denominator, and obligate/same-clan pair flags. This module
    no longer computes anything itself: every function here builds the small long-format
    domain table genus_analysis expects from the app's protein DataFrame (accession,
    organism, pfam_ids, reviewed, length), calls genus_analysis, and renames the result
    back to the column names app_tabs.py and the viz_*.py chart builders already expect,
    so those modules did not need to change.

    get_species_summary, compute_interpro_occurrence and get_species_exclusive_domains
    are not part of this consolidation: the first two operate on quantities (protein/
    length counts, InterPro accessions) that were never duplicated elsewhere, and the
    third is now a thin view over the corrected compute_domain_occurrence below rather
    than an independent implementation.

References:
    Parikesit et al. (2011) Genes 2(4):912-924.
    Parikesit et al. (2018) JBI 14(2):185-190. doi:10.47349/jbi/14022018/185
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

import genus_analysis as _ga
from domain_dataset import CLADE_BY_SPECIES, species_coverage_table as _species_coverage_table

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PFAM_CLANS_PATH = _REPO_ROOT / "data" / "frozen" / "Pfam-A.clans.tsv.gz"

_CLAN_COLUMNS = ["domain_accession", "clan_accession", "clan_id", "domain_name", "description"]


@functools.lru_cache(maxsize=1)
def _load_clans() -> pd.DataFrame:
    """
    Load the Pfam clan table (short names + clan membership) if the frozen copy is present.

    Falls back to an empty frame with the right columns when it is not, so the app still
    runs (labelled by bare accession) without the manuscript's data/frozen/ deposit. Cached
    for the life of the process: this file does not change while the app is running.
    """
    if not _PFAM_CLANS_PATH.exists():
        return pd.DataFrame(columns=_CLAN_COLUMNS)
    try:
        return _ga.load_pfam_clans(_PFAM_CLANS_PATH)
    except Exception:
        return pd.DataFrame(columns=_CLAN_COLUMNS)


def _display_label(accession: str, name) -> str:
    """'{short name} ({accession})', falling back to the bare accession when the name is unknown."""
    if isinstance(name, str) and name.strip():
        return f"{name} ({accession})"
    return accession


def _to_pipeline_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt the app's protein schema (organism, pfam_ids, ...) to the species/clade/reviewed
    columns genus_analysis and domain_dataset.species_coverage_table expect.
    """
    out = df.copy()
    out["species"] = out["organism"]
    out["clade"] = out["species"].map(CLADE_BY_SPECIES).fillna("unassigned")
    if "reviewed" not in out.columns:
        out["reviewed"] = False
    return out


_DOMAIN_TABLE_COLUMNS = [
    "accession", "species", "clade", "reviewed", "length",
    "domain_accession", "domain_name", "copy_number",
]


def _pfam_domain_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the app's pfam_ids lists into the long-format, accession-keyed domain table
    genus_analysis consumes (one row per protein x Pfam family it carries).

    A cross-reference list carries each family once per protein regardless of how UniProt's
    curated DOMAIN feature enumerates repeated instances, so copy_number is 1 by
    construction here; the instance-count question only arises for the ft_domain layer,
    which this app no longer uses for family-level statistics.
    """
    pf = _to_pipeline_frame(df)
    records = []
    for row in pf.itertuples():
        ids = getattr(row, "pfam_ids", None) or []
        for acc in dict.fromkeys(ids):
            records.append(
                {
                    "accession": row.accession,
                    "species": row.species,
                    "clade": row.clade,
                    "reviewed": row.reviewed,
                    "length": row.length,
                    "domain_accession": acc,
                    "domain_name": None,
                    "copy_number": 1,
                }
            )
    return pd.DataFrame.from_records(records, columns=_DOMAIN_TABLE_COLUMNS)


def _coverage_table(df: pd.DataFrame, domains: pd.DataFrame) -> pd.DataFrame:
    """Per-species annotation depth, the input the depth-conditioned avoidance null needs."""
    return _species_coverage_table(_to_pipeline_frame(df), domains)


# ----------------------------------------------------------------------------------
# Species summary (unchanged: not one of the statistics the reviewers found duplicated)
# ----------------------------------------------------------------------------------

def get_species_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise protein and domain counts per Plasmodium species.

    Returns
    -------
    pd.DataFrame
        Columns: organism, n_proteins, n_with_domains, pct_with_domains,
                 unique_domains, total_domains, avg_length, max_length.
    """
    def _unique_domains(series):
        return len({d for lst in series for d in lst})

    agg = df.groupby("organism", as_index=False).agg(
        n_proteins=("accession", "count"),
        n_with_domains=("n_domains", lambda x: (x > 0).sum()),
        total_domains=("n_domains", "sum"),
        avg_length=("length", "mean"),
        max_length=("length", "max"),
    )
    unique_counts = (
        df.groupby("organism")["domain_names"]
        .apply(_unique_domains)
        .reset_index()
        .rename(columns={"domain_names": "unique_domains"})
    )
    agg = agg.merge(unique_counts, on="organism", how="left")
    agg["avg_length"] = agg["avg_length"].round(1)
    agg["pct_with_domains"] = (
        agg["n_with_domains"] / agg["n_proteins"] * 100
    ).round(1)
    return agg.sort_values("n_proteins", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Occurrence (delegates to genus_analysis.compute_domain_occurrence)
# ----------------------------------------------------------------------------------

_OCCURRENCE_COLUMNS = [
    "domain_accession", "domain_name", "count", "species_count", "pct_proteins", "species",
    "n_reviewed_carriers", "n_unreviewed_carriers", "reviewed_fraction",
]


def compute_domain_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Protein-level occurrence per Pfam family, keyed by accession.

    domain_name carries a display label ("short name (accession)", or the bare accession
    when the name is unknown) so every chart and table names each family unambiguously by
    accession rather than by a short name alone, which is what let the submitted manuscript
    report two unrelated Pfam families (TSP_N, TSP_C) as an N-/C-terminal pair.

    Also reports the reviewed vs. unreviewed carrier split, so the curation-bias question
    Reviewer 3 raised for TSP-like families is answerable directly from this table.

    Returns
    -------
    pd.DataFrame sorted by count descending, columns as _OCCURRENCE_COLUMNS.
    """
    domains = _pfam_domain_table(df)
    if domains.empty:
        return pd.DataFrame(columns=_OCCURRENCE_COLUMNS)

    occ = _ga.compute_domain_occurrence(domains, n_proteins_total=len(df))
    occ = _ga.annotate_with_pfam(occ, _load_clans())
    occ["domain_name"] = [
        _display_label(a, n) for a, n in zip(occ["domain_accession"], occ["domain_name"])
    ]
    occ = occ.rename(
        columns={
            "n_proteins": "count",
            "n_species": "species_count",
            "pct_of_proteome": "pct_proteins",
            "species_list": "species",
        }
    )
    return occ[_OCCURRENCE_COLUMNS].sort_values("count", ascending=False).reset_index(drop=True)


def compute_species_domain_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a species x domain presence matrix, keyed by Pfam accession.

    Cell value = number of proteins in that species carrying that family. Column labels
    use the same "{name} ({accession})" convention as compute_domain_occurrence.

    Returns
    -------
    pd.DataFrame
        Rows = species, columns = domain labels.
    """
    domains = _pfam_domain_table(df)
    if domains.empty:
        return pd.DataFrame()

    matrix = (
        domains.groupby(["species", "domain_accession"])["accession"]
        .nunique()
        .unstack(fill_value=0)
    )
    clans = _load_clans()
    name_map = clans.set_index("domain_accession")["domain_name"].to_dict() if not clans.empty else {}
    matrix.columns = [_display_label(acc, name_map.get(acc)) for acc in matrix.columns]
    return matrix


def get_species_exclusive_domains(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Find Pfam families carried by proteins of exactly one Plasmodium species.

    Returns
    -------
    dict
        {species_name: [exclusive_domain_label, ...]}, labelled "{name} ({accession})".
    """
    occurrence = compute_domain_occurrence(df)
    if occurrence.empty:
        return {}

    exclusive = occurrence[occurrence["species_count"] == 1].copy()
    if exclusive.empty:
        return {}

    domains = _pfam_domain_table(df)
    single_species = domains.groupby("domain_accession")["species"].first()

    result: Dict[str, List[str]] = {}
    for _, row in exclusive.iterrows():
        sp = single_species.get(row["domain_accession"])
        if sp is None:
            continue
        result.setdefault(sp, []).append(row["domain_name"])
    return result


# ----------------------------------------------------------------------------------
# Avoidance (delegates to genus_analysis.compute_domain_avoidance)
# ----------------------------------------------------------------------------------

_AVOIDANCE_COLUMNS = [
    "domain_accession", "domain_name", "n_species_present", "n_species_absent",
    "avoidance_score", "min_binom_pvalue", "avoidance_pvalue_adj",
    "expected_species_absent", "excess_absence",
    "n_clades_present", "n_clades_absent", "clade_avoidance_score",
    "species_with", "species_without",
]


def compute_domain_avoidance(df: pd.DataFrame, min_carriers: int = 2) -> pd.DataFrame:
    """
    Domain avoidance under a null conditioned on per-species annotation depth.

    Replaces the uniform-probability binomial null that Reviewer 2 and the editor
    rejected: for family d and species s, the null probability that s carries no copy of
    d is (1 - p)^n_s, where n_s is the number of *annotated* proteins in species s and p is
    d's per-protein frequency across every other species (leave-one-species-out, to avoid
    testing a species against a frequency it helped define). A species with few annotated
    proteins therefore contributes little evidence of absence, rather than counting as a
    full unit against a genus-wide average the way the previous binomial null did.

    min_carriers filters by total carrier *proteins* across the genus (genus_analysis's
    convention), not by minimum species count as the previous min_species_presence did;
    the parameter is renamed to make that shift in meaning explicit. Domains carried by
    every species are still reported at avoidance_score 0 rather than dropped, since the
    corrected finding for several manuscript-cited families (LCCL, CLAG, EBA-175, RAP) is
    exactly that they are not avoided, and that needs to remain visible in this table.

    clade_avoidance_score collapses closely related species into one host-defined clade
    (Laverania, primate, rodent, avian) before scoring, since counting several rodent
    species as independent absences overstates the evidence for what is really one
    lineage-level loss.

    Returns
    -------
    pd.DataFrame sorted by excess_absence descending, columns as _AVOIDANCE_COLUMNS.
    """
    if df["organism"].nunique() < 2:
        warnings.warn(
            "compute_domain_avoidance: df contains only one species; the depth-conditioned "
            "null cannot be calibrated against other species and avoidance scores will be "
            "unreliable.",
            UserWarning,
            stacklevel=2,
        )

    domains = _pfam_domain_table(df)
    if domains.empty:
        return pd.DataFrame(columns=_AVOIDANCE_COLUMNS)

    coverage = _coverage_table(df, domains)
    if coverage["species"].nunique() < 2:
        return pd.DataFrame(columns=_AVOIDANCE_COLUMNS)

    result = _ga.compute_domain_avoidance(domains, coverage, min_carriers=min_carriers)
    if result.empty:
        return pd.DataFrame(columns=_AVOIDANCE_COLUMNS)

    result = _ga.annotate_with_pfam(result, _load_clans())
    result["domain_name"] = [
        _display_label(a, n) for a, n in zip(result["domain_accession"], result["domain_name"])
    ]
    result = result.rename(
        columns={
            "poisson_binomial_p": "min_binom_pvalue",
            "q_value": "avoidance_pvalue_adj",
            "present_species": "species_with",
            "absent_species": "species_without",
        }
    )
    return result[_AVOIDANCE_COLUMNS].reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Co-occurrence (delegates to genus_analysis.compute_cooccurrence)
# ----------------------------------------------------------------------------------

_COOCCURRENCE_COLUMNS = [
    "domain_A", "domain_B", "name_A", "name_B",
    "n_AB", "n_A", "n_B", "n_neither", "N",
    "jaccard", "lift", "pmi", "fisher_pvalue", "fisher_pvalue_adj",
    "obligate_pair", "same_clan", "headline_eligible",
]


def compute_domain_cooccurrence_stats(
    df: pd.DataFrame,
    min_pair_count: int = 5,
    max_domains: int | None = 60,
) -> pd.DataFrame:
    """
    Pairwise Pfam co-occurrence statistics across all proteins.

    Three corrections relative to the previous implementation, per Reviewer 4:

    1. N is the number of proteins carrying at least one Pfam family in the analysed set,
       not the number retaining one of the top max_domains families; the previous
       denominator inflated every lift value and was undocumented.
    2. min_pair_count defaults to 5 rather than 2: at min_pair_count=2 the previous top
       ranking was seven pairs tied at the threshold itself, each supported by two
       proteins.
    3. obligate_pair (n_AB == n_A == n_B) and same_clan pairs are flagged via
       headline_eligible, since these are the defining architecture of a single protein
       family rather than evidence of domain combination.

    Returns
    -------
    pd.DataFrame sorted by lift descending, columns as _COOCCURRENCE_COLUMNS.
    """
    domains = _pfam_domain_table(df)
    if domains.empty:
        return pd.DataFrame(columns=_COOCCURRENCE_COLUMNS)

    result = _ga.compute_cooccurrence(
        domains, _load_clans(), min_pair_count=min_pair_count, max_families=max_domains
    )
    if result.empty:
        return pd.DataFrame(columns=_COOCCURRENCE_COLUMNS)

    result["n_neither"] = result["N"] - result["n_A"] - result["n_B"] + result["n_AB"]
    result = result.rename(
        columns={
            "domain_a": "domain_A",
            "domain_b": "domain_B",
            "name_a": "name_A",
            "name_b": "name_B",
            "pmi_log2": "pmi",
            "fisher_p": "fisher_pvalue",
            "q_value": "fisher_pvalue_adj",
        }
    )
    return result[_COOCCURRENCE_COLUMNS].sort_values("lift", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------
# InterPro (unchanged: already accession-keyed, not part of the duplication the
# reviewers identified)
# ----------------------------------------------------------------------------------

def compute_interpro_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count InterPro entry occurrence across all proteins.
    Mirrors compute_domain_occurrence() but for InterPro cross-references.

    Returns
    -------
    pd.DataFrame
        Columns: interpro_id, count, species_count, pct_proteins, species.
    """
    records = []
    for _, row in df.iterrows():
        for ipr in row.get("interpro_ids", []):
            records.append({"interpro_id": ipr, "accession": row["accession"], "organism": row["organism"]})

    if not records:
        return pd.DataFrame(columns=["interpro_id", "count", "species_count", "pct_proteins", "species"])

    exp = pd.DataFrame(records)
    total = len(df)

    agg = (
        exp.groupby("interpro_id")
        .agg(
            count=("accession", "nunique"),
            species_count=("organism", "nunique"),
            species=("organism", lambda x: sorted(set(x))),
        )
        .reset_index()
    )
    agg["pct_proteins"] = (agg["count"] / total * 100).round(2)
    return agg.sort_values("count", ascending=False).reset_index(drop=True)
