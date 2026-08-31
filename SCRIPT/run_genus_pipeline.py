#!/usr/bin/env python3
"""
Module: run_genus_pipeline
Purpose: Run every reported analysis from the frozen snapshot and write accession-keyed tables.
Author: Dr. Arli Aditya Parikesit, Dr. Arif Nur Muhammad Ansori, and Moch. Royhan Afnani.,M.Sc
Date: 2026

Rationale (response to reviewer):
    The editor asked that all figures and reported numbers be regenerated directly from
    the revised pipeline, and that one identifier system be applied across occurrence,
    avoidance, co-occurrence, network and pan-core analyses. This script is the single
    entry point that produces every table behind the manuscript. The figure script reads
    only its outputs, so no reported value can come from a source other than the frozen
    snapshot.

    Both dataset scopes run through the same code path:
      scope="reference" : the 22 Plasmodium reference proteomes (primary analysis)
      scope="swissprot" : the reviewed subset (the submitted manuscript's dataset)
    Running both is the sensitivity analysis that answers whether the original
    conclusions were properties of the genus or of Swiss-Prot curation coverage.

Parameters:
    --snapshot   : frozen UniProt snapshot (default data/frozen/...)
    --clans      : Pfam-A.clans.tsv.gz
    --outdir     : results/tables
    --min-pair   : minimum co-occurrence support (default 5; 2 for the sparse subset)
    --bootstrap  : bootstrap replicates for the degree-distribution fit (default 1000)

References:
    Clauset A., Shalizi C.R., Newman M.E.J. (2009) SIAM Review 51(4):661-703. doi:10.1137/070710111
    Tettelin H. et al. (2005) PNAS 102(39):13950-13955. doi:10.1073/pnas.0506758102
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from domain_dataset import (
    CLADE_BY_SPECIES,
    build_domain_table,
    load_snapshot,
    species_coverage_table,
    summarise_dataset,
)
from genus_analysis import (
    annotate_with_pfam,
    build_network,
    compute_cooccurrence,
    compute_domain_avoidance,
    compute_domain_occurrence,
    compute_pan_core,
    describe_network,
    fit_degree_distribution,
    load_pfam_clans,
)

logger = logging.getLogger(__name__)


def run_scope(
    df: pd.DataFrame,
    clans: pd.DataFrame,
    scope: str,
    outdir: Path,
    min_pair: int,
    bootstrap: int,
) -> dict:
    """Run the full analysis for one dataset scope and write its tables."""
    prefix = f"{scope}_"
    logger.info("scope=%s proteins=%d species=%d", scope, len(df), df["species"].nunique())

    summary = summarise_dataset(df)
    domains = build_domain_table(df, layer="pfam")
    coverage = species_coverage_table(df, domains)
    coverage.to_csv(outdir / f"{prefix}table1_species_coverage.csv", index=False)

    occurrence = compute_domain_occurrence(domains, n_proteins_total=len(df))
    occurrence = annotate_with_pfam(occurrence, clans)
    occurrence.to_csv(outdir / f"{prefix}occurrence.csv", index=False)

    # Species-by-family carrier matrix for the heatmap that Figure 3's caption promised
    # but the submitted figure did not contain. Written here so the figure script reads
    # only pipeline output and cannot introduce a number of its own.
    if not domains.empty:
        top_families = occurrence.head(25)["domain_accession"].tolist()
        matrix = (
            domains[domains["domain_accession"].isin(top_families)]
            .groupby(["domain_accession", "species"])["accession"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(index=top_families, fill_value=0)
        )
        matrix = matrix.reindex(columns=sorted(df["species"].unique()), fill_value=0)
        names = dict(zip(occurrence["domain_accession"], occurrence["domain_name"]))
        matrix.insert(0, "domain_name", [names.get(a) for a in matrix.index])
        matrix.to_csv(outdir / f"{prefix}species_domain_matrix.csv")

    avoidance = compute_domain_avoidance(domains, coverage)
    if not avoidance.empty:
        avoidance = annotate_with_pfam(avoidance, clans)
    avoidance.to_csv(outdir / f"{prefix}avoidance.csv", index=False)

    cooccurrence = compute_cooccurrence(domains, clans, min_pair_count=min_pair)
    cooccurrence.to_csv(outdir / f"{prefix}cooccurrence.csv", index=False)

    graph = build_network(cooccurrence) if not cooccurrence.empty else build_network(pd.DataFrame())
    topology = describe_network(graph)
    degrees = np.array([d for _, d in graph.degree()]) if graph.number_of_nodes() else np.array([])
    pd.DataFrame({"domain_accession": list(dict(graph.degree()).keys()), "degree": list(dict(graph.degree()).values())}).to_csv(
        outdir / f"{prefix}degree_sequence.csv", index=False
    )
    fits = fit_degree_distribution(degrees, bootstrap=bootstrap) if len(degrees) else {"error": "empty graph"}

    species = sorted(df["species"].unique())
    pan_core = compute_pan_core(domains, species)
    if not pan_core.empty:
        pan_core = annotate_with_pfam(pan_core, clans)
    pan_core.to_csv(outdir / f"{prefix}pan_core.csv", index=False)

    pan_core_counts = (
        pan_core["category"].value_counts().to_dict() if not pan_core.empty else {}
    )

    # Sensitivity analysis: repeat the pan-core partition after dropping the species whose
    # annotation depth falls in the lowest quartile, which is the check the reviewers asked
    # for on whether the partition is driven by shallow proteomes.
    depth_cut = coverage["n_annotated_proteins"].quantile(0.25)
    deep_species = coverage.loc[coverage["n_annotated_proteins"] >= depth_cut, "species"].tolist()
    pan_core_deep = compute_pan_core(domains, deep_species)
    pan_core_deep_counts = (
        pan_core_deep["category"].value_counts().to_dict() if not pan_core_deep.empty else {}
    )

    result = {
        "scope": scope,
        "dataset": summary.to_dict(),
        "n_species": len(species),
        "n_clades": int(df["clade"].nunique()),
        "clades": sorted(df["clade"].unique().tolist()),
        "n_pfam_families": int(domains["domain_accession"].nunique()) if not domains.empty else 0,
        "n_annotated_proteins": int(domains["accession"].nunique()) if not domains.empty else 0,
        "occurrence_top10": occurrence.head(10)[
            ["domain_accession", "domain_name", "n_proteins", "n_species", "pct_of_proteome", "reviewed_fraction"]
        ].to_dict("records"),
        "avoidance_significant": int(avoidance["significant_q05"].sum()) if not avoidance.empty else 0,
        "cooccurrence_pairs": int(len(cooccurrence)),
        "cooccurrence_significant": int(cooccurrence["significant_q05"].sum()) if not cooccurrence.empty else 0,
        "cooccurrence_headline_eligible": int(cooccurrence["headline_eligible"].sum()) if not cooccurrence.empty else 0,
        "network": topology,
        "degree_fits": fits,
        "pan_core": pan_core_counts,
        "pan_core_deep_species_only": {
            "n_species": len(deep_species),
            "depth_threshold": float(depth_cut),
            "counts": pan_core_deep_counts,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/frozen/plasmodium_reference_proteomes.tsv.gz"))
    parser.add_argument(
        "--swissprot-snapshot",
        type=Path,
        default=Path("data/frozen/plasmodium_swissprot_reviewed.tsv.gz"),
        help="frozen copy of the submitted manuscript's query (taxonomy_id:5820 AND reviewed:true)",
    )
    parser.add_argument("--clans", type=Path, default=Path("data/frozen/Pfam-A.clans.tsv.gz"))
    parser.add_argument("--outdir", type=Path, default=Path("results/tables"))
    parser.add_argument("--min-pair", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_snapshot(args.snapshot)
    clans = load_pfam_clans(args.clans)

    # One protein can appear in two reference proteomes of the same species (the two
    # P. falciparum isolates). Deduplicating on accession keeps protein-level counts honest.
    df = df[~df["accession"].duplicated(keep="first")].copy()

    results = {
        "clade_assignment": CLADE_BY_SPECIES,
        "scopes": {},
    }

    results["scopes"]["reference"] = run_scope(
        df, clans, "reference", args.outdir, args.min_pair, args.bootstrap
    )

    # The Swiss-Prot scope reads its own frozen snapshot of the submitted manuscript's
    # query rather than the reviewed subset of the reference proteomes. The two are not
    # the same set: reviewed proteins exist for species that have no reference proteome,
    # and reviewed entries attach to non-reference isolates. Reproducing the submitted
    # numbers requires the query that produced them.
    swissprot = load_snapshot(args.swissprot_snapshot)
    swissprot = swissprot[~swissprot["accession"].duplicated(keep="first")].copy()
    results["scopes"]["swissprot"] = run_scope(
        swissprot, clans, "swissprot", args.outdir, max(2, args.min_pair // 2), args.bootstrap
    )

    ref = results["scopes"]["reference"]
    sp = results["scopes"]["swissprot"]
    results["comparison"] = {
        "swissprot_fraction_of_reference_proteins": round(
            sp["dataset"]["n_unique_accessions"] / ref["dataset"]["n_unique_accessions"], 5
        ),
        "swissprot_fraction_of_reference_families": round(
            sp["n_pfam_families"] / ref["n_pfam_families"], 5
        )
        if ref["n_pfam_families"]
        else None,
    }

    out = Path(args.outdir).parent / "analysis_summary.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results["comparison"], indent=2))
    for scope, res in results["scopes"].items():
        print(f"\n=== {scope} ===")
        print(json.dumps({k: res[k] for k in ("dataset", "n_species", "n_pfam_families", "network", "pan_core")}, indent=2, default=str))


if __name__ == "__main__":
    main()
