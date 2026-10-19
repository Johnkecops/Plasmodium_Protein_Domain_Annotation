#!/usr/bin/env python3
"""
Module: robustness_analysis
Purpose: Sensitivity and null-model analyses on the frozen reference-proteome snapshot.
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    --replicates : curveball null replicates (default 200)
    --trades     : curveball trades per replicate, as a multiple of the protein count (default 5)
    --workers    : processes for the null replicates (default os.cpu_count())
    --seed       : base random seed (default 20261019)
References:
    - Strona G, Nappo D, Boccacci F, Fattorini S, San-Miguel-Ayanz J, 2014. A fast and
      unbiased procedure to randomize ecological binary matrices with fixed row and column
      totals. Nature Communications 5, 4114.
    - Carstens CJ, 2015. Proof of uniform sampling of binary matrices with fixed row sums and
      column sums for the fast curveball algorithm. Physical Review E 91, 042812.

Five analyses, all on the frozen input, none re-sampling or re-fetching data:

1. Annotation coverage by member database (why Pfam and InterPro coverage differ).
2. One representative proteome per species (P. falciparum 3D7 only, P. yoelii 17X only),
   repeating avoidance and pan-core, to show the duplicate isolates do not drive results.
3. Clade-level inference: a Poisson-binomial test over host-defined clades rather than
   species, and a clade jackknife of the species-level test.
4. Core composition: copy number, promiscuity, single-copy core and a PANTHER
   gene-family core as an orthology-closer comparison.
5. Architecture-preserving network null: the protein-by-family incidence matrix is
   randomised with the curveball algorithm, which fixes every protein's family count and
   every family's carrier count, and the co-occurrence network is rebuilt with the same
   support and FDR filters.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse, stats

from domain_dataset import XREF_COLUMNS, build_domain_table, load_snapshot, species_coverage_table
from genus_analysis import _bh_fdr, _poisson_binomial_sf, compute_domain_avoidance, compute_pan_core

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
# Second reference proteome of a species with two; dropped in the single-representative run.
# Kept proteome per species is the one with the higher BUSCO completeness (3D7 99.1% vs
# NF54 80.0%; P. yoelii 17X 96.0% vs yoelii yoelii 64.7%).
SECONDARY_PROTEOMES = {"UP000030673", "UP000008553"}


# ----------------------------------------------------------------------------------
# 1. Member-database coverage
# ----------------------------------------------------------------------------------

def member_db_coverage(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    has = {db: df[col].fillna("").str.strip().ne("") for db, col in XREF_COLUMNS.items()}
    rows = [{"database": db, "n_proteins": int(m.sum()), "coverage_pct": round(100 * m.sum() / n, 2)} for db, m in has.items()]
    members = [db for db in has if db != "InterPro"]
    any_member = np.logical_or.reduce([has[db] for db in members])
    rows.append({"database": "any listed member", "n_proteins": int(any_member.sum()), "coverage_pct": round(100 * any_member.sum() / n, 2)})
    no_pfam = has["InterPro"] & ~has["Pfam"]
    for db in members:
        if db == "Pfam":
            continue
        rows.append({"database": f"InterPro without Pfam, matched by {db}", "n_proteins": int((no_pfam & has[db]).sum()),
                     "coverage_pct": round(100 * (no_pfam & has[db]).sum() / n, 2)})
    rows.append({"database": "InterPro without Pfam (total)", "n_proteins": int(no_pfam.sum()), "coverage_pct": round(100 * no_pfam.sum() / n, 2)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# 2. Single representative proteome per species
# ----------------------------------------------------------------------------------

def single_representative(df: pd.DataFrame, full_avoid: pd.DataFrame, full_pan: pd.DataFrame) -> dict:
    sub = df[~df["proteome_id"].isin(SECONDARY_PROTEOMES)]
    dom = build_domain_table(sub, layer="pfam")
    cov = species_coverage_table(sub, dom)
    avoid = compute_domain_avoidance(dom, cov)
    pan = compute_pan_core(dom, sorted(sub["species"].unique()))
    sig_full = set(full_avoid.loc[full_avoid["significant_q05"], "domain_accession"])
    sig_sub = set(avoid.loc[avoid["significant_q05"], "domain_accession"])
    merged = full_pan.set_index("domain_accession")["category"]
    single = pan.set_index("domain_accession")["category"]
    common = merged.index.intersection(single.index)
    return {
        "n_proteins": int(len(sub)),
        "n_species": int(sub["species"].nunique()),
        "n_pfam_families": int(dom["domain_accession"].nunique()),
        "avoidance_significant": len(sig_sub),
        "avoidance_overlap_with_merged": len(sig_full & sig_sub),
        "avoidance_only_merged": sorted(sig_full - sig_sub),
        "avoidance_only_single": sorted(sig_sub - sig_full),
        "pan_core": pan["category"].value_counts().to_dict(),
        "pan_core_category_agreement": round(float((merged[common] == single[common]).mean()), 4),
        "families_lost": int(len(merged.index.difference(single.index))),
    }


# ----------------------------------------------------------------------------------
# 3. Clade-level inference
# ----------------------------------------------------------------------------------

def clade_avoidance(domains: pd.DataFrame, coverage: pd.DataFrame, min_carriers: int = 3) -> pd.DataFrame:
    """
    Poisson-binomial test over clades instead of species.

    For family d and clade c, q(d,c) = (1 - p_d(-c)) ** n_c, with n_c the annotated proteins
    summed over the clade's species and p_d(-c) the per-protein frequency of d outside c.
    Each clade is one observation, so related species cannot add evidence twice.
    """
    depth = coverage.groupby("clade")["n_annotated_proteins"].sum().to_dict()
    clades = sorted(depth)
    total = sum(depth.values())
    carriers = domains.groupby(["domain_accession", "clade"])["accession"].nunique().unstack(fill_value=0)
    carriers = carriers.reindex(columns=clades, fill_value=0)
    totals = carriers.sum(axis=1)
    carriers = carriers[totals >= min_carriers]
    totals = totals[totals >= min_carriers]
    n = np.array([depth[c] for c in clades], dtype=float)
    counts = carriers.to_numpy(dtype=float)
    p_out = np.clip((totals.to_numpy()[:, None] - counts) / (total - n)[None, :], 0, 1)
    q = (1 - p_out) ** n[None, :]
    absent = counts == 0
    pvals = np.array([_poisson_binomial_sf(q[i], int(absent[i].sum())) for i in range(len(q))])
    out = pd.DataFrame({
        "domain_accession": carriers.index,
        "n_clades_absent": absent.sum(axis=1),
        "expected_clades_absent": q.sum(axis=1).round(3),
        "clade_p": pvals,
        "absent_clades": ["; ".join(c for c, a in zip(clades, row) if a) for row in absent],
    })
    out["clade_q"] = _bh_fdr(out["clade_p"].to_numpy())
    out["clade_significant_q05"] = out["clade_q"] < 0.05
    return out


def clade_jackknife(domains: pd.DataFrame, coverage: pd.DataFrame, sig_full: set[str]) -> pd.DataFrame:
    rows = []
    for clade in sorted(coverage["clade"].unique()):
        keep = coverage.loc[coverage["clade"] != clade, "species"]
        av = compute_domain_avoidance(domains[domains["species"].isin(keep)], coverage[coverage["species"].isin(keep)])
        sig = set(av.loc[av["significant_q05"], "domain_accession"])
        rows.append({"dropped_clade": clade, "n_species": int(len(keep)), "n_significant": len(sig),
                     "retained_from_full": len(sig & sig_full), "new": len(sig - sig_full)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# 4. Core composition and an orthology-closer comparison
# ----------------------------------------------------------------------------------

def core_composition(df: pd.DataFrame, domains: pd.DataFrame, pan: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sub = df[~df["proteome_id"].isin(SECONDARY_PROTEOMES)]
    dom1 = domains[domains["accession"].isin(set(sub["accession"]))]
    species = sorted(sub["species"].unique())
    copies = dom1.groupby(["domain_accession", "species"])["accession"].nunique().unstack(fill_value=0).reindex(columns=species, fill_value=0)

    sets = domains.groupby("accession")["domain_accession"].apply(list)
    partners: dict[str, set] = {}
    for fams in sets:
        for f in fams:
            partners.setdefault(f, set()).update(x for x in fams if x != f)
    promiscuity = pd.Series({f: len(p) for f, p in partners.items()}, name="n_partner_families")

    core = pan.loc[pan["category"] == "core", "domain_accession"]
    tab = pd.DataFrame(index=core)
    tab["min_copies"] = copies.reindex(core).min(axis=1)
    tab["median_copies"] = copies.reindex(core).median(axis=1)
    tab["max_copies"] = copies.reindex(core).max(axis=1)
    tab["single_copy_all_species"] = (copies.reindex(core) == 1).all(axis=1)
    tab["n_partner_families"] = promiscuity.reindex(core).fillna(0).astype(int)
    cut = promiscuity.quantile(0.95)
    tab["promiscuous_top5pct"] = tab["n_partner_families"] >= cut
    tab["multicopy_median_gt2"] = tab["median_copies"] > 2

    panther = build_domain_table(sub.assign(xref_panther=sub["xref_panther"].fillna("").str.replace(r"PTHR\d+:SF\d+;?", "", regex=True)), layer="PANTHER")
    panther_core = panther.groupby("domain_accession")["species"].nunique()
    summary = {
        "n_core": int(len(core)),
        "single_copy_core": int(tab["single_copy_all_species"].sum()),
        "core_with_median_le_2_copies": int((~tab["multicopy_median_gt2"]).sum()),
        "promiscuity_cut_top5pct_partners": float(cut),
        "core_promiscuous_top5pct": int(tab["promiscuous_top5pct"].sum()),
        "core_excluding_promiscuous_and_multicopy": int((~tab["promiscuous_top5pct"] & ~tab["multicopy_median_gt2"]).sum()),
        "core_promiscuity_median": float(tab["n_partner_families"].median()),
        "core_partner_zero": int((tab["n_partner_families"] == 0).sum()),
        "panther_families": int(len(panther_core)),
        "panther_core_all_species": int((panther_core == len(species)).sum()),
        "n_species": len(species),
    }
    return tab.reset_index(names="domain_accession"), summary


# ----------------------------------------------------------------------------------
# 5. Architecture-preserving network null (curveball)
# ----------------------------------------------------------------------------------

def curveball(rows: list[np.ndarray], n_trades: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Curveball randomisation; row and column sums of the binary matrix are preserved."""
    rows = [set(r.tolist()) for r in rows]
    n = len(rows)
    pairs = rng.integers(0, n, size=(n_trades, 2))
    for i, j in pairs:
        if i == j:
            continue
        a, b = rows[i], rows[j]
        only_a, only_b = a - b, b - a
        if not only_a or not only_b:
            continue
        pool = list(only_a | only_b)
        rng.shuffle(pool)
        k = len(only_a)
        shared = a & b
        rows[i] = shared | set(pool[:k])
        rows[j] = shared | set(pool[k:])
    return [np.fromiter(r, dtype=np.int64) for r in rows]


def network_from_rows(rows: list[np.ndarray], n_fam: int, min_pair: int = 5) -> dict:
    """
    Metrics for two graphs built from one incidence matrix: 'filtered' uses the same filters
    as the observed network (support >= min_pair, one-sided Fisher, BH q < 0.05); 'raw'
    joins every pair of families that share at least one protein, which is the graph whose
    topology the protein-architecture constraint acts on directly.
    """
    indptr = np.cumsum([0] + [len(r) for r in rows])
    X = sparse.csr_matrix((np.ones(indptr[-1]), np.concatenate(rows), indptr), shape=(len(rows), n_fam))
    N = len(rows)
    C = (X.T @ X).tocoo()
    freq = np.asarray(X.sum(axis=0)).ravel()
    upper = C.row < C.col
    m = upper & (C.data >= min_pair)
    a, b, nab = C.row[m], C.col[m], C.data[m]
    p = stats.hypergeom.sf(nab - 1, N, freq[a], freq[b])
    q = _bh_fdr(p) if len(p) else p
    keep = q < 0.05
    out = {f"filtered_{k}": v for k, v in _graph_metrics(zip(a[keep].tolist(), b[keep].tolist())).items()}
    out.update({f"raw_{k}": v for k, v in _graph_metrics(zip(C.row[upper].tolist(), C.col[upper].tolist())).items()})
    return out


def _graph_metrics(edges) -> dict:
    g = nx.Graph()
    g.add_edges_from(edges)
    if g.number_of_nodes() == 0:
        return {"n_nodes": 0}
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    deg = np.array([d for _, d in g.degree()])
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_components": len(comps),
        "largest_component_nodes": comps[0],
        "largest_component_fraction": comps[0] / g.number_of_nodes(),
        "mean_degree": float(deg.mean()),
        "max_degree": int(deg.max()),
        "degree_cv": float(deg.std() / deg.mean()),
        "avg_clustering": nx.average_clustering(g),
        "transitivity": nx.transitivity(g),
    }


def _replicate(args) -> dict:
    rows, n_fam, trades, seed = args
    rng = np.random.default_rng(seed)
    return network_from_rows(curveball(rows, trades, rng), n_fam)


def curveball_null(domains: pd.DataFrame, replicates: int, trade_mult: int, workers: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fams = {f: i for i, f in enumerate(sorted(domains["domain_accession"].unique()))}
    rows = [np.array(sorted({fams[f] for f in fs})) for fs in domains.groupby("accession")["domain_accession"].apply(list)]
    observed = network_from_rows(rows, len(fams))
    trades = trade_mult * len(rows)
    jobs = [(rows, len(fams), trades, seed + i) for i in range(replicates)]
    # ponytail: the full incidence list is pickled per job (~MBs); fine for 200 jobs on 8 cores,
    # use a pool initializer if replicates grow into the thousands.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        null = pd.DataFrame(list(ex.map(_replicate, jobs, chunksize=max(1, replicates // (workers * 4)))))
    out = []
    for k, obs in observed.items():
        v = null[k].to_numpy(dtype=float)
        sd = v.std(ddof=1)
        lo = (np.sum(v <= obs) + 1) / (len(v) + 1)
        hi = (np.sum(v >= obs) + 1) / (len(v) + 1)
        out.append({"metric": k, "observed": obs, "null_mean": v.mean(), "null_sd": sd,
                    "null_2.5pct": np.percentile(v, 2.5), "null_97.5pct": np.percentile(v, 97.5),
                    "z": (obs - v.mean()) / sd if sd > 0 else np.nan, "empirical_p_two_sided": min(1.0, 2 * min(lo, hi))})
    return pd.DataFrame(out), null


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--trades", type=int, default=5)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tables = args.outdir / "tables"

    df = load_snapshot(ROOT / "data" / "frozen" / "plasmodium_reference_proteomes.tsv.gz")
    domains = build_domain_table(df, layer="pfam")
    coverage = species_coverage_table(df, domains)
    avoid = pd.read_csv(tables / "reference_avoidance.csv")
    pan = pd.read_csv(tables / "reference_pan_core.csv")
    sig_full = set(avoid.loc[avoid["significant_q05"], "domain_accession"])
    summary: dict = {}

    member_db_coverage(df).to_csv(tables / "reference_member_db_coverage.csv", index=False)
    logger.info("member-database coverage written")

    summary["single_representative"] = single_representative(df, avoid, pan)
    logger.info("single representative: %s", summary["single_representative"])

    clade = clade_avoidance(domains, coverage)
    clade = clade.merge(avoid[["domain_accession", "q_value", "significant_q05"]], on="domain_accession", how="left")
    clade.to_csv(tables / "reference_clade_avoidance.csv", index=False)
    jk = clade_jackknife(domains, coverage, sig_full)
    jk.to_csv(tables / "reference_clade_jackknife.csv", index=False)
    sig_clade = set(clade.loc[clade["clade_significant_q05"], "domain_accession"])
    summary["clade_inference"] = {
        "n_clades": int(coverage["clade"].nunique()),
        "families_tested": int(len(clade)),
        "clade_significant": len(sig_clade),
        "species_significant": len(sig_full),
        "both": len(sig_clade & sig_full),
        "species_only": sorted(sig_full - sig_clade),
        "clade_only": sorted(sig_clade - sig_full),
        "jackknife_min_retained": int(jk["retained_from_full"].min()),
        "jackknife_max_new": int(jk["new"].max()),
    }
    logger.info("clade inference: %s", summary["clade_inference"])

    core_tab, core_sum = core_composition(df, domains, pan)
    core_tab.to_csv(tables / "reference_core_composition.csv", index=False)
    summary["core_composition"] = core_sum
    logger.info("core composition: %s", core_sum)

    null_tab, null_raw = curveball_null(domains, args.replicates, args.trades, args.workers, args.seed)
    null_tab.to_csv(tables / "reference_network_null.csv", index=False)
    null_raw.to_csv(tables / "reference_network_null_replicates.csv", index=False)
    summary["network_null"] = {"replicates": args.replicates, "trades_per_protein": args.trades, "seed": args.seed}
    logger.info("network null:\n%s", null_tab.to_string())

    with (args.outdir / "robustness_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
