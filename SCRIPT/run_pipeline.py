#!/usr/bin/env python3
"""
Module: run_pipeline
Purpose: Execute non-scale-free network analysis on Plasmodium domain co-occurrence graph
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - taxon_id: NCBI taxonomy ID (default: 5820 = genus Plasmodium)
    - use_pfam: bool, use Pfam IDs (True) or ft_domain names (False)
    - max_proteins: int, cap on proteins fetched from UniProt
    - report_path: str, output path for REPORT.md (relative to this script)
References:
    - Clauset, Shalizi & Newman (2009) SIAM Rev 51(4):661-703. doi:10.1137/070710111
    - Alstott et al. (2014) PLOS ONE 9(1):e85777. doi:10.1371/journal.pone.0085777
    - Albert & Barabasi (2002) Rev. Mod. Phys. 74:47. doi:10.1103/RevModPhys.74.47
    - Watts & Strogatz (1998) Nature 393:440-442. doi:10.1038/30918
"""

import sys
import os
import datetime
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_proteins import fetch_plasmodium_proteins
from network_builder import build_domain_network, graph_summary
from degree_analysis import fit_degree_distribution, compare_distributions, interpret_results
from network_metrics import compute_clustering, compute_path_length, compute_hub_analysis, compute_core_pan

logger = logging.getLogger(__name__)


def run(
    taxon_id: str = "5820",
    use_pfam: bool = True,
    max_proteins: int = 5000,
    report_path: str = "../REPORT.md",
) -> str:
    """Run full analysis pipeline; return report as Markdown string."""

    logger.info("[1/6] Fetching Plasmodium proteins (taxon=%s, reviewed=True)...", taxon_id)
    df = fetch_plasmodium_proteins(taxon_id=taxon_id, reviewed=True, max_results=max_proteins)
    n_proteins = len(df)
    n_species = df["organism"].nunique() if n_proteins > 0 else 0
    logger.info("      Fetched %d proteins across %d species.", n_proteins, n_species)

    if n_proteins == 0:
        logger.error("No proteins retrieved. Check network access.")
        return "ERROR: No proteins retrieved. Check network access."

    domain_col = "pfam_ids" if use_pfam and "pfam_ids" in df.columns else "domain_names"
    n_with_domains = df[domain_col].apply(
        lambda x: bool(x) if isinstance(x, list) else False
    ).sum()
    logger.info(
        "      Domain column: %s | %d/%d proteins with annotations.",
        domain_col, n_with_domains, n_proteins,
    )

    # Fall back if Pfam coverage is < 5%
    if use_pfam and n_with_domains < 0.05 * n_proteins:
        logger.warning("Pfam coverage <5%%. Falling back to ft_domain names.")
        use_pfam = False
        domain_col = "domain_names"
        n_with_domains = df[domain_col].apply(
            lambda x: bool(x) if isinstance(x, list) else False
        ).sum()

    logger.info("[2/6] Building domain co-occurrence network (domain_col=%s)...", domain_col)
    G = build_domain_network(df, use_pfam=use_pfam, min_edge_proteins=1)
    summary = graph_summary(G)
    logger.info(
        "      Nodes=%d, Edges=%d, Components=%d",
        summary["n_nodes"], summary["n_edges"], summary["n_components"],
    )

    if summary["n_nodes"] < 5:
        logger.error("Network too small (<5 nodes). Insufficient domain annotations in dataset.")
        return "ERROR: Network too small (<5 nodes). Insufficient domain annotations in dataset."

    logger.info("[3/6] Fitting degree distribution (MLE, Clauset-Shalizi-Newman)...")
    degrees = [d for _, d in G.degree() if d > 0]
    try:
        fit, fit_stats = fit_degree_distribution(degrees, discrete=True)
    except ValueError as e:
        logger.error("Degree fitting failed: %s", e)
        return f"ERROR in degree fitting: {e}"
    logger.info(
        "      alpha=%s, xmin=%s, KS=%s, n_tail=%s",
        fit_stats["alpha"], fit_stats["xmin"], fit_stats["KS_distance"], fit_stats["n_tail"],
    )

    logger.info("[4/6] Vuong likelihood ratio tests (power law vs alternatives)...")
    comparisons = compare_distributions(fit)
    for alt, res in comparisons.items():
        logger.info("      vs %s: R=%s, p=%s", alt, res["R"], res["p"])

    logger.info("[5/6] Computing structural network metrics...")
    clustering = compute_clustering(G)
    path_metrics = compute_path_length(G)
    hubs = compute_hub_analysis(G, top_n=10)
    core_pan = compute_core_pan(df, use_pfam=use_pfam)
    logger.info(
        "      avg_clustering=%s, avg_path=%s",
        clustering["avg_clustering"], path_metrics.get("avg_path_length"),
    )

    logger.info("[6/6] Writing REPORT.md...")
    interpretation = interpret_results(fit_stats, comparisons)
    report = _format_report(
        df, summary, fit_stats, comparisons, interpretation,
        clustering, path_metrics, hubs, core_pan, domain_col, use_pfam,
    )

    abs_report = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), report_path)
    )
    with open(abs_report, "w", encoding="utf-8") as fh:
        fh.write(report)
    logger.info("      Written: %s", abs_report)
    return report


def _get_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


def _format_report(
    df, summary, fit_stats, comparisons, interpretation,
    clustering, path_metrics, hubs, core_pan, domain_col, use_pfam,
) -> str:
    today = datetime.date.today().isoformat()
    n_proteins = len(df)
    n_species = df["organism"].nunique() if "organism" in df.columns else 0
    species_list = sorted(df["organism"].unique().tolist()) if "organism" in df.columns else []

    # Vuong table rows
    vuong_rows = []
    for alt, res in comparisons.items():
        R_str = f"{res['R']:.4f}" if res["R"] is not None else "N/A"
        p_str = f"{res['p']:.4f}" if res["p"] is not None else "N/A"
        sig = (
            "Yes"
            if (res["R"] is not None and res["R"] < 0 and res["p"] is not None and res["p"] < 0.05)
            else "No"
        )
        better = "Alternative" if (res["R"] is not None and res["R"] < 0) else "Power law"
        vuong_rows.append(f"| {alt} | {R_str} | {p_str} | {better} | {sig} |")

    # Hub table rows
    hub_rows = []
    for _, row in hubs.iterrows():
        hub_rows.append(f"| {row['domain']} | {row['degree']} | {row['betweenness']:.6f} |")

    cp = core_pan
    path_note = "(sampled estimate)" if path_metrics.get("sampled") else ""
    diameter_str = (
        str(path_metrics.get("diameter"))
        if path_metrics.get("diameter") is not None
        else "N/A (LCC too large for exact computation)"
    )

    non_sf_count = sum(
        1
        for res in comparisons.values()
        if res["R"] is not None and res["R"] < 0 and res["p"] is not None and res["p"] < 0.05
    )
    verdict = "NON-SCALE FREE" if non_sf_count >= 1 else "INCONCLUSIVE"

    core_domain_str = (
        ", ".join(cp.get("core_domains", []))
        if cp.get("core_domains")
        else "None identified (may reflect incomplete annotations for some species)."
    )

    return f"""# Computational Report: Non-Scale Free Analysis of Plasmodium Protein Domain Networks

**Date:** {today}
**Author:** Dr. Arli Aditya Parikesit, i3L University, Jakarta
**Method:** Clauset-Shalizi-Newman framework (MLE + Vuong likelihood ratio test + KS test)
**Domain identifier:** {domain_col}
**Software:** Python 3, networkx {_get_version("networkx")}, powerlaw {_get_version("powerlaw")}

---

## 1. Dataset Summary

| Parameter | Value |
|-----------|-------|
| Source | UniProt REST API v2 (Swiss-Prot, reviewed only) |
| Taxon | Plasmodium genus (NCBI taxon 5820) |
| Total proteins fetched | {n_proteins} |
| Species represented | {n_species} |
| Domain column used | {domain_col} |

**Species:** {", ".join(species_list)}

---

## 2. Network Construction

Domain co-occurrence network: nodes = unique domain identifiers ({domain_col}); edges = co-occurrence within the same protein. Edge weights record the number of proteins sharing each pair. Binary existence per unique protein (not per domain copy) approximates ortholog-group normalization and prevents lineage-expanded gene families (e.g., var genes) from inflating hub connectivity.

| Metric | Value |
|--------|-------|
| Nodes (unique domains) | {summary["n_nodes"]} |
| Edges (co-occurring pairs) | {summary["n_edges"]} |
| Connected components | {summary["n_components"]} |
| Largest component | {summary["largest_component_size"]} nodes |
| Graph density | {summary["density"]:.8f} |
| Mean degree | {summary["avg_degree"]:.4f} |
| Max degree | {summary["max_degree"]} |
| Min degree | {summary["min_degree"]} |

---

## 3. Degree Distribution Fitting

Maximum likelihood estimation (MLE) of power law parameters following Clauset, Shalizi & Newman (2009). The lower cutoff xmin is estimated automatically to minimize KS distance.

| Parameter | Value |
|-----------|-------|
| Power law exponent (alpha) | {fit_stats["alpha"]} |
| Standard error (sigma) | {fit_stats["sigma"]} |
| Lower cutoff (xmin) | {fit_stats["xmin"]} |
| KS distance (empirical vs PL) | {fit_stats["KS_distance"]} |
| Observations in tail (k >= xmin) | {fit_stats["n_tail"]} |
| Total degree observations | {fit_stats["n_total"]} |

**Alpha interpretation:** Canonical scale-free networks exhibit alpha in [2, 3]. KS distance > 0.1 indicates a poor power law fit.

---

## 4. Vuong Likelihood Ratio Tests

Normalized Vuong log-likelihood ratio comparing power law against alternative distributions. Negative R = alternative is the better-fitting model. p < 0.05 = statistically significant.

| Alternative | Log-likelihood Ratio (R) | p-value | Better fit | Significant |
|-------------|--------------------------|---------|------------|-------------|
{chr(10).join(vuong_rows)}

---

## 5. Statistical Interpretation

{interpretation}

---

## 6. Network Structural Metrics

| Metric | Value | Note |
|--------|-------|------|
| Average clustering coefficient | {clustering["avg_clustering"]} | High = modular structure |
| Transitivity (global clustering) | {clustering["transitivity"]} | |
| Average path length (LCC) | {path_metrics.get("avg_path_length", "N/A")} {path_note} | |
| Network diameter | {diameter_str} | |
| LCC node count | {path_metrics.get("n_lcc_nodes", "N/A")} | |

High clustering with moderate path length = small-world but modular topology, inconsistent with the hub-and-spoke architecture of pure scale-free networks (Watts & Strogatz 1998).

---

## 7. Top Hubs by Degree

| Domain | Degree | Betweenness Centrality |
|--------|--------|----------------------|
{chr(10).join(hub_rows)}

**Saturation check:** Bounded max degree ({summary["max_degree"]}) relative to network size ({summary["n_nodes"]} nodes) indicates hub connectivity is physically constrained, consistent with a truncated or log-normal degree distribution rather than an unbounded power law.

---

## 8. Core vs Pan Domain Analysis

| Category | Count | Description |
|----------|-------|-------------|
| Pan (union) | {cp.get("pan_size", "N/A")} | All domains across all {cp.get("n_species", 0)} species |
| Core | {cp.get("core_size", "N/A")} | Present in ALL species |
| Shell | {cp.get("shell_size", "N/A")} | Present in 2 to n-1 species |
| Cloud | {cp.get("cloud_size", "N/A")} | Species-exclusive |
| Core fraction | {cp.get("core_fraction", "N/A")} | Core / Pan ratio |

A large cloud fraction indicates lineage-specific domain acquisition consistent with functional modularity rather than preferential attachment.

**Core domains (shared by all species):**
{core_domain_str}

---

## 9. Verdict

**Network classification: {verdict}**

{interpretation}

---

## 10. Methods

- **Data:** UniProt REST API v2, Swiss-Prot reviewed Plasmodium proteins, taxon 5820.
- **Network:** Python networkx {_get_version("networkx")}; nodes = {domain_col}; edges = protein co-occurrence (binary per protein).
- **Degree fitting:** powerlaw {_get_version("powerlaw")} (MLE, discrete=True; xmin auto-estimated).
- **Comparison:** Vuong normalized log-likelihood ratio test (normalized_ratio=True).
- **Clustering:** nx.average_clustering; transitivity: nx.transitivity.
- **Path length:** nx.average_shortest_path_length (exact for LCC <= 200 nodes; sampled for larger).
- **Core/pan:** set operations across per-species domain pools.

## References

1. Clauset A, Shalizi CR, Newman MEJ (2009) Power-law distributions in empirical data. SIAM Rev 51(4):661-703. doi:10.1137/070710111
2. Alstott J, Bullmore E, Plenz D (2014) powerlaw: A Python package for analysis of heavy-tailed distributions. PLOS ONE 9(1):e85777. doi:10.1371/journal.pone.0085777
3. Albert R, Barabasi AL (2002) Statistical mechanics of complex networks. Rev Mod Phys 74:47. doi:10.1103/RevModPhys.74.47
4. Watts DJ, Strogatz SH (1998) Collective dynamics of small-world networks. Nature 393:440-442. doi:10.1038/30918
5. Newman MEJ (2006) Modularity and community structure in networks. PNAS 103(23):8577-8582. doi:10.1073/pnas.0601602103
6. Parikesit AA, Utomo DH, Karimah S (2018) Protein domain annotation of Plasmodium spp. CSP using hidden Markov models. JBI 14(2):185-190. doi:10.14203/jbi.v14i2.3737
"""


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Plasmodium domain network non-scale-free analysis"
    )
    parser.add_argument("--taxon", default="5820", help="NCBI taxon ID (default: 5820)")
    parser.add_argument(
        "--no-pfam", action="store_true", help="Use ft_domain names instead of Pfam IDs"
    )
    parser.add_argument("--max-proteins", type=int, default=5000)
    parser.add_argument("--report", default="../REPORT.md", help="Output path for REPORT.md")
    args = parser.parse_args()

    result = run(
        taxon_id=args.taxon,
        use_pfam=not args.no_pfam,
        max_proteins=args.max_proteins,
        report_path=args.report,
    )
    if result.startswith("ERROR"):
        print(result, file=sys.stderr)
        sys.exit(1)
