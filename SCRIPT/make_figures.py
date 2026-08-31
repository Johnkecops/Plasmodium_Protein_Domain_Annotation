#!/usr/bin/env python3
"""
Module: make_figures
Purpose: Generate every manuscript figure from pipeline output alone.
Author: Dr. Arli Aditya Parikesit, Dr. Arif Nur Muhammad Ansori, and Moch. Royhan Afnani.,M.Sc
Date: 2026

Rationale (response to reviewer):
    Reviewer tabulated five figures whose captions did not describe their panels, and
    concluded from the palette and metric-tile layout that the submitted figures were
    screen captures of the Streamlit application rather than figures generated for
    publication. Figures 3 and 4 described a heatmap and a network that were never drawn.

    This script draws them. It reads only the CSV tables written by run_genus_pipeline.py,
    so no figure can display a number that the pipeline did not compute, and every panel
    is reproducible from the frozen snapshot. Captions are written from the rendered
    panels after this script runs, not before.

Parameters:
    --tables : directory of pipeline CSV output (default results/tables)
    --outdir : figure destination (default results/figures)
    --scope  : which dataset scope to draw (default reference)

References:
    Hunter J.D. (2007) Computing in Science and Engineering 9(3):90-95. doi:10.1109/MCSE.2007.55
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)

INK = "#1a1a1a"
ACCENT = "#2b6cb0"
MUTED = "#a0aec0"
WARN = "#c05621"


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {name}.png / .pdf")


def _italic_species(name: str) -> str:
    """Render "Plasmodium berghei" as an italic abbreviated binomial for axis labels."""
    parts = str(name).split()
    if len(parts) < 2:
        return str(name)
    epithet = "\\ ".join(parts[1:])
    return "$P.\\ " + epithet + "$"


def figure1_dataset(tables: Path, outdir: Path) -> None:
    """Sampling depth per species in both dataset scopes, and annotation coverage."""
    ref = pd.read_csv(tables / "reference_table1_species_coverage.csv")
    sp = pd.read_csv(tables / "swissprot_table1_species_coverage.csv")
    sp_by_species = dict(zip(sp["species"], sp["n_proteins"]))

    ref = ref.sort_values("n_proteins", ascending=True)
    labels = [_italic_species(s) for s in ref["species"]]
    y = np.arange(len(ref))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    ax.barh(y, ref["n_proteins"], color=MUTED, height=0.72, label="Reference proteome")
    ax.barh(
        y,
        [sp_by_species.get(s, 0) for s in ref["species"]],
        color=ACCENT,
        height=0.72,
        label="Swiss-Prot reviewed",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(0.8, ref["n_proteins"].max() * 1.6)
    ax.set_xlabel("Proteins per species (log scale)")
    ax.set_title("a  Sampling depth by dataset scope", loc="left", weight="bold")
    ax.legend(loc="lower right")
    for i, (n_ref, s) in enumerate(zip(ref["n_proteins"], ref["species"])):
        n_sp = sp_by_species.get(s, 0)
        ax.text(n_ref * 1.1, i, f"{n_ref:,} / {n_sp}", va="center", fontsize=6.5, color=INK)

    ax = axes[1]
    ax.barh(y, ref["annotation_coverage"] * 100, color=ACCENT, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proteins carrying at least one Pfam family (%)")
    ax.set_title("b  Annotation coverage", loc="left", weight="bold")
    mean_cov = ref["annotation_coverage"].mean() * 100
    ax.axvline(mean_cov, color=WARN, lw=1, ls="--")
    ax.text(mean_cov + 1.5, len(ref) - 1.5, f"mean {mean_cov:.1f}%", color=WARN, fontsize=7)

    _save(fig, outdir, "Figure1_dataset_composition")


def figure2_occurrence(tables: Path, outdir: Path, scope: str) -> None:
    """The most frequently annotated families, labelled by Pfam accession."""
    occ = pd.read_csv(tables / f"{scope}_occurrence.csv").head(20).iloc[::-1]
    y = np.arange(len(occ))

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    colours = plt.cm.viridis(occ["n_species"] / occ["n_species"].max())
    ax.barh(y, occ["n_proteins"], color=colours, height=0.74)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  ({a})" for a, n in zip(occ["domain_accession"], occ["domain_name"])], fontsize=7)
    ax.set_xlabel("Proteins carrying the family")
    ax.set_title(
        "Twenty most frequently annotated Pfam families, labelled by accession",
        loc="left",
        weight="bold",
    )
    for i, (n, ns) in enumerate(zip(occ["n_proteins"], occ["n_species"])):
        ax.text(n * 1.01, i, f"{n:,}  ({ns} spp.)", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(0, occ["n_proteins"].max() * 1.18)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, occ["n_species"].max()))
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Species carrying the family", fontsize=7)

    _save(fig, outdir, "Figure2_domain_occurrence")


def figure3_heatmap(tables: Path, outdir: Path, scope: str) -> None:
    """The species-by-family heatmap that the submitted Figure 3 caption described."""
    matrix = pd.read_csv(tables / f"{scope}_species_domain_matrix.csv", index_col=0)
    names = matrix.pop("domain_name")
    matrix = matrix.iloc[::-1]
    names = names.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.6, 6.8))
    data = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_where(data == 0, data)
    cmap = plt.cm.magma_r.copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=LogNorm(vmin=1, vmax=max(2, data.max())))

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([_italic_species(s) for s in matrix.columns], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"{n}  ({a})" for a, n in zip(matrix.index, names)], fontsize=7)
    ax.set_title(
        "Carrier proteins per species for the 25 most frequent families\n"
        "Grey cells indicate absence from that species",
        loc="left",
        weight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cbar.set_label("Carrier proteins (log scale)", fontsize=7)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    _save(fig, outdir, "Figure3_species_domain_heatmap")


def figure4_network(tables: Path, outdir: Path, scope: str, summary: dict) -> None:
    """
    The network rendering the submitted Figure 4 caption described.

    Both panels are drawn because the component structure is the finding: the graph is a
    field of small disjoint cliques, and a single large-component view would misrepresent
    it exactly as the earlier small-world characterisation did.
    """
    co = pd.read_csv(tables / f"{scope}_cooccurrence.csv")
    edges = co[co["q_value"] < 0.05]
    graph = nx.Graph()
    for row in edges.itertuples():
        graph.add_edge(row.domain_a, row.domain_b, weight=row.n_AB)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = graph.subgraph(components[0])
    topology = summary["scopes"][scope]["network"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4))

    ax = axes[0]
    pos = nx.spring_layout(graph, seed=20260831, k=0.32, iterations=60)
    degrees = dict(graph.degree())
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#cbd5e0", width=0.4)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=[6 + 5 * degrees[n] for n in graph.nodes()],
        node_color=[degrees[n] for n in graph.nodes()],
        cmap="viridis",
        linewidths=0,
    )
    ax.set_axis_off()
    ax.set_title(
        f"a  Complete graph: {topology['n_nodes']} nodes, {topology['n_edges']} edges, "
        f"{topology['n_components']} components",
        loc="left",
        weight="bold",
    )

    ax = axes[1]
    pos2 = nx.spring_layout(largest, seed=20260831, k=0.45, iterations=100)
    deg2 = dict(largest.degree())
    nx.draw_networkx_edges(largest, pos2, ax=ax, edge_color="#cbd5e0", width=0.6)
    nx.draw_networkx_nodes(
        largest,
        pos2,
        ax=ax,
        node_size=[18 + 9 * deg2[n] for n in largest.nodes()],
        node_color=[deg2[n] for n in largest.nodes()],
        cmap="viridis",
        linewidths=0,
    )
    # Hub labels are offset above their node and boxed; drawn at the node centre they
    # overlap each other wherever the hubs are adjacent, which they are by construction.
    hubs = sorted(deg2, key=deg2.get, reverse=True)[:5]
    for rank, hub in enumerate(hubs):
        hx, hy = pos2[hub]
        ax.annotate(
            f"{hub} (k = {deg2[hub]})",
            xy=(hx, hy),
            xytext=(hx, hy + 0.10 + 0.05 * (rank % 2)),
            ha="center",
            fontsize=6,
            color=INK,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#cbd5e0", "lw": 0.4},
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#718096"},
        )
    ax.set_axis_off()
    ax.set_title(
        f"b  Largest component: {topology['largest_component_nodes']} nodes "
        f"({topology['largest_component_fraction'] * 100:.1f}% of the graph)",
        loc="left",
        weight="bold",
    )

    _save(fig, outdir, "Figure4_cooccurrence_network")


def figure5_cooccurrence(tables: Path, outdir: Path, scope: str) -> None:
    """
    Top co-occurring pairs, ranked by Jaccard rather than by lift.

    Reviewer 4 showed that ranking by lift ranks by rarity: for a pair that co-occurs
    obligately, lift reduces to N / n_AB, so the highest values belong to the rarest
    pairs and several tie at the support threshold. Jaccard is bounded in [0, 1] and
    measures the overlap itself, so it is used for the ranking. Lift, support and the
    corrected q-value are printed for every pair.
    """
    co = pd.read_csv(tables / f"{scope}_cooccurrence.csv")
    top = co[co["headline_eligible"] & co["significant_q05"]].nlargest(20, "jaccard").iloc[::-1]
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    ax.barh(y, top["jaccard"], color=ACCENT, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{a} x {b}\n({pa} x {pb})"
            for a, b, pa, pb in zip(top["name_a"], top["name_b"], top["domain_a"], top["domain_b"])
        ],
        fontsize=6.2,
    )
    ax.set_xlabel("Jaccard index of the two carrier sets")
    ax.set_title(
        "Twenty strongest domain pairs by Jaccard overlap,\n"
        "excluding obligate partners and pairs within one Pfam clan",
        loc="left",
        weight="bold",
    )
    for i, (j, lift, n_ab, q) in enumerate(zip(top["jaccard"], top["lift"], top["n_AB"], top["q_value"])):
        ax.text(
            j + 0.012,
            i,
            f"J = {j:.2f}   lift = {lift:,.0f}   n = {n_ab}   q = {q:.0e}",
            va="center",
            fontsize=6.2,
            color=INK,
        )
    ax.set_xlim(0, 1.42)

    _save(fig, outdir, "Figure5_cooccurrence_pairs")


def figure6_pan_core(tables: Path, outdir: Path, summary: dict) -> None:
    """Pan-core partition under both dataset scopes, side by side."""
    order = ["core", "soft_core", "shell", "cloud"]
    colours = {"core": "#22543d", "soft_core": "#48bb78", "shell": "#a0aec0", "cloud": "#e2e8f0"}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax, scope, title in zip(
        axes,
        ("reference", "swissprot"),
        ("a  Reference proteomes (20 species)", "b  Swiss-Prot reviewed (16 species)"),
    ):
        counts = summary["scopes"][scope]["pan_core"]
        n_species = summary["scopes"][scope]["n_species"]
        total = sum(counts.values())
        values = [counts.get(k, 0) for k in order]
        bottom = 0.0
        for key, value in zip(order, values):
            if value == 0:
                continue
            ax.bar(0, value, bottom=bottom, color=colours[key], width=0.55, edgecolor="white")
            ax.text(
                0,
                bottom + value / 2,
                f"{key.replace('_', ' ')}: {value:,}  ({value / total * 100:.1f}%)",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if key in ("core", "soft_core") else INK,
            )
            bottom += value
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_ylabel(f"Pfam families (pan set = {total:,})")
        ax.set_title(f"{title}\ncore = present in all {n_species} species", loc="left", weight="bold")

    _save(fig, outdir, "Figure6_pan_core")


def figure7_degree_distribution(tables: Path, outdir: Path, scope: str, summary: dict) -> None:
    """Empirical degree distribution with the candidate fits and bootstrap interval."""
    import powerlaw

    degrees = pd.read_csv(tables / f"{scope}_degree_sequence.csv")["degree"].to_numpy()
    degrees = degrees[degrees > 0]
    fits = summary["scopes"][scope]["degree_fits"]
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))

    ax = axes[0]
    fit.plot_ccdf(ax=ax, color=INK, marker="o", markersize=3, linestyle="none", label="Empirical")
    fit.power_law.plot_ccdf(ax=ax, color=ACCENT, ls="--", label=f"Power law (alpha = {fits['alpha']:.2f})")
    fit.truncated_power_law.plot_ccdf(ax=ax, color=WARN, ls="-.", label="Truncated power law")
    fit.lognormal.plot_ccdf(ax=ax, color="#805ad5", ls=":", label="Log-normal")
    fit.exponential.plot_ccdf(ax=ax, color="#38a169", ls="-", lw=0.9, label="Exponential")
    ax.set_xlabel("Degree k")
    ax.set_ylabel("P(K >= k)")
    ax.set_title(f"a  Degree distribution, x_min = {fits['xmin']:.0f}", loc="left", weight="bold")
    ax.legend(fontsize=6.5, loc="lower left")

    ax = axes[1]
    comparisons = [
        ("truncated\npower law", "vs_truncated_power_law"),
        ("log-normal", "vs_lognormal"),
        ("exponential", "vs_exponential"),
        ("stretched\nexponential", "vs_stretched_exponential"),
        ("Poisson", "vs_poisson"),
    ]
    labels, ratios, colours = [], [], []
    for label, key in comparisons:
        entry = fits.get(key, {})
        if "loglikelihood_ratio" not in entry:
            continue
        labels.append(label)
        ratios.append(entry["loglikelihood_ratio"])
        colours.append(ACCENT if entry["p_value"] < 0.05 else MUTED)
    x = np.arange(len(labels))
    ax.bar(x, ratios, color=colours, width=0.6)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Normalised log-likelihood ratio\n(positive favours the power law)")
    ax.set_title("b  Model comparison against the power law", loc="left", weight="bold")

    span = max(abs(min(ratios)), abs(max(ratios)))
    ax.set_ylim(min(min(ratios), 0) - 0.45 * span, max(max(ratios), 0) + 0.35 * span)
    for i, (r, (_, key)) in enumerate(zip(ratios, comparisons)):
        p = fits[key]["p_value"]
        # A p-value below the reporting resolution is stated as a bound rather than as
        # an exact zero, which no likelihood-ratio test produces.
        text = "p < 1e-5" if p < 1e-5 else f"p = {p:.3g}"
        offset = 0.08 * span if r >= 0 else -0.08 * span
        ax.text(i, r + offset, text, ha="center", va="bottom" if r >= 0 else "top", fontsize=6.5)

    ax.text(0.02, 0.95, "Filled bars: p < 0.05", transform=ax.transAxes, fontsize=6.5, color=ACCENT)

    boot = fits.get("bootstrap", {})
    if boot:
        fig.text(
            0.5,
            -0.04,
            f"Power-law exponent alpha = {fits['alpha']:.2f}, 95% CI "
            f"[{boot['alpha_ci95'][0]:.2f}, {boot['alpha_ci95'][1]:.2f}] "
            f"from {boot['replicates']} bootstrap replicates of the degree sequence; "
            f"Kolmogorov-Smirnov distance {fits['ks_distance']:.3f} over {fits['n_tail']} tail nodes.",
            ha="center",
            fontsize=6.8,
            color=INK,
        )

    _save(fig, outdir, "Figure7_degree_distribution")


def figure8_avoidance(tables: Path, outdir: Path, scope: str) -> None:
    """
    Avoidance under the depth-conditioned null, with q-values on the panel.

    Reviewer 4 objected that the submitted Figure 3 presented four families as bars of
    equal weight with no significance annotation, so the figure asserted four findings
    where the analysis supported one. Every bar here carries its q-value, and the
    families whose absence is fully explained by annotation depth are shown alongside.
    """
    av = pd.read_csv(tables / f"{scope}_avoidance.csv")

    # The four families the submitted manuscript reported as avoided are shown alongside
    # the families that actually reach significance. Without them the panel would carry
    # only significant bars and the reader could not see what the correction changed.
    claimed = ["PF03815", "PF03805", "PF11556", "PF08373"]
    top = av.nlargest(12, "excess_absence")
    shown = pd.concat([top, av[av["domain_accession"].isin(claimed)]]).drop_duplicates("domain_accession")
    shown = shown.sort_values("excess_absence").reset_index(drop=True)
    y = np.arange(len(shown))

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    colours = [ACCENT if s else MUTED for s in shown["significant_q05"]]
    ax.barh(y, shown["n_species_absent"], color=colours, height=0.72)
    ax.scatter(
        shown["expected_species_absent"],
        y,
        color=WARN,
        s=20,
        zorder=3,
        marker="D",
        label="Expected absences under the depth-conditioned null",
    )
    labels = []
    for acc, name in zip(shown["domain_accession"], shown["domain_name"]):
        marker = "  *" if acc in claimed else ""
        labels.append(f"{name}  ({acc}){marker}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Species lacking the family (of 20)")
    ax.set_title(
        "Domain avoidance tested against annotation depth rather than a uniform null",
        loc="left",
        weight="bold",
    )
    for i, (obs, q) in enumerate(zip(shown["n_species_absent"], shown["q_value"])):
        text = "q < 1e-250" if q < 1e-250 else f"q = {q:.2g}"
        ax.text(obs + 0.2, i, text, va="center", fontsize=6.3, color=INK)
    ax.set_xlim(0, 20 * 1.32)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACCENT),
        plt.Rectangle((0, 0), 1, 1, color=MUTED),
        plt.Line2D([], [], color=WARN, marker="D", ls="none", ms=5),
    ]
    ax.legend(
        handles,
        [
            "Observed absences, q < 0.05",
            "Observed absences, not significant",
            "Expected under the depth-conditioned null",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        fontsize=6.8,
    )
    fig.text(
        0.5,
        -0.06,
        "* families reported as avoided in the submitted manuscript. All four are carried by all 20 species "
        "in the reference proteomes, so their avoidance scores are zero.",
        ha="center",
        fontsize=6.8,
        color=INK,
    )

    _save(fig, outdir, "Figure8_domain_avoidance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("results/tables"))
    parser.add_argument("--outdir", type=Path, default=Path("results/figures"))
    parser.add_argument("--summary", type=Path, default=Path("results/analysis_summary.json"))
    parser.add_argument("--scope", default="reference")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())

    figure1_dataset(args.tables, args.outdir)
    figure2_occurrence(args.tables, args.outdir, args.scope)
    figure3_heatmap(args.tables, args.outdir, args.scope)
    figure4_network(args.tables, args.outdir, args.scope, summary)
    figure5_cooccurrence(args.tables, args.outdir, args.scope)
    figure6_pan_core(args.tables, args.outdir, summary)
    figure7_degree_distribution(args.tables, args.outdir, args.scope, summary)
    figure8_avoidance(args.tables, args.outdir, args.scope)


if __name__ == "__main__":
    main()
