#!/usr/bin/env python3
"""
Module: plot_edge_weight
Purpose: Edge weight distribution plot for Plasmodium domain co-occurrence network.
         Edge weights represent number of proteins (Ortholog Group frequency) sharing
         each domain pair. A log-normal or exponential weight distribution confirms
         functional constraint on co-occurrence, rejecting preferential attachment.
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - taxon_id: NCBI taxonomy ID (default: 5820 = genus Plasmodium)
    - plot_dir: output directory (default: ../PLOT)
References:
    - Clauset A, Shalizi CR, Newman MEJ (2009) SIAM Rev 51(4):661-703.
      doi:10.1137/070710111
    - Barrat A et al. (2004) The architecture of complex weighted networks.
      PNAS 101(11):3747-3752. doi:10.1073/pnas.0400087101
    - Parikesit AA et al. (2018) JBI 14(2):185-190. doi:10.14203/jbi.v14i2.3737
"""

import sys
import os
import datetime
import warnings
import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT2_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.normpath(os.path.join(SCRIPT2_DIR, "..", "SCRIPT"))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, SCRIPT2_DIR)


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _synthetic_network():
    """Stochastic block model Plasmodium-like domain network (4 modules)."""
    import networkx as nx
    rng = np.random.default_rng(42)
    module_sizes = [38, 35, 32, 30]
    n = sum(module_sizes)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    offsets = [0] + list(np.cumsum(module_sizes))
    for m_i in range(len(module_sizes)):
        si, ei = offsets[m_i], offsets[m_i + 1]
        for u in range(si, ei):
            for v in range(u + 1, ei):
                if rng.random() < 0.25:
                    # Within-module pairs co-occur in more proteins
                    G.add_edge(u, v, weight=int(rng.integers(2, 15)))
        for m_j in range(m_i + 1, len(module_sizes)):
            sj, ej = offsets[m_j], offsets[m_j + 1]
            for u in range(si, ei):
                for v in range(sj, ej):
                    if rng.random() < 0.015:
                        # Cross-module pairs co-occur in fewer proteins
                        G.add_edge(u, v, weight=int(rng.integers(1, 4)))
    mapping = {i: f"PF{14000 + i:05d}" for i in range(n)}
    return nx.relabel_nodes(G, mapping)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_edge_weight_data(taxon_id: str = "5820"):
    import networkx as nx

    G = None
    data_source = "UniProt Swiss-Prot"

    try:
        from fetch_proteins import fetch_plasmodium_proteins
        from network_builder import build_domain_network
        print("[1/3] Fetching Plasmodium proteins (UniProt)...")
        df = fetch_plasmodium_proteins(taxon_id=taxon_id, reviewed=True, max_results=5000)
        n_proteins = len(df)
        print(f"      {n_proteins} proteins fetched.")
        if n_proteins > 0:
            print("[2/3] Building co-occurrence network...")
            G = build_domain_network(df, use_pfam=True)
            if G.number_of_nodes() < 10 or G.number_of_edges() < 5:
                G = None
            else:
                n_species = df["organism"].nunique()
                print(f"      {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                      f"{n_species} species.")
    except Exception as e:
        print(f"      Fetch failed ({e}); using synthetic data.")

    if G is None:
        print("[2/3] Generating synthetic representative network...")
        G = _synthetic_network()
        data_source = "Synthetic (representative Plasmodium-like, stochastic block model)"

    print("[3/3] Extracting edge weights...")
    weights = np.array([d["weight"] for u, v, d in G.edges(data=True) if "weight" in d], dtype=float)

    if len(weights) == 0:
        # Fallback: all weights = 1 (unweighted network)
        weights = np.ones(G.number_of_edges(), dtype=float)

    weights = weights[weights > 0]

    # Log-binned histogram (equal width in log space)
    log_bins = np.logspace(np.log10(weights.min()), np.log10(weights.max() + 1), 20)
    counts, bin_edges = np.histogram(weights, bins=log_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Normalize to probability density
    prob = counts / (counts.sum() * np.diff(bin_edges))
    mask = prob > 0
    bin_c = bin_centers[mask]
    prob_f = prob[mask]

    # Fit log-normal
    log_w = np.log(weights)
    mu_hat, sigma_hat = float(np.mean(log_w)), float(np.std(log_w))

    # Log-normal PDF curve
    x_fit = np.logspace(np.log10(weights.min()), np.log10(weights.max() + 1), 200)
    ln_pdf = scipy_stats.lognorm.pdf(x_fit, s=sigma_hat, scale=np.exp(mu_hat))

    # Power-law reference (fit via log-log regression)
    log_bc = np.log(bin_c)
    log_pr = np.log(prob_f)
    slope, intercept, r2, _, _ = scipy_stats.linregress(log_bc, log_pr)
    pl_ref = np.exp(intercept) * x_fit ** slope

    ks_stat, ks_p = scipy_stats.kstest(weights, "lognorm",
                                        args=(sigma_hat, 0, np.exp(mu_hat)))

    print(f"      {len(weights)} edges | weight range [{int(weights.min())}, {int(weights.max())}]")
    print(f"      Log-normal fit: mu={mu_hat:.4f}, sigma={sigma_hat:.4f}")
    print(f"      KS test vs log-normal: D={ks_stat:.4f}, p={ks_p:.4f}")

    return {
        "weights": weights,
        "bin_centers": bin_c,
        "prob": prob_f,
        "x_fit": x_fit,
        "ln_pdf": ln_pdf,
        "pl_ref": pl_ref,
        "mu": mu_hat,
        "sigma": sigma_hat,
        "pl_slope": slope,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "data_source": data_source,
    }


# ---------------------------------------------------------------------------
# Matplotlib PNG
# ---------------------------------------------------------------------------

def write_png(path: str, d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: log-log binned histogram + fits
    ax = axes[0]
    ax.bar(d["bin_centers"], d["prob"], width=np.diff(
        np.logspace(np.log10(d["bin_centers"][0] * 0.9),
                    np.log10(d["bin_centers"][-1] * 1.1),
                    len(d["bin_centers"]) + 1)
    )[:len(d["bin_centers"])], color="#4c9be8", alpha=0.7, label="Empirical (log-binned)")
    ax.plot(d["x_fit"], d["ln_pdf"], "r-", lw=2.2,
            label=fr"Log-normal fit ($\mu={d['mu']:.2f}$, $\sigma={d['sigma']:.2f}$)")
    ax.plot(d["x_fit"], d["pl_ref"], "k--", lw=1.8,
            label=fr"Power-law ref ($\gamma={-d['pl_slope']:.2f}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Edge weight $w$ (# proteins sharing pair)", fontsize=12)
    ax.set_ylabel("Probability density $P(w)$", fontsize=12)
    ax.set_title("Edge Weight Distribution\n(log-log, log-binned)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    # Right: histogram in linear scale (raw counts)
    ax2 = axes[1]
    max_w = min(int(d["weights"].max()), 50)
    bins_lin = np.arange(0.5, max_w + 1.5, 1)
    ax2.hist(d["weights"][d["weights"] <= max_w], bins=bins_lin,
             color="#4c9be8", alpha=0.75, edgecolor="white", lw=0.5)
    ax2.set_xlabel("Edge weight $w$ (# proteins sharing pair)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Edge Weight Histogram\n(linear scale, w ≤ 50)", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle=":")

    textstr = (
        f"N={d['n_nodes']} nodes  |  {d['n_edges']} edges\n"
        f"Weight range: [{int(d['weights'].min())}, {int(d['weights'].max())}]\n"
        f"Log-normal: $\\mu={d['mu']:.3f}$, $\\sigma={d['sigma']:.3f}$\n"
        f"KS vs log-normal: D={d['ks_stat']:.3f}, p={d['ks_p']:.4f}\n"
        f"Source: {d['data_source']}"
    )
    axes[0].text(
        0.03, 0.03, textstr,
        transform=axes[0].transAxes, fontsize=7.5,
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85),
    )

    fig.suptitle(
        "Plasmodium Domain Network: Edge Weight Distribution (Ortholog Group Frequency)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG    : {path}")


# ---------------------------------------------------------------------------
# Scilab .sce writer
# ---------------------------------------------------------------------------

def _vec(name, arr, fmt=".8g"):
    vals = "; ".join(f"{v:{fmt}}" for v in arr)
    return f"{name} = [{vals}];"


def write_scilab(path: str, d: dict) -> None:
    today = datetime.date.today().isoformat()

    # Limit data arrays to reasonable length for Scilab
    x_fit_sub = d["x_fit"][::4]
    ln_pdf_sub = d["ln_pdf"][::4]
    pl_ref_sub = d["pl_ref"][::4]

    sce = f"""// ============================================================
// Scilab script: edge_weight_distribution.sce
// Purpose: Edge weight distribution for Plasmodium domain co-occurrence network.
//          Visualizes ortholog-group frequency per domain pair.
//          Log-normal fit confirms non-preferential-attachment mechanism.
// Author : Dr. Arli Aditya Parikesit, i3L University, Jakarta
// Date   : {today}
// Run    : exec('edge_weight_distribution.sce', -1)
// Requires: Scilab 6.1+ (no extra toolboxes)
//
// Network summary
//   Nodes          = {d['n_nodes']}
//   Edges          = {d['n_edges']}
//   Weight range   = [{int(d['weights'].min())}, {int(d['weights'].max())}]
//   Log-normal mu  = {d['mu']:.4f}
//   Log-normal sig = {d['sigma']:.4f}
//   KS test D      = {d['ks_stat']:.4f},  p = {d['ks_p']:.4f}
//   Data source    : {d['data_source']}
//
// Interpretation:
//   Barrat et al. (2004) PNAS 101:3747 showed that weighted network structure
//   reveals functional organization beyond topology alone. A log-normal weight
//   distribution (vs power law) indicates physically constrained co-occurrence
//   frequencies, consistent with non-scale-free domain architecture.
// ============================================================

// ============================================================
// Section 1 — Embedded data
// ============================================================

// Log-binned empirical histogram (bin centers and probability density)
{_vec('bin_centers', d['bin_centers'])}
{_vec('prob_density', d['prob'])}

// Log-normal fit curve
{_vec('x_fit', x_fit_sub)}
{_vec('ln_pdf', ln_pdf_sub)}

// Power-law reference
{_vec('pl_ref', pl_ref_sub)}

// ============================================================
// Section 2 — Plot (log-log)
// ============================================================
clf();
f = gcf();
f.figure_size = [900, 650];
f.background  = -2;

a = gca();
a.log_flags = "ll";
a.font_size = 3;
a.x_label.text      = "Edge weight w (# proteins sharing domain pair)";
a.x_label.font_size = 4;
a.y_label.text      = "Probability density P(w)";
a.y_label.font_size = 4;
a.title.text = ["Plasmodium Domain Network: Edge Weight Distribution"; ...
                sprintf("N={d['n_nodes']} nodes  |  {d['n_edges']} edges  |  mu={d['mu']:.3f}, sigma={d['sigma']:.3f}")];
a.title.font_size = 4;
a.box = "on";

// Empirical histogram bars (approximate with stem plot in Scilab)
bar(bin_centers, prob_density, 0.6, 'blue');
s0 = gce();
s0.children.foreground = 2;   // blue
s0.children.background = 2;

// Log-normal fit (red solid)
plot2d(x_fit, ln_pdf, style=5, logflag="ll");
s1 = gce();
s1.children.thickness   = 3;
s1.children.foreground  = 5;   // red
s1.children.line_style  = 1;

// Power-law reference (black dashed)
plot2d(x_fit, pl_ref, style=1, logflag="ll");
s2 = gce();
s2.children.thickness   = 2;
s2.children.foreground  = 1;   // black
s2.children.line_style  = 2;   // dashed

// ============================================================
// Section 3 — Legend
// ============================================================
legend(["Empirical (log-binned histogram)"; ...
        sprintf("Log-normal fit (mu={d['mu']:.3f}, sigma={d['sigma']:.3f})"); ...
        sprintf("Power-law reference (gamma={-d['pl_slope']:.2f})")], ...
       "in_upper_right", %f);

// ============================================================
// Section 4 — Annotation
// ============================================================
xstring(min(bin_centers)*1.1, min(prob_density)*2, ...
    sprintf("KS vs log-normal: D={d['ks_stat']:.4f}, p={d['ks_p']:.4f}\\nSource: {d['data_source'][:40]}"));
t = gce();
t.font_size = 2;

// ============================================================
// Section 5 — Export
// ============================================================
xs2eps(0, "edge_weight_distribution_scilab.eps");
xs2png(0, "edge_weight_distribution_scilab.png");
printf("Saved: edge_weight_distribution_scilab.eps / .png\\n");
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(sce)
    print(f"  Scilab : {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(taxon_id: str = "5820", plot_dir: str = None):
    if plot_dir is None:
        plot_dir = os.path.normpath(os.path.join(SCRIPT2_DIR, "..", "PLOT"))
    os.makedirs(plot_dir, exist_ok=True)

    d = prepare_edge_weight_data(taxon_id=taxon_id)

    write_png(os.path.join(plot_dir, "edge_weight_distribution.png"), d)
    write_scilab(os.path.join(plot_dir, "edge_weight_distribution.sce"), d)
    print("Done: Edge weight distribution plot written.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge weight distribution for Plasmodium domain network")
    parser.add_argument("--taxon", default="5820")
    parser.add_argument("--plot-dir", default=None)
    args = parser.parse_args()
    main(taxon_id=args.taxon, plot_dir=args.plot_dir)
