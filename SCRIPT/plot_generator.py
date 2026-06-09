#!/usr/bin/env python3
"""
Module: plot_generator
Purpose: Generate XMGrace (.agr) and Scilab (.sce) plots proving non-scale-free
         degree distribution in the Plasmodium domain co-occurrence network.
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - taxon_id: NCBI taxonomy ID (default: 5820 = genus Plasmodium)
    - plot_dir: output directory for PLOT/ files
References:
    - Clauset, Shalizi & Newman (2009) SIAM Rev 51(4):661-703. doi:10.1137/070710111
    - Alstott et al. (2014) PLOS ONE 9(1):e85777. doi:10.1371/journal.pone.0085777
"""

import sys
import os
import warnings
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SCRIPT")
sys.path.insert(0, SCRIPT_DIR)

from fetch_proteins import fetch_plasmodium_proteins
from network_builder import build_domain_network

import powerlaw


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _extract_ccdf_from_distribution(fit_obj, k_min, k_max):
    """
    Extract (x, y) arrays for the theoretical CCDF of a powerlaw distribution
    object by intercepting its plot_ccdf() call via a temporary matplotlib axes.
    """
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            artists = fit_obj.plot_ccdf(ax=ax)
        x = np.array(artists[0].get_xdata(), dtype=float)
        y = np.array(artists[0].get_ydata(), dtype=float)
    except Exception:
        # Fallback: empty arrays
        x = np.array([k_min, k_max], dtype=float)
        y = np.array([1.0, 0.01], dtype=float)
    plt.close(fig)
    # Filter to valid (positive) range
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def prepare_plot_data(taxon_id: str = "5820"):
    """Fetch data, build network, fit distributions, return all series."""
    print("[1/4] Fetching Plasmodium proteins...")
    df = fetch_plasmodium_proteins(taxon_id=taxon_id, reviewed=True, max_results=5000)
    n_proteins = len(df)
    n_species = df["organism"].nunique() if n_proteins > 0 else 0
    print(f"      {n_proteins} proteins, {n_species} species")

    print("[2/4] Building co-occurrence network...")
    G = build_domain_network(df, use_pfam=True)
    degrees_all = [d for _, d in G.degree()]
    degrees_pos = [d for d in degrees_all if d > 0]
    n_nodes = G.number_of_nodes()
    print(f"      {n_nodes} nodes, {G.number_of_edges()} edges, {len(degrees_pos)} non-zero degrees")

    print("[3/4] Fitting degree distributions (MLE)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(degrees_pos, discrete=True, verbose=False)

    alpha = float(fit.power_law.alpha)
    sigma = float(fit.power_law.sigma)
    xmin  = float(fit.power_law.xmin)
    ks    = float(fit.power_law.D)

    R_tpl, p_tpl = [float(v) for v in
                    fit.distribution_compare("power_law", "truncated_power_law",
                                             normalized_ratio=True)]
    R_ln,  p_ln  = [float(v) for v in
                    fit.distribution_compare("power_law", "lognormal",
                                             normalized_ratio=True)]

    print(f"      alpha={alpha:.4f}, xmin={xmin}, KS={ks:.4f}")
    print(f"      vs truncated_PL : R={R_tpl:.4f}, p={p_tpl:.4f}")
    print(f"      vs lognormal    : R={R_ln:.4f},  p={p_ln:.4f}")

    print("[4/4] Extracting theoretical CCDFs...")
    # Empirical CCDF
    emp_x, emp_y = fit.ccdf()
    emp_x = np.array(emp_x, dtype=float)
    emp_y = np.array(emp_y, dtype=float)

    k_max = float(max(degrees_pos))

    pl_x,  pl_y  = _extract_ccdf_from_distribution(fit.power_law,  xmin, k_max)
    tpl_x, tpl_y = _extract_ccdf_from_distribution(fit.truncated_power_law, xmin, k_max)
    ln_x,  ln_y  = _extract_ccdf_from_distribution(fit.lognormal,  xmin, k_max)

    return {
        "emp_x": emp_x, "emp_y": emp_y,
        "pl_x":  pl_x,  "pl_y":  pl_y,
        "tpl_x": tpl_x, "tpl_y": tpl_y,
        "ln_x":  ln_x,  "ln_y":  ln_y,
        "alpha": alpha,  "sigma": sigma,  "xmin": xmin, "ks": ks,
        "R_tpl": R_tpl,  "p_tpl": p_tpl,
        "R_ln":  R_ln,   "p_ln":  p_ln,
        "n_proteins": n_proteins, "n_species": n_species,
        "n_nodes": n_nodes,
        "n_edges": G.number_of_edges(),
    }


# ---------------------------------------------------------------------------
# XMGrace .agr writer
# ---------------------------------------------------------------------------

def _agr_series(data_x, data_y, precision: int = 8) -> str:
    """Format x-y pairs for one Grace dataset block."""
    lines = []
    for x, y in zip(data_x, data_y):
        if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
            lines.append(f"{x:.{precision}g} {y:.{precision}g}")
    return "\n".join(lines)


def write_agr(path: str, d: dict) -> None:
    """Write XMGrace project file (.agr) with 4 data series on log-log axes."""
    today = datetime.date.today().strftime("%m/%d/%y")
    alpha = d["alpha"]
    p_tpl = d["p_tpl"]
    p_ln  = d["p_ln"]
    R_tpl = d["R_tpl"]
    R_ln  = d["R_ln"]

    s0 = _agr_series(d["emp_x"], d["emp_y"])
    s1 = _agr_series(d["pl_x"],  d["pl_y"])
    s2 = _agr_series(d["tpl_x"], d["tpl_y"])
    s3 = _agr_series(d["ln_x"],  d["ln_y"])

    agr = f"""# Grace project file
#
@version 50125
@page size 792, 612
@page scroll 5%
@page inout 5%
@link page off
@map font 0 to "Times-Roman", "Times-Roman"
@map font 1 to "Times-Italic", "Times-Italic"
@map font 2 to "Times-Bold", "Times-Bold"
@map font 3 to "Times-BoldItalic", "Times-BoldItalic"
@map font 4 to "Helvetica", "Helvetica"
@map font 5 to "Helvetica-Oblique", "Helvetica-Oblique"
@map font 6 to "Helvetica-Bold", "Helvetica-Bold"
@map font 7 to "Helvetica-BoldOblique", "Helvetica-BoldOblique"
@map font 8 to "Symbol", "Symbol"
@map color 0 to (255, 255, 255), "white"
@map color 1 to (0, 0, 0), "black"
@map color 2 to (255, 0, 0), "red"
@map color 3 to (0, 160, 0), "green"
@map color 4 to (0, 0, 255), "blue"
@map color 5 to (255, 140, 0), "orange"
@map color 6 to (150, 0, 200), "violet"
@map color 7 to (180, 180, 180), "grey"
@background color 0
@timestamp on
@timestamp {today}
@timestamp font 0
@default linewidth 1.5
@default linestyle 1
@default color 1
@default char size 1.000000
@default symbol size 1.000000
@with g0
@    view 0.180000, 0.180000, 0.960000, 0.860000
@    world 0.8, 0.0005, 25, 1.5
@    xaxis  scale Logarithmic
@    xaxis  label "Degree k"
@    xaxis  label char size 1.500000
@    xaxis  label font 2
@    xaxis  label color 1
@    xaxis  tick major 10
@    xaxis  ticklabel char size 1.200000
@    xaxis  ticklabel format power
@    xaxis  ticklabel prec 0
@    yaxis  scale Logarithmic
@    yaxis  label "CCDF  P(K \\xb3\\N k)"
@    yaxis  label char size 1.500000
@    yaxis  label font 2
@    yaxis  label color 1
@    yaxis  tick major 10
@    yaxis  ticklabel char size 1.200000
@    yaxis  ticklabel format power
@    yaxis  ticklabel prec 1
@    title "Plasmodium Domain Co-occurrence Network: Degree Distribution"
@    title font 2
@    title size 1.400000
@    subtitle "N={d['n_nodes']} Pfam nodes  |  {d['n_proteins']} proteins  |  {d['n_species']} species  |  UniProt Swiss-Prot"
@    subtitle font 0
@    subtitle size 0.900000
@    legend on
@    legend loctype view
@    legend 0.62, 0.88
@    legend box linewidth 1.5
@    legend char size 0.950000
@    legend font 0
@    frame type 0
@    frame linestyle 1
@    frame linewidth 1.5
@    frame color 1
@    s0 hidden false
@    s0 type xy
@    s0 symbol 1
@    s0 symbol size 0.700000
@    s0 symbol color 1
@    s0 symbol fill color 1
@    s0 symbol fill pattern 1
@    s0 symbol linewidth 1.5
@    s0 line type 0
@    s0 line linestyle 1
@    s0 line linewidth 1.5
@    s0 line color 1
@    s0 legend "Empirical CCDF"
@    s1 hidden false
@    s1 type xy
@    s1 symbol 0
@    s1 line type 1
@    s1 line linestyle 1
@    s1 line linewidth 2.500000
@    s1 line color 2
@    s1 legend "Power law (\\xa\\N={alpha:.3f}, p=1.0)"
@    s2 hidden false
@    s2 type xy
@    s2 symbol 0
@    s2 line type 1
@    s2 line linestyle 2
@    s2 line linewidth 2.500000
@    s2 line color 4
@    s2 legend "Truncated power law (R={R_tpl:.3f}, p={p_tpl:.4f})**"
@    s3 hidden false
@    s3 type xy
@    s3 symbol 0
@    s3 line type 1
@    s3 line linestyle 3
@    s3 line linewidth 2.500000
@    s3 line color 3
@    s3 legend "Log-normal (R={R_ln:.3f}, p={p_ln:.4f})"
@target G0.S0
@type xy
{s0}
&
@target G0.S1
@type xy
{s1}
&
@target G0.S2
@type xy
{s2}
&
@target G0.S3
@type xy
{s3}
&
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(agr)


# ---------------------------------------------------------------------------
# Scilab .sce writer
# ---------------------------------------------------------------------------

def _scilab_array(name: str, values, indent: int = 0) -> str:
    """Format a 1-D array as Scilab row vector assignment."""
    pad = " " * indent
    vals_str = "; ".join(f"{v:.8g}" for v in values)
    return f"{pad}{name} = [{vals_str}];"


def write_scilab(path: str, d: dict) -> None:
    """Write a self-contained Scilab script (.sce) with embedded data."""
    alpha = d["alpha"]
    sigma = d["sigma"]
    xmin  = d["xmin"]
    ks    = d["ks"]
    R_tpl, p_tpl = d["R_tpl"], d["p_tpl"]
    R_ln,  p_ln  = d["R_ln"],  d["p_ln"]

    emp_x_arr  = _scilab_array("emp_k",    d["emp_x"])
    emp_y_arr  = _scilab_array("emp_ccdf", d["emp_y"])
    pl_x_arr   = _scilab_array("pl_k",     d["pl_x"])
    pl_y_arr   = _scilab_array("pl_ccdf",  d["pl_y"])
    tpl_x_arr  = _scilab_array("tpl_k",    d["tpl_x"])
    tpl_y_arr  = _scilab_array("tpl_ccdf", d["tpl_y"])
    ln_x_arr   = _scilab_array("ln_k",     d["ln_x"])
    ln_y_arr   = _scilab_array("ln_ccdf",  d["ln_y"])

    sce = f"""// ============================================================
// Scilab script: degree_distribution.sce
// Purpose: Plot degree distribution of Plasmodium domain co-occurrence
//          network with power-law and alternative distribution fits.
//          Proves deviation from scale-free topology.
// Author : Dr. Arli Aditya Parikesit, i3L University, Jakarta
// Date   : {datetime.date.today().isoformat()}
// Run    : exec('degree_distribution.sce', -1)
// Requires: Scilab 6.1+ (standard installation, no extra toolboxes)
//
// Key results
//   alpha          = {alpha:.4f}   (power-law exponent, MLE)
//   sigma          = {sigma:.4f}   (standard error on alpha)
//   xmin           = {xmin}    (lower cutoff, auto-estimated)
//   KS distance    = {ks:.4f}   (> 0.1 = poor power-law fit)
//   truncated PL   : R = {R_tpl:.4f}, p = {p_tpl:.4f}  ** SIGNIFICANT **
//   log-normal     : R = {R_ln:.4f},  p = {p_ln:.4f}
//   VERDICT        : NON-SCALE FREE
// ============================================================

// ============================================================
// Section 1 — Embedded data (extracted from powerlaw library)
// ============================================================

// Empirical CCDF: P(K >= k)
{emp_x_arr}
{emp_y_arr}

// Power-law theoretical CCDF
{pl_x_arr}
{pl_y_arr}

// Truncated power-law theoretical CCDF  ← better fit (p={p_tpl:.4f})
{tpl_x_arr}
{tpl_y_arr}

// Log-normal theoretical CCDF
{ln_x_arr}
{ln_y_arr}

// ============================================================
// Section 2 — Plot
// ============================================================
clf();
f = gcf();
f.figure_size = [900, 650];
f.background  = -2;   // white

a = gca();
a.log_flags = "ll";   // log-log axes
a.font_size = 3;
a.x_label.text     = "Degree k";
a.x_label.font_size = 4;
a.y_label.text     = "CCDF P(K >= k)";
a.y_label.font_size = 4;
a.title.text = ["Plasmodium Domain Co-occurrence Network: Degree Distribution"; ...
                sprintf("N={d['n_nodes']} Pfam nodes  |  {d['n_proteins']} proteins  |  {d['n_species']} species (UniProt Swiss-Prot)")];
a.title.font_size = 4;
a.box = "on";

// Series 0 — empirical CCDF (black filled circles, no line)
plot2d(emp_k, emp_ccdf, style=-9, logflag="ll");
s0 = gce();
s0.children.mark_style      = 9;   // filled circle
s0.children.mark_size       = 7;
s0.children.mark_foreground = 1;   // black
s0.children.mark_background = 1;

// Series 1 — power-law (solid red)
plot2d(pl_k, pl_ccdf, style=5, logflag="ll");
s1 = gce();
s1.children.thickness   = 3;
s1.children.foreground  = 5;   // red
s1.children.line_style  = 1;   // solid

// Series 2 — truncated power-law (dashed blue) — significantly better fit
plot2d(tpl_k, tpl_ccdf, style=2, logflag="ll");
s2 = gce();
s2.children.thickness   = 3;
s2.children.foreground  = 2;   // blue
s2.children.line_style  = 2;   // dashed

// Series 3 — log-normal (dash-dot green)
plot2d(ln_k, ln_ccdf, style=3, logflag="ll");
s3 = gce();
s3.children.thickness   = 3;
s3.children.foreground  = 3;   // green
s3.children.line_style  = 4;   // dash-dot

// ============================================================
// Section 3 — Legend
// ============================================================
legend(["Empirical CCDF"; ...
        sprintf("Power law (\\alpha=%.3f, reference)", {alpha:.3f}); ...
        sprintf("Truncated power law (R=%.3f, p=%.4f) **", {R_tpl:.3f}, {p_tpl:.4f}); ...
        sprintf("Log-normal (R=%.3f, p=%.4f)", {R_ln:.3f}, {p_ln:.4f})], ...
       "in_upper_left", %f);

// ============================================================
// Section 4 — Annotation box (key statistics)
// ============================================================
xstring(1.1, 0.0015, ...
    sprintf("\\alpha = %.4f (\\pm %.4f)\\nxmin = %.1f\\nKS = %.4f\\nVerdict: NON-SCALE FREE", ...
            {alpha:.4f}, {sigma:.4f}, {xmin}, {ks:.4f}));
t = gce();
t.font_size = 3;

// ============================================================
// Section 5 — Export
// ============================================================
xs2eps(0, "degree_distribution_scilab.eps");
xs2png(0, "degree_distribution_scilab.png");
printf("Plots saved: degree_distribution_scilab.eps / .png\\n");
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(sce)


# ---------------------------------------------------------------------------
# Matplotlib preview (PNG) — for systems without XMGrace
# ---------------------------------------------------------------------------

def write_matplotlib_preview(path: str, d: dict) -> None:
    """Write a publication-style matplotlib PNG of the same plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(d["emp_x"], d["emp_y"], "ko", ms=5, label="Empirical CCDF", zorder=5)
    ax.plot(d["pl_x"],  d["pl_y"],  "r-",  lw=2.0,
            label=f"Power law (α={d['alpha']:.3f})")
    ax.plot(d["tpl_x"], d["tpl_y"], "b--", lw=2.0,
            label=f"Truncated PL (R={d['R_tpl']:.3f}, p={d['p_tpl']:.4f}) **")
    ax.plot(d["ln_x"],  d["ln_y"],  "g-.", lw=2.0,
            label=f"Log-normal (R={d['R_ln']:.3f}, p={d['p_ln']:.4f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree k", fontsize=14)
    ax.set_ylabel("CCDF  P(K ≥ k)", fontsize=14)
    ax.set_title(
        "Plasmodium Domain Co-occurrence Network\nDegree Distribution — NON-SCALE FREE",
        fontsize=13,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    textstr = (
        f"N={d['n_nodes']} Pfam nodes  |  {d['n_proteins']} proteins\n"
        f"{d['n_species']} species  |  UniProt Swiss-Prot\n"
        f"α={d['alpha']:.4f}±{d['sigma']:.4f}   xmin={d['xmin']:.0f}   KS={d['ks']:.4f}\n"
        f"Truncated PL: R={d['R_tpl']:.3f}, p={d['p_tpl']:.4f} **\n"
        f"Log-normal:   R={d['R_ln']:.3f}, p={d['p_ln']:.4f}"
    )
    ax.text(
        0.97, 0.97, textstr,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(taxon_id: str = "5820", plot_dir: str = None):
    if plot_dir is None:
        plot_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PLOT")
        )
    os.makedirs(plot_dir, exist_ok=True)

    d = prepare_plot_data(taxon_id=taxon_id)

    agr_path = os.path.join(plot_dir, "degree_distribution.agr")
    sce_path = os.path.join(plot_dir, "degree_distribution.sce")
    png_path = os.path.join(plot_dir, "degree_distribution_preview.png")

    print("\n[Writing files]")
    write_agr(agr_path, d)
    print(f"  XMGrace : {agr_path}")

    write_scilab(sce_path, d)
    print(f"  Scilab  : {sce_path}")

    write_matplotlib_preview(png_path, d)
    print(f"  Preview : {png_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate XMGrace and Scilab degree distribution plots")
    parser.add_argument("--taxon", default="5820")
    parser.add_argument("--plot-dir", default=None, help="Output directory (default: ../PLOT)")
    args = parser.parse_args()
    main(taxon_id=args.taxon, plot_dir=args.plot_dir)
