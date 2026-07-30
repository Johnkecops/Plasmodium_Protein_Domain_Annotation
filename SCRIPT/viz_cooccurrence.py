#!/usr/bin/env python3
"""
Module: viz_cooccurrence
Purpose: Domain co-occurrence heatmap and top-pairs bar chart.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from viz_common import _empty_figure, _VALID_METRICS


def plot_domain_cooccurrence_heatmap(
    cooc_stats: pd.DataFrame,
    metric: str = "jaccard",
    top_n: int = 30,
) -> go.Figure:
    """
    Symmetric heatmap of pairwise domain co-occurrence.

    Parameters
    ----------
    cooc_stats : output of compute_domain_cooccurrence_stats()
    metric     : 'jaccard', 'lift', or 'pmi'
    top_n      : limit to the top_n most frequent domains (by total n_AB)
    """
    if cooc_stats.empty:
        return _empty_figure("No domain co-occurrence data available.")

    if metric not in _VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(_VALID_METRICS)!r}, got {metric!r}")

    # Find most represented domains in pairs
    freq = (
        pd.concat([
            cooc_stats[["domain_A", "n_AB"]].rename(columns={"domain_A": "domain"}),
            cooc_stats[["domain_B", "n_AB"]].rename(columns={"domain_B": "domain"}),
        ])
        .groupby("domain")["n_AB"].sum()
        .nlargest(top_n)
    )
    top_domains = freq.index.tolist()

    sub = cooc_stats[
        cooc_stats["domain_A"].isin(top_domains) & cooc_stats["domain_B"].isin(top_domains)
    ][["domain_A", "domain_B", metric]].copy()

    # Build symmetric matrix via pivot (stack both directions, no iterrows)
    both = pd.concat([
        sub,
        sub.rename(columns={"domain_A": "domain_B", "domain_B": "domain_A"}),
    ], ignore_index=True)
    mat = (
        both.pivot_table(index="domain_A", columns="domain_B", values=metric, aggfunc="first")
        .reindex(index=top_domains, columns=top_domains)
    )
    arr = mat.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 0.0 if metric in ("jaccard", "pmi") else 1.0)
    mat = pd.DataFrame(arr, index=mat.index, columns=mat.columns)
    mat = mat.fillna(0.0)  # zero co-occurrence, not missing data

    metric_labels = {
        "jaccard": "Jaccard similarity",
        "lift":    "Lift (> 1 = co-enriched)",
        "pmi":     "PMI log2",
    }

    fig = px.imshow(
        mat,
        color_continuous_scale="Blues" if metric == "jaccard" else "RdBu_r",
        color_continuous_midpoint=1.0 if metric == "lift" else 0.0,
        labels=dict(color=metric_labels.get(metric, metric)),
        title=f"Domain co-occurrence heatmap — {metric_labels.get(metric, metric)}",
        aspect="auto",
    )
    fig.update_layout(
        height=max(500, len(top_domains) * 24),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        plot_bgcolor="white",
    )
    return fig


def plot_domain_cooccurrence_top_pairs(
    cooc_stats: pd.DataFrame,
    metric: str = "lift",
    top_n: int = 25,
    sig_only: bool = False,
    min_n_AB: int = 0,
) -> go.Figure:
    """
    Horizontal bar chart of top domain pairs ranked by chosen metric.

    Parameters
    ----------
    cooc_stats : output of compute_domain_cooccurrence_stats()
    metric     : column to rank by ('lift', 'jaccard', 'pmi')
    top_n      : number of pairs to show
    sig_only   : if True, restrict to pairs with fisher_pvalue_adj < 0.05
    min_n_AB   : minimum co-occurrence count; filters rare pairs before ranking
    """
    if cooc_stats.empty:
        return _empty_figure("No domain co-occurrence pairs found.")

    df = cooc_stats.copy()
    if sig_only:
        if "fisher_pvalue_adj" not in df.columns:
            warnings.warn(
                "sig_only=True requested but 'fisher_pvalue_adj' column absent; "
                "significance filter skipped.",
                UserWarning,
                stacklevel=2,
            )
        else:
            df = df[df["fisher_pvalue_adj"] < 0.05]

    if min_n_AB > 0 and "n_AB" in df.columns:
        df = df[df["n_AB"] >= min_n_AB]

    if df.empty:
        return _empty_figure("No statistically significant co-occurrence pairs (adj p < 0.05).")

    df = df.sort_values(metric, ascending=False).head(top_n).copy()
    df["pair"] = df["domain_A"] + "  ×  " + df["domain_B"]
    df = df.sort_values(metric)  # ascending for horizontal bar (longest bar at top)

    has_padj = "fisher_pvalue_adj" in df.columns
    hover = {
        "n_AB": True,
        "n_A": True,
        "n_B": True,
        metric: ":.4f",
    }
    if has_padj:
        hover["fisher_pvalue_adj"] = ":.2e"

    metric_labels = {
        "lift":    "Lift (> 1 = co-enriched)",
        "jaccard": "Jaccard similarity",
        "pmi":     "PMI log2",
    }

    fig = px.bar(
        df,
        x=metric,
        y="pair",
        orientation="h",
        color=metric,
        color_continuous_scale="Teal",
        labels={
            "pair":              "Domain pair",
            metric:              metric_labels.get(metric, metric),
            "n_AB":              "Co-occurring proteins",
            "n_A":               "Proteins with domain A",
            "n_B":               "Proteins with domain B",
            "fisher_pvalue_adj": "Adj p-value (BH-FDR)",
        },
        title=f"Top {len(df)} domain pairs by {metric_labels.get(metric, metric)}",
        hover_data=hover,
    )
    fig.update_layout(
        height=max(420, len(df) * 28),
        xaxis_title=metric_labels.get(metric, metric),
        yaxis_title=None,
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE")

    # Add vertical reference line for lift=1 or pmi=0
    if metric in ("lift", "pmi"):
        ref = 1.0 if metric == "lift" else 0.0
        fig.add_vline(
            x=ref,
            line_dash="dash",
            line_color="#888888",
            annotation_text="Independence",
            annotation_position="top right",
        )

    return fig
