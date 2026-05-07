#!/usr/bin/env python3
"""
Module: Visualization functions for Plasmodium protein domain analysis.
Purpose: Generate interactive Plotly charts for domain occurrence,
         avoidance, species comparison, and protein domain maps.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


# ─── Colour constants ─────────────────────────────────────────────────────────
_BLUE      = "#2E86AB"
_RED       = "#E84855"
_AMBER     = "#F18F01"
_PURPLE    = "#A23B72"
_TEAL      = "#3BB273"
_QUALSET1  = px.colors.qualitative.Set1
_QUALSET2  = px.colors.qualitative.Set2


# ─── Species overview ─────────────────────────────────────────────────────────
def plot_species_overview(summary_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: total proteins, proteins with domains, unique domain
    types — one group per Plasmodium species.
    """
    if summary_df.empty:
        return _empty_figure("No species data available.")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Total proteins",
        x=summary_df["organism"],
        y=summary_df["n_proteins"],
        marker_color=_BLUE,
    ))
    fig.add_trace(go.Bar(
        name="With domain annotations",
        x=summary_df["organism"],
        y=summary_df["n_with_domains"],
        marker_color=_PURPLE,
    ))
    fig.add_trace(go.Bar(
        name="Unique domain types",
        x=summary_df["organism"],
        y=summary_df["unique_domains"],
        marker_color=_AMBER,
    ))
    fig.update_layout(
        barmode="group",
        title="Protein and domain coverage per Plasmodium species",
        xaxis=dict(tickangle=30, title="Species"),
        yaxis_title="Count",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEEEEE")
    return fig


def plot_domain_length_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Box plot of protein length distribution per species.
    """
    if df.empty:
        return _empty_figure("No protein data.")

    fig = px.box(
        df,
        x="organism",
        y="length",
        color="organism",
        color_discrete_sequence=_QUALSET2,
        title="Protein length distribution per Plasmodium species",
        labels={"length": "Protein length (aa)", "organism": "Species"},
        points="outliers",
    )
    fig.update_layout(
        xaxis=dict(tickangle=30),
        showlegend=False,
        height=450,
        plot_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#EEEEEE")
    return fig


# ─── Domain occurrence ────────────────────────────────────────────────────────
def plot_domain_occurrence_bar(
    occurrence_df: pd.DataFrame,
    top_n: int = 20,
    color_by: str = "count",
) -> go.Figure:
    """
    Horizontal bar chart of top domain occurrences.

    Parameters
    ----------
    occurrence_df : output of compute_domain_occurrence()
    top_n         : number of domains to display
    color_by      : 'count' or 'species_count'
    """
    if occurrence_df.empty:
        return _empty_figure("No domain annotations found.")

    df = occurrence_df.head(top_n).sort_values("count")  # ascending → longest bar at top

    fig = px.bar(
        df,
        x="count",
        y="domain_name",
        orientation="h",
        color=color_by,
        color_continuous_scale="Blues",
        labels={
            "count": "Number of proteins",
            "domain_name": "Domain",
            "species_count": "Species count",
            "pct_proteins": "% of proteome",
        },
        title=f"Top {len(df)} domain occurrences in Plasmodium spp.",
        hover_data={"count": True, "species_count": True, "pct_proteins": True},
    )
    fig.update_layout(
        height=max(420, len(df) * 26),
        xaxis_title="Number of proteins",
        yaxis_title=None,
        coloraxis_colorbar=dict(
            title="Species count" if color_by == "species_count" else "Protein count"
        ),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    return fig


def plot_species_domain_heatmap(matrix: pd.DataFrame, max_domains: int = 50) -> go.Figure:
    """
    Heatmap of species × domain occurrence counts.

    Rows = species, columns = domain names (limited to top max_domains by total).
    """
    if matrix.empty:
        return _empty_figure("No domain-species matrix available.")

    # Limit columns to top domains by total count
    totals = matrix.sum(axis=0)
    top_cols = totals.nlargest(max_domains).index.tolist()
    m = matrix[top_cols]

    fig = px.imshow(
        m,
        labels=dict(x="Domain", y="Species", color="Protein count"),
        color_continuous_scale="YlOrRd",
        title="Domain occurrence heatmap across Plasmodium species",
        aspect="auto",
    )
    fig.update_layout(
        height=max(350, len(m.index) * 55),
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        coloraxis_colorbar=dict(title="Proteins"),
        plot_bgcolor="white",
    )
    return fig


# ─── Domain avoidance ─────────────────────────────────────────────────────────
def plot_domain_avoidance(avoidance_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Horizontal bar chart of domain avoidance scores coloured by -log10(adj p-value).
    """
    if avoidance_df.empty:
        return _empty_figure("No domain avoidance patterns detected.")

    df = avoidance_df.head(top_n).sort_values("avoidance_score").copy()

    has_pval = "avoidance_pvalue_adj" in df.columns
    if has_pval:
        df["neg_log10_padj"] = -np.log10(df["avoidance_pvalue_adj"].clip(lower=1e-300))
        color_col = "neg_log10_padj"
        color_label = "-log10(adj p-value)"
        colorscale = "Reds"
        hover_extra = {
            "n_species_present": True,
            "n_species_absent": True,
            "avoidance_score": ":.3f",
            "min_binom_pvalue": ":.2e",
            "avoidance_pvalue_adj": ":.2e",
            "neg_log10_padj": False,
        }
    else:
        color_col = "avoidance_score"
        color_label = "Avoidance score"
        colorscale = "RdYlGn_r"
        hover_extra = {
            "n_species_present": True,
            "n_species_absent": True,
            "avoidance_score": ":.3f",
        }

    fig = px.bar(
        df,
        x="avoidance_score",
        y="domain_name",
        orientation="h",
        color=color_col,
        color_continuous_scale=colorscale,
        range_color=[0, 1] if not has_pval else None,
        labels={
            "avoidance_score": "Avoidance score",
            "domain_name": "Domain",
            "n_species_present": "Species with domain",
            "n_species_absent": "Species without domain",
            "neg_log10_padj": "-log10(adj p)",
            "min_binom_pvalue": "Min binomial p",
            "avoidance_pvalue_adj": "Adj p-value (BH)",
        },
        title=f"Domain avoidance in Plasmodium spp. (top {len(df)} by score)",
        hover_data=hover_extra,
    )
    fig.update_layout(
        height=max(420, len(df) * 26),
        xaxis=dict(range=[0, 1], title="Avoidance score (fraction of species lacking domain)"),
        yaxis_title=None,
        coloraxis_colorbar=dict(title=color_label),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    return fig


# ─── Domain co-occurrence ─────────────────────────────────────────────────────
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

    # Build symmetric matrix for the chosen metric
    sub = cooc_stats[
        cooc_stats["domain_A"].isin(top_domains) & cooc_stats["domain_B"].isin(top_domains)
    ].copy()

    mat = pd.DataFrame(
        np.nan, index=top_domains, columns=top_domains, dtype=float
    )
    np.fill_diagonal(mat.values, 0.0 if metric in ("jaccard", "pmi") else 1.0)

    for _, row in sub.iterrows():
        val = row[metric]
        mat.loc[row["domain_A"], row["domain_B"]] = val
        mat.loc[row["domain_B"], row["domain_A"]] = val

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
) -> go.Figure:
    """
    Horizontal bar chart of top domain pairs ranked by chosen metric.

    Parameters
    ----------
    cooc_stats : output of compute_domain_cooccurrence_stats()
    metric     : column to rank by ('lift', 'jaccard', 'pmi')
    top_n      : number of pairs to show
    sig_only   : if True, restrict to pairs with fisher_pvalue_adj < 0.05
    """
    if cooc_stats.empty:
        return _empty_figure("No domain co-occurrence pairs found.")

    df = cooc_stats.copy()
    if sig_only and "fisher_pvalue_adj" in df.columns:
        df = df[df["fisher_pvalue_adj"] < 0.05]

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


# ─── Protein domain map ───────────────────────────────────────────────────────
def plot_protein_domain_map(protein_row: pd.Series) -> go.Figure:
    """
    Linear protein domain map for a single protein entry.
    Domains are drawn as coloured boxes along the sequence backbone.
    """
    length = int(protein_row.get("length", 0))
    domains = protein_row.get("domains", [])
    name = protein_row.get("protein_name", "Unknown")
    acc = protein_row.get("accession", "")

    if length == 0:
        return _empty_figure("Sequence length unavailable.")

    fig = go.Figure()

    # Grey backbone
    fig.add_shape(
        type="rect",
        x0=0, x1=length,
        y0=0.35, y1=0.65,
        line=dict(color="#BBBBBB", width=1),
        fillcolor="#DDDDDD",
    )

    # N- and C-terminus labels
    fig.add_annotation(x=0,      y=0.5, text="N", showarrow=False, font=dict(size=11, color="#444"))
    fig.add_annotation(x=length, y=0.5, text="C", showarrow=False, font=dict(size=11, color="#444"))

    unique_names = list(dict.fromkeys(d["name"] for d in domains))
    color_map = {n: _QUALSET1[i % len(_QUALSET1)] for i, n in enumerate(unique_names)}

    for i, domain in enumerate(domains):
        d_name = domain["name"]
        start  = domain["start"]
        end    = domain["end"]
        color  = color_map.get(d_name, "#888888")
        mid    = (start + end) / 2

        fig.add_shape(
            type="rect",
            x0=start, x1=end,
            y0=0.2, y1=0.8,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.85,
        )
        # Annotation above or below alternating to avoid overlap
        ay_offset = -35 if i % 2 == 0 else 35
        fig.add_annotation(
            x=mid, y=0.8 if ay_offset < 0 else 0.2,
            text=f"<b>{d_name}</b><br>{start}–{end}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            arrowwidth=1.5,
            ax=0, ay=ay_offset,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color,
            borderwidth=1,
        )

    fig.update_layout(
        title=f"{name} ({acc})  —  {length} aa",
        xaxis=dict(
            title="Amino acid position",
            range=[-length * 0.02, length * 1.08],
            showgrid=False,
        ),
        yaxis=dict(visible=False, range=[-0.5, 1.8]),
        height=260,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=40),
    )

    # Legend for domain colours
    for d_name, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=color, symbol="square"),
            name=d_name,
            showlegend=True,
        ))
    if color_map:
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.25),
        )

    return fig


# ─── InterPro occurrence ──────────────────────────────────────────────────────
def plot_interpro_occurrence_bar(
    ipr_df: pd.DataFrame,
    ipr_name_map: dict,
    top_n: int = 20,
) -> go.Figure:
    """
    Horizontal bar chart for InterPro cross-reference occurrence.
    Uses human-readable domain names from ipr_name_map where available.
    """
    if ipr_df.empty:
        return _empty_figure("No InterPro cross-references found.")

    df = ipr_df.head(top_n).copy()
    df["label"] = df["interpro_id"].map(lambda x: ipr_name_map.get(x, x))
    df = df.sort_values("count")

    fig = px.bar(
        df,
        x="count",
        y="label",
        orientation="h",
        color="species_count",
        color_continuous_scale="Teal",
        labels={
            "count": "Number of proteins",
            "label": "InterPro entry",
            "species_count": "Species count",
        },
        title=f"Top {len(df)} InterPro entries in Plasmodium spp.",
        hover_data={"interpro_id": True, "count": True, "species_count": True, "pct_proteins": True},
    )
    fig.update_layout(
        height=max(420, len(df) * 26),
        xaxis_title="Number of proteins",
        yaxis_title=None,
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEEEEE")
    return fig


# ─── Helper ───────────────────────────────────────────────────────────────────
def _empty_figure(message: str = "No data.") -> go.Figure:
    """Return a blank figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#888888"),
    )
    fig.update_layout(
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig
