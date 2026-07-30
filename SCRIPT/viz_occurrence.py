#!/usr/bin/env python3
"""
Module: viz_occurrence
Purpose: Domain occurrence charts (top-N bar chart, species x domain heatmap).
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from viz_common import _empty_figure


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
        height=min(max(350, len(m.index) * 55), 1200),
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        coloraxis_colorbar=dict(title="Proteins"),
        plot_bgcolor="white",
    )
    return fig
