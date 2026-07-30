#!/usr/bin/env python3
"""
Module: viz_overview
Purpose: Species-overview charts (protein/domain coverage, length distribution).
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from viz_common import _empty_figure, _BLUE, _PURPLE, _AMBER, _QUALSET2


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
