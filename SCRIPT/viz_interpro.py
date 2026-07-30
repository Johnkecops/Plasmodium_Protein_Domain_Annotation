#!/usr/bin/env python3
"""
Module: viz_interpro
Purpose: InterPro cross-reference occurrence bar chart.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from viz_common import _empty_figure


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
