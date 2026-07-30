#!/usr/bin/env python3
"""
Module: viz_avoidance
Purpose: Domain avoidance bar chart.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from viz_common import _empty_figure


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
