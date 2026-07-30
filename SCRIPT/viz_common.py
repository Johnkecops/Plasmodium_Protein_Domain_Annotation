#!/usr/bin/env python3
"""
Module: viz_common
Purpose: Shared constants and the blank-state figure helper used by every
         viz_*.py chart module. Split out of the former single 571-line
         visualize.py, which bundled nine unrelated chart builders together.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import plotly.express as px
import plotly.graph_objects as go

_VALID_METRICS = frozenset(("jaccard", "lift", "pmi"))

# ─── Colour constants ─────────────────────────────────────────────────────────
_BLUE      = "#2E86AB"
_RED       = "#E84855"
_AMBER     = "#F18F01"
_PURPLE    = "#A23B72"
_TEAL      = "#3BB273"
_QUALSET1  = px.colors.qualitative.Set1
_QUALSET2  = px.colors.qualitative.Set2


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
