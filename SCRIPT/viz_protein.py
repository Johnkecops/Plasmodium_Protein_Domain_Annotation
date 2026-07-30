#!/usr/bin/env python3
"""
Module: viz_protein
Purpose: Per-protein linear domain map chart.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import pandas as pd
import plotly.graph_objects as go

from viz_common import _empty_figure, _QUALSET1


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
