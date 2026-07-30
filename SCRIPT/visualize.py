#!/usr/bin/env python3
"""
Module: Visualization functions for Plasmodium protein domain analysis.
Purpose: Public re-export facade over the viz_*.py chart modules (species
         overview, occurrence, avoidance, co-occurrence, protein map,
         InterPro). Split into one module per chart family because the
         original 571-line single-file version bundled nine unrelated
         Plotly chart builders together; this facade keeps the existing
         `from visualize import plot_x` call sites unchanged.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

from viz_overview import plot_species_overview, plot_domain_length_distribution
from viz_occurrence import plot_domain_occurrence_bar, plot_species_domain_heatmap
from viz_avoidance import plot_domain_avoidance
from viz_cooccurrence import plot_domain_cooccurrence_heatmap, plot_domain_cooccurrence_top_pairs
from viz_protein import plot_protein_domain_map
from viz_interpro import plot_interpro_occurrence_bar
from viz_common import _empty_figure

__all__ = [
    "plot_species_overview",
    "plot_domain_length_distribution",
    "plot_domain_occurrence_bar",
    "plot_species_domain_heatmap",
    "plot_domain_avoidance",
    "plot_domain_cooccurrence_heatmap",
    "plot_domain_cooccurrence_top_pairs",
    "plot_protein_domain_map",
    "plot_interpro_occurrence_bar",
]
