#!/usr/bin/env python3
"""
Module: plot_utils
Purpose: Shared helpers for the SCRIPT/plot_*.py output scripts. Extracted
         because plot_edge_weight.py, plot_clustering_vs_degree.py,
         plot_robustness.py, and plot_path_length_heatmap.py each carried
         their own near-identical copy of a synthetic fallback network
         generator, a Scilab literal formatter, and the PLOT/ directory
         resolution boilerplate (flagged by code-review-graph as duplicated
         high-fan-out hub functions across 3-4 files).
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import os
import numpy as np
import networkx as nx


def synthetic_domain_network(
    module_sizes=(38, 35, 32, 30),
    p_in=0.25,
    p_out=0.015,
    within_weight_range=(1, 8),
    cross_weight_range=(1, 3),
    seed=42,
):
    """
    Stochastic block model Plasmodium-like domain network (4 modules).
    Within-module edge probability p_in, between-module p_out; edge weights
    drawn uniformly from within_weight_range / cross_weight_range respectively.
    """
    rng = np.random.default_rng(seed)
    n = sum(module_sizes)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    offsets = [0] + list(np.cumsum(module_sizes))
    for m_i in range(len(module_sizes)):
        si, ei = offsets[m_i], offsets[m_i + 1]
        for u in range(si, ei):
            for v in range(u + 1, ei):
                if rng.random() < p_in:
                    G.add_edge(u, v, weight=int(rng.integers(*within_weight_range)))
        for m_j in range(m_i + 1, len(module_sizes)):
            sj, ej = offsets[m_j], offsets[m_j + 1]
            for u in range(si, ei):
                for v in range(sj, ej):
                    if rng.random() < p_out:
                        G.add_edge(u, v, weight=int(rng.integers(*cross_weight_range)))
    mapping = {i: f"PF{14000 + i:05d}" for i in range(n)}
    return nx.relabel_nodes(G, mapping)


def scilab_vector(name: str, arr, fmt: str = ".8g") -> str:
    """Format a 1-D array as a Scilab row-vector literal: 'name = [v1; v2; ...];'."""
    vals = "; ".join(f"{v:{fmt}}" for v in arr)
    return f"{name} = [{vals}];"


def scilab_matrix(name: str, arr, fmt: str = ".6g") -> str:
    """Format a 1-D or 2-D array as a Scilab matrix() literal, NaN-safe."""
    arr = np.asarray(arr)
    flat = arr.flatten()
    vals = "; ".join("Nan" if np.isnan(v) else f"{v:{fmt}}" for v in flat)
    rows, cols = arr.shape if arr.ndim == 2 else (1, len(flat))
    return f"{name} = matrix([{vals}], {rows}, {cols});"


def resolve_plot_dir(script_file: str, plot_dir: str = None) -> str:
    """Resolve (and create) the PLOT/ output directory relative to a plot script's own location."""
    if plot_dir is None:
        plot_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(script_file)), "..", "PLOT")
        )
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def fetch_network_or_synthetic(
    taxon_id: str,
    synthetic_fn,
    min_nodes: int = 10,
    min_edges: int = 0,
    label: str = "network",
):
    """
    Try to fetch live UniProt data and build a domain co-occurrence network;
    fall back to synthetic_fn() on any fetch error or if the result is too
    small to analyze. Returns (graph, data_source_label).
    """
    from fetch_proteins import fetch_plasmodium_proteins
    from network_builder import build_domain_network

    G = None
    try:
        print(f"[1/3] Fetching Plasmodium proteins (UniProt) for {label}...")
        df = fetch_plasmodium_proteins(taxon_id=taxon_id, reviewed=True, max_results=5000)
        n_proteins = len(df)
        print(f"      {n_proteins} proteins fetched.")
        if n_proteins > 0:
            print("[2/3] Building co-occurrence network...")
            G = build_domain_network(df, use_pfam=True)
            if G.number_of_nodes() < min_nodes or G.number_of_edges() < min_edges:
                G = None
            else:
                n_species = df["organism"].nunique()
                print(
                    f"      {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                    f"{n_species} species."
                )
    except Exception as e:
        print(f"      Fetch failed ({e}); using synthetic data.")

    if G is None:
        print("[2/3] Generating synthetic representative network...")
        G = synthetic_fn()
        return G, "Synthetic (representative Plasmodium-like, stochastic block model)"
    return G, "UniProt Swiss-Prot"
