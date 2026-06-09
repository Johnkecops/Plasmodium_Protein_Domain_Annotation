#!/usr/bin/env python3
"""
Module: network_builder
Purpose: Build domain co-occurrence network from UniProt Plasmodium data
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - df: pd.DataFrame from fetch_plasmodium_proteins()
    - use_pfam: bool, use Pfam IDs (True) or ft_domain names (False)
    - min_edge_proteins: int, minimum proteins sharing a pair to include edge
References:
    - Clauset et al. (2009) SIAM Review 51(4):661-703. doi:10.1137/070710111
    - Albert & Barabasi (2002) Rev. Mod. Phys. 74:47. doi:10.1103/RevModPhys.74.47
"""

import networkx as nx
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple


def build_domain_network(
    df: pd.DataFrame,
    use_pfam: bool = True,
    min_edge_proteins: int = 1,
) -> nx.Graph:
    """
    Build domain co-occurrence network from protein DataFrame.

    Nodes = unique domain identifiers (Pfam IDs or ft_domain names).
    Edges = domain pairs that co-occur in >= min_edge_proteins proteins.
    Edge weight = number of proteins sharing the pair.

    Binary per-protein counting (not per domain copy) approximates ortholog-group
    normalization, preventing lineage-expanded gene families from inflating hub
    connectivity (Clauset et al. 2009).
    """
    G = nx.Graph()

    domain_col = "pfam_ids" if use_pfam and "pfam_ids" in df.columns else "domain_names"

    pair_counts: Dict[Tuple[str, str], int] = {}
    node_counts: Dict[str, int] = {}

    for _, row in df.iterrows():
        domains = row.get(domain_col, [])
        if not isinstance(domains, list):
            continue
        domains = sorted({d for d in domains if d and str(d).strip()})
        if not domains:
            continue
        for d in domains:
            node_counts[d] = node_counts.get(d, 0) + 1
        for d1, d2 in combinations(domains, 2):
            pair = (d1, d2)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    for domain, count in node_counts.items():
        G.add_node(domain, protein_count=count)

    for (d1, d2), count in pair_counts.items():
        if count >= min_edge_proteins:
            G.add_edge(d1, d2, weight=count)

    return G


def get_largest_component(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a new graph."""
    if G.number_of_nodes() == 0:
        return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def graph_summary(G: nx.Graph) -> Dict:
    """Return basic network statistics dict."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {
            "n_nodes": 0, "n_edges": 0, "n_components": 0,
            "largest_component_size": 0, "density": 0.0,
            "avg_degree": 0.0, "max_degree": 0, "min_degree": 0,
        }
    components = list(nx.connected_components(G))
    degrees = [d for _, d in G.degree()]
    return {
        "n_nodes": n,
        "n_edges": m,
        "n_components": len(components),
        "largest_component_size": max(len(c) for c in components),
        "density": round(nx.density(G), 8),
        "avg_degree": round(sum(degrees) / len(degrees), 4),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
    }
