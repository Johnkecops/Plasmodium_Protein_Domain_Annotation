#!/usr/bin/env python3
"""
Module: network_metrics
Purpose: Compute structural metrics for Plasmodium domain co-occurrence graph
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - G: nx.Graph, domain co-occurrence network
    - df: pd.DataFrame from fetch_plasmodium_proteins()
References:
    - Watts & Strogatz (1998) Nature 393:440-442. doi:10.1038/30918
    - Newman (2006) PNAS 103(23):8577-8582. doi:10.1073/pnas.0601602103
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def compute_clustering(G: nx.Graph) -> Dict:
    """Average clustering coefficient and transitivity."""
    if G.number_of_nodes() < 3:
        return {"avg_clustering": 0.0, "transitivity": 0.0}
    return {
        "avg_clustering": round(nx.average_clustering(G), 6),
        "transitivity": round(nx.transitivity(G), 6),
    }


def compute_path_length(G: nx.Graph, sample_size: int = 500) -> Dict:
    """
    Average shortest path length on the largest connected component.

    For LCC with <= 200 nodes: exact computation.
    For larger graphs: sample-based estimate from sample_size source nodes.
    Diameter computed only for small LCC (expensive for large graphs).
    """
    components = list(nx.connected_components(G))
    if not components:
        return {"avg_path_length": None, "diameter": None, "n_lcc_nodes": 0, "sampled": False}

    lcc_nodes = max(components, key=len)
    H = G.subgraph(lcc_nodes).copy()
    n = H.number_of_nodes()

    if n < 2:
        return {"avg_path_length": None, "diameter": None, "n_lcc_nodes": n, "sampled": False}

    if n <= 200:
        avg_pl = nx.average_shortest_path_length(H)
        diameter = nx.diameter(H)
        sampled = False
    else:
        nodes = list(H.nodes())
        rng = np.random.default_rng(42)
        src = rng.choice(nodes, size=min(sample_size, n), replace=False)
        lengths = []
        for node in src:
            lengths.extend(nx.single_source_shortest_path_length(H, node).values())
        avg_pl = float(np.mean(lengths))
        diameter = None
        sampled = True

    return {
        "avg_path_length": round(avg_pl, 4),
        "diameter": diameter,
        "n_lcc_nodes": n,
        "sampled": sampled,
    }


def compute_hub_analysis(G: nx.Graph, top_n: int = 10) -> pd.DataFrame:
    """
    Top-N hubs by degree with betweenness centrality.
    Betweenness computed on the largest connected component.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["domain", "degree", "betweenness"])

    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]

    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes)
    between = nx.betweenness_centrality(H, normalized=True)

    rows = [
        {
            "domain": node,
            "degree": degrees[node],
            "betweenness": round(between.get(node, 0.0), 6),
        }
        for node in top_nodes
    ]
    return pd.DataFrame(rows)


def compute_core_pan(df: pd.DataFrame, use_pfam: bool = True) -> Dict:
    """
    Core vs pan domain analysis across Plasmodium species.

    Core: present in ALL species.
    Pan: union of all domains.
    Shell: present in 2 to n_species-1 species.
    Cloud: species-exclusive (present in exactly 1 species).
    """
    domain_col = "pfam_ids" if use_pfam and "pfam_ids" in df.columns else "domain_names"

    species_domains: Dict[str, set] = {}
    for _, row in df.iterrows():
        sp = row.get("organism", "Unknown")
        domains = row.get(domain_col, [])
        if not isinstance(domains, list):
            continue
        domain_set = {d for d in domains if d and str(d).strip()}
        if sp not in species_domains:
            species_domains[sp] = set()
        species_domains[sp].update(domain_set)

    if not species_domains:
        return {}

    n_species = len(species_domains)
    pan = set.union(*species_domains.values())
    core = set.intersection(*species_domains.values()) if n_species > 1 else pan.copy()

    species_presence = {
        d: sum(1 for sp_d in species_domains.values() if d in sp_d)
        for d in pan
    }
    shell = {d for d, cnt in species_presence.items() if 1 < cnt < n_species}
    cloud = {d for d, cnt in species_presence.items() if cnt == 1}

    return {
        "n_species": n_species,
        "pan_size": len(pan),
        "core_size": len(core),
        "shell_size": len(shell),
        "cloud_size": len(cloud),
        "core_fraction": round(len(core) / len(pan), 4) if pan else 0.0,
        "species_list": sorted(species_domains.keys()),
        "core_domains": sorted(core),
        "cloud_domains": sorted(cloud),
    }
