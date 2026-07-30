import networkx as nx

from network_metrics import (
    compute_clustering,
    compute_path_length,
    compute_hub_analysis,
    compute_core_pan,
)


def test_compute_clustering_complete_graph():
    G = nx.complete_graph(4)
    result = compute_clustering(G)
    assert result["avg_clustering"] == 1.0
    assert result["transitivity"] == 1.0


def test_compute_clustering_too_small():
    G = nx.Graph()
    G.add_edge("a", "b")
    result = compute_clustering(G)
    assert result == {"avg_clustering": 0.0, "transitivity": 0.0}


def test_compute_path_length_exact_small_graph():
    G = nx.path_graph(5)  # 0-1-2-3-4
    result = compute_path_length(G)
    assert result["sampled"] is False
    assert result["n_lcc_nodes"] == 5
    assert result["avg_path_length"] == round(nx.average_shortest_path_length(G), 4)
    assert result["diameter"] == nx.diameter(G)


def test_compute_path_length_uses_largest_component():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])  # LCC size 3
    G.add_node(99)  # isolated node, smaller component
    result = compute_path_length(G)
    assert result["n_lcc_nodes"] == 3


def test_compute_path_length_empty_graph():
    result = compute_path_length(nx.Graph())
    assert result["avg_path_length"] is None
    assert result["n_lcc_nodes"] == 0


def test_compute_hub_analysis_star_graph():
    G = nx.star_graph(4)  # center node 0, leaves 1-4
    df = compute_hub_analysis(G, top_n=1)
    assert len(df) == 1
    assert df.iloc[0]["domain"] == 0
    assert df.iloc[0]["degree"] == 4
    assert df.iloc[0]["betweenness"] == 1.0


def test_compute_hub_analysis_empty_graph():
    df = compute_hub_analysis(nx.Graph())
    assert list(df.columns) == ["domain", "degree", "betweenness"]
    assert len(df) == 0


def test_compute_core_pan(sample_protein_df):
    result = compute_core_pan(sample_protein_df, use_pfam=True)

    assert result["n_species"] == 3
    assert result["pan_size"] == 4
    assert result["core_size"] == 1
    assert result["shell_size"] == 1
    assert result["cloud_size"] == 2
    assert result["core_fraction"] == 0.25
    assert result["core_domains"] == ["PF00001"]
    assert result["cloud_domains"] == ["PF00003", "PF00004"]
    assert result["species_list"] == [
        "Plasmodium falciparum",
        "Plasmodium malariae",
        "Plasmodium vivax",
    ]


def test_compute_core_pan_empty_df():
    import pandas as pd
    result = compute_core_pan(pd.DataFrame(columns=["organism", "pfam_ids"]))
    assert result == {}
