from network_builder import build_domain_network, get_largest_component, graph_summary


def test_build_domain_network_nodes_and_edge_weights(sample_protein_df):
    G = build_domain_network(sample_protein_df, use_pfam=True)

    assert set(G.nodes()) == {"PF00001", "PF00002", "PF00003", "PF00004"}
    # PF00001+PF00002 co-occur in P1 and P4 -> weight 2; no other pair co-occurs.
    assert G.number_of_edges() == 1
    assert G["PF00001"]["PF00002"]["weight"] == 2
    assert G.nodes["PF00001"]["protein_count"] == 4
    assert G.nodes["PF00003"]["protein_count"] == 1


def test_build_domain_network_min_edge_proteins_filters_weak_edges(sample_protein_df):
    G = build_domain_network(sample_protein_df, use_pfam=True, min_edge_proteins=3)
    assert G.number_of_edges() == 0
    # Nodes are still added even when their only edge is filtered out.
    assert G.number_of_nodes() == 4


def test_build_domain_network_domain_names_column(sample_protein_df):
    G = build_domain_network(sample_protein_df, use_pfam=False)
    assert "Core domain" in G.nodes()
    assert "Shell domain" in G.nodes()


def test_get_largest_component(sample_protein_df):
    G = build_domain_network(sample_protein_df, use_pfam=True)
    lcc = get_largest_component(G)
    assert lcc.number_of_nodes() == 2
    assert set(lcc.nodes()) == {"PF00001", "PF00002"}


def test_get_largest_component_empty_graph():
    import networkx as nx
    G = nx.Graph()
    assert get_largest_component(G).number_of_nodes() == 0


def test_graph_summary(sample_protein_df):
    G = build_domain_network(sample_protein_df, use_pfam=True)
    summary = graph_summary(G)

    assert summary["n_nodes"] == 4
    assert summary["n_edges"] == 1
    assert summary["n_components"] == 3
    assert summary["largest_component_size"] == 2
    assert summary["max_degree"] == 1
    assert summary["min_degree"] == 0
    assert summary["avg_degree"] == 0.5


def test_graph_summary_empty_graph():
    import networkx as nx
    summary = graph_summary(nx.Graph())
    assert summary["n_nodes"] == 0
    assert summary["density"] == 0.0
