import os

import numpy as np
import pytest

from plot_utils import (
    synthetic_domain_network,
    scilab_vector,
    scilab_matrix,
    resolve_plot_dir,
    fetch_network_or_synthetic,
)


def test_synthetic_domain_network_is_deterministic():
    G1 = synthetic_domain_network(seed=42)
    G2 = synthetic_domain_network(seed=42)
    assert set(G1.nodes()) == set(G2.nodes())
    assert set(G1.edges()) == set(G2.edges())
    assert G1.number_of_nodes() == sum((38, 35, 32, 30))


def test_synthetic_domain_network_respects_weight_ranges():
    G = synthetic_domain_network(within_weight_range=(2, 15), cross_weight_range=(1, 4), seed=42)
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    assert all(1 <= w < 15 for w in weights)


def test_scilab_vector_format():
    out = scilab_vector("x", [1.0, 2.5, 3.0], fmt=".2g")
    assert out == "x = [1; 2.5; 3];"


def test_scilab_matrix_1d():
    out = scilab_matrix("x", [1.0, 2.0], fmt=".2g")
    assert out == "x = matrix([1; 2], 1, 2);"


def test_scilab_matrix_2d_handles_nan():
    arr = np.array([[1.0, np.nan], [3.0, 4.0]])
    out = scilab_matrix("m", arr, fmt=".2g")
    assert "Nan" in out
    assert out.startswith("m = matrix([")
    assert out.endswith("], 2, 2);")


def test_resolve_plot_dir_creates_directory(tmp_path):
    fake_script = tmp_path / "SCRIPT" / "plot_fake.py"
    fake_script.parent.mkdir()
    fake_script.write_text("# fake")

    plot_dir = resolve_plot_dir(str(fake_script))
    assert os.path.isdir(plot_dir)
    assert os.path.basename(plot_dir) == "PLOT"


def test_resolve_plot_dir_explicit_override(tmp_path):
    target = tmp_path / "custom_out"
    result = resolve_plot_dir(__file__, plot_dir=str(target))
    assert result == str(target)
    assert os.path.isdir(target)


def test_fetch_network_or_synthetic_falls_back_on_fetch_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("network unavailable in test")

    import fetch_proteins
    monkeypatch.setattr(fetch_proteins, "fetch_plasmodium_proteins", _boom)

    G, source = fetch_network_or_synthetic(
        "5820", lambda: synthetic_domain_network(seed=1), min_nodes=1, label="test"
    )
    assert "Synthetic" in source
    assert G.number_of_nodes() > 0
