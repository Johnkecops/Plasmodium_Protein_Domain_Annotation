#!/usr/bin/env python3
"""
Module: test_robustness
Purpose: Guard the network null, the clade-level test and the pan-core partition.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from robustness_analysis import clade_avoidance, curveball

ROOT = Path(__file__).resolve().parent.parent


def test_curveball_preserves_row_and_column_sums():
    rng = np.random.default_rng(1)
    rows = [np.array(sorted(rng.choice(30, size=rng.integers(1, 6), replace=False))) for _ in range(200)]
    out = curveball(rows, 2000, np.random.default_rng(2))
    assert [len(r) for r in out] == [len(r) for r in rows]
    assert np.array_equal(np.bincount(np.concatenate(rows), minlength=30), np.bincount(np.concatenate(out), minlength=30))
    assert any(set(a) != set(b) for a, b in zip(rows, out))


def test_clade_test_flags_clade_restricted_family_only():
    clades = {"s1": "A", "s2": "A", "s3": "B", "s4": "C"}
    rows = []
    for sp, cl in clades.items():
        for i in range(200):
            acc = f"{sp}_{i}"
            rows.append((acc, "PFcommon", sp, cl))
            if cl == "A" and i < 40:
                rows.append((acc, "PFrestricted", sp, cl))
    domains = pd.DataFrame(rows, columns=["accession", "domain_accession", "species", "clade"])
    coverage = pd.DataFrame({"species": list(clades), "clade": list(clades.values()), "n_annotated_proteins": 200})
    out = clade_avoidance(domains, coverage).set_index("domain_accession")
    assert out.loc["PFrestricted", "clade_significant_q05"]
    assert not out.loc["PFcommon", "clade_significant_q05"]


@pytest.mark.skipif(not (ROOT / "results/tables/reference_pan_core.csv").exists(), reason="pipeline output absent")
def test_pan_core_categories_partition_by_species_count():
    pan = pd.read_csv(ROOT / "results/tables/reference_pan_core.csv")
    bounds = {"core": (20, 20), "soft_core": (19, 19), "shell": (3, 18), "cloud": (1, 2)}
    for cat, (lo, hi) in bounds.items():
        n = pan.loc[pan["category"] == cat, "n_species"]
        assert n.between(lo, hi).all(), cat
    assert pan["domain_accession"].is_unique
    assert pan["category"].value_counts().to_dict() == {"core": 1618, "shell": 458, "soft_core": 250, "cloud": 201}
