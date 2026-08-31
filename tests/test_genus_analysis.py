#!/usr/bin/env python3
"""
Module: test_genus_analysis
Purpose: Guard the statistical defects the reviewers identified in the submitted analysis.
Author: Dr. Arli Aditya Parikesit, Dr. Arif Nur Muhammad Ansori, and Moch. Royhan Afnani.,M.Sc
Date: 2026
"""

import numpy as np
import pandas as pd
import pytest

from domain_dataset import CLADE_BY_SPECIES, normalise_columns
from genus_analysis import (
    _bh_fdr,
    _poisson_binomial_sf,
    compute_cooccurrence,
    compute_domain_avoidance,
    compute_domain_occurrence,
    compute_pan_core,
)


@pytest.fixture
def clans():
    return pd.DataFrame(
        {
            "domain_accession": ["PF00001", "PF00002", "PF00003", "PF00004"],
            "clan_accession": ["CL0001", "CL0001", "", "CL0002"],
            "clan_id": ["ClanA", "ClanA", "", "ClanB"],
            "domain_name": ["Alpha", "Beta", "Gamma", "Delta"],
            "description": ["a", "b", "c", "d"],
        }
    )


def _domains(rows):
    return pd.DataFrame(
        rows,
        columns=["accession", "species", "clade", "domain_accession", "copy_number", "reviewed"],
    )


def test_every_clade_assignment_is_named():
    """An unassigned clade silently distorts the clade-collapsed avoidance statistic."""
    assert "unassigned" not in set(CLADE_BY_SPECIES.values())
    assert all(isinstance(v, str) and v for v in CLADE_BY_SPECIES.values())


def test_poisson_binomial_reduces_to_binomial_under_equal_probabilities():
    """Sanity check against the closed form the previous version assumed throughout."""
    from scipy import stats

    p, n, k = 0.3, 10, 4
    expected = float(stats.binom.sf(k - 1, n, p))
    assert _poisson_binomial_sf(np.full(n, p), k) == pytest.approx(expected, rel=1e-9)


def test_poisson_binomial_handles_unequal_probabilities():
    """P(X >= 0) is 1 and P(X >= n) is the product of the individual probabilities."""
    probs = np.array([0.1, 0.5, 0.9])
    assert _poisson_binomial_sf(probs, 0) == pytest.approx(1.0)
    assert _poisson_binomial_sf(probs, 3) == pytest.approx(0.1 * 0.5 * 0.9)


def test_bh_fdr_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    q = _bh_fdr(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(q <= 1.0)
    assert np.all(np.diff(q) >= -1e-12)


def test_avoidance_weights_species_by_annotation_depth():
    """
    Reviewer 2: absence in a deeply annotated species is strong evidence, absence in a
    shallow one is weak. The depth-conditioned null must rank the deep absence as more
    surprising even though both have the same raw avoidance score.
    """
    rows = []
    # A family carried by many proteins of species A, absent from B (deep) and C (shallow).
    for i in range(40):
        rows.append([f"A{i}", "Species A", "cladeA", "PF00001", 1, False])
    domains = _domains(rows)
    coverage = pd.DataFrame(
        {
            "species": ["Species A", "Species B", "Species C"],
            "clade": ["cladeA", "cladeB", "cladeC"],
            "n_annotated_proteins": [1000, 1000, 5],
        }
    )
    result = compute_domain_avoidance(domains, coverage)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_species_absent"] == 2
    # Under the null, the shallow species is expected to lack the family, the deep one
    # is not, so the expected number of absences sits strictly between 1 and 2.
    assert 1.0 < row["expected_species_absent"] < 2.0


def test_avoidance_reports_clade_collapsed_score():
    """Closely related species are pseudo-replicates; the clade statistic counts each once."""
    rows = [[f"R{i}", sp, "rodent", "PF00002", 1, False] for i, sp in enumerate(["S1", "S2", "S3"])]
    domains = _domains(rows)
    coverage = pd.DataFrame(
        {
            "species": ["S1", "S2", "S3", "S4"],
            "clade": ["rodent", "rodent", "rodent", "avian"],
            "n_annotated_proteins": [500, 500, 500, 500],
        }
    )
    result = compute_domain_avoidance(domains, coverage)
    row = result.iloc[0]
    assert row["n_species_absent"] == 1
    assert row["avoidance_score"] == pytest.approx(0.25)
    # One of two clades lacks the family, so the clade score is 0.5 rather than 0.25.
    assert row["clade_avoidance_score"] == pytest.approx(0.5)


def test_cooccurrence_denominator_is_all_annotated_proteins(clans):
    """
    Reviewer 4: the previous lift denominator counted only proteins retaining one of the
    top sixty families, which inflated every value. N must be the annotated set.
    """
    rows = []
    for i in range(10):
        rows.append([f"P{i}", "S1", "c", "PF00001", 1, False])
        rows.append([f"P{i}", "S1", "c", "PF00002", 1, False])
    for i in range(10, 100):
        rows.append([f"P{i}", "S1", "c", "PF00003", 1, False])
    result = compute_cooccurrence(_domains(rows), clans, min_pair_count=5)
    assert (result["N"] == 100).all(), "denominator must be every annotated protein"


def test_cooccurrence_flags_obligate_and_same_clan_pairs(clans):
    """
    Reviewer 4: obligate partners and within-clan pairs are the defining architecture of
    single protein families, not evidence of domain combination, and must be separable
    from the headline ranking.
    """
    rows = []
    for i in range(8):
        rows.append([f"P{i}", "S1", "c", "PF00001", 1, False])
        rows.append([f"P{i}", "S1", "c", "PF00002", 1, False])
    for i in range(8, 30):
        rows.append([f"P{i}", "S1", "c", "PF00003", 1, False])
        if i < 20:
            rows.append([f"P{i}", "S1", "c", "PF00004", 1, False])
    result = compute_cooccurrence(_domains(rows), clans, min_pair_count=5)
    obligate = result[(result.domain_a == "PF00001") & (result.domain_b == "PF00002")].iloc[0]
    assert bool(obligate["obligate_pair"]) is True
    assert bool(obligate["same_clan"]) is True
    assert bool(obligate["headline_eligible"]) is False

    mixed = result[(result.domain_a == "PF00003") & (result.domain_b == "PF00004")].iloc[0]
    assert bool(mixed["obligate_pair"]) is False
    assert bool(mixed["headline_eligible"]) is True


def test_cooccurrence_reports_support_count(clans):
    """A co-enrichment maximum resting on two proteins must be visible as such."""
    rows = []
    for i in range(6):
        rows.append([f"P{i}", "S1", "c", "PF00001", 1, False])
        rows.append([f"P{i}", "S1", "c", "PF00003", 1, False])
    result = compute_cooccurrence(_domains(rows), clans, min_pair_count=5)
    assert "n_AB" in result.columns
    assert int(result.iloc[0]["n_AB"]) == 6


def test_occurrence_separates_reviewed_from_unreviewed_carriers():
    """Reviewer 3 asked whether domain dominance is a curation artefact; the table answers it."""
    rows = [
        ["P1", "S1", "c", "PF00001", 1, True],
        ["P2", "S1", "c", "PF00001", 1, False],
        ["P3", "S2", "c", "PF00001", 1, False],
    ]
    result = compute_domain_occurrence(_domains(rows), n_proteins_total=100)
    row = result.iloc[0]
    assert int(row["n_reviewed_carriers"]) == 1
    assert int(row["n_unreviewed_carriers"]) == 2
    assert row["reviewed_fraction"] == pytest.approx(1 / 3, abs=1e-4)
    assert row["pct_of_proteome"] == pytest.approx(3.0)


def test_pan_core_partition_is_exhaustive():
    """Every family lands in exactly one category, so the reported fractions sum to one."""
    rows = []
    for sp in ["S1", "S2", "S3", "S4"]:
        rows.append([f"P_{sp}_core", sp, "c", "PF00001", 1, False])
    rows.append(["P_a", "S1", "c", "PF00002", 1, False])
    rows.append(["P_b", "S2", "c", "PF00002", 1, False])
    rows.append(["P_c", "S1", "c", "PF00003", 1, False])
    result = compute_pan_core(_domains(rows), ["S1", "S2", "S3", "S4"])
    assert len(result) == 3
    assert result.set_index("domain_accession").loc["PF00001", "category"] == "core"
    assert set(result["category"]) <= {"core", "soft_core", "shell", "cloud"}
