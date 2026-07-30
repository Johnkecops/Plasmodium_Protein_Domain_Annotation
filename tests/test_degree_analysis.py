import pytest

from degree_analysis import fit_degree_distribution, compare_distributions, interpret_results


@pytest.fixture(scope="module")
def degree_fit():
    # Skewed synthetic degree sequence, deterministic, len >= 5.
    degrees = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 8, 13, 21, 34]
    return fit_degree_distribution(degrees)


def test_fit_degree_distribution_returns_expected_keys(degree_fit):
    fit, stats = degree_fit
    for key in ("alpha", "xmin", "sigma", "KS_distance", "n_total", "n_tail"):
        assert key in stats
    assert stats["n_total"] == 14
    assert stats["alpha"] > 0


def test_fit_degree_distribution_rejects_too_few_observations():
    with pytest.raises(ValueError):
        fit_degree_distribution([1, 2, 3])


def test_compare_distributions_has_all_alternatives(degree_fit):
    fit, _ = degree_fit
    comparisons = compare_distributions(fit)
    assert set(comparisons.keys()) == {
        "lognormal", "exponential", "truncated_power_law", "lognormal_positive",
    }
    for res in comparisons.values():
        assert "verdict" in res


def test_interpret_results_flags_non_scale_free_when_alternative_wins():
    fit_stats = {"alpha": 2.5, "KS_distance": 0.2}
    comparisons = {"lognormal": {"R": -2.0, "p": 0.01, "verdict": "..."}}
    text = interpret_results(fit_stats, comparisons)
    assert "NON-SCALE FREE" in text
    assert "lognormal is significantly BETTER" in text


def test_interpret_results_inconclusive_when_no_alternative_wins():
    fit_stats = {"alpha": 2.5, "KS_distance": 0.05}
    comparisons = {"lognormal": {"R": 1.0, "p": 0.9, "verdict": "..."}}
    text = interpret_results(fit_stats, comparisons)
    assert "Evidence is insufficient to reject the scale-free hypothesis" in text
