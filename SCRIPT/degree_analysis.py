#!/usr/bin/env python3
"""
Module: degree_analysis
Purpose: Fit degree distribution and test for non-scale-free properties
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - degrees: list of node degrees
    - discrete: bool, use discrete powerlaw fitting (True for integer degrees)
References:
    - Clauset, Shalizi & Newman (2009) SIAM Review 51(4):661-703. doi:10.1137/070710111
    - Alstott, Bullmore & Plenz (2014) PLOS ONE 9(1):e85777. doi:10.1371/journal.pone.0085777
"""

import warnings
import numpy as np
import powerlaw
from typing import Dict, List, Tuple


def fit_degree_distribution(
    degrees: List[int],
    discrete: bool = True,
) -> Tuple[powerlaw.Fit, Dict]:
    """
    Fit degree distribution with powerlaw library (MLE, Clauset-Shalizi-Newman).

    Returns the Fit object and a stats dict with:
    - alpha: power law exponent
    - xmin: lower bound for power law fit (auto-estimated)
    - sigma: standard error on alpha
    - KS_distance: Kolmogorov-Smirnov statistic vs power law
    - n_tail: number of observations in the tail (k >= xmin)
    """
    if len(degrees) < 5:
        raise ValueError(f"Too few degree observations ({len(degrees)}) to fit.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(degrees, discrete=discrete, verbose=False)

    stats = {
        "alpha": round(float(fit.power_law.alpha), 4),
        "xmin": float(fit.power_law.xmin),
        "sigma": round(float(fit.power_law.sigma), 4),
        "KS_distance": round(float(fit.power_law.D), 4),
        "n_total": len(degrees),
        "n_tail": int(fit.power_law.n),
    }
    return fit, stats


def compare_distributions(fit: powerlaw.Fit) -> Dict:
    """
    Vuong's normalized log-likelihood ratio test: power law vs alternatives.

    Returns dict of {alternative: {'R': ratio, 'p': p_value, 'verdict': str}}
    Negative R = alternative is better fit than power law.
    p < 0.05 = comparison is statistically significant.
    """
    alternatives = ["lognormal", "exponential", "truncated_power_law", "lognormal_positive"]
    results = {}

    for alt in alternatives:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
            R, p = float(R), float(p)
            if R < 0 and p < 0.05:
                verdict = f"BETTER: {alt} > power_law (R={R:.4f}, p={p:.4f})"
            elif R > 0 and p < 0.05:
                verdict = f"power_law > {alt} (R={R:.4f}, p={p:.4f})"
            else:
                verdict = f"INCONCLUSIVE: no significant difference (R={R:.4f}, p={p:.4f})"
            results[alt] = {"R": round(R, 4), "p": round(p, 4), "verdict": verdict}
        except Exception as e:
            results[alt] = {"R": None, "p": None, "verdict": f"Error: {e}"}

    return results


def interpret_results(fit_stats: Dict, comparisons: Dict) -> str:
    """Generate biological interpretation of scale-free test results."""
    alpha = fit_stats.get("alpha", 0.0)
    ks = fit_stats.get("KS_distance", 1.0)

    lines = []

    if 2.0 <= alpha <= 3.0:
        lines.append(
            f"Power law exponent alpha={alpha:.4f} falls within the canonical scale-free range "
            f"(2-3), but additional tests are required."
        )
    else:
        lines.append(
            f"Power law exponent alpha={alpha:.4f} is OUTSIDE the canonical scale-free range "
            f"(2-3) — inconsistent with a pure power law."
        )

    if ks > 0.1:
        lines.append(
            f"KS distance={ks:.4f} > 0.1 indicates a poor power law fit to the empirical "
            f"degree distribution."
        )
    else:
        lines.append(
            f"KS distance={ks:.4f} indicates a moderate fit; Vuong tests provide the "
            f"definitive comparison."
        )

    non_sf_count = 0
    for alt, res in comparisons.items():
        R, p = res.get("R"), res.get("p")
        if R is not None and R < 0 and p is not None and p < 0.05:
            lines.append(
                f"Vuong test: {alt} is significantly BETTER than power law "
                f"(R={R:.4f}, p={p:.4f})."
            )
            non_sf_count += 1

    if non_sf_count >= 1:
        lines.append(
            f"\nCONCLUSION: {non_sf_count} alternative model(s) significantly outperform "
            f"power law (Clauset-Shalizi-Newman framework). The degree distribution of this "
            f"Plasmodium domain co-occurrence network is NON-SCALE FREE."
        )
        lines.append(
            "Biological basis: The parasitic lifestyle, extreme AT-genomic bias, and "
            "physical constraints of domain architecture limit hub growth and prevent "
            "the emergence of an unbounded preferential-attachment topology."
        )
    else:
        lines.append(
            "\nCONCLUSION: No alternative model significantly outperforms power law with "
            "the current dataset. Evidence is insufficient to reject the scale-free hypothesis; "
            "larger or more complete domain annotation data may clarify the result."
        )

    return "\n".join(lines)
