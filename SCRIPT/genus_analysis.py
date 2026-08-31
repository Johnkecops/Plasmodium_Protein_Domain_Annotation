#!/usr/bin/env python3
"""
Module: genus_analysis
Purpose: Occurrence, avoidance, co-occurrence, network and pan-core analyses on one identifier system.
Author: Dr. Arli Aditya Parikesit, Dr. Arif Nur Muhammad Ansori, and Moch. Royhan Afnani.,M.Sc
Date: 2026

Rationale (response to reviewers):
    One of the reviewer and the editor rejected the previous binomial avoidance null because it
    ignored annotation depth, species sampling and phylogenetic dependence. Reviewer 4
    showed that the lift denominator counted only proteins retaining one of the top sixty
    domains, that the top-ranked pairs were obligate partners tied at the minimum support
    threshold, and that within-family instance pairs dominated the ranking.

    This module replaces all of that:
      - avoidance uses a depth-conditioned Poisson-binomial null with leave-one-species-out
        family frequencies, plus a clade-collapsed statistic that removes the
        pseudo-replication introduced by closely related species;
      - co-occurrence uses the full annotated-protein denominator, reports support counts,
        and separates obligate and same-clan pairs from the headline ranking;
      - the network and the co-occurrence table are built from the same Pfam accession keys,
        so Figures 4 and 5 describe one graph rather than two.

Parameters:
    --snapshot   : frozen UniProt snapshot
    --clans      : Pfam-A.clans.tsv.gz, supplying family names and clan membership
    --min-pair   : minimum co-occurrence support for a reported pair (default 5)
    --bootstrap  : bootstrap replicates for degree-distribution fits (default 1000)

References:
    Clauset A., Shalizi C.R., Newman M.E.J. (2009) SIAM Review 51(4):661-703. doi:10.1137/070710111
    Alstott J., Bullmore E., Plenz D. (2014) PLoS ONE 9(1):e85777. doi:10.1371/journal.pone.0085777
    Benjamini Y., Hochberg Y. (1995) J R Stat Soc B 57(1):289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
    Vuong Q.H. (1989) Econometrica 57(2):307-333. doi:10.2307/1912557
"""

from __future__ import annotations

import gzip
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# Pfam metadata
# ----------------------------------------------------------------------------------

def load_pfam_clans(path: Path) -> pd.DataFrame:
    """
    Load the Pfam clan table: accession, clan, short name, description.

    Supplies the stable short names and descriptions that the manuscript must print
    beside every accession, and the clan membership used to exclude same-clan pairs
    from the co-occurrence ranking.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        df = pd.read_csv(
            fh,
            sep="\t",
            header=None,
            names=["domain_accession", "clan_accession", "clan_id", "domain_name", "description"],
            dtype=str,
        )
    return df.fillna({"clan_accession": "", "clan_id": ""})


def annotate_with_pfam(table: pd.DataFrame, clans: pd.DataFrame, key: str = "domain_accession") -> pd.DataFrame:
    """Attach Pfam short name, description and clan to any accession-keyed table."""
    return table.merge(
        clans[["domain_accession", "domain_name", "description", "clan_accession", "clan_id"]],
        left_on=key,
        right_on="domain_accession",
        how="left",
        suffixes=("", "_pfam"),
    )


# ----------------------------------------------------------------------------------
# Layer 1: occurrence
# ----------------------------------------------------------------------------------

def compute_domain_occurrence(domains: pd.DataFrame, n_proteins_total: int) -> pd.DataFrame:
    """
    Protein-level occurrence per domain family, keyed by accession.

    The denominator is stated in the output rather than being implicit: pct_of_proteome
    uses every protein in the analysed set, annotated or not, which is the quantity a
    reader assumes when a percentage is printed beside a domain family.

    Also reports the reviewed and unreviewed carrier counts separately, so that the
    curation-bias question raised for TSP and similar families is answerable directly
    from the table.
    """
    if domains.empty:
        return pd.DataFrame()

    agg = domains.groupby("domain_accession").agg(
        n_proteins=("accession", "nunique"),
        n_instances=("copy_number", "sum"),
        n_species=("species", "nunique"),
        n_clades=("clade", "nunique"),
        n_reviewed_carriers=("reviewed", "sum"),
        species_list=("species", lambda s: "; ".join(sorted(set(s)))),
    )
    agg["n_unreviewed_carriers"] = agg["n_proteins"] - agg["n_reviewed_carriers"]
    agg["reviewed_fraction"] = (agg["n_reviewed_carriers"] / agg["n_proteins"]).round(4)
    agg["pct_of_proteome"] = (agg["n_proteins"] / n_proteins_total * 100).round(3)
    agg["instances_per_carrier"] = (agg["n_instances"] / agg["n_proteins"]).round(3)
    return agg.sort_values("n_proteins", ascending=False).reset_index()


# ----------------------------------------------------------------------------------
# Avoidance, conditioned on annotation depth
# ----------------------------------------------------------------------------------

def _poisson_binomial_sf(probs: np.ndarray, k: int) -> float:
    """
    P(X >= k) for X = sum of independent Bernoulli variables with unequal probabilities.

    Exact, by convolution. With at most a few dozen species this costs nothing, and it
    avoids the equal-probability assumption of the binomial null that Reviewer 2 rejected.
    """
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[k:].sum())


def compute_domain_avoidance(
    domains: pd.DataFrame,
    coverage: pd.DataFrame,
    min_carriers: int = 3,
) -> pd.DataFrame:
    """
    Depth-conditioned avoidance analysis.

    For family d and species s, the null probability that s carries no copy of d is

        q_ds = (1 - p_d(-s)) ** n_s

    where n_s is the number of annotated proteins in species s and p_d(-s) is the
    per-protein frequency of d across every species except s. Leaving s out of its own
    expectation removes the circularity of testing a species against a frequency it
    helped define.

    The number of species lacking d is then Poisson-binomial under the null, which gives
    an exact one-sided p-value for excess absence. Species with deep annotation contribute
    strong evidence and species with shallow annotation contribute weak evidence, which is
    exactly the behaviour the previous equal-weight binomial lacked.

    A clade-collapsed score is reported alongside. Closely related species are not
    independent observations, so counting four rodent species as four absences
    overstates the evidence; the clade statistic counts each host-defined clade once.
    """
    if domains.empty:
        return pd.DataFrame()

    depth = coverage.set_index("species")["n_annotated_proteins"].to_dict()
    species_all = sorted(depth)
    clade_of = coverage.set_index("species")["clade"].to_dict()
    clades_all = sorted(set(clade_of.values()))
    total_annotated = sum(depth.values())

    carriers = domains.groupby(["domain_accession", "species"])["accession"].nunique().unstack(fill_value=0)
    carriers = carriers.reindex(columns=species_all, fill_value=0)
    totals = carriers.sum(axis=1)

    rows = []
    for acc, per_species in carriers.iterrows():
        total_d = int(totals[acc])
        if total_d < min_carriers:
            continue

        present_species = [s for s in species_all if per_species[s] > 0]
        absent_species = [s for s in species_all if per_species[s] == 0]

        q = []
        for s in absent_species:
            n_s = depth.get(s, 0)
            denom = total_annotated - n_s
            p_out = (total_d - int(per_species[s])) / denom if denom > 0 else 0.0
            p_out = min(max(p_out, 0.0), 1.0)
            q.append((1 - p_out) ** n_s if n_s > 0 else 1.0)
        q_present = []
        for s in present_species:
            n_s = depth.get(s, 0)
            denom = total_annotated - n_s
            p_out = (total_d - int(per_species[s])) / denom if denom > 0 else 0.0
            p_out = min(max(p_out, 0.0), 1.0)
            q_present.append((1 - p_out) ** n_s if n_s > 0 else 1.0)

        probs = np.array(q + q_present, dtype=float)
        n_absent = len(absent_species)
        p_value = _poisson_binomial_sf(probs, n_absent)
        expected_absent = float(probs.sum())

        present_clades = {clade_of.get(s, "unassigned") for s in present_species}
        absent_clades = [c for c in clades_all if c not in present_clades]

        rows.append(
            {
                "domain_accession": acc,
                "n_carriers": total_d,
                "n_species_present": len(present_species),
                "n_species_absent": n_absent,
                "avoidance_score": round(n_absent / len(species_all), 4),
                "expected_species_absent": round(expected_absent, 3),
                "excess_absence": round(n_absent - expected_absent, 3),
                "poisson_binomial_p": p_value,
                "n_clades_present": len(present_clades),
                "n_clades_absent": len(absent_clades),
                "clade_avoidance_score": round(len(absent_clades) / len(clades_all), 4),
                "absent_clades": "; ".join(absent_clades),
                "present_species": "; ".join(present_species),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = _bh_fdr(out["poisson_binomial_p"].to_numpy())
    out["significant_q05"] = out["q_value"] < 0.05
    return out.sort_values(["excess_absence", "avoidance_score"], ascending=False).reset_index(drop=True)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR correction."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ----------------------------------------------------------------------------------
# Co-occurrence
# ----------------------------------------------------------------------------------

def compute_cooccurrence(
    domains: pd.DataFrame,
    clans: pd.DataFrame,
    min_pair_count: int = 5,
    max_families: int | None = None,
) -> pd.DataFrame:
    """
    Pairwise within-protein co-occurrence on Pfam accessions.

    Three corrections relative to the previous implementation:

    1. N is the number of proteins carrying at least one Pfam family in the analysed set,
       not the number retaining one of the top sixty families. The previous denominator
       inflated every lift value and was not documented in the methods.
    2. Support (n_AB) is reported for every pair, so a co-enrichment maximum resting on a
       handful of proteins is visible rather than tied for first place.
    3. Obligate pairs (n_AB == n_A == n_B) and same-clan pairs are flagged. These are the
       defining architecture of single protein families rather than evidence of domain
       combination, and are excluded from the headline ranking.
    """
    if domains.empty:
        return pd.DataFrame()

    protein_sets = domains.groupby("accession")["domain_accession"].apply(set)
    protein_sets = protein_sets[protein_sets.map(len) > 0]
    N = int(len(protein_sets))

    freq = domains.groupby("domain_accession")["accession"].nunique()
    if max_families is not None:
        keep = set(freq.nlargest(max_families).index)
        protein_sets = protein_sets.map(lambda s: s & keep)
        protein_sets = protein_sets[protein_sets.map(len) > 0]

    pair_counts: dict[tuple[str, str], int] = {}
    for members in protein_sets:
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    clan_of = clans.set_index("domain_accession")["clan_accession"].to_dict()

    rows = []
    for (a, b), n_ab in pair_counts.items():
        if n_ab < min_pair_count:
            continue
        n_a = int(freq.get(a, 0))
        n_b = int(freq.get(b, 0))
        if n_a == 0 or n_b == 0:
            continue
        expected = n_a * n_b / N
        lift = (n_ab / N) / ((n_a / N) * (n_b / N))
        jaccard = n_ab / (n_a + n_b - n_ab)
        pmi = float(np.log2(lift)) if lift > 0 else float("-inf")
        table = [[n_ab, n_a - n_ab], [n_b - n_ab, N - n_a - n_b + n_ab]]
        _, p = stats.fisher_exact(table, alternative="greater")
        clan_a, clan_b = clan_of.get(a, ""), clan_of.get(b, "")
        rows.append(
            {
                "domain_a": a,
                "domain_b": b,
                "n_AB": n_ab,
                "n_A": n_a,
                "n_B": n_b,
                "N": N,
                "expected_AB": round(expected, 3),
                "jaccard": round(jaccard, 4),
                "lift": round(lift, 3),
                "pmi_log2": round(pmi, 3),
                "fisher_p": p,
                "obligate_pair": bool(n_ab == n_a == n_b),
                "same_clan": bool(clan_a and clan_a == clan_b),
                "clan_a": clan_a,
                "clan_b": clan_b,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = _bh_fdr(out["fisher_p"].to_numpy())
    out["significant_q05"] = out["q_value"] < 0.05
    out["headline_eligible"] = ~(out["obligate_pair"] | out["same_clan"])
    names = clans.set_index("domain_accession")["domain_name"].to_dict()
    out["name_a"] = out["domain_a"].map(names)
    out["name_b"] = out["domain_b"].map(names)
    return out.sort_values(["lift", "n_AB"], ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------------
# Network and degree distribution
# ----------------------------------------------------------------------------------

def build_network(cooccurrence: pd.DataFrame, q_threshold: float = 0.05):
    """
    Build the co-occurrence graph from the same table that produces the reported pairs.

    A single edge set now underlies both the network statistics and the pair ranking,
    which removes the discrepancy between Figures 4 and 5 in the previous version.
    """
    import networkx as nx

    edges = cooccurrence[cooccurrence["q_value"] < q_threshold]
    graph = nx.Graph()
    for row in edges.itertuples():
        graph.add_edge(row.domain_a, row.domain_b, weight=row.n_AB, lift=row.lift, q=row.q_value)
    return graph


def describe_network(graph) -> dict:
    """Topology summary, including the component structure that the previous version omitted."""
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0}

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = graph.subgraph(components[0])
    degrees = [d for _, d in graph.degree()]
    summary = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 6),
        "n_components": len(components),
        "largest_component_nodes": largest.number_of_nodes(),
        "largest_component_fraction": round(largest.number_of_nodes() / graph.number_of_nodes(), 4),
        "mean_degree": round(float(np.mean(degrees)), 3),
        "max_degree": int(max(degrees)),
        "avg_clustering": round(nx.average_clustering(graph), 4),
        "transitivity": round(nx.transitivity(graph), 4),
    }
    if largest.number_of_nodes() > 1:
        summary["lcc_avg_path_length"] = round(nx.average_shortest_path_length(largest), 4)
        summary["lcc_diameter"] = int(nx.diameter(largest))
    return summary


def _poisson_loglike(degrees: np.ndarray) -> float:
    lam = degrees.mean()
    return float(stats.poisson.logpmf(degrees, lam).sum())


def fit_degree_distribution(
    degrees: np.ndarray,
    bootstrap: int = 1000,
    seed: int = 20260831,
) -> dict:
    """
    Fit and compare candidate degree distributions with bootstrap confidence intervals.

    Reviewers 2 and 3 both asked for comparison beyond power law versus truncated power
    law. Log-normal, exponential, stretched exponential and Poisson are added, each by
    likelihood ratio against the power law under the Clauset-Shalizi-Newman framework,
    with the Vuong normalised statistic and its two-sided p-value.

    Bootstrap resampling of the degree sequence gives confidence intervals for alpha and
    x_min, which the previous version reported as point estimates.
    """
    import powerlaw

    degrees = np.asarray([d for d in degrees if d > 0], dtype=float)
    if len(degrees) < 20:
        return {"error": "degree sequence too short for distribution fitting", "n": int(len(degrees))}

    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    result = {
        "n_nodes_with_degree": int(len(degrees)),
        "alpha": round(float(fit.alpha), 4),
        "xmin": float(fit.xmin),
        "sigma": round(float(fit.sigma), 4),
        "ks_distance": round(float(fit.power_law.D), 4),
        "n_tail": int((degrees >= fit.xmin).sum()),
    }

    for alternative in ("truncated_power_law", "lognormal", "exponential", "stretched_exponential"):
        try:
            R, p = fit.distribution_compare("power_law", alternative, normalized_ratio=True)
            result[f"vs_{alternative}"] = {
                "loglikelihood_ratio": round(float(R), 4),
                "p_value": round(float(p), 5),
                "favoured": "power_law" if R > 0 else alternative,
            }
        except Exception as exc:  # a candidate can fail to converge on a short tail
            result[f"vs_{alternative}"] = {"error": str(exc)}

    # Poisson is not offered by the powerlaw package. It is the natural random-graph
    # null and both Reviewer 2 and the editor asked for it, so it is fitted directly on
    # the tail and compared by the same Vuong statistic.
    tail = degrees[degrees >= fit.xmin].astype(int)
    if len(tail) > 5:
        ll_pl = float(fit.power_law.loglikelihoods(tail).sum())
        ll_pois = _poisson_loglike(tail)
        diffs = fit.power_law.loglikelihoods(tail) - stats.poisson.logpmf(tail, tail.mean())
        sd = float(np.std(diffs, ddof=1))
        R = (ll_pl - ll_pois) / (sd * np.sqrt(len(tail))) if sd > 0 else 0.0
        p = float(2 * stats.norm.sf(abs(R)))
        result["vs_poisson"] = {
            "loglikelihood_ratio": round(R, 4),
            "p_value": round(p, 5),
            "favoured": "power_law" if R > 0 else "poisson",
        }

    rng = np.random.default_rng(seed)
    alphas, xmins = [], []
    for _ in range(bootstrap):
        sample = rng.choice(degrees, size=len(degrees), replace=True)
        try:
            boot = powerlaw.Fit(sample, discrete=True, verbose=False)
            alphas.append(float(boot.alpha))
            xmins.append(float(boot.xmin))
        except Exception:
            continue
    if alphas:
        result["bootstrap"] = {
            "replicates": len(alphas),
            "alpha_mean": round(float(np.mean(alphas)), 4),
            "alpha_ci95": [round(float(np.percentile(alphas, 2.5)), 4), round(float(np.percentile(alphas, 97.5)), 4)],
            "xmin_median": float(np.median(xmins)),
            "xmin_ci95": [float(np.percentile(xmins, 2.5)), float(np.percentile(xmins, 97.5))],
        }
    return result


# ----------------------------------------------------------------------------------
# Pan-core
# ----------------------------------------------------------------------------------

def compute_pan_core(domains: pd.DataFrame, species: list[str], soft_core: float = 0.95) -> pd.DataFrame:
    """
    Pan-core classification over a stated species set.

    A soft-core category is added because a strict all-species definition is fragile to a
    single assembly gap: one missing annotation in one proteome removes a family from the
    core. Both the strict and soft counts are reported.
    """
    if domains.empty:
        return pd.DataFrame()

    n_species = len(species)
    per_family = domains[domains["species"].isin(species)].groupby("domain_accession")["species"].nunique()
    out = per_family.reset_index(name="n_species")
    out["species_fraction"] = (out["n_species"] / n_species).round(4)

    def classify(frac: float) -> str:
        if frac >= 1.0:
            return "core"
        if frac >= soft_core:
            return "soft_core"
        if frac >= 0.15:
            return "shell"
        return "cloud"

    out["category"] = out["species_fraction"].map(classify)
    return out.sort_values("n_species", ascending=False).reset_index(drop=True)
