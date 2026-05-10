#!/usr/bin/env python3
"""
Module: Domain occurrence and avoidance analysis for Plasmodium proteins.
Purpose: Compute domain frequency distributions, species-specific patterns,
         and domain avoidance (systematic absence in particular lineages).
Author: Dr. Arli Aditya Parikesit
Date: 2026

References:
    Parikesit et al. (2011) Genes 2(4):912-924.
    Parikesit et al. (2018) JBI 14(2):185-190.
"""

import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.stats import fisher_exact, binomtest
from itertools import combinations
from statsmodels.stats.multitest import multipletests


def get_species_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise protein and domain counts per Plasmodium species.

    Returns
    -------
    pd.DataFrame
        Columns: organism, n_proteins, n_with_domains, pct_with_domains,
                 unique_domains, total_domains, avg_length, max_length.
    """
    def _unique_domains(series):
        return len({d for lst in series for d in lst})

    agg = df.groupby("organism", as_index=False).agg(
        n_proteins=("accession", "count"),
        n_with_domains=("n_domains", lambda x: (x > 0).sum()),
        total_domains=("n_domains", "sum"),
        avg_length=("length", "mean"),
        max_length=("length", "max"),
    )
    # unique domain types per species
    unique_counts = (
        df.groupby("organism")["domain_names"]
        .apply(_unique_domains)
        .reset_index()
        .rename(columns={"domain_names": "unique_domains"})
    )
    agg = agg.merge(unique_counts, on="organism", how="left")
    agg["avg_length"] = agg["avg_length"].round(1)
    agg["pct_with_domains"] = (
        agg["n_with_domains"] / agg["n_proteins"] * 100
    ).round(1)
    return agg.sort_values("n_proteins", ascending=False).reset_index(drop=True)


def compute_domain_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count domain occurrence across all proteins.

    Returns
    -------
    pd.DataFrame
        Columns: domain_name, count, species_count, pct_proteins, species.
        Sorted by count descending.
    """
    records = []
    for _, row in df.iterrows():
        for d in row["domain_names"]:
            records.append({"domain_name": d, "accession": row["accession"], "organism": row["organism"]})

    if not records:
        return pd.DataFrame(columns=["domain_name", "count", "species_count", "pct_proteins", "species"])

    exp = pd.DataFrame(records)
    total = len(df)

    agg = (
        exp.groupby("domain_name")
        .agg(
            count=("accession", "nunique"),
            species_count=("organism", "nunique"),
            species=("organism", lambda x: sorted(set(x))),
        )
        .reset_index()
    )
    agg["pct_proteins"] = (agg["count"] / total * 100).round(2)
    return agg.sort_values("count", ascending=False).reset_index(drop=True)


def compute_species_domain_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a species × domain presence matrix.

    Cell value = number of proteins in that species containing that domain.

    Returns
    -------
    pd.DataFrame
        Rows = species, columns = domain names.
    """
    expanded = (
        df[["organism", "domain_names"]]
        .explode("domain_names")
        .dropna(subset=["domain_names"])
        .rename(columns={"domain_names": "domain_name"})
    )
    expanded = expanded[expanded["domain_name"].str.strip().ne("")]
    if expanded.empty:
        return pd.DataFrame()

    matrix = (
        expanded.groupby(["organism", "domain_name"])
        .size()
        .unstack(fill_value=0)
    )
    return matrix


def compute_domain_avoidance(
    df: pd.DataFrame,
    min_species_presence: int = 2,
) -> pd.DataFrame:
    """
    Identify domains present in some Plasmodium species but absent in others,
    with binomial p-values for statistical significance of each absence.

    'Avoidance score' = fraction of species that lack the domain.
    'binom_pvalue' = probability of observing 0 proteins with this domain in
    an absent species under the null model (random domain assignment).
    'avoidance_pvalue_adj' = Benjamini-Hochberg FDR-corrected minimum p-value
    across all species that lack the domain.

    Returns
    -------
    pd.DataFrame
        Columns: domain_name, n_species_present, n_species_absent,
                 avoidance_score, min_binom_pvalue, avoidance_pvalue_adj,
                 species_with, species_without.
        Sorted by avoidance_score descending.
    """
    if df["organism"].nunique() < 2:
        warnings.warn(
            "compute_domain_avoidance: df contains only one species; "
            "p_global is calibrated on a single-species subset and avoidance p-values will be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    matrix = compute_species_domain_matrix(df)
    if matrix.empty:
        return pd.DataFrame()

    all_species = list(matrix.index)
    n_species = len(all_species)
    n_total_proteins = len(df)

    # Number of proteins per species
    species_protein_counts = df.groupby("organism")["accession"].count().to_dict()

    rows = []
    for domain in matrix.columns:
        species_with = [s for s in all_species if matrix.loc[s, domain] > 0]
        species_without = [s for s in all_species if matrix.loc[s, domain] == 0]

        n_with = len(species_with)
        n_without = len(species_without)

        if n_with < min_species_presence or n_without == 0:
            continue

        # Global prevalence of this domain across all proteins
        n_proteins_with_domain = int(matrix[domain].sum())
        p_global = n_proteins_with_domain / n_total_proteins if n_total_proteins > 0 else 0.0

        # Binomial p-value for each absent species:
        # Under H0 (random assignment), expected count in species S = p_global * n_s
        # Test: P(X=0 | n=n_s, p=p_global) — one-sided (under-representation)
        absent_pvalues = []
        for sp in species_without:
            n_sp = species_protein_counts.get(sp, 0)
            if n_sp == 0 or p_global == 0:
                pval = 1.0
            else:
                # P(X <= 0) = (1 - p_global)^n_sp
                result = binomtest(0, n=n_sp, p=p_global, alternative="less")
                pval = result.pvalue
            absent_pvalues.append(pval)

        min_pval = min(absent_pvalues) if absent_pvalues else 1.0

        rows.append(
            {
                "domain_name": domain,
                "n_species_present": n_with,
                "n_species_absent": n_without,
                "avoidance_score": round(n_without / n_species, 4),
                "min_binom_pvalue": round(min_pval, 6),
                "species_with": species_with,
                "species_without": species_without,
                "_absent_pvalues": absent_pvalues,
            }
        )

    if not rows:
        return pd.DataFrame()

    result = (
        pd.DataFrame(rows)
        .sort_values("avoidance_score", ascending=False)
        .reset_index(drop=True)
    )

    # BH-FDR correction via statsmodels.multipletests
    _, pvals_adj, _, _ = multipletests(result["min_binom_pvalue"].values, method="fdr_bh")
    result["avoidance_pvalue_adj"] = np.minimum(pvals_adj, 1.0).round(6)
    result = result.drop(columns=["_absent_pvalues"])

    return result


def compute_domain_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a symmetric domain co-occurrence matrix.

    Cell (i, j) = number of proteins that contain both domain_i and domain_j.
    Diagonal = single-domain occurrence count.

    Returns
    -------
    pd.DataFrame
        Square matrix with domain names as both index and columns.
    """
    protein_domains = {
        row["accession"]: set(row["domain_names"])
        for _, row in df.iterrows()
        if row["n_domains"] > 0
    }

    if not protein_domains:
        return pd.DataFrame()

    all_domains = sorted({d for ds in protein_domains.values() for d in ds})
    matrix = pd.DataFrame(0, index=all_domains, columns=all_domains)

    for domains in protein_domains.values():
        domain_list = sorted(domains)
        for i, d1 in enumerate(domain_list):
            matrix.loc[d1, d1] += 1
            for d2 in domain_list[i + 1 :]:
                matrix.loc[d1, d2] += 1
                matrix.loc[d2, d1] += 1

    return matrix


def compute_domain_cooccurrence_stats(
    df: pd.DataFrame,
    min_pair_count: int = 2,
    max_domains: int = 60,
) -> pd.DataFrame:
    """
    Pairwise domain co-occurrence statistics across all proteins.

    For each domain pair (A, B) that appear together in at least
    min_pair_count proteins:
    - n_AB       : proteins containing both A and B
    - n_A        : proteins containing A (regardless of B)
    - n_B        : proteins containing B (regardless of A)
    - n_neither  : proteins containing neither
    - N          : total proteins with at least one domain
    - jaccard    : n_AB / (n_A + n_B - n_AB)  [0, 1]
    - lift       : (n_AB / N) / ((n_A / N) * (n_B / N))  [> 1 = positive association]
    - pmi        : log2((n_AB / N) / ((n_A / N) * (n_B / N)))  [positive = co-enriched]
    - fisher_pvalue  : Fisher's exact test (one-sided: greater), H0 = independence
    - fisher_pvalue_adj : BH-FDR corrected Fisher p-value

    Parameters
    ----------
    df : pd.DataFrame
        Output of fetch_plasmodium_proteins().
    min_pair_count : int
        Minimum co-occurrence count to include a pair.
    max_domains : int
        Analyse only the top-N most frequent domains to keep computation tractable.

    Returns
    -------
    pd.DataFrame sorted by lift descending.
    """
    # Restrict to proteins with at least one domain
    protein_domains = {
        row["accession"]: set(row["domain_names"])
        for _, row in df.iterrows()
        if row["n_domains"] > 0
    }

    if not protein_domains:
        return pd.DataFrame()

    # Limit to top max_domains by frequency to keep O(n^2) tractable
    from collections import Counter
    freq = Counter(d for ds in protein_domains.values() for d in ds)
    top_domains = {d for d, _ in freq.most_common(max_domains)}

    # Filter protein_domains to top_domains only
    filtered = {acc: ds & top_domains for acc, ds in protein_domains.items()}
    filtered = {acc: ds for acc, ds in filtered.items() if ds}

    if not filtered:
        return pd.DataFrame()

    N = len(filtered)
    all_domains = sorted(top_domains & {d for ds in filtered.values() for d in ds})

    # Domain marginal counts (proteins containing each domain)
    marginals = {d: sum(1 for ds in filtered.values() if d in ds) for d in all_domains}

    rows = []
    for dA, dB in combinations(all_domains, 2):
        nA = marginals[dA]
        nB = marginals[dB]
        nAB = sum(1 for ds in filtered.values() if dA in ds and dB in ds)

        if nAB < min_pair_count:
            continue

        n_neither = N - nA - nB + nAB

        # Jaccard
        jaccard = nAB / (nA + nB - nAB) if (nA + nB - nAB) > 0 else 0.0

        # Lift
        p_A = nA / N
        p_B = nB / N
        p_AB = nAB / N
        lift = p_AB / (p_A * p_B) if (p_A * p_B) > 0 else 1.0

        # PMI (log2)
        pmi = np.log2(p_AB / (p_A * p_B)) if (p_A * p_B) > 0 and p_AB > 0 else 0.0

        # Fisher's exact — one-sided (alternative='greater' tests positive association)
        contingency = [[nAB, nA - nAB], [nB - nAB, n_neither]]
        _, fisher_p = fisher_exact(contingency, alternative="greater")

        rows.append({
            "domain_A": dA,
            "domain_B": dB,
            "n_AB": nAB,
            "n_A": nA,
            "n_B": nB,
            "n_neither": n_neither,
            "N": N,
            "jaccard": round(jaccard, 4),
            "lift": round(lift, 4),
            "pmi": round(pmi, 4),
            "fisher_pvalue": round(fisher_p, 6),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)

    # BH-FDR on Fisher p-values via statsmodels.multipletests
    _, pvals_adj, _, _ = multipletests(result["fisher_pvalue"].values, method="fdr_bh")
    result["fisher_pvalue_adj"] = np.minimum(pvals_adj, 1.0).round(6)

    return result


def get_species_exclusive_domains(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Find domains found in exactly one Plasmodium species (species-exclusive).

    Returns
    -------
    dict
        {species_name: [exclusive_domain_1, ...]}
    """
    occurrence = compute_domain_occurrence(df)
    if occurrence.empty:
        return {}

    exclusive = occurrence[occurrence["species_count"] == 1].copy()
    result: Dict[str, List[str]] = {}
    for _, row in exclusive.iterrows():
        sp = row["species"][0]
        result.setdefault(sp, []).append(row["domain_name"])
    return result


def compute_interpro_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count InterPro entry occurrence across all proteins.
    Mirrors compute_domain_occurrence() but for InterPro cross-references.

    Returns
    -------
    pd.DataFrame
        Columns: interpro_id, count, species_count, pct_proteins, species.
    """
    records = []
    for _, row in df.iterrows():
        for ipr in row.get("interpro_ids", []):
            records.append({"interpro_id": ipr, "accession": row["accession"], "organism": row["organism"]})

    if not records:
        return pd.DataFrame(columns=["interpro_id", "count", "species_count", "pct_proteins", "species"])

    exp = pd.DataFrame(records)
    total = len(df)

    agg = (
        exp.groupby("interpro_id")
        .agg(
            count=("accession", "nunique"),
            species_count=("organism", "nunique"),
            species=("organism", lambda x: sorted(set(x))),
        )
        .reset_index()
    )
    agg["pct_proteins"] = (agg["count"] / total * 100).round(2)
    return agg.sort_values("count", ascending=False).reset_index(drop=True)
