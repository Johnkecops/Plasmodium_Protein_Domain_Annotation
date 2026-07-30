import pandas as pd
import pytest

from domain_analysis import (
    get_species_summary,
    compute_domain_occurrence,
    compute_species_domain_matrix,
    compute_domain_avoidance,
    compute_domain_cooccurrence,
    compute_domain_cooccurrence_stats,
    get_species_exclusive_domains,
    compute_interpro_occurrence,
)


def test_get_species_summary(sample_protein_df):
    summary = get_species_summary(sample_protein_df)
    by_org = summary.set_index("organism")

    assert by_org.loc["Plasmodium falciparum", "n_proteins"] == 3
    assert by_org.loc["Plasmodium vivax", "n_proteins"] == 2
    assert by_org.loc["Plasmodium malariae", "n_proteins"] == 1
    assert by_org.loc["Plasmodium falciparum", "unique_domains"] == 3
    assert by_org.loc["Plasmodium malariae", "unique_domains"] == 1
    assert by_org.loc["Plasmodium falciparum", "max_length"] == 300
    # Sorted by n_proteins descending.
    assert list(summary["organism"]) == [
        "Plasmodium falciparum", "Plasmodium vivax", "Plasmodium malariae",
    ]


def test_compute_domain_occurrence(sample_protein_df):
    occ = compute_domain_occurrence(sample_protein_df).set_index("domain_name")

    assert occ.loc["Core domain", "count"] == 4
    assert occ.loc["Core domain", "species_count"] == 3
    assert occ.loc["Shell domain", "count"] == 2
    assert occ.loc["Shell domain", "species_count"] == 2
    assert occ.loc["Cloud falciparum domain", "species_count"] == 1
    # Sorted by count descending -> Core domain (4) is first.
    full = compute_domain_occurrence(sample_protein_df)
    assert full.iloc[0]["domain_name"] == "Core domain"


def test_compute_domain_occurrence_empty():
    empty = pd.DataFrame(columns=["accession", "organism", "domain_names"])
    result = compute_domain_occurrence(empty)
    assert result.empty
    assert list(result.columns) == ["domain_name", "count", "species_count", "pct_proteins", "species"]


def test_compute_species_domain_matrix(sample_protein_df):
    matrix = compute_species_domain_matrix(sample_protein_df)
    assert matrix.loc["Plasmodium falciparum", "Core domain"] == 2
    assert matrix.loc["Plasmodium vivax", "Cloud vivax domain"] == 1
    assert matrix.loc["Plasmodium malariae", "Shell domain"] == 0


def test_compute_domain_avoidance(sample_protein_df):
    avoidance = compute_domain_avoidance(sample_protein_df)

    # Core domain is universal (n_species_absent=0) -> excluded.
    # Cloud domains are single-species (n_species_present=1 < min_species_presence) -> excluded.
    # Only Shell domain (present in 2 of 3 species) qualifies.
    assert list(avoidance["domain_name"]) == ["Shell domain"]
    row = avoidance.iloc[0]
    assert row["n_species_present"] == 2
    assert row["n_species_absent"] == 1
    assert row["avoidance_score"] == pytest.approx(1 / 3, abs=1e-4)
    assert 0.0 <= row["avoidance_pvalue_adj"] <= 1.0


def test_compute_domain_avoidance_warns_on_single_species(sample_protein_df):
    single = sample_protein_df[sample_protein_df["organism"] == "Plasmodium falciparum"]
    with pytest.warns(UserWarning, match="only one species"):
        compute_domain_avoidance(single)


def test_compute_domain_cooccurrence(sample_protein_df):
    matrix = compute_domain_cooccurrence(sample_protein_df)
    assert matrix.loc["Core domain", "Core domain"] == 4
    assert matrix.loc["Shell domain", "Shell domain"] == 2
    assert matrix.loc["Core domain", "Shell domain"] == 2
    assert matrix.loc["Shell domain", "Core domain"] == 2  # symmetric
    assert matrix.loc["Cloud falciparum domain", "Cloud vivax domain"] == 0


def test_compute_domain_cooccurrence_stats(sample_protein_df):
    stats = compute_domain_cooccurrence_stats(sample_protein_df, min_pair_count=2)

    assert len(stats) == 1
    row = stats.iloc[0]
    assert {row["domain_A"], row["domain_B"]} == {"Core domain", "Shell domain"}
    assert row["n_AB"] == 2
    assert row["n_A"] == 4
    assert row["n_B"] == 2
    assert row["N"] == 6
    assert row["jaccard"] == pytest.approx(0.5, abs=1e-4)
    assert row["lift"] == pytest.approx(1.5, abs=1e-4)
    assert 0.0 <= row["fisher_pvalue"] <= 1.0
    assert 0.0 <= row["fisher_pvalue_adj"] <= 1.0


def test_get_species_exclusive_domains(sample_protein_df):
    exclusive = get_species_exclusive_domains(sample_protein_df)
    assert exclusive["Plasmodium falciparum"] == ["Cloud falciparum domain"]
    assert exclusive["Plasmodium vivax"] == ["Cloud vivax domain"]
    assert "Plasmodium malariae" not in exclusive


def test_compute_interpro_occurrence(sample_protein_df):
    occ = compute_interpro_occurrence(sample_protein_df).set_index("interpro_id")
    assert occ.loc["IPR00001", "count"] == 4
    assert occ.loc["IPR00001", "species_count"] == 3
    assert occ.loc["IPR00003", "species_count"] == 1
