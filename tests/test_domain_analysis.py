"""
Tests for the domain_analysis facade over genus_analysis.

Occurrence, avoidance and co-occurrence are now Pfam-accession-keyed (domain_analysis.py
delegates to genus_analysis.py rather than computing anything itself), matching the
manuscript's corrected pipeline. sample_protein_df's domain_names and pfam_ids correspond
1:1 by position (PF00001="Core domain", PF00002="Shell domain", PF00003="Cloud falciparum
domain", PF00004="Cloud vivax domain"; see conftest.py), so assertions below are pinned to
accessions rather than to the retired curated-note identifiers.

Expected numeric values (avoidance excess_absence, q_value, etc.) were derived by running
genus_analysis directly against this fixture's shape before writing these assertions, not
guessed, since the depth-conditioned null does not reduce to simple fractions.
"""

import pandas as pd
import pytest

from domain_analysis import (
    get_species_summary,
    compute_domain_occurrence,
    compute_species_domain_matrix,
    compute_domain_avoidance,
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
    occ = compute_domain_occurrence(sample_protein_df).set_index("domain_accession")

    assert occ.loc["PF00001", "count"] == 4
    assert occ.loc["PF00001", "species_count"] == 3
    assert occ.loc["PF00002", "count"] == 2
    assert occ.loc["PF00002", "species_count"] == 2
    assert occ.loc["PF00003", "species_count"] == 1
    # Sorted by count descending -> PF00001 (4 carriers) is first.
    full = compute_domain_occurrence(sample_protein_df)
    assert full.iloc[0]["domain_accession"] == "PF00001"


def test_compute_domain_occurrence_reports_accession_and_display_label(sample_protein_df):
    """
    Every family is reported by accession, per Reviewer 4: the submitted manuscript's
    reliance on short names alone let two unrelated Pfam families (TSP_N, TSP_C) be
    reported as an N-/C-terminal pair. domain_name is a display label that always embeds
    the accession, so a chart or table built from this column cannot lose it.
    """
    occ = compute_domain_occurrence(sample_protein_df).set_index("domain_accession")
    assert "PF00001" in occ.loc["PF00001", "domain_name"]


def test_compute_domain_occurrence_reports_reviewed_split(sample_protein_df):
    """Reviewer 3 asked whether domain dominance is a curation artefact; this answers it."""
    occ = compute_domain_occurrence(sample_protein_df).set_index("domain_accession")
    row = occ.loc["PF00001"]
    assert row["n_reviewed_carriers"] == 4
    assert row["n_unreviewed_carriers"] == 0
    assert row["reviewed_fraction"] == pytest.approx(1.0)


def test_compute_domain_occurrence_empty():
    empty = pd.DataFrame(columns=["accession", "organism", "pfam_ids", "reviewed", "length"])
    result = compute_domain_occurrence(empty)
    assert result.empty
    assert list(result.columns) == [
        "domain_accession", "domain_name", "count", "species_count", "pct_proteins", "species",
        "n_reviewed_carriers", "n_unreviewed_carriers", "reviewed_fraction",
    ]


def test_compute_species_domain_matrix(sample_protein_df, monkeypatch):
    import domain_analysis

    # Force the bare-accession label path so this test does not depend on whichever Pfam
    # names happen to be in data/frozen/Pfam-A.clans.tsv.gz (or whether it exists at all).
    monkeypatch.setattr(domain_analysis, "_load_clans", lambda: pd.DataFrame(columns=domain_analysis._CLAN_COLUMNS))

    matrix = compute_species_domain_matrix(sample_protein_df)
    assert matrix.loc["Plasmodium falciparum", "PF00001"] == 2
    assert matrix.loc["Plasmodium vivax", "PF00004"] == 1
    assert matrix.loc["Plasmodium malariae", "PF00002"] == 0


def test_compute_domain_avoidance_includes_universal_domains(sample_protein_df, monkeypatch):
    """
    Reviewer 4 / the editor: several manuscript-cited families (LCCL, CLAG, EBA-175, RAP)
    turned out to be present in every species once annotation depth was accounted for, and
    that correction only stays visible if universal domains remain in the output at
    avoidance_score 0 rather than being dropped, as the previous facade dropped them.
    """
    import domain_analysis

    monkeypatch.setattr(domain_analysis, "_load_clans", lambda: pd.DataFrame(columns=domain_analysis._CLAN_COLUMNS))

    avoidance = compute_domain_avoidance(sample_protein_df, min_carriers=2)

    # PF00003 and PF00004 (1 carrier each) fall below min_carriers=2 and are not tested.
    # PF00001 (universal, 4 carriers) and PF00002 (2 carriers, absent from 1 species) both
    # clear the threshold and both appear, including the universal one.
    assert set(avoidance["domain_accession"]) == {"PF00001", "PF00002"}

    row1 = avoidance.set_index("domain_accession").loc["PF00001"]
    assert row1["n_species_present"] == 3
    assert row1["n_species_absent"] == 0
    assert row1["avoidance_score"] == pytest.approx(0.0)

    row2 = avoidance.set_index("domain_accession").loc["PF00002"]
    assert row2["n_species_present"] == 2
    assert row2["n_species_absent"] == 1
    assert row2["avoidance_score"] == pytest.approx(1 / 3, abs=1e-4)

    # Sorted by excess_absence descending: PF00002's absence is less fully explained by
    # annotation depth than PF00001's (which has none), so it ranks first.
    assert list(avoidance["domain_accession"]) == ["PF00002", "PF00001"]

    for col in ("min_binom_pvalue", "avoidance_pvalue_adj"):
        assert (avoidance[col].between(0.0, 1.0)).all()


def test_compute_domain_avoidance_warns_on_single_species(sample_protein_df):
    single = sample_protein_df[sample_protein_df["organism"] == "Plasmodium falciparum"]
    with pytest.warns(UserWarning, match="only one species"):
        compute_domain_avoidance(single)


def test_compute_domain_cooccurrence_stats(sample_protein_df, monkeypatch):
    import domain_analysis

    # PF00001/PF00002 are real accessions (7tm_1/7tm_2) that happen to share a Pfam clan
    # in the current release; isolate the obligate/same-clan flag from that real-world
    # coincidence so this test exercises the fixture's synthetic pairing, not Pfam content.
    monkeypatch.setattr(domain_analysis, "_load_clans", lambda: pd.DataFrame(columns=domain_analysis._CLAN_COLUMNS))

    stats = compute_domain_cooccurrence_stats(sample_protein_df, min_pair_count=2)

    assert len(stats) == 1
    row = stats.iloc[0]
    assert {row["domain_A"], row["domain_B"]} == {"PF00001", "PF00002"}
    assert row["n_AB"] == 2
    assert row["n_A"] == 4
    assert row["n_B"] == 2
    assert row["N"] == 6
    assert row["jaccard"] == pytest.approx(0.5, abs=1e-4)
    assert row["lift"] == pytest.approx(1.5, abs=1e-4)
    assert 0.0 <= row["fisher_pvalue"] <= 1.0
    assert 0.0 <= row["fisher_pvalue_adj"] <= 1.0
    # PF00001+PF00002 co-occur in 2 of PF00001's 4 carriers, so this is not an obligate
    # pair, and headline_eligible reflects that.
    assert bool(row["obligate_pair"]) is False
    assert bool(row["headline_eligible"]) is True


def test_get_species_exclusive_domains(sample_protein_df):
    exclusive = get_species_exclusive_domains(sample_protein_df)
    assert len(exclusive["Plasmodium falciparum"]) == 1
    assert "PF00003" in exclusive["Plasmodium falciparum"][0]
    assert len(exclusive["Plasmodium vivax"]) == 1
    assert "PF00004" in exclusive["Plasmodium vivax"][0]
    assert "Plasmodium malariae" not in exclusive


def test_compute_interpro_occurrence(sample_protein_df):
    occ = compute_interpro_occurrence(sample_protein_df).set_index("interpro_id")
    assert occ.loc["IPR00001", "count"] == 4
    assert occ.loc["IPR00001", "species_count"] == 3
    assert occ.loc["IPR00003", "species_count"] == 1
