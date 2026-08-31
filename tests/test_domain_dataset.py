#!/usr/bin/env python3
"""
Module: test_domain_dataset
Purpose: Guard the identifier-normalisation defects that the reviewers identified.
Author: Dr. Arli Aditya Parikesit, Dr. Arif Nur Muhammad Ansori, and Moch. Royhan Afnani.,M.Sc
Date: 2026

Each test below corresponds to a specific reviewer finding, so a regression reintroduces
a published defect and fails loudly rather than silently changing a reported number.
"""

import pandas as pd
import pytest

from domain_dataset import (
    build_domain_table,
    normalise_columns,
    normalise_species,
    species_coverage_table,
    strip_instance_suffix,
    summarise_dataset,
)


@pytest.fixture
def snapshot():
    """
    A minimal snapshot exercising repeats, note-less features and multi-proteome species.

    Built from raw UniProt TSV headers and passed through normalise_columns, so the tests
    exercise the same entry path as the pipeline rather than a hand-made schema.
    """
    raw = pd.DataFrame(
        {
            "Entry": ["A0001", "A0002", "A0003", "A0004"],
            "Reviewed": ["reviewed", "unreviewed", "unreviewed", "reviewed"],
            "Organism": [
                "Plasmodium falciparum (isolate 3D7)",
                "Plasmodium berghei (strain Anka)",
                "Plasmodium yoelii 17X",
                "Plasmodium ovale wallikeri",
            ],
            "Length": ["500", "1200", "300", "800"],
            "Domain [FT]": [
                'DOMAIN 25..80; /note="6-Cys 1"; /evidence="ECO:0000255"; DOMAIN 120..190; /note="6-Cys 2"',
                'DOMAIN 10..60; /note="ABC transporter 1"; DOMAIN 300..360; /note="ABC transporter 2"',
                "DOMAIN 5..70;",
                "",
            ],
            "Pfam": ["PF07422;", "PF00005;PF00664;", "", "PF00069;"],
            "InterPro": ["IPR010884;", "IPR003439;IPR011527;", "IPR000001;", "IPR000719;"],
            "proteome_id": ["UP000001450", "UP000074855", "UP000018538", "UP000078555"],
        }
    )
    return normalise_columns(raw)


def test_strip_instance_suffix_removes_uniprot_repeat_counter():
    """Reviewer 4: "6-Cys 4" and "6-Cys 7" are instances of one family, not two families."""
    assert strip_instance_suffix("6-Cys 4") == "6-Cys"
    assert strip_instance_suffix("6-Cys 7") == "6-Cys"
    assert strip_instance_suffix("ABC transporter 2") == "ABC transporter"


def test_strip_instance_suffix_preserves_names_ending_in_a_meaningful_digit():
    """Family names whose final token is not a bare repeat counter must survive intact."""
    assert strip_instance_suffix("EF-hand") == "EF-hand"
    assert strip_instance_suffix("TSP_1") == "TSP_1"
    assert strip_instance_suffix("Sm-like") == "Sm-like"


def test_ft_domain_layer_collapses_repeats_into_copy_number(snapshot):
    """Repeated instances raise copy_number rather than creating extra families."""
    table = build_domain_table(snapshot, layer="ft_domain")
    six_cys = table[(table["accession"] == "A0001")]
    assert len(six_cys) == 1, "two 6-Cys instances must collapse to one family row"
    assert int(six_cys.iloc[0]["copy_number"]) == 2
    assert six_cys.iloc[0]["domain_name"] == "6-Cys"


def test_note_less_domain_features_are_excluded_not_merged(snapshot):
    """
    Reviewer 4: assigning "Unknown domain" to every note-less DOMAIN feature merged
    unrelated domains into a single graph node. Such features must not become a family.
    """
    table = build_domain_table(snapshot, layer="ft_domain")
    assert "Unknown domain" not in set(table["domain_name"])
    assert "A0003" not in set(table["accession"])


def test_pfam_layer_is_keyed_by_accession(snapshot):
    """The canonical layer keys on Pfam accessions, never on short names."""
    table = build_domain_table(snapshot, layer="pfam")
    assert set(table["domain_accession"]) == {"PF07422", "PF00005", "PF00664", "PF00069"}
    assert table["domain_name"].isna().all(), "names are attached later, from the Pfam clan table"


def test_species_normalisation_merges_isolates_and_strains(snapshot):
    """Two reference proteomes of one species must not inflate the species count."""
    assert normalise_species("Plasmodium falciparum (isolate 3D7)") == "Plasmodium falciparum"
    assert normalise_species("Plasmodium falciparum (isolate NF54)") == "Plasmodium falciparum"
    assert normalise_species("Plasmodium yoelii 17X") == "Plasmodium yoelii"
    assert normalise_species("Plasmodium yoelii yoelii") == "Plasmodium yoelii"
    assert normalise_species("Plasmodium berghei (strain Anka)") == "Plasmodium berghei"


def test_species_normalisation_keeps_distinct_subspecies():
    """P. ovale wallikeri is reported separately from P. ovale in UniProt and stays distinct."""
    assert normalise_species("Plasmodium ovale wallikeri") == "Plasmodium ovale wallikeri"
    assert normalise_species("Plasmodium vinckei petteri") == "Plasmodium vinckei petteri"


def test_coverage_table_reports_the_requested_columns(snapshot):
    """Reviewer 1 asked for proteins, domains, domains per protein, missing fraction, coverage."""
    domains = build_domain_table(snapshot, layer="pfam")
    coverage = species_coverage_table(snapshot, domains)
    for column in (
        "n_proteins",
        "n_annotated_proteins",
        "n_domain_instances",
        "n_unique_families",
        "domains_per_protein",
        "annotation_coverage",
        "unannotated_fraction",
    ):
        assert column in coverage.columns
    assert (coverage["annotation_coverage"] + coverage["unannotated_fraction"]).round(4).eq(1.0).all()


def test_summary_coverage_denominator_is_all_proteins(snapshot):
    """
    Reviewer 4: the reported dark fraction must state its denominator. Coverage is
    computed over every protein in the analysed set, annotated or not.
    """
    summary = summarise_dataset(snapshot)
    assert summary.n_protein_rows == 4
    assert summary.pfam_protein_coverage == pytest.approx(3 / 4)
    assert summary.interpro_protein_coverage == pytest.approx(4 / 4)


def test_interpro_coverage_cannot_fall_below_pfam_coverage(snapshot):
    """
    Reviewer 4's construction argument: InterPro integrates Pfam, so a protein with an
    integrated Pfam signature necessarily carries an InterPro entry. Layer 2 coverage
    below Layer 1 coverage indicates a pipeline error, not a biological finding.
    """
    summary = summarise_dataset(snapshot)
    assert summary.interpro_protein_coverage >= summary.pfam_protein_coverage
