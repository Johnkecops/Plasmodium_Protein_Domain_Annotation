"""
Offline tests for fetch_proteins.py. Everything here runs without touching
the live UniProt API: pure parsing functions are tested directly, and HTTP
calls are mocked with `responses` against a small fixture TSV
(tests/fixtures/sample_uniprot.tsv) instead of the real endpoint. This closes
the gap noted in MEMORY.md, where a prior session could not validate the
pipeline at all because the sandbox had no outbound network access.
"""

import os

import pandas as pd
import pytest
import responses

import fetch_proteins
from fetch_proteins import (
    parse_ft_domain,
    parse_semicolon_ids,
    _normalise_organism_name,
    _normalise,
    _extract_next_url,
    _parse_go_terms,
    _request_with_retry,
    fetch_plasmodium_proteins,
    UNIPROT_SEARCH,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_uniprot.tsv")


def _fixture_text() -> str:
    with open(FIXTURE_PATH) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Pure parsing functions
# ---------------------------------------------------------------------------

def test_parse_ft_domain_extracts_name_and_span():
    ft = 'DOMAIN 309..375; /note="Thrombospondin type-1"; /evidence="ECO:0000255"'
    result = parse_ft_domain(ft)
    assert result == [{"name": "Thrombospondin type-1", "start": 309, "end": 375}]


def test_parse_ft_domain_multiple_domains():
    ft = 'DOMAIN 1..50; /note="First"; DOMAIN 60..120; /note="Second"'
    result = parse_ft_domain(ft)
    assert [d["name"] for d in result] == ["First", "Second"]


@pytest.mark.parametrize("value", [None, "", "nan", "NaN"])
def test_parse_ft_domain_empty_inputs(value):
    assert parse_ft_domain(value) == []


def test_parse_semicolon_ids():
    assert parse_semicolon_ids("IPR000884;IPR036465") == ["IPR000884", "IPR036465"]
    assert parse_semicolon_ids("") == []
    assert parse_semicolon_ids(None) == []


def test_normalise_organism_name_strips_qualifiers():
    assert _normalise_organism_name(
        "Plasmodium falciparum (isolate 3D7) (Malaria parasite)"
    ) == "Plasmodium falciparum"
    assert _normalise_organism_name("Plasmodium vivax") == "Plasmodium vivax"
    assert _normalise_organism_name("Salvador I") == "Salvador I"


def test_parse_go_terms_strips_go_id_suffix():
    assert _parse_go_terms("carbohydrate binding [GO:0030246]") == ["carbohydrate binding"]
    assert _parse_go_terms(None) == []


def test_extract_next_url_present():
    header = '<https://rest.uniprot.org/uniprotkb/search?cursor=abc>; rel="next"'
    assert _extract_next_url(header) == "https://rest.uniprot.org/uniprotkb/search?cursor=abc"


def test_extract_next_url_absent():
    assert _extract_next_url("") is None
    assert _extract_next_url('<https://example.com>; rel="prev"') is None


# ---------------------------------------------------------------------------
# _normalise() on a realistic UniProt TSV chunk
# ---------------------------------------------------------------------------

def test_normalise_parses_fixture_correctly():
    raw = pd.read_csv(FIXTURE_PATH, sep="\t")
    df = _normalise(raw)

    assert len(df) == 2
    row0 = df.iloc[0]
    assert row0["accession"] == "Q8I1R6"
    assert row0["organism"] == "Plasmodium falciparum"  # qualifier stripped
    assert row0["domain_names"] == ["Thrombospondin type-1"]
    assert row0["n_domains"] == 1
    assert row0["pfam_ids"] == ["PF00090"]
    assert row0["interpro_ids"] == ["IPR000884", "IPR036465"]
    assert row0["has_signal_peptide"] is True or bool(row0["has_signal_peptide"]) is True
    assert row0["has_gpi_anchor"] is True or bool(row0["has_gpi_anchor"]) is True

    row1 = df.iloc[1]
    assert row1["organism"] == "Plasmodium vivax"
    assert row1["n_domains"] == 0
    assert row1["domain_names"] == []
    assert bool(row1["has_signal_peptide"]) is False


# ---------------------------------------------------------------------------
# fetch_plasmodium_proteins() with mocked HTTP (no live network)
# ---------------------------------------------------------------------------

@responses.activate
def test_fetch_plasmodium_proteins_single_page():
    responses.add(
        responses.GET, UNIPROT_SEARCH,
        body=_fixture_text(), status=200,
        content_type="text/tab-separated-values",
    )

    df = fetch_plasmodium_proteins(taxon_id="5820", reviewed=True, max_results=5000)

    assert len(df) == 2
    assert set(df["organism"]) == {"Plasmodium falciparum", "Plasmodium vivax"}


@responses.activate
def test_fetch_plasmodium_proteins_paginates_via_link_header(monkeypatch):
    monkeypatch.setattr(fetch_proteins.time, "sleep", lambda *_: None)

    page1_lines = _fixture_text().splitlines()
    page1 = "\n".join(page1_lines[:2])  # header + first data row only
    page2 = "\n".join([page1_lines[0], page1_lines[2]])  # header + second data row
    next_url = UNIPROT_SEARCH + "?cursor=page2"

    # Both pages share the same base URL (UniProt paginates via cursor on the
    # same endpoint), so a single callback keyed on the request URL is used
    # instead of two static responses.add() registrations to avoid ambiguous
    # query-string matching between them.
    def _callback(request):
        if "cursor=page2" in request.url:
            return (200, {}, page2)
        return (200, {"Link": f'<{next_url}>; rel="next"'}, page1)

    responses.add_callback(responses.GET, UNIPROT_SEARCH, callback=_callback)
    responses.add_callback(responses.GET, next_url, callback=_callback)

    df = fetch_plasmodium_proteins(taxon_id="5820", max_results=5000)

    assert len(df) == 2
    assert set(df["accession"]) == {"Q8I1R6", "Q8IB24"}


@responses.activate
def test_fetch_plasmodium_proteins_respects_max_results(monkeypatch):
    monkeypatch.setattr(fetch_proteins.time, "sleep", lambda *_: None)
    responses.add(
        responses.GET, UNIPROT_SEARCH,
        body=_fixture_text(), status=200,
    )

    df = fetch_plasmodium_proteins(taxon_id="5820", max_results=1)
    assert len(df) == 1


@responses.activate
def test_fetch_plasmodium_proteins_empty_result():
    responses.add(responses.GET, UNIPROT_SEARCH, body="", status=200)
    df = fetch_plasmodium_proteins(taxon_id="999999")
    assert df.empty


# ---------------------------------------------------------------------------
# Retry / backoff behaviour
# ---------------------------------------------------------------------------

@responses.activate
def test_request_with_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(fetch_proteins.time, "sleep", lambda *_: None)

    responses.add(responses.GET, UNIPROT_SEARCH, status=503)
    responses.add(responses.GET, UNIPROT_SEARCH, status=503)
    responses.add(responses.GET, UNIPROT_SEARCH, body="ok", status=200)

    resp = _request_with_retry(UNIPROT_SEARCH)
    assert resp.status_code == 200
    assert resp.text == "ok"


@responses.activate
def test_request_with_retry_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(fetch_proteins.time, "sleep", lambda *_: None)

    responses.add(responses.GET, UNIPROT_SEARCH, status=503)
    responses.add(responses.GET, UNIPROT_SEARCH, status=503)
    responses.add(responses.GET, UNIPROT_SEARCH, status=503)

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        _request_with_retry(UNIPROT_SEARCH)
