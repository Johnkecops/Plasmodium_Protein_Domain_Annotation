"""
Pytest configuration: put SCRIPT/ on sys.path (mirrors the sys.path.insert
pattern app.py and run_pipeline.py already use) so tests can import the
pipeline modules directly, and provide a small synthetic protein DataFrame
fixture shared across test modules. No network access required anywhere here.
"""

import os
import sys

SCRIPT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "SCRIPT"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import pandas as pd
import pytest


@pytest.fixture
def sample_protein_df() -> pd.DataFrame:
    """
    Small synthetic Plasmodium-like protein table matching the schema produced
    by fetch_plasmodium_proteins(): accession, organism, length, n_domains,
    domain_names (list[str]), pfam_ids (list[str]), interpro_ids (list[str]).

    Layout (3 species):
      - PF00001 ("core"): present in all 3 species.
      - PF00002 ("shell"): present in falciparum + vivax only.
      - PF00003 ("cloud_falciparum"): falciparum only.
      - PF00004 ("cloud_vivax"): vivax only.
    """
    rows = [
        # Plasmodium falciparum: 3 proteins
        dict(accession="P1", organism="Plasmodium falciparum", length=250,
             n_domains=2, domain_names=["Core domain", "Shell domain"],
             pfam_ids=["PF00001", "PF00002"], interpro_ids=["IPR00001"]),
        dict(accession="P2", organism="Plasmodium falciparum", length=180,
             n_domains=1, domain_names=["Cloud falciparum domain"],
             pfam_ids=["PF00003"], interpro_ids=["IPR00003"]),
        dict(accession="P3", organism="Plasmodium falciparum", length=300,
             n_domains=1, domain_names=["Core domain"],
             pfam_ids=["PF00001"], interpro_ids=["IPR00001"]),
        # Plasmodium vivax: 2 proteins
        dict(accession="P4", organism="Plasmodium vivax", length=220,
             n_domains=2, domain_names=["Core domain", "Shell domain"],
             pfam_ids=["PF00001", "PF00002"], interpro_ids=["IPR00001"]),
        dict(accession="P5", organism="Plasmodium vivax", length=150,
             n_domains=1, domain_names=["Cloud vivax domain"],
             pfam_ids=["PF00004"], interpro_ids=["IPR00004"]),
        # Plasmodium malariae: 1 protein
        dict(accession="P6", organism="Plasmodium malariae", length=210,
             n_domains=1, domain_names=["Core domain"],
             pfam_ids=["PF00001"], interpro_ids=["IPR00001"]),
    ]
    return pd.DataFrame(rows)
