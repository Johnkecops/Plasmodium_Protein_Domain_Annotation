#!/usr/bin/env python3
"""
Module: app_data
Purpose: Cached data-loading helpers for app.py (UniProt/InterPro fetches
         wrapped in st.cache_data, plus JSON<->DataFrame round-tripping).
         Split out of app.py to keep that file focused on page layout.
Author: Dr. Arli Aditya Parikesit
Date: 2026
"""

import io
import json

import pandas as pd
import streamlit as st

from fetch_proteins import (
    fetch_plasmodium_proteins,
    build_interpro_name_map,
    fetch_fasta_sequences,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_proteins(taxon_id: str, reviewed: bool, max_results: int) -> str:
    """Fetch protein table and return as JSON string for cache compatibility."""
    df = fetch_plasmodium_proteins(
        taxon_id=taxon_id,
        reviewed=reviewed,
        max_results=max_results,
    )
    return df.to_json(orient="records")


@st.cache_data(ttl=3600, show_spinner=False)
def load_fasta(taxon_id: str, reviewed: bool, max_fasta: int) -> str:
    return fetch_fasta_sequences(taxon_id=taxon_id, reviewed=reviewed, max_results=max_fasta)


@st.cache_data(ttl=7200, show_spinner=False)
def get_interpro_names(ipr_ids_json: str) -> str:
    """Fetch InterPro names and return as JSON string."""
    ipr_ids = json.loads(ipr_ids_json)
    name_map = build_interpro_name_map(ipr_ids[:80])  # cap to avoid slow load
    return json.dumps(name_map)


def json_to_df(json_str: str) -> pd.DataFrame:
    """Restore DataFrame from JSON, fixing list columns."""
    if not json_str:
        return pd.DataFrame()
    df = pd.read_json(io.StringIO(json_str), orient="records")
    for col in ["domains", "domain_names", "interpro_ids", "pfam_ids", "go_functions", "keywords"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    return df
