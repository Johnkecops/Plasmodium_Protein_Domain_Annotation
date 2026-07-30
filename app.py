#!/usr/bin/env python3
"""
Plasmodium Protein Domain Annotator
Streamlit application for interactive HMM-based domain annotation
across all reviewed Plasmodium species in UniProt.

Author : Dr. Arli Aditya Parikesit, i3L University Jakarta
Date   : 2026

References:
    Parikesit, A. A., Utomo, D. H., & Karimah, N. (2018). Protein Domain
    Annotation of Plasmodium spp. Circumsporozoite Protein (CSP) Using Hidden
    Markov Model-based Tools. Jurnal Biologi Indonesia, 14(2), 185-190.
    https://doi.org/10.14203/jbi.v14i2.3737

    Widjaja, V., Lim, A., Aini, B., Gandasasmita, G. A., Darmawan, J. T., &
    Parikesit, A. A. (2022). Identification of Uncharacterized Plasmodium
    falciparum Proteins via In-silico Analysis. BIOEDUSCIENCE, 6(2).
    https://doi.org/10.22236/J.BES/628770
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SCRIPT"))

import json
import streamlit as st

from fetch_proteins import SPECIES_TAXONS
from domain_analysis import (
    compute_domain_occurrence,
    compute_species_domain_matrix,
    compute_domain_avoidance,
    compute_domain_cooccurrence_stats,
    compute_interpro_occurrence,
    get_species_summary,
    get_species_exclusive_domains,
)

from app_data import load_proteins, get_interpro_names, json_to_df
from app_tabs import (
    render_species_overview_tab,
    render_domain_occurrence_tab,
    render_domain_avoidance_tab,
    render_domain_cooccurrence_tab,
    render_interpro_tab,
    render_protein_browser_tab,
    render_raw_data_tab,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plasmodium Domain Annotator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .app-title  { font-size:2.0rem; font-weight:700; color:#1A1A2E; }
    .app-sub    { font-size:0.95rem; color:#555; margin-bottom:1.2rem; }
    .section-h  { font-size:1.15rem; font-weight:600; margin-top:0.5rem; }
    .info-box   { background:#f0f7ff; border-left:4px solid #2E86AB;
                  padding:0.7rem 1rem; border-radius:6px; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Plasmodium Domain Annotator")
    st.markdown(
        "Retrieve reviewed *Plasmodium* proteins from UniProt and analyse "
        "protein domain occurrence and avoidance patterns."
    )
    st.markdown("---")

    species_label = st.selectbox(
        "Plasmodium taxon",
        options=list(SPECIES_TAXONS.keys()),
        index=0,
        help="Use 'All Plasmodium spp.' to analyse the full genus at once.",
    )
    taxon_id = SPECIES_TAXONS[species_label]

    reviewed_only = st.checkbox(
        "Swiss-Prot (reviewed) only",
        value=True,
        help="Restrict to manually curated UniProt entries. Strongly recommended.",
    )

    max_results = st.slider(
        "Max proteins to fetch",
        min_value=50,
        max_value=5000,
        value=2000,
        step=50,
    )

    fetch_ipr_names = st.checkbox(
        "Fetch InterPro domain names",
        value=False,
        help="Calls InterPro REST API to resolve IPR accessions to human-readable names. "
             "Adds ~30–60 s for the first load; cached afterwards.",
    )

    st.markdown("---")
    run_btn = st.button("Fetch / Refresh data", type="primary", use_container_width=True)

    st.markdown(
        """
---
**Data sources**
- [UniProt REST API](https://rest.uniprot.org)
- [InterPro REST API](https://www.ebi.ac.uk/interpro)

**References**
- Parikesit et al. (2018) *JBI* 14(2):185–190
- Widjaja et al. (2022) *BIOEDUSCIENCE* 6(2):198–210
        """
    )


# ─── Session state ────────────────────────────────────────────────────────────
if "proteins_json" not in st.session_state:
    st.session_state.proteins_json = None
if "active_taxon" not in st.session_state:
    st.session_state.active_taxon = None

need_reload = (
    run_btn
    or st.session_state.proteins_json is None
    or st.session_state.active_taxon != taxon_id
)

# ─── Data fetch ───────────────────────────────────────────────────────────────
if need_reload:
    with st.spinner(f"Fetching *{species_label}* proteins from UniProt..."):
        try:
            proteins_json = load_proteins(
                taxon_id=taxon_id,
                reviewed=reviewed_only,
                max_results=max_results,
            )
            st.session_state.proteins_json = proteins_json
            st.session_state.active_taxon = taxon_id
        except RuntimeError as exc:
            st.error(f"Data fetch failed: {exc}")
            st.stop()

df = json_to_df(st.session_state.proteins_json)

if df is None or df.empty:
    st.warning("No proteins returned. Try a different taxon or uncheck 'reviewed only'.")
    st.stop()

# ─── Analyses ─────────────────────────────────────────────────────────────────
occurrence_df  = compute_domain_occurrence(df)
matrix_df      = compute_species_domain_matrix(df)
avoidance_df   = compute_domain_avoidance(df, min_species_presence=2)
summary_df     = get_species_summary(df)
ipr_occ_df     = compute_interpro_occurrence(df)
exclusive_map  = get_species_exclusive_domains(df)
cooc_stats_df  = compute_domain_cooccurrence_stats(df, min_pair_count=2, max_domains=60)

# InterPro name resolution (optional, slow)
ipr_name_map: dict = {}
if fetch_ipr_names and not ipr_occ_df.empty:
    top_ipr_ids = ipr_occ_df["interpro_id"].head(80).tolist()
    with st.spinner("Resolving InterPro domain names..."):
        ipr_names_json = get_interpro_names(json.dumps(top_ipr_ids))
        ipr_name_map = json.loads(ipr_names_json)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-title">Plasmodium Protein Domain Annotator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-sub">'
    'HMM-based domain annotation for reviewed <i>Plasmodium</i> species '
    '— domain occurrence, avoidance, and protein-level domain maps. '
    'Data sourced from UniProt Swiss-Prot and annotated via InterPro.'
    '</div>',
    unsafe_allow_html=True,
)

# ─── Top metrics ──────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Proteins", f"{len(df):,}")
m2.metric("Species", df["organism"].nunique())
m3.metric("Curated domains", len(occurrence_df))
m4.metric("InterPro entries", len(ipr_occ_df))
n_ann = (df["n_domains"] > 0).sum()
m5.metric("With domain annotation", f"{n_ann} ({n_ann/len(df)*100:.1f}%)")

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_ov, tab_occ, tab_avoid, tab_cooc, tab_ipr, tab_browse, tab_raw = st.tabs([
    "Species Overview",
    "Domain Occurrence",
    "Domain Avoidance",
    "Domain Co-occurrence",
    "InterPro Cross-refs",
    "Protein Browser",
    "Raw Data & Export",
])

render_species_overview_tab(tab_ov, summary_df, df, exclusive_map)
render_domain_occurrence_tab(tab_occ, occurrence_df, matrix_df, taxon_id)
render_domain_avoidance_tab(tab_avoid, avoidance_df, taxon_id)
render_domain_cooccurrence_tab(tab_cooc, cooc_stats_df, taxon_id)
render_interpro_tab(tab_ipr, ipr_occ_df, ipr_name_map, fetch_ipr_names, taxon_id)
render_protein_browser_tab(tab_browse, df, ipr_name_map)
render_raw_data_tab(tab_raw, df, taxon_id, reviewed_only)
