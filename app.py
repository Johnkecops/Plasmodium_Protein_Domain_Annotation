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

import io
import json
import time
import streamlit as st
import pandas as pd
import numpy as np

from fetch_proteins import (
    fetch_plasmodium_proteins,
    build_interpro_name_map,
    fetch_fasta_sequences,
    SPECIES_TAXONS,
)
from domain_analysis import (
    compute_domain_occurrence,
    compute_species_domain_matrix,
    compute_domain_avoidance,
    compute_domain_cooccurrence,
    compute_domain_cooccurrence_stats,
    compute_interpro_occurrence,
    get_species_summary,
    get_species_exclusive_domains,
)
from visualize import (
    plot_species_overview,
    plot_domain_length_distribution,
    plot_domain_occurrence_bar,
    plot_species_domain_heatmap,
    plot_domain_avoidance,
    plot_protein_domain_map,
    plot_interpro_occurrence_bar,
    plot_domain_cooccurrence_heatmap,
    plot_domain_cooccurrence_top_pairs,
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


# ─── Cached data loaders ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_proteins(taxon_id: str, reviewed: bool, max_results: int) -> str:
    """Fetch protein table and return as JSON string for cache compatibility."""
    df = fetch_plasmodium_proteins(
        taxon_id=taxon_id,
        reviewed=reviewed,
        max_results=max_results,
    )
    return df.to_json(orient="records")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_fasta(taxon_id: str, reviewed: bool, max_fasta: int) -> str:
    return fetch_fasta_sequences(taxon_id=taxon_id, reviewed=reviewed, max_results=max_fasta)


@st.cache_data(ttl=7200, show_spinner=False)
def _get_interpro_names(ipr_ids_json: str) -> str:
    """Fetch InterPro names and return as JSON string."""
    ipr_ids = json.loads(ipr_ids_json)
    name_map = build_interpro_name_map(ipr_ids[:80])  # cap to avoid slow load
    return json.dumps(name_map)


def _json_to_df(json_str: str) -> pd.DataFrame:
    """Restore DataFrame from JSON, fixing list columns."""
    if not json_str:
        return pd.DataFrame()
    df = pd.read_json(io.StringIO(json_str), orient="records")
    for col in ["domains", "domain_names", "interpro_ids", "pfam_ids", "go_functions", "keywords"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    return df


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
            proteins_json = _load_proteins(
                taxon_id=taxon_id,
                reviewed=reviewed_only,
                max_results=max_results,
            )
            st.session_state.proteins_json = proteins_json
            st.session_state.active_taxon = taxon_id
        except RuntimeError as exc:
            st.error(f"Data fetch failed: {exc}")
            st.stop()

df = _json_to_df(st.session_state.proteins_json)

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
        ipr_names_json = _get_interpro_names(json.dumps(top_ipr_ids))
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Species Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_ov:
    st.subheader("Species-level summary")

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(plot_species_overview(summary_df), use_container_width=True)
    with col_table:
        disp_sum = summary_df.rename(columns={
            "organism":          "Species",
            "n_proteins":        "Proteins",
            "n_with_domains":    "With domains",
            "pct_with_domains":  "% annotated",
            "unique_domains":    "Unique domains",
            "total_domains":     "Total domains",
            "avg_length":        "Avg len (aa)",
            "max_length":        "Max len (aa)",
        })
        st.dataframe(disp_sum, use_container_width=True, hide_index=True)

    st.subheader("Protein length distribution")
    st.plotly_chart(plot_domain_length_distribution(df), use_container_width=True)

    if exclusive_map:
        st.subheader("Species-exclusive domains")
        st.markdown(
            "Domains found in proteins of exactly **one** species — "
            "potential lineage-specific adaptations."
        )
        for sp, doms in sorted(exclusive_map.items()):
            with st.expander(f"{sp} ({len(doms)} exclusive domains)"):
                st.write(", ".join(doms))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Domain Occurrence
# ══════════════════════════════════════════════════════════════════════════════
with tab_occ:
    st.subheader("Domain occurrence — manually curated UniProt annotations")

    if occurrence_df.empty:
        st.info(
            "No manually annotated domain features found for this taxon.\n\n"
            "UniProt Swiss-Prot domain annotations (Feature type: Domain) are the "
            "gold-standard source. If none appear here, check the InterPro Cross-refs "
            "tab for computationally predicted entries."
        )
    else:
        ctl1, ctl2 = st.columns(2)
        top_n = ctl1.slider("Top N domains", 5, min(50, len(occurrence_df)), 20, key="occ_n")
        color_by = ctl2.radio(
            "Colour by",
            ["count", "species_count"],
            format_func=lambda x: "Protein count" if x == "count" else "Species count",
            horizontal=True,
        )

        st.plotly_chart(
            plot_domain_occurrence_bar(occurrence_df, top_n=top_n, color_by=color_by),
            use_container_width=True,
        )

        if not matrix_df.empty:
            st.subheader("Domain × species heatmap")
            max_heatmap = st.slider("Max domain columns", 5, 60, 30, key="heat_n")
            st.plotly_chart(
                plot_species_domain_heatmap(matrix_df, max_domains=max_heatmap),
                use_container_width=True,
            )

        st.subheader("Full occurrence table")
        disp_occ = occurrence_df.copy()
        disp_occ["species"] = disp_occ["species"].apply("; ".join)
        st.dataframe(
            disp_occ.rename(columns={
                "domain_name":  "Domain",
                "count":        "Proteins",
                "species_count":"Species",
                "pct_proteins": "% of proteome",
                "species":      "Species list",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download occurrence table (CSV)",
            disp_occ.to_csv(index=False),
            f"plasmodium_{taxon_id}_domain_occurrence.csv",
            "text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Domain Avoidance
# ══════════════════════════════════════════════════════════════════════════════
with tab_avoid:
    st.subheader("Domain avoidance analysis")
    st.markdown(
        '<div class="info-box">'
        "<b>Domain avoidance</b> identifies domains present in some "
        "<i>Plasmodium</i> species but systematically absent in others. "
        "A high avoidance score means the domain is widespread across the genus "
        "but lost (or never acquired) in a specific lineage — a potential "
        "signature of functional divergence or evolutionary selection."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if avoidance_df.empty:
        st.info(
            "No avoidance patterns detected. This occurs when: "
            "(1) too few species are represented; "
            "(2) all species share the same domains; or "
            "(3) no proteins have manually annotated domains."
        )
    else:
        av_c1, av_c2 = st.columns(2)
        top_avoid = av_c1.slider("Top N", 5, min(40, len(avoidance_df)), 20, key="avoid_n")
        sig_filter = av_c2.checkbox(
            "Show only significant avoidance (adj p < 0.05)",
            value=False,
            help="Filter domains whose absence in at least one species is statistically "
                 "significant after Benjamini-Hochberg FDR correction.",
        )

        disp_av_src = avoidance_df.copy()
        if sig_filter and "avoidance_pvalue_adj" in disp_av_src.columns:
            disp_av_src = disp_av_src[disp_av_src["avoidance_pvalue_adj"] < 0.05]

        if disp_av_src.empty:
            st.info("No domains pass the significance threshold at adj p < 0.05.")
        else:
            st.plotly_chart(
                plot_domain_avoidance(disp_av_src, top_n=top_avoid),
                use_container_width=True,
            )

            if "avoidance_pvalue_adj" in disp_av_src.columns:
                st.markdown(
                    '<div class="info-box">'
                    "<b>Statistical significance:</b> Colour intensity reflects "
                    "-log<sub>10</sub>(BH-adjusted p-value) from a one-sided binomial test. "
                    "Under H<sub>0</sub>, domain absence is modelled as a random draw given "
                    "each species' proteome size and the genus-wide domain prevalence. "
                    "Higher -log<sub>10</sub>(adj p) = stronger statistical evidence of avoidance."
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

            st.subheader("Avoidance detail table")
            disp_av = disp_av_src.copy()
            disp_av["species_with"]    = disp_av["species_with"].apply("; ".join)
            disp_av["species_without"] = disp_av["species_without"].apply("; ".join)

            rename_map = {
                "domain_name":          "Domain",
                "n_species_present":    "# species with",
                "n_species_absent":     "# species without",
                "avoidance_score":      "Avoidance score",
                "min_binom_pvalue":     "Min binomial p",
                "avoidance_pvalue_adj": "Adj p-value (BH-FDR)",
                "species_with":         "Species with domain",
                "species_without":      "Species without domain",
            }
            st.dataframe(
                disp_av.rename(columns=rename_map),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download avoidance table (CSV)",
                disp_av.to_csv(index=False),
                f"plasmodium_{taxon_id}_domain_avoidance.csv",
                "text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Domain Co-occurrence
# ══════════════════════════════════════════════════════════════════════════════
with tab_cooc:
    st.subheader("Domain co-occurrence analysis")
    st.markdown(
        '<div class="info-box">'
        "<b>Domain co-occurrence</b> measures the statistical tendency of two domains "
        "to appear together within the same protein. Three metrics are provided:<br>"
        "<b>Jaccard similarity</b> = n<sub>AB</sub> / (n<sub>A</sub> + n<sub>B</sub> - n<sub>AB</sub>) "
        "— proportion of proteins that share both domains out of those that carry either.<br>"
        "<b>Lift</b> = P(A ∩ B) / (P(A) × P(B)) — ratio of observed to expected co-occurrence "
        "under independence; lift &gt; 1 indicates positive association.<br>"
        "<b>PMI</b> = log<sub>2</sub>[ P(A,B) / (P(A)×P(B)) ] — pointwise mutual information; "
        "positive PMI = co-enriched beyond chance.<br>"
        "Fisher's exact test (one-sided) and BH-FDR correction assess statistical significance."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if cooc_stats_df.empty:
        st.info(
            "No co-occurrence pairs found. This occurs when proteins have fewer than "
            "2 annotated domains or too few proteins share domain combinations. "
            "Try fetching a broader taxon or unchecking 'reviewed only'."
        )
    else:
        cooc_c1, cooc_c2, cooc_c3 = st.columns(3)
        cooc_metric = cooc_c1.selectbox(
            "Metric",
            options=["lift", "jaccard", "pmi"],
            format_func=lambda x: {"lift": "Lift", "jaccard": "Jaccard similarity", "pmi": "PMI log2"}[x],
            key="cooc_metric",
        )
        cooc_top_n = cooc_c2.slider("Top N pairs", 5, min(50, len(cooc_stats_df)), 20, key="cooc_n")
        cooc_sig_only = cooc_c3.checkbox(
            "Significant only (adj p < 0.05)",
            value=False,
            key="cooc_sig",
            help="Fisher's exact test, BH-FDR corrected.",
        )

        st.subheader("Top co-occurring domain pairs")
        st.plotly_chart(
            plot_domain_cooccurrence_top_pairs(
                cooc_stats_df,
                metric=cooc_metric,
                top_n=cooc_top_n,
                sig_only=cooc_sig_only,
            ),
            use_container_width=True,
        )

        st.subheader("Co-occurrence heatmap")
        heatmap_metric = st.radio(
            "Heatmap colour metric",
            options=["jaccard", "lift", "pmi"],
            format_func=lambda x: {"jaccard": "Jaccard similarity", "lift": "Lift", "pmi": "PMI log2"}[x],
            horizontal=True,
            key="heatmap_metric",
        )
        heatmap_top_n = st.slider("Max domains in heatmap", 5, 40, 20, key="heatmap_n")
        st.plotly_chart(
            plot_domain_cooccurrence_heatmap(
                cooc_stats_df,
                metric=heatmap_metric,
                top_n=heatmap_top_n,
            ),
            use_container_width=True,
        )

        st.subheader("Co-occurrence statistics table")
        disp_cooc = cooc_stats_df.copy()
        if cooc_sig_only and "fisher_pvalue_adj" in disp_cooc.columns:
            disp_cooc = disp_cooc[disp_cooc["fisher_pvalue_adj"] < 0.05]

        st.dataframe(
            disp_cooc.rename(columns={
                "domain_A":          "Domain A",
                "domain_B":          "Domain B",
                "n_AB":              "Co-occurring proteins",
                "n_A":               "Proteins with A",
                "n_B":               "Proteins with B",
                "n_neither":         "Proteins with neither",
                "N":                 "Total (annotated)",
                "jaccard":           "Jaccard",
                "lift":              "Lift",
                "pmi":               "PMI log2",
                "fisher_pvalue":     "Fisher p-value",
                "fisher_pvalue_adj": "Adj p-value (BH-FDR)",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download co-occurrence table (CSV)",
            disp_cooc.to_csv(index=False),
            f"plasmodium_{taxon_id}_domain_cooccurrence.csv",
            "text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 (was 4) — InterPro Cross-references
# ══════════════════════════════════════════════════════════════════════════════
with tab_ipr:
    st.subheader("InterPro cross-reference occurrence")
    st.markdown(
        "InterPro entries are computationally predicted domain annotations "
        "integrated from SUPERFAMILY, Pfam, SMART, TIGRFAMs, and other "
        "member databases — broadening coverage beyond manually curated domains."
    )

    if ipr_occ_df.empty:
        st.info("No InterPro cross-references found for this taxon.")
    else:
        if not fetch_ipr_names:
            st.caption(
                "Enable 'Fetch InterPro domain names' in the sidebar to resolve "
                "IPR accessions to human-readable names."
            )

        top_ipr = st.slider("Top N entries", 5, min(50, len(ipr_occ_df)), 20, key="ipr_n")
        st.plotly_chart(
            plot_interpro_occurrence_bar(ipr_occ_df, ipr_name_map, top_n=top_ipr),
            use_container_width=True,
        )

        st.subheader("InterPro occurrence table")
        disp_ipr = ipr_occ_df.copy()
        disp_ipr["domain_name"] = disp_ipr["interpro_id"].map(
            lambda x: ipr_name_map.get(x, x)
        )
        disp_ipr["species"] = disp_ipr["species"].apply("; ".join)
        st.dataframe(
            disp_ipr[["interpro_id", "domain_name", "count", "species_count", "pct_proteins", "species"]].rename(
                columns={
                    "interpro_id":   "IPR accession",
                    "domain_name":   "Domain name",
                    "count":         "Proteins",
                    "species_count": "Species",
                    "pct_proteins":  "% of proteome",
                    "species":       "Species list",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download InterPro table (CSV)",
            disp_ipr.to_csv(index=False),
            f"plasmodium_{taxon_id}_interpro_occurrence.csv",
            "text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Protein Browser
# ══════════════════════════════════════════════════════════════════════════════
with tab_browse:
    st.subheader("Protein browser")

    # Filter controls
    f1, f2 = st.columns(2)
    sp_filter = f1.multiselect(
        "Filter by species",
        options=sorted(df["organism"].unique()),
        default=[],
        placeholder="All species",
    )
    search = f2.text_input("Search protein name", placeholder="e.g. circumsporozoite")
    dom_only = st.checkbox("Only proteins with annotated domains", value=False)

    filtered = df.copy()
    if sp_filter:
        filtered = filtered[filtered["organism"].isin(sp_filter)]
    if search:
        filtered = filtered[filtered["protein_name"].str.contains(search, case=False, na=False)]
    if dom_only:
        filtered = filtered[filtered["n_domains"] > 0]

    st.caption(f"Showing **{len(filtered)}** proteins")

    display_cols = ["accession", "protein_name", "organism", "length", "n_domains", "n_interpro"]
    tbl = filtered[display_cols].rename(columns={
        "accession":    "Accession",
        "protein_name": "Protein name",
        "organism":     "Species",
        "length":       "Length (aa)",
        "n_domains":    "# Domains",
        "n_interpro":   "# InterPro",
    })

    selection = st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # ── Detail panel for selected protein ────────────────────────────────────
    if selection and selection.selection.rows:
        row_idx = selection.selection.rows[0]
        prow = filtered.iloc[row_idx]

        st.markdown("---")
        st.markdown(f"### {prow['protein_name']} ({prow['accession']})")
        c_info1, c_info2 = st.columns(2)
        c_info1.markdown(f"**Species:** {prow['organism']}")
        c_info1.markdown(f"**Length:** {prow['length']} aa")
        c_info2.markdown(f"**Annotated domains:** {prow['n_domains']}")
        c_info2.markdown(f"**InterPro entries:** {prow['n_interpro']}")

        if prow["has_signal_peptide"]:
            st.info("Signal peptide detected — likely secreted or membrane-targeted.")
        if prow["has_gpi_anchor"]:
            st.info("GPI anchor signal detected — likely membrane-anchored surface protein.")

        if prow["domains"]:
            st.markdown("**Domain annotations:**")
            for d in prow["domains"]:
                st.markdown(f"- **{d['name']}** — positions {d['start']}–{d['end']} aa")
            st.plotly_chart(plot_protein_domain_map(prow), use_container_width=True)
        else:
            st.info("No manually annotated domain features for this protein.")

        if prow["interpro_ids"]:
            ipr_labels = [
                f"{x} ({ipr_name_map.get(x, 'resolve name in InterPro tab')})"
                for x in prow["interpro_ids"]
            ]
            st.markdown("**InterPro cross-references:** " + ", ".join(ipr_labels))

        if prow["go_functions"]:
            st.markdown("**GO molecular functions:**")
            for go in prow["go_functions"][:8]:
                st.markdown(f"- {go}")

        if prow["keywords"]:
            st.markdown("**Keywords:** " + "; ".join(prow["keywords"][:10]))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Raw Data & Export
# ══════════════════════════════════════════════════════════════════════════════
with tab_raw:
    st.subheader("Raw protein data")

    raw_cols = [
        "accession", "protein_name", "organism", "taxon_id",
        "length", "n_domains", "n_interpro",
        "has_signal_peptide", "has_gpi_anchor",
    ]
    st.dataframe(
        df[raw_cols].rename(columns={
            "accession":          "Accession",
            "protein_name":       "Protein name",
            "organism":           "Species",
            "taxon_id":           "Taxon ID",
            "length":             "Length (aa)",
            "n_domains":          "# Domains",
            "n_interpro":         "# InterPro",
            "has_signal_peptide": "Signal peptide",
            "has_gpi_anchor":     "GPI anchor",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download protein table (CSV)",
        df[raw_cols].to_csv(index=False),
        f"plasmodium_{taxon_id}_proteins.csv",
        "text/csv",
    )

    st.subheader("Download FASTA sequences")
    if st.button("Fetch FASTA (up to 500 sequences)"):
        with st.spinner("Fetching FASTA from UniProt..."):
            try:
                fasta = _load_fasta(taxon_id=taxon_id, reviewed=reviewed_only, max_fasta=500)
                st.download_button(
                    "Download FASTA",
                    fasta,
                    f"plasmodium_{taxon_id}.fasta",
                    "text/plain",
                )
                seq_count = fasta.count(">")
                st.success(f"Fetched {seq_count} sequences.")
            except RuntimeError as exc:
                st.error(f"FASTA fetch failed: {exc}")

    st.subheader("Dataset statistics")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"- Total proteins: **{len(df):,}**")
        st.markdown(f"- Species: **{df['organism'].nunique()}**")
        st.markdown(f"- With signal peptide: **{df['has_signal_peptide'].sum()}**")
        st.markdown(f"- With GPI anchor: **{df['has_gpi_anchor'].sum()}**")
    with s2:
        st.markdown(f"- With domain annotations: **{(df['n_domains'] > 0).sum()}**")
        st.markdown(f"- With InterPro entries: **{(df['n_interpro'] > 0).sum()}**")
        if len(df) > 0:
            max_idx = df["length"].idxmax()
            st.markdown(f"- Average length: **{df['length'].mean():.0f} aa**")
            st.markdown(
                f"- Longest protein: **{df.loc[max_idx, 'length']} aa** "
                f"({df.loc[max_idx, 'accession']})"
            )
