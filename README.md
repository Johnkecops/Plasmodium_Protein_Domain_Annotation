# Plasmodium Protein Domain Annotator (v1.1)

A Streamlit application for HMM-based protein domain annotation across all reviewed *Plasmodium* species available in UniProt. The tool retrieves Swiss-Prot-curated protein entries, parses domain feature annotations, and provides interactive visualisations of domain occurrence and avoidance patterns at both the whole-genus and per-species level.

---

## Background

Malaria remains a significant global health burden, caused primarily by *Plasmodium falciparum* and other *Plasmodium* species that infect humans and non-human primates. Understanding the protein domain architecture of these parasites is foundational to vaccine design, drug target identification, and functional genomics.

This application implements, automates, and extends the domain annotation workflow described in Parikesit et al. (2018) and Widjaja et al. (2022), replacing manual web-server queries with a reproducible, species-agnostic pipeline.

---

## Features

| Feature | Description |
|---|---|
| Multi-species support | All reviewed *Plasmodium* species in UniProt, selectable by taxon |
| Domain occurrence | Frequency tables and bar charts for curated domain annotations |
| Domain avoidance | Domains present in some species but systematically absent in others |
| InterPro cross-refs | Computationally predicted domain entries from SUPERFAMILY, Pfam, SMART |
| Protein browser | Search and filter proteins; click to view a per-protein domain map |
| FASTA export | Download protein sequences for downstream alignment or phylogenetics |
| CSV downloads | All tables available for download |

---

## Installation

```bash
# Clone or copy this project directory
cd "Plasmodium Protein Domain Annotation"

# Create a clean environment (recommended)
conda create -n plasmodium-domains python=3.11
conda activate plasmodium-domains

# Install dependencies
pip install -r requirements.txt
```

No API keys or `.env` files are required. The application queries public REST APIs (UniProt, InterPro).

---

## Running the Application

```bash
streamlit run app.py
```

Open the URL printed to the terminal (typically `http://localhost:8501`).

---

## Project Structure

```
Plasmodium Protein Domain Annotation/
├── app.py                     # Main Streamlit application (loads cached data via file I/O)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE.txt                # MIT licence
└── SCRIPT/
    ├── __init__.py
    ├── fetch_proteins.py      # UniProt REST API client and TSV parser
    ├── domain_analysis.py     # Occurrence, avoidance, co-occurrence logic
    └── visualize.py           # Plotly chart generators
```

**Integration pattern:** `app.py` and `SCRIPT/` are decoupled by design. `SCRIPT/` functions generate analysis outputs; `app.py` loads them via `st.cache_data`-wrapped file I/O (JSON/FASTA). There are no import-level dependencies between `app.py` and `SCRIPT/`. The schema contract between what `SCRIPT/` writes and what `app.py` reads is enforced by column names documented in each function's docstring.

---

## Methodology

### Data retrieval

Reviewed (*Swiss-Prot*) protein entries are fetched from the UniProt REST API v2 for the selected *Plasmodium* taxon (default: genus taxon 5820). Pagination is handled automatically with exponential backoff retry (3 attempts), with a configurable result cap.

**Coverage caveat:** `reviewed:true` restricts retrieval to Swiss-Prot manually curated entries. For *Plasmodium* species other than *P. falciparum* and *P. vivax*, Swiss-Prot coverage is sparse. Species-level comparisons will be biased toward well-annotated species. Set `reviewed=False` to include TrEMBL entries for broader coverage (at reduced annotation quality).

**Organism name normalization:** Strain and isolate qualifiers (e.g. "isolate 3D7", "Salvador I") are stripped from organism names during data ingestion. All entries are grouped at genus + species level to prevent phantom species from strain name variants in UniProt records.

### Domain annotation sources

1. **Manually curated domains** — UniProt Feature type `Domain` (`ft_domain` field). These annotations are derived from the SUPERFAMILY HMM library, Pfam, PROSITE, and other member databases integrated into UniProt during curation. This mirrors the SUPERFAMILY HMM approach used in Parikesit et al. (2018).

2. **InterPro cross-references** — `xref_interpro` field. These record computationally predicted domain entries from SUPERFAMILY, Pfam, SMART, TIGRFAMs, HAMAP, and CDD — consistent with the multi-database approach of Widjaja et al. (2022).

### Domain occurrence

For each domain type, the application counts:
- Number of distinct proteins containing the domain
- Number of distinct species containing the domain
- Percentage of total proteins in the dataset

### Domain avoidance

Domain avoidance identifies domains that are present in a subset of *Plasmodium* species but systematically absent in others. The avoidance score for domain *d* in a dataset of *N* species is:

```
avoidance_score(d) = (number of species lacking d) / N
```

Only domains present in at least two species are considered, to avoid singleton noise. High avoidance scores indicate lineage-specific domain loss or gain, which may reflect evolutionary adaptation to host range or parasite biology.

---

## Improvements Over Benchmark

This application was benchmarked against [github.com/Johnkecops/protein-domain](https://github.com/Johnkecops/protein-domain). Key improvements:

| Pitfall in benchmark | Fix in this application |
|---|---|
| Hardcoded *S. cerevisiae* taxon | Full *Plasmodium* genus support; any taxon ID accepted |
| API cap at 300 proteins | Automatic pagination; configurable up to 5,000 |
| No error recovery on API failure | `try/except` with user-facing `st.error` messages |
| XMGrace dependency for plots | Pure Plotly — no external tools needed |
| No domain avoidance analysis | Full avoidance score computation and table |
| No per-protein visualisation | Interactive linear domain map per selected protein |
| Matrix memory issues at scale | Columns capped to top-N domains; `fill_value=0` sparse matrix |

---

## Version Comparison

| Component                      | v1.0                                                                                                      | v1.1                                                                                          |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **API resilience**             | Single-shot `requests.get`; one network error aborts entire fetch                                         | Exponential backoff retry (3 attempts: 2s, 4s, 8s) on all API calls                           |
| **Organism normalization**     | Raw UniProt `organism` string used; strain/isolate variants create phantom species                        | Qualifiers stripped (e.g. "isolate 3D7"); all entries grouped at genus + species level        |
| **BH-FDR correction**          | Hand-rolled `numpy` implementation in both avoidance and co-occurrence modules; untested on tied p-values | `statsmodels.multipletests(method="fdr_bh")` — reference implementation, correct monotonicity |
| **Species input guard**        | Silent miscalibration if single-species `df` passed to `compute_domain_avoidance`                         | `UserWarning` emitted when `df` has < 2 species                                               |
| **Matrix build performance**   | `iterrows()` loop in `compute_species_domain_matrix`                                                      | `DataFrame.explode()` + `groupby` (vectorized)                                                |
| **Cooccurrence heatmap build** | `iterrows()` symmetric fill loop                                                                          | `pivot_table` + concat of both directions (vectorized)                                        |
| **Cooccurrence heatmap NaN**   | Zero co-occurrence pairs rendered as blank/grey patches (ambiguous)                                       | Filled with `0.0` explicitly; visually unambiguous                                            |
| **Metric validation**          | Unknown `metric` string raises `KeyError` inside loop with no useful message                              | `ValueError` raised before loop with clear message                                            |
| **`sig_only` filter**          | Silently no-ops if `fisher_pvalue_adj` column absent                                                      | `UserWarning` emitted when filter is skipped                                                  |
| **Heatmap height**             | `n_species * 55` unbounded; cramped at 20+ species                                                        | Capped at 1200px                                                                              |
| **Rare-pair filtering**        | No minimum co-occurrence guard; lift inflates for low-count pairs                                         | `min_n_AB` parameter added to `plot_domain_cooccurrence_top_pairs`                            |
| **Coverage documentation**     | `reviewed:true` default undocumented; sparse coverage for non-falciparum species implicit                 | Caveat documented in Methodology; `reviewed=False` option described                           |
| **Integration pattern**        | `app.py` / `SCRIPT/` decoupling undocumented; schema contract implicit                                    | File I/O integration pattern documented; schema contract referenced to docstrings             |
| **Dependencies**               | `scipy` missing from `requirements.txt` despite active import                                             | `scipy>=1.10.0` and `statsmodels>=0.14.0` added                                               |

---

## References

Parikesit, A. A., Utomo, D. H., & Karimah, N. (2018). Protein Domain Annotation of *Plasmodium* spp. Circumsporozoite Protein (CSP) Using Hidden Markov Model-based Tools. *Jurnal Biologi Indonesia*, *14*(2), 185–190. https://doi.org/10.14203/jbi.v14i2.3737

Widjaja, V., Lim, A., Aini, B., Gandasasmita, G. A., Darmawan, J. T., & Parikesit, A. A. (2022). Identification of Uncharacterized *Plasmodium falciparum* Proteins via In-silico Analysis. *BIOEDUSCIENCE*, *6*(2). https://doi.org/10.22236/J.BES/628770

---

## Licence

MIT — see `LICENSE.txt`.

---

## Contact

Dr. Arli Aditya Parikesit  
Department of Bioinformatics, i3L University, Jakarta  
arli.parikesit@i3l.ac.id  
ORCID: 0000-0001-8716-3926
