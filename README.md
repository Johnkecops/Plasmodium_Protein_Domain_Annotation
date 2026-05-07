# Plasmodium Protein Domain Annotator

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
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE.txt                # MIT licence
└── SCRIPT/
    ├── __init__.py
    ├── fetch_proteins.py      # UniProt REST API client and TSV parser
    ├── domain_analysis.py     # Occurrence, avoidance, co-occurrence logic
    └── visualize.py           # Plotly chart generators
```

---

## Methodology

### Data retrieval

Reviewed (*Swiss-Prot*) protein entries are fetched from the UniProt REST API v2 for the selected *Plasmodium* taxon (default: genus taxon 5820). Pagination is handled automatically, with a configurable result cap.

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
