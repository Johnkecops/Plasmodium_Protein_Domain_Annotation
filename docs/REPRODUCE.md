# Reproducing every reported value

All numbers in the manuscript are computed from the frozen snapshot in `data/frozen/`.
No step below queries UniProt except the optional direct HMM scan (step 5), which
downloads sequences because the frozen table does not store them.

## 0. Environment

```bash
git clone https://github.com/Johnkecops/Plasmodium_Protein_Domain_Annotation.git
cd Plasmodium_Protein_Domain_Annotation
conda env create -f environments/environment.yml && conda activate plasmodium-domains
# or: python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
```

Tested on macOS 15 (arm64, 8 cores, 16 GB RAM). Steps 1 to 4 need under 5 min; step 5 about 15 min, mostly download.

## 1. Check the frozen input

```bash
shasum -a 256 data/frozen/plasmodium_reference_proteomes.tsv.gz data/frozen/proteome_index.tsv data/frozen/Pfam-A.clans.tsv.gz
```

| File | SHA-256 |
|---|---|
| plasmodium_reference_proteomes.tsv.gz | a471975ff8efe312630d88a8c6b4d855c40d803344887034bbf2e8d478468cf4 |
| plasmodium_swissprot_reviewed.tsv.gz | c46a9b440996b2a7fec9d5ce591b113a767b9a0453879dd61ce1b6e09efb7cc8 |
| proteome_index.tsv | f7720838d6ceb7420701f0a40811be33e6a99caedc3db66344f31a988556873b |
| Pfam-A.clans.tsv.gz | 86062b7ef1a0e0caee0c28cef479ac0d294c80789c51cbf75e513b368ce3a6f6 |

Snapshot: UniProt release 2026_02, retrieved 31 August 2026; InterPro 109.0; Pfam 38.2.

## 2. Main pipeline (Tables 1, Figures 1 to 8)

```bash
export MPLCONFIGDIR="$PWD/.mplcache"
python SCRIPT/run_genus_pipeline.py        # writes results/tables/*.csv and results/analysis_summary.json
python SCRIPT/make_figures.py              # writes results/figures/Figure1..8 (.png, .pdf)
```

Expected headline values (reference scope): 128,121 proteins, 20 species, 2,527 Pfam families,
76,215 annotated proteins, Pfam coverage 59.49%, InterPro 67.8%; 21 families with significant
avoidance (q < 0.05); pan-core 1,618 core / 250 soft core / 458 shell / 201 cloud; network
1,260 nodes, 1,513 edges, 337 components, largest component 91 nodes.

## 3. Robustness analyses (Tables 2 to 5)

```bash
python SCRIPT/robustness_analysis.py --replicates 200 --workers 8 --seed 20261019
```

Writes `results/robustness_summary.json` and `results/tables/reference_{member_db_coverage,clade_avoidance,clade_jackknife,core_composition,network_null,network_null_replicates}.csv`.
About 3 min on 8 cores.

## 4. Tests

```bash
python -m pytest -q
```

## 5. Direct HMM scan (Table 6)

```bash
python SCRIPT/hmm_validation.py --step all --threads 8
```

Downloads the 22 proteome FASTA files and 26 Pfam HMMs into `data/raw/` (git-ignored),
runs `hmmsearch` with Pfam gathering thresholds via pyhmmer, and writes
`results/tables/hmm_validation_*.csv` and `results/hmm_validation_summary.json`. Because
UniProt and Pfam move on, exact counts can differ slightly from the manuscript, which used
UniProt 2026_03 sequences (19 October 2026); the summary file records the releases used.

## Repository layout expected by the scripts

Scripts read and write `results/tables/`, `results/figures/` and `results/*.json`. These
directories must be committed together with the code for the manuscript version.
