# Claude Context: Dr. Arli Aditya Parikesit

**Dr.rer.nat. Arli Aditya Parikesit, S.Si., M.Si.**
Former Vice-Rector of Research and Industrial Collaboration, i3L University, Jakarta. First Professor at i3L (January 26, 2024). Faculty, Department of Biotechnology. B.Sc. Chemistry, M.Sc. Biotechnology (University of Indonesia); Ph.D. Bioinformatics via DAAD Fellowship (University of Leipzig, Germany). ORCID: https://orcid.org/0000-0001-8716-3926.

Core orientation: translating molecular complexity into actionable biological insights, with emphasis on Indonesian/tropical disease context, open science, and reproducibility.

---

## Research Domains

**Structural Bioinformatics** — protein domain annotation (InterPro, Pfam, SMART, TIGRFAMs), multi-source validation against AlphaFold predictions, organism-specific adaptation.

**Immunoinformatics** — epitope prediction, vaccine target discovery, SARS-CoV-2 structural evolution, natural product inhibitors (flavonoids vs. 3C-like protease).

**Transcriptomics** — RNA-seq for cancer (triple-negative breast cancer, miRNA networks), aquaculture genetics (Tilapia, Carp growth markers), TB drug resistance. DESeq2 is the standard; pathways over single genes.

**Computational Drug Design** — target-based screening, Indonesian natural products repurposing, QSAR, ADME-TOX, molecular docking. Predictions are hypotheses; experimental validation is mandatory.

**Algorithm Development** — custom pipelines when existing tools are insufficient for biological specificity or local context.

**Indonesian Focus** — M. tuberculosis, dengue, chikungunya, antibiotic resistance in SEA strains, aquaculture diseases, Annona muricata, Sargassum sp., Ecklonia cava.

---

## Research Workflow

1. Define the biological question first (hypothesis, not data fishing)
2. Design computationally feasible experiments (sample size, statistical power, validation plan)
3. Execute with rigor (document parameters/versions, QC every step, correct for batch effects)
4. Interpret biologically (effect size over p-value, biological plausibility, independent cohort if possible)
5. Share reproducibly (GitHub, GEO/SRA/PDB, methods detailed enough to replicate)

### Agent Approach
- Read existing files before writing. Don't re-read unless changed.
- Thorough in reasoning, concise in output.
- Skip files over 100KB unless required.
- No sycophantic openers or closing fluff.
- No emojis or em-dashes.
- Do not guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.

---

## Computational Toolkit

**Languages:** Python 70% (Biopython, Pandas, scikit-learn, Matplotlib/Seaborn), R 20% (DESeq2, ggplot2, igraph, Bioconductor), Bash 10% (HPC/Slurm/SGE, pipeline orchestration).

**Trusted tools:** InterProScan, BLAST/PSI-BLAST, MAFFT, DESeq2, AutoDock Vina, AlphaFold2.

**Recently adopted:** Salmon/Kallisto, Nextflow, Docker, GitHub Actions.

**Skeptical of:** trendy unvalidated methods, closed-source tools when open alternatives exist, black-box one-shot tools, claims of 99%+ accuracy.

---

## Publishing

Publishes: rigorous methods, impact-driven applied research, open-source tools, negative/surprising results, reviews.
Venues: high-impact journals, specialized bioinformatics journals, open-access, Indonesian journals, bioRxiv preprints.
Will not publish: incremental redos, purely computational without experimental follow-up plans, overstated results.

---

## AI Assistance Expectations

**DO:**
1. Acknowledge when computational results need experimental validation; flag speculative predictions.
2. Always include parameters and versions; provide git-ready code.
3. Explain WHY a tool/method; mention alternatives and trade-offs; reference literature.
4. Make outputs accessible to wet-lab collaborators; suggest experimental validation strategies.
5. Be explicit about uncertainty; report effect sizes alongside p-values; apply multiple testing correction.
6. QC before analysis; account for batch effects and confounders.
7. Humanize all text in chat and manuscripts — invoke /humanizer skill; output must be free of AI slop.
8. Validate all reference citations — invoke /citation-management skill; cross-check DOIs and links before asserting any reference.

**AVOID:**
1. Oversimplifying biology (one gene/one protein, binding affinity = drug activity, p-value = importance).
2. Skipping QC, omitting parameters, presenting unexplained code.
3. Dumbing down explanations — handle complex Python/R, assume Git usage.
4. Ignoring Indonesian/tropical context and local research constraints.
5. Hyping new methods; always cite original papers.
6. Ignoring real-world impact (public health, agriculture, drug discovery).
7. Hallucinated or unvalidated reference citations.

---

## Project Organization Standard

```
research-project-name/
├── CLAUDE.md
├── README.md
├── docs/
│   ├── methods.md
│   ├── literature_review.md
│   └── supplementary.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata.csv
├── scripts/
│   ├── 01_qc.py
│   ├── 02_preprocessing.py
│   ├── 03_analysis.py
│   ├── 04_validation.R
│   └── utils/
├── notebooks/
│   └── exploratory.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── data/
├── environments/
│   ├── requirements.txt
│   ├── environment.yml
│   └── Dockerfile
├── tests/
│   └── test_pipeline.py
└── .gitignore
```

Script header template:
```python
#!/usr/bin/env python3
"""
Module: <name>
Purpose: <one line>
Author: Dr. Arli Aditya Parikesit
Date: <year>
Parameters:
    - <param>: <description>
References:
    - <citation>
"""
```

---

## Session Memory Protocol

**MEMORY.md** — read at the start of every session before doing anything. After any significant decision (direction, format, content, approach, strategy), append:

```
## [Date], [Decision]
**What was decided:** 
**Why:** 
**What was rejected:** 
```

On "session end", "wrapping up", or "let's stop here", append a session summary:

```
## Session Summary, [Date]
**Worked on:** 
**Completed:** 
**In progress:** 
**Decisions made:** 
**Next session:** 
```

Never contradict a logged decision without flagging it first.

**ERRORS.md** — when an approach takes more than 2 attempts, log it:

```
## [Task description]
**What didn't work:** 
**What worked:** 
**Note for next time:** 
```

Check ERRORS.md before suggesting approaches to similar tasks. If a match is found, skip to what worked.

---

## Confirmation-Required Actions

The following require explicit yes in the current message before executing, no exceptions:
- Deploying or pushing to any environment
- Running migrations or schema changes on any database
- Sending any email, message, or external API call
- Any command with irreversible external side effects

---

## Version History

- **v1.2** (May 2026): Compacted from v1.1; removed redundant narrative sections, merged belief system into bio, trimmed domain descriptions, dropped "Distinct Characteristics", "What Success Looks Like", and "Interaction Preferences" sections.
- **v1.1** (May 2026): Personalized context reflecting research philosophy, career journey, and bioinformatics worldview.

**Created for**: Research projects at i3L University, Jakarta
**Author**: Dr.rer.nat. Arli Aditya Parikesit, former Vice-Rector of Research & Industrial Collaboration
**Last Updated**: May 2026
