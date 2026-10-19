#!/usr/bin/env python3
"""
Module: hmm_validation
Purpose: Direct HMMER3 scan of the reference proteomes for the key families, compared with
         the precomputed Pfam assignments in the frozen snapshot.
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    --step     : download | scan | all (default all)
    --outdir   : results directory (default results/)
    --rawdir   : where proteome FASTA and Pfam HMMs are cached (default data/raw/)
    --threads  : HMMER worker threads (default os.cpu_count())
References:
    - Eddy SR, 2011. Accelerated profile HMM searches. PLoS Comput Biol 7, e1002195.
    - Larralde M, Zeller G, 2023. PyHMMER. Bioinformatics 39, btad214.
    - Mistry J et al., 2021. Pfam: the protein families database in 2021. NAR 49, D412-D419.

The precomputed assignments are what every other analysis uses. This script answers one
question: does an independent hmmsearch with Pfam gathering thresholds, run today on the
sequences of the same accessions, find the same carriers per species for the families the
avoidance result rests on? Families scanned: every family significant for avoidance, the
four genus-wide families LCCL, CLAG, EBA-175 and RAP, and TSP_1.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from domain_dataset import build_domain_table, load_snapshot, species_coverage_table
from genus_analysis import compute_domain_avoidance

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/stream"
PFAM_HMM = "https://www.ebi.ac.uk/interpro/api/entry/pfam/{acc}?annotation=hmm"
EXTRA_FAMILIES = ["PF03815", "PF03805", "PF11556", "PF08373", "PF00090"]
# One proteome per host-defined clade, highest BUSCO in its clade; highlighted as the
# representative subset the reviewer asked for. All 22 are scanned regardless.
REPRESENTATIVE = {
    "UP000001450": "Laverania (human)",
    "UP000027581": "Laverania (ape)",
    "UP000219813": "primate (human)",
    "UP000195521": "primate (simian)",
    "UP000031513": "primate (simian, zoonotic)",
    "UP000074855": "rodent",
    "UP000220797": "avian",
}


def _get(url: str, params: dict | None = None, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=600)
            if r.status_code == 200:
                return r
            logger.warning("HTTP %s for %s", r.status_code, url)
        except requests.RequestException as exc:
            logger.warning("%s on %s", exc, url)
        time.sleep(3 * 2**attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}")


def target_families(outdir: Path) -> list[str]:
    avoid = pd.read_csv(outdir / "tables" / "reference_avoidance.csv")
    sig = avoid.loc[avoid["significant_q05"], "domain_accession"].tolist()
    return sorted(set(sig) | set(EXTRA_FAMILIES))


def download(rawdir: Path, families: list[str]) -> dict:
    fasta_dir = rawdir / "proteome_fasta"
    hmm_dir = rawdir / "pfam_hmm"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    hmm_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(ROOT / "data" / "frozen" / "proteome_index.tsv", sep="\t")
    release = None
    for pid in index["proteome_id"]:
        out = fasta_dir / f"{pid}.fasta.gz"
        if out.exists() and out.stat().st_size > 0:
            continue
        r = _get(UNIPROT_FASTA, {"query": f"proteome:{pid}", "format": "fasta", "compressed": "true"})
        release = r.headers.get("x-uniprot-release", release)
        out.write_bytes(r.content)
        logger.info("fasta %s %d bytes", pid, len(r.content))
    for acc in families:
        out = hmm_dir / f"{acc}.hmm"
        if out.exists() and out.stat().st_size > 0:
            continue
        r = _get(PFAM_HMM.format(acc=acc))
        out.write_bytes(gzip.decompress(r.content))
    if release:
        (fasta_dir / "UNIPROT_RELEASE").write_text(release)
    return {"uniprot_release": release}


def scan(rawdir: Path, families: list[str], threads: int) -> pd.DataFrame:
    """hmmsearch with gathering cutoffs; one row per (family, accession) hit."""
    import pyhmmer

    alphabet = pyhmmer.easel.Alphabet.amino()
    hmms = []
    for acc in families:
        with pyhmmer.plan7.HMMFile(rawdir / "pfam_hmm" / f"{acc}.hmm") as fh:
            hmms.append(fh.read())
    rows = []
    for path in sorted((rawdir / "proteome_fasta").glob("*.fasta.gz")):
        # File object, not path: pyhmmer shells out to gzip for .gz paths, which breaks on spaces.
        with gzip.open(path, "rb") as gz, pyhmmer.easel.SequenceFile(gz, format="fasta", digital=True, alphabet=alphabet) as sf:
            seqs = sf.read_block()
        for hits in pyhmmer.hmmsearch(hmms, seqs, cpus=threads, bit_cutoffs="gathering"):
            fam = hits.query.accession.decode().split(".")[0] if isinstance(hits.query.accession, bytes) else str(hits.query.accession).split(".")[0]
            for hit in hits.included:
                name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
                rows.append({"domain_accession": fam, "accession": name.split("|")[1], "proteome_id": path.name.split(".")[0], "score": hit.score})
        logger.info("scanned %s (%d sequences)", path.name, len(seqs))
    return pd.DataFrame(rows)


def compare(hits: pd.DataFrame, families: list[str], rawdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per family and species: precomputed carriers vs direct-scan carriers, on shared accessions."""
    df = load_snapshot(ROOT / "data" / "frozen" / "plasmodium_reference_proteomes.tsv.gz")
    # Accessions present in today's FASTA; anything else was obsoleted or merged since the freeze.
    import pyhmmer

    current: set[str] = set()
    for path in (rawdir / "proteome_fasta").glob("*.fasta.gz"):
        with gzip.open(path, "rb") as gz, pyhmmer.easel.SequenceFile(gz, format="fasta", digital=False) as sf:
            for seq in sf:
                n = seq.name.decode() if isinstance(seq.name, bytes) else seq.name
                current.add(n.split("|")[1])
    shared = df[df["accession"].isin(current)]
    pre = {
        fam: set(shared.loc[shared["xref_pfam"].fillna("").str.contains(fam), "accession"])
        for fam in families
    }
    hits = hits[hits["accession"].isin(set(shared["accession"]))]
    species_of = dict(zip(shared["accession"], shared["species"]))
    rows = []
    for fam in families:
        direct = set(hits.loc[hits["domain_accession"] == fam, "accession"])
        for sp in sorted(shared["species"].unique()):
            a = {x for x in pre[fam] if species_of[x] == sp}
            b = {x for x in direct if species_of[x] == sp}
            rows.append({"domain_accession": fam, "species": sp, "n_precomputed": len(a), "n_direct": len(b), "n_both": len(a & b)})
    per = pd.DataFrame(rows)
    per["presence_agrees"] = (per["n_precomputed"] > 0) == (per["n_direct"] > 0)
    fam = per.groupby("domain_accession")[["n_precomputed", "n_direct", "n_both"]].sum()
    fam["sensitivity"] = (fam["n_both"] / fam["n_precomputed"]).round(4)
    fam["precision"] = (fam["n_both"] / fam["n_direct"]).round(4)
    fam["jaccard"] = (fam["n_both"] / (fam["n_precomputed"] + fam["n_direct"] - fam["n_both"])).round(4)
    fam["species_present_precomputed"] = per[per.n_precomputed > 0].groupby("domain_accession").size()
    fam["species_present_direct"] = per[per.n_direct > 0].groupby("domain_accession").size()
    fam["species_presence_agreement"] = per.groupby("domain_accession")["presence_agrees"].mean().round(4)
    fam = fam.fillna(0).reset_index()
    meta = {
        "frozen_accessions": int(len(df)),
        "accessions_still_current": int(len(shared)),
        "species_presence_cells": int(len(per)),
        "species_presence_agreement": round(float(per["presence_agrees"].mean()), 4),
        "absences_precomputed": int((per.n_precomputed == 0).sum()),
        "absences_confirmed_by_scan": int(((per.n_precomputed == 0) & (per.n_direct == 0)).sum()),
    }
    return per, fam, meta


def avoidance_with_direct_carriers(hits: pd.DataFrame, outdir: Path) -> dict:
    """Repeat the species-level avoidance test with the scanned families' carriers taken from hmmsearch."""
    df = load_snapshot(ROOT / "data" / "frozen" / "plasmodium_reference_proteomes.tsv.gz")
    domains = build_domain_table(df, layer="pfam")
    coverage = species_coverage_table(df, domains)
    meta = df.set_index("accession")[["species", "clade"]]
    direct = hits[hits["accession"].isin(meta.index)].drop_duplicates(["domain_accession", "accession"]).join(meta, on="accession")
    scanned = set(hits["domain_accession"])
    alt = pd.concat([domains[~domains["domain_accession"].isin(scanned)], direct[["accession", "domain_accession", "species", "clade"]]])
    rerun = compute_domain_avoidance(alt, coverage)
    orig = pd.read_csv(outdir / "tables" / "reference_avoidance.csv")
    s0 = set(orig.loc[orig["significant_q05"], "domain_accession"])
    s1 = set(rerun.loc[rerun["significant_q05"], "domain_accession"])
    return {"significant_precomputed": len(s0), "significant_direct": len(s1), "shared": len(s0 & s1),
            "lost": sorted(s0 - s1), "gained": sorted(s1 - s0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["download", "scan", "all"], default="all")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    parser.add_argument("--rawdir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    families = target_families(args.outdir)
    logger.info("%d families: %s", len(families), ", ".join(families))
    if args.step in ("download", "all"):
        info = download(args.rawdir, families)
        logger.info("download done %s", info)
    if args.step in ("scan", "all"):
        import json
        import pyhmmer

        hits = scan(args.rawdir, families, args.threads)
        hits.to_csv(args.outdir / "tables" / "hmm_direct_hits.csv", index=False)
        per, fam, meta = compare(hits, families, args.rawdir)
        per.to_csv(args.outdir / "tables" / "hmm_validation_per_species.csv", index=False)
        fam.to_csv(args.outdir / "tables" / "hmm_validation_per_family.csv", index=False)
        meta["avoidance_rerun_with_direct_carriers"] = avoidance_with_direct_carriers(hits, args.outdir)
        meta["pyhmmer_version"] = pyhmmer.__version__
        rel = args.rawdir / "proteome_fasta" / "UNIPROT_RELEASE"
        meta["uniprot_release_fasta"] = rel.read_text().strip() if rel.exists() else None
        meta["pfam_hmm_versions"] = {
            line.split()[1].split(".")[0]: line.split()[1]
            for f in sorted((args.rawdir / "pfam_hmm").glob("*.hmm"))
            for line in f.read_text().splitlines()[:5] if line.startswith("ACC")
        }
        meta["representative_proteomes"] = REPRESENTATIVE
        with (args.outdir / "hmm_validation_summary.json").open("w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("%s", meta)


if __name__ == "__main__":
    main()
