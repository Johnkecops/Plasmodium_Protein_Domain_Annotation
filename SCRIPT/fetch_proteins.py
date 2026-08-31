#!/usr/bin/env python3
"""
Module: UniProt data fetcher for Plasmodium protein domain annotation.
Purpose: Retrieve reviewed Plasmodium protein sequences and domain features
         via UniProt REST API v2. No external tools or .env files required.
Author: Dr. Arli Aditya Parikesit, Dr. Arif Nur Muhammad Ansori, and Moch. Royhan Afnani.,M.Sc 
Date: 2026

References:
    Parikesit et al. (2018) JBI 14(2):185-190. doi:10.14203/jbi.v14i2.3737
    Widjaja et al. (2022) BIOEDUSCIENCE 6(2):198-210. doi:10.22236/J.BES/628770
"""

import io
import re
import time
import json
import logging
import requests
import pandas as pd
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
INTERPRO_ENTRY = "https://www.ebi.ac.uk/interpro/api/entry/interpro/{ipr_id}/?format=json"

PLASMODIUM_TAXON = "5820"

# Species-rank NCBI taxonomy identifiers, each verified against the UniProt taxonomy
# endpoint by tests/test_taxonomy_map.py rather than transcribed by hand.
#
# Three entries were wrong in the submitted version and are corrected here:
#   P. ovale     was 36329, which is the P. falciparum 3D7 isolate; the species is 36330
#   P. chabaudi  was 5826, the subspecies P. chabaudi adami; the species is 5825
#   P. yoelii    was 73239, the subspecies P. yoelii yoelii; the species is 5861
# The first two were reported by a reviewer; the third was found while verifying them.
SPECIES_TAXONS: Dict[str, str] = {
    "All Plasmodium spp. (genus)": "5820",
    "Plasmodium falciparum": "5833",
    "Plasmodium vivax": "5855",
    "Plasmodium malariae": "5858",
    "Plasmodium knowlesi": "5850",
    "Plasmodium berghei": "5821",
    "Plasmodium yoelii": "5861",
    "Plasmodium chabaudi": "5825",
    "Plasmodium ovale": "36330",
}

# UniProt TSV fields to request
_FIELDS = ",".join([
    "accession",
    "reviewed",
    "protein_name",
    "organism_name",
    "organism_id",
    "length",
    "ft_domain",
    "ft_signal",
    "ft_propep",
    "xref_interpro",
    "xref_pfam",
    "go_f",
    "keyword",
])

# Regex to parse UniProt ft_domain text entries
# Handles: "DOMAIN start..end; /note="name"; /evidence="...""
_DOMAIN_RE = re.compile(
    r'DOMAIN\s+(\d+)\.\.(\d+)(?:\s*;\s*/note="([^"]*)")?',
    re.IGNORECASE,
)

# Strip parenthetical/bracketed qualifiers from organism names (e.g. "(isolate 3D7)")
_ORGANISM_QUALIFIER_RE = re.compile(r'\s*[\(\[].*', re.DOTALL)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt


def _request_with_retry(
    url: str,
    params=None,
    timeout: int = 60,
) -> requests.Response:
    """GET with exponential backoff retry on transient failures."""
    delay = _RETRY_BASE_DELAY
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                logger.error(
                    "UniProt API request failed after %d attempts: %s", _MAX_RETRIES, exc
                )
                raise RuntimeError(
                    f"UniProt API request failed after {_MAX_RETRIES} attempts: {exc}"
                ) from exc
            logger.warning(
                "UniProt API request failed (attempt %d/%d): %s; retrying in %.0fs",
                attempt + 1, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable")  # pragma: no cover


def _normalise_organism_name(name) -> str:
    """Strip strain/isolate qualifiers; keep genus + species only."""
    if not isinstance(name, str):
        return str(name)
    clean = _ORGANISM_QUALIFIER_RE.sub("", name).strip()
    parts = clean.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else clean


def fetch_plasmodium_proteins(
    taxon_id: str = PLASMODIUM_TAXON,
    reviewed: bool = True,
    max_results: int = 5000,
) -> pd.DataFrame:
    """
    Fetch Plasmodium proteins from UniProt REST API (TSV format, paginated).

    Parameters
    ----------
    taxon_id : str
        NCBI taxonomy ID. 5820 = genus Plasmodium.
    reviewed : bool
        If True, restrict to Swiss-Prot (manually reviewed) entries.
    max_results : int
        Hard cap on total proteins fetched.

    Returns
    -------
    pd.DataFrame
        One row per protein with parsed domain annotations.
    """
    query = f"taxonomy_id:{taxon_id}"
    if reviewed:
        query += " AND reviewed:true"

    params = {
        "query": query,
        "format": "tsv",
        "fields": _FIELDS,
        "size": 200,
    }

    all_frames: List[pd.DataFrame] = []
    url: Optional[str] = UNIPROT_SEARCH
    total_fetched = 0
    page = 0

    logger.info("Fetching Plasmodium proteins: taxon=%s reviewed=%s max_results=%d",
                taxon_id, reviewed, max_results)

    while url and total_fetched < max_results:
        page += 1
        resp = _request_with_retry(url, params=params)
        text = resp.text.strip()
        if not text:
            logger.debug("Page %d: empty response body, stopping pagination.", page)
            break

        try:
            chunk = pd.read_csv(io.StringIO(text), sep="\t", low_memory=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse UniProt TSV: {exc}") from exc

        if chunk.empty:
            logger.debug("Page %d: no rows, stopping pagination.", page)
            break

        # Trim to stay within max_results
        remaining = max_results - total_fetched
        chunk = chunk.head(remaining)

        all_frames.append(chunk)
        total_fetched += len(chunk)
        logger.debug("Page %d: %d rows fetched (%d total so far).", page, len(chunk), total_fetched)

        # Follow pagination
        link_header = resp.headers.get("Link", "")
        url = _extract_next_url(link_header)
        params = None  # params only for first request; subsequent requests use full URL
        time.sleep(0.4)

    if not all_frames:
        logger.info("No proteins retrieved for taxon=%s.", taxon_id)
        return pd.DataFrame()

    raw = pd.concat(all_frames, ignore_index=True)
    logger.info("Fetched %d proteins across %d page(s).", total_fetched, page)
    return _normalise(raw)


def _extract_next_url(link_header: str) -> Optional[str]:
    """Pull next-page URL from HTTP Link header."""
    if not link_header or 'rel="next"' not in link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names and parse domain feature strings.
    Returns a clean DataFrame ready for analysis.
    """
    # Rename UniProt TSV columns to stable internal names
    col_map = {
        "Entry": "accession",
        "Reviewed": "reviewed",
        "Protein names": "protein_name",
        "Organism": "organism",
        "Organism ID": "taxon_id",
        "Length": "length",
        "Domain [FT]": "ft_domain_raw",
        "Signal peptide": "ft_signal_raw",
        "Propeptide": "ft_propep_raw",
        "InterPro": "interpro_raw",
        "Pfam": "pfam_raw",
        "Gene Ontology (molecular function)": "go_f_raw",
        "Keywords": "keywords_raw",
    }
    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    # Ensure required columns exist
    for col in ["accession", "protein_name", "organism", "length"]:
        if col not in raw.columns:
            raw[col] = ""

    raw["length"] = pd.to_numeric(raw.get("length", 0), errors="coerce").fillna(0).astype(int)
    raw["taxon_id"] = raw.get("taxon_id", pd.Series("", index=raw.index)).astype(str).str.strip()
    raw["organism"] = raw["organism"].apply(_normalise_organism_name)

    # Parse domain features
    raw["domains"] = raw.get("ft_domain_raw", pd.Series("", index=raw.index)).apply(parse_ft_domain)
    raw["domain_names"] = raw["domains"].apply(lambda ds: [d["name"] for d in ds])
    raw["n_domains"] = raw["domains"].apply(len)

    # Signal peptide presence
    raw["has_signal_peptide"] = (
        raw.get("ft_signal_raw", pd.Series("", index=raw.index))
        .fillna("")
        .str.strip()
        .ne("")
    )

    # GPI anchor (flagged by keyword or propeptide note)
    keywords_col = raw.get("keywords_raw", pd.Series("", index=raw.index)).fillna("")
    propep_col = raw.get("ft_propep_raw", pd.Series("", index=raw.index)).fillna("")
    raw["has_gpi_anchor"] = (
        keywords_col.str.contains("GPI-anchor", case=False, na=False)
        | propep_col.str.contains("GPI", case=False, na=False)
    )

    # InterPro cross-references
    raw["interpro_ids"] = (
        raw.get("interpro_raw", pd.Series("", index=raw.index))
        .apply(parse_semicolon_ids)
    )
    raw["n_interpro"] = raw["interpro_ids"].apply(len)

    # Pfam cross-references
    raw["pfam_ids"] = (
        raw.get("pfam_raw", pd.Series("", index=raw.index))
        .apply(parse_semicolon_ids)
    )

    # GO molecular functions
    raw["go_functions"] = (
        raw.get("go_f_raw", pd.Series("", index=raw.index))
        .apply(_parse_go_terms)
    )

    # Keywords
    raw["keywords"] = (
        raw.get("keywords_raw", pd.Series("", index=raw.index))
        .apply(lambda x: [k.strip() for k in str(x).split(";") if k.strip()] if isinstance(x, str) else [])
    )

    # Clean protein name (take first name before semicolon / bracket)
    raw["protein_name"] = (
        raw["protein_name"]
        .astype(str)
        .str.split(r"\s*\(", n=1)
        .str[0]
        .str.strip()
        .replace("nan", "Unknown protein")
    )

    keep = [
        "accession", "protein_name", "organism", "taxon_id", "length",
        "domains", "domain_names", "n_domains",
        "interpro_ids", "pfam_ids", "n_interpro",
        "has_signal_peptide", "has_gpi_anchor",
        "go_functions", "keywords",
    ]
    return raw[[c for c in keep if c in raw.columns]].reset_index(drop=True)


def parse_ft_domain(ft_str) -> List[Dict]:
    """
    Parse a UniProt 'Domain [FT]' TSV field into a list of domain dicts.

    Each domain dict has keys: name, start, end.
    Handles both old and new UniProt REST API TSV formatting.
    """
    if not ft_str or not isinstance(ft_str, str) or ft_str.strip().lower() in ("nan", ""):
        return []

    domains = []
    for m in _DOMAIN_RE.finditer(ft_str):
        start = int(m.group(1))
        end = int(m.group(2))
        name = m.group(3).strip() if m.group(3) else "Unknown domain"
        domains.append({"name": name, "start": start, "end": end})

    return domains


def parse_semicolon_ids(raw_str) -> List[str]:
    """Parse semicolon-separated accession IDs (e.g., InterPro, Pfam)."""
    if not raw_str or not isinstance(raw_str, str):
        return []
    return [x.strip() for x in raw_str.split(";") if x.strip()]


def _parse_go_terms(raw_str) -> List[str]:
    """Extract GO term labels from UniProt go_f field."""
    if not raw_str or not isinstance(raw_str, str):
        return []
    terms = []
    for part in raw_str.split(";"):
        part = part.strip()
        # Remove GO ID suffix like "[GO:0003700]"
        clean = re.sub(r"\s*\[GO:\d+\]", "", part).strip()
        if clean:
            terms.append(clean)
    return terms


def fetch_interpro_entry(ipr_id: str) -> Dict:
    """
    Fetch name and type for a single InterPro accession.
    Returns dict with keys: id, name, type.
    Falls back gracefully on network errors.
    """
    url = INTERPRO_ENTRY.format(ipr_id=ipr_id)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("metadata", {})
        name_obj = meta.get("name", {})
        name = name_obj if isinstance(name_obj, str) else name_obj.get("name", ipr_id)
        return {
            "id": ipr_id,
            "name": name or ipr_id,
            "type": meta.get("type", "unknown"),
        }
    except Exception:
        return {"id": ipr_id, "name": ipr_id, "type": "unknown"}


def build_interpro_name_map(ipr_ids: List[str], delay: float = 0.15) -> Dict[str, str]:
    """
    Build {IPR_accession: human_readable_name} mapping.
    Only fetches unique IDs; respects rate limits with delay.
    """
    unique_ids = list(dict.fromkeys(ipr_ids))
    mapping: Dict[str, str] = {}
    for ipr_id in unique_ids:
        if ipr_id.startswith("IPR"):
            info = fetch_interpro_entry(ipr_id)
            mapping[ipr_id] = info.get("name", ipr_id)
            time.sleep(delay)
    return mapping


def fetch_fasta_sequences(
    taxon_id: str = PLASMODIUM_TAXON,
    reviewed: bool = True,
    max_results: int = 500,
) -> str:
    """
    Fetch protein FASTA sequences for a Plasmodium taxon.
    Returns raw FASTA string.
    """
    query = f"taxonomy_id:{taxon_id}"
    if reviewed:
        query += " AND reviewed:true"

    params = {
        "query": query,
        "format": "fasta",
        "size": min(max_results, 500),
    }

    sequences: List[str] = []
    url: Optional[str] = UNIPROT_SEARCH
    fetched = 0

    while url and fetched < max_results:
        resp = _request_with_retry(url, params=params)
        text = resp.text.strip()
        if text:
            sequences.append(text)
            fetched += text.count(">")

        link_header = resp.headers.get("Link", "")
        url = _extract_next_url(link_header)
        params = None
        time.sleep(0.4)

    return "\n".join(sequences)
