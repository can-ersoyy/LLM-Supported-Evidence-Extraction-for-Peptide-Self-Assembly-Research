from datetime import datetime, timezone

import requests

import re
from typing import Optional

def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Clean and normalize a DOI string according to ISO 26324 standards.
    
    - Strips whitespace
    - Converts to lowercase
    - Removes 'https://doi.org/', 'http://dx.doi.org/', and 'doi:' prefixes
    """
    if not doi:
        return None
    cleaned = doi.strip().lower()
    cleaned = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", cleaned).strip()
    return cleaned if cleaned else None


def get_utc_now():
    return datetime.now(timezone.utc)

def get_pmcid_from_pmid(api_key, pmid: str) -> str | None:
    """Find corresponding PMCID for a given PMID using ELink."""

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "cmd": "neighbor",
        "retmode": "json",
        "tool": "evidence_extractor",
        "api_key": api_key
    }

    res = requests.get(url, params=params).json()
    try:
        linkset = res["linksets"][0]
        links = linkset["linksetdbs"][0]["links"]
        return links[0]
    except (KeyError, IndexError):
        return None