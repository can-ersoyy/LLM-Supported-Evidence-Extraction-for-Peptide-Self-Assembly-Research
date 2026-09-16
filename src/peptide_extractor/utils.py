from datetime import datetime, timezone

import requests

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