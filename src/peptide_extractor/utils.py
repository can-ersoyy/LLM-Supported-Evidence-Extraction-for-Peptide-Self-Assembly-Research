import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import requests

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

def get_pmcid_from_pmid(api_key: str, pmid: str) -> str | None:
    """Find corresponding PMCID for a given PMID using ELink."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "linkname": "pubmed_pmc",  # Ensures NCBI only returns direct PMC article matches
        "id": pmid,
        "cmd": "neighbor",
        "retmode": "json",
        "tool": "evidence_extractor",
        "api_key": api_key,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    finally:
        time.sleep(0.1)

    try:
        linksets = data.get("linksets", [])
        if not linksets:
            return None

        for linksetdb in linksets[0].get("linksetdbs", []):
            if linksetdb.get("linkname") == "pubmed_pmc":
                links = linksetdb.get("links", [])
                if links:
                    pmc_numeric_id = str(links[0])
                    # Ensure standard "PMC" prefix if not already present
                    return (
                        pmc_numeric_id
                        if pmc_numeric_id.startswith("PMC")
                        else f"PMC{pmc_numeric_id}"
                    )
        return None
    except (KeyError, IndexError):
        return None


def get_body_from_jats_xml(xml_text: str) -> str:
    """Extract and normalize the text content of the <body> tag from JATS XML text."""
    if not xml_text:
        return ""

    try:
        root = ET.fromstring(
            xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text
        )
        body = root.find(".//body")
        if body is not None:
            return " ".join(" ".join(body.itertext()).split())
    except (ET.ParseError, Exception):
        print("There is an error with the format of this xml string.")
        return ""