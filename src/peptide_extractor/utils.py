from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

from sqlmodel import select
from sqlmodel import Session
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


def doi_to_filename(doi: Optional[str]) -> Optional[str]:
    """replaces / -> _ and . -> _ to make it a valid filename (for blob storage usage)"""
    clean_doi = normalize_doi(doi)  # first normalize just in case
    if not clean_doi:
        return None
    clean_doi = clean_doi.replace("/", "_").replace(".", "_")
    return clean_doi


def normalize_database_dois(session: Optional[Session] = None) -> int:
    """Normalizes all existing DOIs in the database to lowercase, prefix-free format."""
    from peptide_extractor.db.database import get_session_context
    from peptide_extractor.db.paper_metadata import PaperMetadata

    def _normalize(s: Session) -> int:
        papers = s.exec(
            select(PaperMetadata).where(PaperMetadata.doi.is_not(None))
        ).all()
        updated_count = 0
        for paper in papers:
            norm = normalize_doi(paper.doi)
            if norm != paper.doi:
                paper.doi = norm
                s.add(paper)
                updated_count += 1
        s.commit()
        return updated_count

    if session is not None:
        return _normalize(session)
    with get_session_context() as s:
        return _normalize(s)


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


def convert_pmc_to_pmid_batch(
    email: str, pmc_ids: list[str], api_key: Optional[str] = None
) -> dict[str, dict]:
    """Converts PMC IDs to PMIDs and DOIs using the NCBI ID Converter API."""
    if not pmc_ids:
        return {}

    formatted_ids = [
        f"PMC{i}" if not str(i).startswith("PMC") else str(i) for i in pmc_ids
    ]  # normally should already receive formatted but just in case

    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "ids": ",".join(formatted_ids),
        "format": "json",
        "tool": "evidence_extractor",
        "email": email,
        "api_key": api_key,
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    results = {pid: {"pmid": None, "doi": None, "pmcid": None} for pid in formatted_ids}

    for record in data.get("records", []):
        req_id = record.get("requested-id")
        if req_id in results:
            results[req_id] = {
                "pmid": str(record.get("pmid", "")),
                "doi": str(record.get("doi", "")),
                "pmcid": str(record.get("pmcid", "")),
            }

    return results


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
        return ""
    except (ET.ParseError, Exception):
        logger.info("There is an error with the format of this xml string.")
        return ""


def to_uppercase(text: Optional[str]) -> Optional[str]:
    """Convert text to uppercase, if text is empty string or 'NP' return None."""
    if not text:
        return None
    return text.strip().upper()


def to_lowercase(text: Optional[str]) -> Optional[str]:
    """Convert text to lowercase, if text is empty string or 'NP' return None."""
    if not text:
        return None
    return text.strip().lower()
