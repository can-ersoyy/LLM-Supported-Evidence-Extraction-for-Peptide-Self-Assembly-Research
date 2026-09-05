import csv
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

from peptide_extractor.db import get_session_context, init_db
from peptide_extractor.providers import NCBIProvider
import os

GOLDEN_DATASET_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "literature"
    / "golden_dataset_initial.csv"
)


def get_golden_dataset_pmids(csv_path: Optional[Path | str] = None) -> List[str]:
    """Parse golden_dataset_initial.csv and return unique PubMed IDs."""
    path = Path(csv_path) if csv_path else GOLDEN_DATASET_CSV_PATH
    pmids: List[str] = []

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("pmid", "").strip()
            if pmid and pmid not in pmids:
                pmids.append(pmid)

    return pmids


# Pre-parsed list of golden dataset PubMed IDs
golden_dataset_pmids: List[str] = get_golden_dataset_pmids()

def retrieve_golden_dataset(ncbi_provider):
    raw_records = ncbi_provider.search_by_ids(golden_dataset_pmids)
    golden_dataset_metadatas = ncbi_provider.parse_raw_search_records(raw_records)

    init_db()
    with get_session_context() as session:
        session.add_all(golden_dataset_metadatas)
        session.commit()


def retrive_ncbi_by_query(ncbi_provider):
    query = """((tetrapeptide[tiab] OR pentapeptide[tiab] OR hexapeptide[tiab] OR heptapeptide[tiab] OR octapeptide[tiab] OR nonapeptide[tiab] OR decapeptide[tiab] OR undecapeptide[tiab] OR dodecapeptide[tiab] OR tridecapeptide[tiab] OR tetradecapeptide[tiab] OR pentadecapeptide[tiab] OR hexadecapeptide[tiab] OR heptadecapeptide[tiab] OR octadecapeptide[tiab] OR nonadecapeptide[tiab] OR eicosapeptide[tiab] OR icosapeptide[tiab]) AND (nanostructure[tiab] OR nanotube[tiab] OR nanofiber[tiab] OR nanofibre[tiab] OR nanorod[tiab] OR hydrogel[tiab] OR nanosphere[tiab] OR nanoparticle[tiab] OR "self-assembly"[tiab])) AND (english[la]) AND ("free full text"[filter]) NOT (review[pt])"""
    raw_records = ncbi_provider.search_by_query(query)
    ncbi_metadatas = ncbi_provider.parse_raw_search_records(raw_records)

    init_db()
    with get_session_context() as session:
        session.add_all(ncbi_metadatas)
        session.commit()


if __name__ == "__main__":
    find_dotenv("..")
    load_dotenv()

    api_key = os.getenv("NCBI_API_KEY")

    ncbi_provider = NCBIProvider(api_key, batch_size=150)

    #retrieve_golden_dataset(ncbi_provider)
    #retrieve_ncbi_by_query(ncbi_provider)
    
