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


if __name__ == "__main__":
    find_dotenv("..")
    load_dotenv()

    api_key = os.getenv("NCBI_API_KEY")

    ncbi_provider = NCBIProvider(api_key, batch_size=150)

    raw_records = ncbi_provider.search_by_ids(golden_dataset_pmids)
    golden_dataset_metadatas = ncbi_provider.parse_raw_search_records(raw_records)

    init_db()
    with get_session_context() as session:
        session.add_all(golden_dataset_metadatas)
        session.commit()
