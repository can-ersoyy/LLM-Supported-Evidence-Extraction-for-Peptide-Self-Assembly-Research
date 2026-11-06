import logging

logger = logging.getLogger(__name__)
from peptide_extractor.db import get_session_context, init_db
import csv
from typing import List
from typing import Optional
from pathlib import Path


GOLDEN_DATASET_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "golden_dataset_final.csv"
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
# golden_dataset_pmids: List[str] = get_golden_dataset_pmids()


def retrieve_golden_dataset(ncbi_provider, golden_dataset_pmids: List[str]):
    raw_records = ncbi_provider.search_by_ids(golden_dataset_pmids)
    golden_dataset_metadatas = ncbi_provider.parse_raw_search_records(raw_records)
    logger.info(f"Retrieved {golden_dataset_metadatas}")

    init_db()
    with get_session_context() as session:
        session.add_all(golden_dataset_metadatas)
        session.commit()
