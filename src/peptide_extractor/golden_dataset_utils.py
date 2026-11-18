import csv
import logging
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional

from sqlmodel import Session, select

from peptide_extractor.db import (
    ExtractionMode,
    PaperMetadata,
    PeptideEntry,
    get_session_context,
    init_db,
)
from peptide_extractor.db.peptide_entry import (
    PeptideAssemblyType,
    PeptideConjugateMixture,
    PeptideLinearity,
)
from peptide_extractor.utils import to_lowercase, to_uppercase

logger = logging.getLogger(__name__)

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


def _clean_csv_str(value: Optional[str]) -> Optional[str]:
    """Return stripped string, or None when blank or 'NP'."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "NP":
        return None
    return cleaned


def _parse_csv_float(value: Optional[str], suffix: Optional[str] = None) -> Optional[float]:
    """Parse a numeric CSV cell, optionally stripping a unit suffix; return None on NP/mismatch."""
    cleaned = _clean_csv_str(value)
    if cleaned is None:
        return None
    if suffix and cleaned.lower().endswith(suffix.lower()):
        cleaned = cleaned[: -len(suffix)].strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_csv_enum(value: Optional[str], enum_cls: type[Enum]) -> Optional[str]:
    """Normalize a CSV value to lowercase and validate against a string Enum."""
    cleaned = _clean_csv_str(value)
    if cleaned is None:
        return None
    lowered = to_lowercase(cleaned)
    valid_values = {member.value for member in enum_cls}
    return lowered if lowered in valid_values else None


def _row_to_peptide_entry(
    row: dict[str, str],
    pmid_to_paper_id: dict[str, int],
    run_identifier: int,
) -> PeptideEntry:
    """Convert a single golden-dataset CSV row into a PeptideEntry."""
    sequence = to_uppercase(row["sequence"].strip())
    raw_length = _clean_csv_str(row.get("length"))
    try:
        length = int(raw_length) if raw_length is not None else len(sequence)
    except ValueError:
        length = len(sequence)

    pmid = row["pmid"].strip()
    return PeptideEntry(
        paper_id=pmid_to_paper_id.get(pmid),
        pmid=pmid,
        sequence=sequence,
        length=length,
        type=_parse_csv_enum(row.get("type"), PeptideAssemblyType),
        n_terminal=_clean_csv_str(row.get("n_terminal")),
        c_terminal=_clean_csv_str(row.get("c_terminal")),
        temperature=_parse_csv_float(row.get("temperature"), suffix="°C"),
        ph=_parse_csv_float(row.get("ph")),
        peptide_concentration=_parse_csv_float(
            row.get("peptide_concentration"), suffix="mg/ml"
        ),
        solution=_clean_csv_str(row.get("solution")),
        solvent=_clean_csv_str(row.get("solvent")),
        volume_ratio=_clean_csv_str(row.get("volume_ratio")),
        incubation_period=_parse_csv_float(row.get("incubation_period")),
        peptide_conjugate_mixture=_parse_csv_enum(
            row.get("peptide/conjugate/mixture"), PeptideConjugateMixture
        ),
        conjugate_partner=_clean_csv_str(row.get("conjugate_partner")),
        linear_cyclic=_parse_csv_enum(row.get("linear/cyclic"), PeptideLinearity),
        chirality=None,
        detection_method=_clean_csv_str(row.get("detection_method")),
        preparation_protocol=_clean_csv_str(row.get("preparation_protocol")),
        evidence_span=None,
        database=_clean_csv_str(row.get("database")),
        extraction_mode=ExtractionMode.MANUAL,
        model=None,
        run_identifier=run_identifier,
    )


def ingest_golden_dataset_entries(
    csv_path: Optional[Path | str] = None,
    databases: Optional[Iterable[str]] = ("peptideminer/sapdb",),
    run_identifier: int = 0,
    skip_existing: bool = True,
    session: Optional[Session] = None,
) -> List[PeptideEntry]:
    """Ingest golden-dataset CSV rows into ``peptide_entry`` as manual entries.

    Args:
        csv_path: Path to the golden dataset CSV. Defaults to ``GOLDEN_DATASET_CSV_PATH``.
        databases: Database names to include (case-insensitive). If None, includes all rows.
        run_identifier: Integer run identifier to assign (defaults to ``0``).
        skip_existing: If True, skips insertion when manual entries for ``run_identifier``
            already exist.
        session: Optional SQLModel session. Uses ``get_session_context()`` if omitted.

    Returns:
        List of newly inserted ``PeptideEntry`` instances.
    """
    path = Path(csv_path) if csv_path else GOLDEN_DATASET_CSV_PATH
    allowed_dbs = {db.strip().lower() for db in databases} if databases is not None else None

    with open(path, mode="r", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if allowed_dbs is None
            or (row.get("database") or "").strip().lower() in allowed_dbs
        ]

    def _ingest(s: Session) -> List[PeptideEntry]:
        if skip_existing:
            existing = s.exec(
                select(PeptideEntry).where(
                    PeptideEntry.run_identifier == run_identifier,
                    PeptideEntry.extraction_mode == ExtractionMode.MANUAL,
                )
            ).first()
            if existing is not None:
                logger.info(
                    "Manual entries for run_identifier=%s already exist; skipping.",
                    run_identifier,
                )
                return []

        pmids = {row["pmid"].strip() for row in rows if row.get("pmid")}
        papers = s.exec(
            select(PaperMetadata).where(PaperMetadata.pmid.in_(pmids))  # type: ignore[union-attr]
        ).all()
        pmid_to_paper_id = {p.pmid: p.id for p in papers if p.pmid and p.id is not None}

        entries = [
            _row_to_peptide_entry(row, pmid_to_paper_id, run_identifier) for row in rows
        ]
        s.add_all(entries)
        s.commit()
        return entries

    if session is not None:
        return _ingest(session)

    init_db()
    with get_session_context() as s:
        return _ingest(s)
