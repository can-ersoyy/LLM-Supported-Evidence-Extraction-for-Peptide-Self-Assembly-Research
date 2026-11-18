import csv
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from peptide_extractor.db import ExtractionMode, PaperMetadata, PeptideEntry
from peptide_extractor.golden_dataset_utils import ingest_golden_dataset_entries


@pytest.fixture
def memory_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            PaperMetadata(
                id=10,
                title="Sample Paper",
                pmid="18070345",
                is_open_access=True,
            )
        )
        session.commit()
        yield session


def _write_sample_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "golden.csv"
    fieldnames = [
        "sequence",
        "length",
        "type",
        "n_terminal",
        "c_terminal",
        "temperature",
        "ph",
        "peptide_concentration",
        "solution",
        "solvent",
        "volume_ratio",
        "incubation_period",
        "peptide/conjugate/mixture",
        "conjugate_partner",
        "linear/cyclic",
        "detection_method",
        "preparation_protocol",
        "pmid",
        "database",
    ]
    rows = [
        {
            "sequence": "ff",
            "length": "2",
            "type": "Hydrogel",
            "n_terminal": "Fmoc",
            "c_terminal": "Free",
            "temperature": "37 °C",
            "ph": "7",
            "peptide_concentration": "5.025 mg/ml",
            "solution": "DMSO/PBS",
            "solvent": "NP",
            "volume_ratio": "5/95",
            "incubation_period": "NP",
            "peptide/conjugate/mixture": "Peptide",
            "conjugate_partner": "NP",
            "linear/cyclic": "Linear",
            "detection_method": "Confocal Microscopy",
            "preparation_protocol": "Dissolved in DMSO.",
            "pmid": "18070345",
            "database": "peptideminer/sapdb",
        },
        {
            "sequence": "KLVFF",
            "length": "5",
            "type": "unknown_morphology",
            "n_terminal": "NP",
            "c_terminal": "Free",
            "temperature": "Room temperature",
            "ph": "NP",
            "peptide_concentration": "10 uM",
            "solution": "H2O",
            "solvent": "Aqueous",
            "volume_ratio": "1",
            "incubation_period": "25",
            "peptide/conjugate/mixture": "Conjugate",
            "conjugate_partner": "PEG",
            "linear/cyclic": "Cyclic",
            "detection_method": "TEM",
            "preparation_protocol": "Incubated.",
            "pmid": "99999999",
            "database": "peptideminer/sapdb",
        },
        {
            "sequence": "VQIVYK",
            "length": "6",
            "type": "amyloid",
            "n_terminal": "Free",
            "c_terminal": "Free",
            "temperature": "25 °C",
            "ph": "7.4",
            "peptide_concentration": "1 mg/ml",
            "solution": "PBS",
            "solvent": "PBS",
            "volume_ratio": "1",
            "incubation_period": "60",
            "peptide/conjugate/mixture": "Peptide",
            "conjugate_partner": "NP",
            "linear/cyclic": "Linear",
            "detection_method": "ThT",
            "preparation_protocol": "Shaken.",
            "pmid": "18070345",
            "database": "cross-beta",
        },
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_ingest_golden_dataset_entries_filters_and_maps_fields(
    memory_session: Session, tmp_path: Path
):
    csv_path = _write_sample_csv(tmp_path)

    inserted = ingest_golden_dataset_entries(
        csv_path=csv_path,
        databases=("peptideminer/sapdb",),
        run_identifier=0,
        session=memory_session,
    )

    assert len(inserted) == 2

    rows = memory_session.exec(
        select(PeptideEntry).order_by(PeptideEntry.id)  # type: ignore[arg-type]
    ).all()
    assert len(rows) == 2

    first, second = rows
    assert first.paper_id == 10
    assert first.pmid == "18070345"
    assert first.sequence == "FF"
    assert first.length == 2
    assert first.type == "hydrogel"
    assert first.temperature == 37.0
    assert first.ph == 7.0
    assert first.peptide_concentration == 5.025
    assert first.solvent is None
    assert first.incubation_period is None
    assert first.conjugate_partner is None
    assert first.peptide_conjugate_mixture == "peptide"
    assert first.linear_cyclic == "linear"
    assert first.chirality is None
    assert first.evidence_span is None
    assert first.database == "peptideminer/sapdb"
    assert first.extraction_mode == ExtractionMode.MANUAL
    assert first.run_identifier == 0

    # Non-matching fields (unknown enum, non-numeric temp/conc) and NP become None
    assert second.paper_id is None
    assert second.type is None
    assert second.n_terminal is None
    assert second.temperature is None
    assert second.ph is None
    assert second.peptide_concentration is None
    assert second.incubation_period == 25.0
    assert second.peptide_conjugate_mixture == "conjugate"
    assert second.conjugate_partner == "PEG"
    assert second.linear_cyclic == "cyclic"

    # Re-running with skip_existing=True should be a no-op
    second_run = ingest_golden_dataset_entries(
        csv_path=csv_path,
        databases=("peptideminer/sapdb",),
        run_identifier=0,
        session=memory_session,
    )
    assert second_run == []

