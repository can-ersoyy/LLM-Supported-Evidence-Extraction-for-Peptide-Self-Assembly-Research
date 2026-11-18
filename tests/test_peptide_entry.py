import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from peptide_extractor.db import (
    ExtractionMode,
    PaperMetadata,
    PeptideEntry,
    split_peptide_entries_by_detection_method,
)


@pytest.fixture
def session_with_peptide_entries():
    """Create an in-memory SQLite session seeded with PeptideEntry rows."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        paper = PaperMetadata(
            id=1, title="Test Paper", pmid="12345678", is_open_access=True
        )
        session.add(paper)

        e1 = PeptideEntry(
            paper_id=1,
            pmid="12345678",
            sequence="KLVFF",
            length=5,
            type="fiber",
            temperature=37.0,
            ph=7.4,
            detection_method="TEM, AFM, CD",
            extraction_mode=ExtractionMode.LLM,
            model="test-model",
            run_identifier=101,
        )
        e2 = PeptideEntry(
            paper_id=1,
            pmid="12345678",
            sequence="FF",
            length=2,
            type="hydrogel",
            detection_method="ThT",
            extraction_mode=ExtractionMode.LLM,
            model="test-model",
            run_identifier=101,
        )
        e3 = PeptideEntry(
            paper_id=1,
            pmid="12345678",
            sequence="VQIVYK",
            length=6,
            type="amyloid",
            detection_method=None,
            extraction_mode=ExtractionMode.LLM,
            model="test-model",
            run_identifier=101,
        )
        e_other_run = PeptideEntry(
            paper_id=1,
            pmid="12345678",
            sequence="AAAA",
            length=4,
            detection_method="TEM, Cryo-EM",
            extraction_mode=ExtractionMode.LLM,
            model="test-model",
            run_identifier=202,
        )
        session.add_all([e1, e2, e3, e_other_run])
        session.commit()
        yield session


def test_split_peptide_entries_default_does_not_save(session_with_peptide_entries):
    session = session_with_peptide_entries

    separated = split_peptide_entries_by_detection_method(
        run_identifier=101, session=session
    )

    # 3 from e1 ("TEM", "AFM", "CD") + 1 from e2 ("ThT") + 1 from e3 (None)
    assert len(separated) == 5
    assert [e.detection_method for e in separated] == ["TEM", "AFM", "CD", "ThT", None]

    # Verify fields were preserved on expanded entries
    klvff_entries = [e for e in separated if e.sequence == "KLVFF"]
    assert len(klvff_entries) == 3
    for entry in klvff_entries:
        assert entry.paper_id == 1
        assert entry.pmid == "12345678"
        assert entry.length == 5
        assert entry.type == "fiber"
        assert entry.temperature == 37.0
        assert entry.ph == 7.4
        assert entry.run_identifier == 101
        assert entry.model == "test-model"

    # Database should remain unchanged (save=False by default)
    db_rows = session.exec(
        select(PeptideEntry).where(PeptideEntry.run_identifier == 101)
    ).all()
    assert len(db_rows) == 3


def test_split_peptide_entries_with_save_persists_new_entries(
    session_with_peptide_entries,
):
    session = session_with_peptide_entries

    separated = split_peptide_entries_by_detection_method(
        run_identifier=101, save=True, session=session
    )

    assert len(separated) == 5
    assert [e.detection_method for e in separated] == ["TEM", "AFM", "CD", "ThT", None]
    assert all(e.id is not None for e in separated)

    # Original 3 rows + 3 newly saved separated rows from e1 = 6 rows for run 101
    db_rows = session.exec(
        select(PeptideEntry).where(PeptideEntry.run_identifier == 101)
    ).all()
    assert len(db_rows) == 6
    methods_in_db = sorted(r.detection_method or "" for r in db_rows)
    assert methods_in_db == ["", "AFM", "CD", "TEM", "TEM, AFM, CD", "ThT"]

    # Other run_identifier must be untouched
    other_rows = session.exec(
        select(PeptideEntry).where(PeptideEntry.run_identifier == 202)
    ).all()
    assert len(other_rows) == 1


def test_split_peptide_entries_no_matches(session_with_peptide_entries):
    separated = split_peptide_entries_by_detection_method(
        run_identifier=999, session=session_with_peptide_entries
    )
    assert separated == []
