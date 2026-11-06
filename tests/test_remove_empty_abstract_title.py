from peptide_extractor.db.paper_metadata import DatasetSubset
from peptide_extractor.db import PaperMetadata
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from peptide_extractor.pipeline import remove_empty_abstract_title_papers


@pytest.fixture
def test_session():
    """Provide an in-memory SQLite database session for testing."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_remove_empty_abstract_papers(test_session):
    p1 = PaperMetadata(
        title="Paper 1",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
        abstract="Abstract 1",
    )
    p2 = PaperMetadata(
        title="Paper 2",
        doi="10.1000/2",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
        abstract="",
    )
    p3 = PaperMetadata(
        title="Paper 3",
        doi="10.1000/3",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
        abstract="Abstract 3",
    )
    test_session.add_all([p1, p2, p3])
    test_session.commit()

    deleted = remove_empty_abstract_title_papers(session=test_session)

    assert deleted == 1
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 2
    assert p1 in remaining
    assert p3 in remaining
    assert p2 not in remaining


def test_remove_empty_title_papers(test_session):
    p1 = PaperMetadata(
        title="Paper 1",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
        abstract="Abstract 1",
    )
    p2 = PaperMetadata(
        title="",
        doi="10.1000/2",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
        abstract="Abstract 2",
    )
    p3 = PaperMetadata(
        title="Paper 3",
        doi="10.1000/3",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
        abstract="Abstract 3",
    )
    test_session.add_all([p1, p2, p3])
    test_session.commit()

    deleted = remove_empty_abstract_title_papers(session=test_session)

    assert deleted == 1
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 2
    assert p1 in remaining
    assert p3 in remaining
    assert p2 not in remaining
