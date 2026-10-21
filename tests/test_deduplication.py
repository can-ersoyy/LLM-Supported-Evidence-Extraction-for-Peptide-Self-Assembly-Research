import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from peptide_extractor.db.paper_metadata import (
    DatasetSubset,
    PaperMetadata,
)
from peptide_extractor.pipeline import deduplicate_by_doi


@pytest.fixture
def test_session():
    """Provide an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_deduplicate_no_duplicates(test_session):
    """Ensure papers with distinct DOIs are not deleted."""
    p1 = PaperMetadata(
        title="Paper 1",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p2 = PaperMetadata(
        title="Paper 2",
        doi="10.1000/2",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p1, p2])
    test_session.commit()

    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 0
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 2


def test_deduplicate_doi_duplicates(test_session):
    """Ensure papers with the same DOI are treated as duplicates."""
    p1 = PaperMetadata(
        title="Paper 1",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p2 = PaperMetadata(
        title="Paper 2",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p3 = PaperMetadata(
        title="Paper 3",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p4 = PaperMetadata(
        title="Paper 4",
        doi="10.1000/2",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p1, p2, p3, p4])
    test_session.commit()
    print(test_session.exec(select(PaperMetadata)).all())
    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 2
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 2
    assert p1 in remaining
    assert p4 in remaining
    assert p2 not in remaining
    assert p3 not in remaining

def test_deduplicate_doi_duplicates_normalize(test_session):
    """Same test with doi normalize"""
    p1 = PaperMetadata(
        title="Paper 1",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p2 = PaperMetadata(
        title="Paper 2",
        doi="10.1000/1    ",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p3 = PaperMetadata(
        title="Paper 3",
        doi="   https://doi.org/10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p4 = PaperMetadata(
        title="Paper 4",
        doi="10.1000/2",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p1, p2, p3, p4])
    test_session.commit()
    print(test_session.exec(select(PaperMetadata)).all())
    deleted = deduplicate_by_doi(session=test_session)
    
    assert deleted == 2
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 2
    assert p1 in remaining
    assert p4 in remaining
    assert p2 not in remaining
    assert p3 not in remaining

def test_empty_doi(test_session):
    """Removes empty by default"""
    p1 = PaperMetadata(
        title="Paper 1",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p2 = PaperMetadata(
        title="Paper 2",
        doi="",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p3 = PaperMetadata(
        title="Paper 3",
        doi="   https://doi.org/10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p4 = PaperMetadata(
        title="Paper 4",
        doi="10.1000/2",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p1, p2, p3, p4])
    test_session.commit()
    print(test_session.exec(select(PaperMetadata)).all())
    deleted = deduplicate_by_doi(session=test_session)
    
    assert deleted == 2
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 2
    assert p1 in remaining
    assert p4 in remaining
    assert p2 not in remaining
    assert p3 not in remaining

def test_empty_doi_2(test_session):
    """Same test with doi normalize"""
    p1 = PaperMetadata(
        title="Paper 1",
        doi="10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p2 = PaperMetadata(
        title="Paper 2",
        doi="",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p3 = PaperMetadata(
        title="Paper 3",
        doi="   https://doi.org/10.1000/1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p4 = PaperMetadata(
        title="Paper 4",
        doi="10.1000/2",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p1, p2, p3, p4])
    test_session.commit()
    print(test_session.exec(select(PaperMetadata)).all())
    deleted = deduplicate_by_doi(session=test_session, remove_empty=False)
    
    assert deleted == 1
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 3
    assert p1 in remaining
    assert p4 in remaining
    assert p2 in remaining
    assert p3 not in remaining