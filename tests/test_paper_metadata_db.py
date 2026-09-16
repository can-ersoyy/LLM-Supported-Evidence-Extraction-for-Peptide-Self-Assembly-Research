import pytest
from sqlmodel import SQLModel, create_engine, Session

from peptide_extractor.db.paper_metadata import (
    DatasetSubset,
    PaperMetadata,
    PaperStatus,
    get_paper_by_id,
    get_paper_metadatas,
    update_paper_status,
)


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def test_session(in_memory_engine):
    """Provide a clean session populated with test PaperMetadata objects."""
    with Session(in_memory_engine) as session:
        paper1 = PaperMetadata(
            title="Paper 1: Tetrapeptide Assembly",
            abstract="Abstract 1 detailing tetrapeptide self-assembly into fibers.",
            doi="10.1000/1",
            pmid="10001",
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
        )
        paper2 = PaperMetadata(
            title="Paper 2: Hexapeptide Hydrogel",
            abstract="Abstract 2 detailing hexapeptide hydrogel formation.",
            doi="10.1000/2",
            pmid="10002",
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.GOLDEN_BENCHMARK,
        )
        paper3 = PaperMetadata(
            title="Paper 3: Discarded Computational Review",
            abstract="Abstract 3 reviewing computational simulations without experimental data.",
            doi="10.1000/3",
            pmid="10003",
            status=PaperStatus.DISCARD,
            is_open_access=False,
            dataset_subset=DatasetSubset.SCRAPED,
        )
        paper4 = PaperMetadata(
            title="Paper 4: Usable Fibril Study",
            abstract="Abstract 4 describing TEM and ThT assays.",
            doi=None,
            pmid="10004",
            status=PaperStatus.USABLE,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
        )
        session.add_all([paper1, paper2, paper3, paper4])
        session.commit()
        for p in [paper1, paper2, paper3, paper4]:
            session.refresh(p)
        yield session


def test_get_paper_by_id(test_session):
    paper = get_paper_by_id(1, session=test_session)
    assert paper is not None
    assert paper.title == "Paper 1: Tetrapeptide Assembly"
    assert paper.pmid == "10001"

    non_existent = get_paper_by_id(999, session=test_session)
    assert non_existent is None


def test_get_paper_metadatas_by_status(test_session):
    not_processed = get_paper_metadatas(status=PaperStatus.NOT_PROCESSED, session=test_session)
    assert len(not_processed) == 2
    titles = [p.title for p in not_processed]
    assert "Paper 1: Tetrapeptide Assembly" in titles
    assert "Paper 2: Hexapeptide Hydrogel" in titles

    discarded = get_paper_metadatas(status=PaperStatus.DISCARD, session=test_session)
    assert len(discarded) == 1
    assert discarded[0].pmid == "10003"


def test_get_paper_metadatas_with_subset(test_session):
    scraped_unprocessed = get_paper_metadatas(
        status=PaperStatus.NOT_PROCESSED,
        dataset_subset=DatasetSubset.SCRAPED,
        session=test_session,
    )
    assert len(scraped_unprocessed) == 1
    assert scraped_unprocessed[0].pmid == "10001"

    golden_unprocessed = get_paper_metadatas(
        status=PaperStatus.NOT_PROCESSED,
        dataset_subset=DatasetSubset.GOLDEN_BENCHMARK,
        session=test_session,
    )
    assert len(golden_unprocessed) == 1
    assert golden_unprocessed[0].pmid == "10002"


def test_get_paper_metadatas_pagination(test_session):
    papers_page1 = get_paper_metadatas(status=PaperStatus.NOT_PROCESSED, limit=1, offset=0, session=test_session)
    assert len(papers_page1) == 1

    papers_page2 = get_paper_metadatas(status=PaperStatus.NOT_PROCESSED, limit=1, offset=1, session=test_session)
    assert len(papers_page2) == 1
    assert papers_page1[0].id != papers_page2[0].id


def test_get_paper_metadatas_all(test_session):
    all_papers = get_paper_metadatas(session=test_session)
    assert len(all_papers) == 4

    all_scraped = get_paper_metadatas(dataset_subset=DatasetSubset.SCRAPED, session=test_session)
    assert len(all_scraped) == 3


def test_get_paper_metadatas_in_list(test_session):
    results = get_paper_metadatas(
        status=[PaperStatus.USABLE, PaperStatus.DISCARD],
        session=test_session,
    )
    assert len(results) == 2
    statuses = {p.status for p in results}
    assert statuses == {PaperStatus.USABLE, PaperStatus.DISCARD}


def test_get_paper_metadatas_none_value(test_session):
    no_doi_papers = get_paper_metadatas(doi=None, session=test_session)
    assert len(no_doi_papers) == 1
    assert no_doi_papers[0].pmid == "10004"


def test_get_paper_metadatas_invalid_column(test_session):
    with pytest.raises(ValueError, match="has no attribute or column 'non_existent_column'"):
        get_paper_metadatas(non_existent_column="test", session=test_session)


def test_update_paper_status(test_session):
    paper = get_paper_by_id(1, session=test_session)
    assert paper.status == PaperStatus.NOT_PROCESSED

    updated = update_paper_status(1, PaperStatus.USABLE, session=test_session)
    assert updated is not None
    assert updated.status == PaperStatus.USABLE

    # Verify state in DB
    refreshed = get_paper_by_id(1, session=test_session)
    assert refreshed.status == PaperStatus.USABLE

