import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from peptide_extractor.db.paper_metadata import (
    DatasetSubset,
    PaperMetadata,
    PaperStatus,
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


def test_deduplicate_rule1_abstract_priority(test_session):
    """Rule 1: Prefer papers with non-empty abstract over missing or whitespace abstract."""
    # Winner: has valid abstract
    p_with_abstract = PaperMetadata(
        title="With Abstract",
        doi="10.1000/dup",
        abstract="Valid abstract content.",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Loser: None abstract
    p_none_abstract = PaperMetadata(
        title="None Abstract",
        doi="10.1000/dup",
        abstract=None,
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Loser: Whitespace/empty abstract
    p_empty_abstract = PaperMetadata(
        title="Empty Abstract",
        doi="10.1000/dup",
        abstract="   ",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p_with_abstract, p_none_abstract, p_empty_abstract])
    test_session.commit()

    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 2
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 1
    assert remaining[0].title == "With Abstract"


def test_deduplicate_rule2_pmcid_priority(test_session):
    """Rule 2: When abstracts are equal, prefer papers with non-empty PMCID."""
    # Winner: has PMCID
    p_with_pmcid = PaperMetadata(
        title="With PMCID",
        doi="10.1000/dup",
        abstract="Abstract present.",
        pmcid="PMC123456",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Loser: missing PMCID
    p_no_pmcid = PaperMetadata(
        title="No PMCID",
        doi="10.1000/dup",
        abstract="Abstract present.",
        pmcid=None,
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Loser: empty PMCID string
    p_empty_pmcid = PaperMetadata(
        title="Empty PMCID",
        doi="10.1000/dup",
        abstract="Abstract present.",
        pmcid="   ",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p_with_pmcid, p_no_pmcid, p_empty_pmcid])
    test_session.commit()

    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 2
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 1
    assert remaining[0].title == "With PMCID"


def test_deduplicate_rule3_ncbi_provider_priority(test_session):
    """Rule 3: When abstract and PMCID match, prefer provider 'ncbi' over other providers."""
    # Winner: provider ncbi
    p_ncbi = PaperMetadata(
        title="NCBI Paper",
        doi="10.1000/dup",
        abstract="Abstract present.",
        pmcid="PMC123456",
        provider="ncbi",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Loser: provider scopus
    p_scopus = PaperMetadata(
        title="Scopus Paper",
        doi="10.1000/dup",
        abstract="Abstract present.",
        pmcid="PMC123456",
        provider="scopus",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Add scopus first (lower ID), then ncbi (higher ID)
    test_session.add_all([p_scopus, p_ncbi])
    test_session.commit()

    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 1
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 1
    # NCBI paper wins even though it had a higher db ID
    assert remaining[0].title == "NCBI Paper"


def test_deduplicate_rule4_lower_id_tie_breaker(test_session):
    """Rule 4: Otherwise, keep the paper with the lower database ID and delete the higher ID."""
    p_older = PaperMetadata(
        title="Older Paper",
        doi="10.1000/dup",
        abstract="Same abstract.",
        pmcid="PMC111",
        provider="ncbi",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p_newer = PaperMetadata(
        title="Newer Paper",
        doi="10.1000/dup",
        abstract="Same abstract.",
        pmcid="PMC111",
        provider="ncbi",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p_older, p_newer])
    test_session.commit()

    # Verify ID order
    assert p_older.id < p_newer.id

    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 1
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 1
    assert remaining[0].id == p_older.id
    assert remaining[0].title == "Older Paper"


def test_deduplicate_case_and_whitespace_insensitive(test_session):
    """DOIs with different casing and whitespace should be treated as duplicates."""
    p1 = PaperMetadata(
        title="Lower DOI",
        doi="10.1016/j.bpj.2020.01.001",
        abstract="Abstract present.",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p2 = PaperMetadata(
        title="Upper and Spaced DOI",
        doi="  10.1016/J.BPJ.2020.01.001  ",
        abstract=None,
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p1, p2])
    test_session.commit()

    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 1
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 1
    assert remaining[0].title == "Lower DOI"


def test_deduplicate_ignores_empty_or_none_dois(test_session):
    """Papers with None or empty DOIs should not be grouped or deleted as duplicates."""
    p1 = PaperMetadata(
        title="No DOI 1",
        doi=None,
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p2 = PaperMetadata(
        title="No DOI 2",
        doi=None,
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    p3 = PaperMetadata(
        title="Empty DOI",
        doi="   ",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p1, p2, p3])
    test_session.commit()

    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 0
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 3


def test_deduplicate_preserves_golden_benchmark(test_session):
    """Golden benchmark papers should not be deleted, even if they share a DOI."""
    golden_paper = PaperMetadata(
        title="Golden Benchmark Paper",
        doi="10.1000/shared",
        abstract=None,
        dataset_subset=DatasetSubset.GOLDEN_BENCHMARK,
        is_open_access=True,
    )
    scraped_paper_1 = PaperMetadata(
        title="Scraped Paper 1",
        doi="10.1000/shared",
        abstract="Scraped abstract 1",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    scraped_paper_2 = PaperMetadata(
        title="Scraped Paper 2",
        doi="10.1000/shared",
        abstract="Scraped abstract 2",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([golden_paper, scraped_paper_1, scraped_paper_2])
    test_session.commit()

    deleted = deduplicate_by_doi(session=test_session)

    # Only one of the scraped duplicates should be deleted
    assert deleted == 1
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 2

    # Golden paper is untouched
    golden_in_db = [p for p in remaining if p.dataset_subset == DatasetSubset.GOLDEN_BENCHMARK]
    assert len(golden_in_db) == 1
    assert golden_in_db[0].title == "Golden Benchmark Paper"


def test_deduplicate_four_way_hierarchy(test_session):
    """Verify full priority ladder among 4 duplicate records."""
    # Rank 1 (lowest): no abstract, no pmcid, scopus, id=1
    p1 = PaperMetadata(
        id=1,
        title="Level 0",
        doi="10.1000/dup",
        abstract=None,
        pmcid=None,
        provider="scopus",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Rank 2: has abstract, no pmcid, scopus, id=2
    p2 = PaperMetadata(
        id=2,
        title="Level 1",
        doi="10.1000/dup",
        abstract="Has abstract",
        pmcid=None,
        provider="scopus",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Rank 3: has abstract, has pmcid, scopus, id=3
    p3 = PaperMetadata(
        id=3,
        title="Level 2",
        doi="10.1000/dup",
        abstract="Has abstract",
        pmcid="PMC123",
        provider="scopus",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Rank 4 (highest): has abstract, has pmcid, ncbi, id=4
    p4 = PaperMetadata(
        id=4,
        title="Level 3 (Winner - Lower ID)",
        doi="10.1000/dup",
        abstract="Has abstract",
        pmcid="PMC123",
        provider="ncbi",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    # Competitor to Rank 4 with identical attributes but higher ID (should be deleted)
    p5 = PaperMetadata(
        id=5,
        title="Level 3 (Higher ID - Deleted)",
        doi="10.1000/dup",
        abstract="Has abstract",
        pmcid="PMC123",
        provider="ncbi",
        dataset_subset=DatasetSubset.SCRAPED,
        is_open_access=True,
    )
    test_session.add_all([p1, p2, p3, p4, p5])
    test_session.commit()

    deleted = deduplicate_by_doi(session=test_session)

    assert deleted == 4
    remaining = test_session.exec(select(PaperMetadata)).all()
    assert len(remaining) == 1
    assert remaining[0].id == 4
    assert remaining[0].title == "Level 3 (Winner - Lower ID)"
