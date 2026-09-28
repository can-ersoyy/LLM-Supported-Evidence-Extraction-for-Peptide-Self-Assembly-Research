import pytest
from sqlmodel import SQLModel, create_engine, Session

from peptide_extractor.db import (
    DatasetSubset,
    PaperContent,
    PaperMetadata,
    PaperRelevance,
    PaperStatus,
    get_paper_contents,
    get_paper_metadatas,
    get_paper_relevances,
    get_records,
)


@pytest.fixture
def test_engine_and_session():
    """Create an in-memory SQLite engine and populated session."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Seed PaperMetadata
        p1 = PaperMetadata(
            id=1,
            title="Paper One",
            abstract="Abstract One",
            doi="10.1000/1",
            pmid="111",
            status=PaperStatus.USABLE,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
        )
        p2 = PaperMetadata(
            id=2,
            title="Paper Two",
            abstract="Abstract Two",
            doi="10.1000/2",
            pmid="222",
            status=PaperStatus.DISCARD,
            is_open_access=False,
            dataset_subset=DatasetSubset.GOLDEN_BENCHMARK,
        )
        p3 = PaperMetadata(
            id=3,
            title="Paper Three",
            abstract=None,
            doi=None,
            pmid="333",
            status=PaperStatus.USABLE,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
        )
        session.add_all([p1, p2, p3])

        # Seed PaperContent
        c1 = PaperContent(
            id=1,
            paper_id=1,
            format="jats_xml",
            blob_url="https://azure.blob/paper-xmls/PMC1.xml.gz",
        )
        c2 = PaperContent(
            id=2,
            paper_id=1,
            format="pdf_text",
            blob_url="https://azure.blob/paper-xmls/PMC1.pdf",
        )
        c3 = PaperContent(
            id=3,
            paper_id=2,
            format="jats_xml",
            blob_url="https://azure.blob/paper-xmls/PMC2.xml.gz",
        )
        session.add_all([c1, c2, c3])

        # Seed PaperRelevance
        r1 = PaperRelevance(
            id=1,
            paper_id=1,
            detected_clues="['nanofiber']",
            explanation="Good",
            is_relevant=True,
            confidence=0.9,
            provider="google_genai",
            model_name="gemma-4-31b-it",
        )
        r2 = PaperRelevance(
            id=2,
            paper_id=2,
            detected_clues="[]",
            explanation="Bad",
            is_relevant=False,
            confidence=0.8,
            provider="ollama",
            model_name="llama3",
        )
        session.add_all([r1, r2])

        session.commit()
        yield engine, session


def test_get_records_with_model_class(test_engine_and_session):
    _, session = test_engine_and_session

    usable = get_records(PaperMetadata, status=PaperStatus.USABLE, session=session)
    assert len(usable) == 2

    # Multiple conditions
    usable_scraped = get_records(
        PaperMetadata,
        status=PaperStatus.USABLE,
        dataset_subset=DatasetSubset.SCRAPED,
        session=session,
    )
    assert len(usable_scraped) == 2


def test_get_records_with_string_table_name(test_engine_and_session):
    _, session = test_engine_and_session

    papers = get_records("paper_metadata", status=PaperStatus.DISCARD, session=session)
    assert len(papers) == 1
    assert papers[0].pmid == "222"

    contents = get_records("paper_content", format="jats_xml", session=session)
    assert len(contents) == 2

    relevances = get_records("paper_relevance", is_relevant=True, session=session)
    assert len(relevances) == 1
    assert relevances[0].provider == "google_genai"


def test_get_records_unknown_table():
    with pytest.raises(ValueError, match="Unknown table name 'unknown_table'"):
        get_records("unknown_table")


def test_get_records_invalid_model_type():
    with pytest.raises(ValueError, match="Expected SQLModel subclass or string"):
        get_records(123)  # type: ignore


def test_get_records_pagination_and_offset(test_engine_and_session):
    _, session = test_engine_and_session

    page1 = get_records(PaperMetadata, limit=1, offset=0, session=session)
    assert len(page1) == 1
    page2 = get_records(PaperMetadata, limit=1, offset=1, session=session)
    assert len(page2) == 1
    assert page1[0].id != page2[0].id


def test_get_paper_contents_helper(test_engine_and_session):
    _, session = test_engine_and_session

    p1_contents = get_paper_contents(paper_id=1, session=session)
    assert len(p1_contents) == 2

    jats_xml_only = get_paper_contents(paper_id=1, format="jats_xml", session=session)
    assert len(jats_xml_only) == 1
    assert jats_xml_only[0].blob_url == "https://azure.blob/paper-xmls/PMC1.xml.gz"


def test_get_records_is_null_and_in_list(test_engine_and_session):
    _, session = test_engine_and_session

    no_doi = get_records(PaperMetadata, doi=None, session=session)
    assert len(no_doi) == 1
    assert no_doi[0].id == 3

    in_list = get_records(
        PaperMetadata,
        status=[PaperStatus.USABLE, PaperStatus.DISCARD],
        session=session,
    )
    assert len(in_list) == 3
