import pytest
from unittest.mock import MagicMock
from sqlmodel import SQLModel, create_engine, Session

from peptide_extractor.db.paper_metadata import (
    PaperMetadata,
    PaperStatus,
    get_paper_by_id,
)
from peptide_extractor.relevance import (
    PaperRelevanceClassifier,
    RelevanceResult,
    get_chat_model,
    screen_unprocessed_papers,
)


@pytest.fixture
def mock_llm():
    """Create a mock BaseChatModel with structured output support."""
    llm = MagicMock()
    structured_mock = MagicMock()

    # Define a default structured result
    sample_result = RelevanceResult(
        detected_clues=["tetrapeptide", "nanofiber", "TEM"],
        explanation="The abstract describes tetrapeptide assembly into nanofibers monitored by TEM.",
        is_relevant=True,
        confidence=0.95,
    )
    structured_mock.invoke.return_value = sample_result
    llm.with_structured_output.return_value = structured_mock
    return llm, structured_mock


def test_relevance_result_schema():
    res = RelevanceResult(
        detected_clues=["hydrogel"],
        explanation="Describes hydrogel formation.",
        is_relevant=True,
        confidence=0.9,
    )
    assert res.is_relevant is True
    assert res.confidence == 0.9
    assert res.detected_clues == ["hydrogel"]


def test_paper_relevance_classifier_mock(mock_llm):
    llm, structured_mock = mock_llm
    classifier = PaperRelevanceClassifier(llm=llm)

    res = classifier.classify_paper(
        title="Diphenylalanine Self-Assembly",
        abstract="Diphenylalanine motifs assemble into hollow nanotubes.",
    )

    assert res.is_relevant is True
    assert res.confidence == 0.95
    assert "tetrapeptide" in res.detected_clues
    structured_mock.invoke.assert_called_once()


def test_screen_unprocessed_papers_workflow(mock_llm):
    llm, structured_mock = mock_llm

    # Create in-memory DB session
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        paper1 = PaperMetadata(
            title="Relevant Paper",
            abstract="Short peptide forms hydrogel.",
            is_open_access=True,
            status=PaperStatus.NOT_PROCESSED,
        )
        paper2 = PaperMetadata(
            title="Irrelevant Review",
            abstract="Literature review on general polymers.",
            is_open_access=True,
            status=PaperStatus.NOT_PROCESSED,
        )
        session.add_all([paper1, paper2])
        session.commit()
        session.refresh(paper1)
        session.refresh(paper2)

        # Mock structured LLM responses: paper1 -> relevant, paper2 -> not relevant
        res1 = RelevanceResult(
            detected_clues=["hydrogel"],
            explanation="Hydrogel formation mentioned.",
            is_relevant=True,
            confidence=0.9,
        )
        res2 = RelevanceResult(
            detected_clues=[],
            explanation="General review without peptide assembly data.",
            is_relevant=False,
            confidence=0.85,
        )
        structured_mock.invoke.side_effect = [res1, res2]

        classifier = PaperRelevanceClassifier(llm=llm)
        screened = screen_unprocessed_papers(batch_size=10, classifier=classifier, session=session, delay_seconds=0)

        assert len(screened) == 2

        updated_p1 = get_paper_by_id(paper1.id, session=session)
        updated_p2 = get_paper_by_id(paper2.id, session=session)

        assert updated_p1.status == PaperStatus.USABLE
        assert updated_p2.status == PaperStatus.DISCARD


def test_get_chat_model_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_chat_model(provider="unknown_provider")

def test_run_relevance_screening_pipeline_function(monkeypatch):
    from peptide_extractor.pipeline import run_relevance_screening

    mock_screen = MagicMock(return_value=[("paper1", "res1"), ("paper2", "res2")])
    mock_classifier_cls = MagicMock()

    monkeypatch.setattr("peptide_extractor.relevance.screen_unprocessed_papers", mock_screen)
    monkeypatch.setattr("peptide_extractor.relevance.PaperRelevanceClassifier", mock_classifier_cls)
    monkeypatch.setattr("peptide_extractor.pipeline.init_db", lambda: None)

    count = run_relevance_screening(provider="google_genai", model_name="gemma-4-31b-it", batch_size=5)
    assert count == 2
    mock_screen.assert_called_once()

def test_screen_unprocessed_papers_skips_failed_api_call(mock_llm):
    llm, structured_mock = mock_llm
    from peptide_extractor.relevance import screen_unprocessed_papers

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        paper1 = PaperMetadata(
            title="Paper 1 Fails API",
            abstract="Abstract 1 causing API exception.",
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
        )
        paper2 = PaperMetadata(
            title="Paper 2 Succeeds",
            abstract="Short peptide forms hydrogel.",
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
        )
        session.add_all([paper1, paper2])
        session.commit()
        session.refresh(paper1)
        session.refresh(paper2)

        # First call raises Runtime/API Exception, second call returns valid RelevanceResult
        res2 = RelevanceResult(
            detected_clues=["hydrogel"],
            explanation="Hydrogel formation.",
            is_relevant=True,
            confidence=0.95,
        )
        structured_mock.invoke.side_effect = [RuntimeError("API Rate Limit Exceeded"), res2]

        classifier = PaperRelevanceClassifier(llm=llm)
        screened = screen_unprocessed_papers(batch_size=10, classifier=classifier, session=session, delay_seconds=0)

        # Only paper2 should be in results
        assert len(screened) == 1
        assert screened[0][0].id == paper2.id

        # Paper 1 should remain NOT_PROCESSED for retry
        refreshed_p1 = session.get(PaperMetadata, paper1.id)
        refreshed_p2 = session.get(PaperMetadata, paper2.id)

        assert refreshed_p1.status == PaperStatus.NOT_PROCESSED
        assert refreshed_p2.status == PaperStatus.USABLE


def test_screen_papers_with_all_papers_flag(mock_llm):
    llm, structured_mock = mock_llm

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        paper1 = PaperMetadata(
            title="Paper 1",
            abstract="Not processed paper.",
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
        )
        paper2 = PaperMetadata(
            title="Paper 2",
            abstract="Previously discarded paper.",
            status=PaperStatus.DISCARD,
            is_open_access=True,
        )
        paper3 = PaperMetadata(
            title="Paper 3",
            abstract="Already usable paper.",
            status=PaperStatus.USABLE,
            is_open_access=True,
        )
        session.add_all([paper1, paper2, paper3])
        session.commit()

        res = RelevanceResult(
            detected_clues=["nanotube"],
            explanation="Nanotube assembly.",
            is_relevant=True,
            confidence=0.9,
        )
        structured_mock.invoke.return_value = res
        classifier = PaperRelevanceClassifier(llm=llm)

        # Default all_papers=False only processes paper1
        screened_default = screen_unprocessed_papers(
            classifier=classifier, session=session, delay_seconds=0, all_papers=False
        )
        assert len(screened_default) == 1
        assert screened_default[0][0].id == paper1.id

        # all_papers=True processes all 3 papers regardless of status
        screened_all = screen_unprocessed_papers(
            classifier=classifier, session=session, delay_seconds=0, all_papers=True
        )
        assert len(screened_all) == 3

        # ignore_status=True also processes all 3 papers
        screened_ignore = screen_unprocessed_papers(
            classifier=classifier, session=session, delay_seconds=0, ignore_status=True
        )
        assert len(screened_ignore) == 3


def test_run_relevance_screening_pipeline_function_all_papers(monkeypatch):
    from peptide_extractor.pipeline import run_relevance_screening

    mock_screen = MagicMock(return_value=[("p1", "r1"), ("p2", "r2"), ("p3", "r3")])
    mock_classifier_cls = MagicMock()

    monkeypatch.setattr("peptide_extractor.relevance.screen_unprocessed_papers", mock_screen)
    monkeypatch.setattr("peptide_extractor.relevance.PaperRelevanceClassifier", mock_classifier_cls)
    monkeypatch.setattr("peptide_extractor.pipeline.init_db", lambda: None)

    count = run_relevance_screening(all_papers=True)
    assert count == 3
    mock_screen.assert_called_once()
    _, kwargs = mock_screen.call_args
    assert kwargs.get("all_papers") is True



