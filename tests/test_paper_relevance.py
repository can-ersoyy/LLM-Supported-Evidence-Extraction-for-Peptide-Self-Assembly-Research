import pytest
from unittest.mock import MagicMock
from sqlmodel import SQLModel, create_engine, Session

from peptide_extractor.db.paper_metadata import (
    PaperMetadata,
    PaperStatus,
    DatasetSubset,
    get_paper_by_id,
)
from peptide_extractor.db.paper_relevance import (
    get_paper_relevances,
    save_paper_relevance,
)
from peptide_extractor.llm import LLMController, get_chat_model
from peptide_extractor.relevance import (
    SYSTEM_PROMPT_GOLDEN_BENCHMARK,
    RelevanceResult,
    screen_papers,
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


def test_llm_controller_relevance_mock(mock_llm):
    llm, structured_mock = mock_llm
    controller = LLMController(
        llm=llm,
        structured_output=RelevanceResult,
        system_prompt=SYSTEM_PROMPT_GOLDEN_BENCHMARK,
    )

    res = controller.classify_paper(
        title="Diphenylalanine Self-Assembly",
        abstract="Diphenylalanine motifs assemble into hollow nanotubes.",
    )

    assert res.is_relevant is True
    assert res.confidence == 0.95
    assert "tetrapeptide" in res.detected_clues
    structured_mock.invoke.assert_called_once()


def test_llm_controller_classify_with_paper_metadata(mock_llm):
    llm, structured_mock = mock_llm
    controller = LLMController(
        llm=llm,
        structured_output=RelevanceResult,
        system_prompt=SYSTEM_PROMPT_GOLDEN_BENCHMARK,
    )

    paper = PaperMetadata(
        title="Diphenylalanine Self-Assembly",
        abstract="Diphenylalanine motifs assemble into hollow nanotubes.",
    )

    res = controller.classify_paper(paper)
    assert res.is_relevant is True
    assert res.confidence == 0.95


def test_screen_papers_workflow(mock_llm):
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

        controller = LLMController(
            llm=llm,
            structured_output=RelevanceResult,
            system_prompt=SYSTEM_PROMPT_GOLDEN_BENCHMARK,
        )
        (succesful_screens, _, _) = screen_papers(batch_size=10, classifier=controller, session=session, delay_seconds=0)

        assert len(succesful_screens) == 2

        updated_p1 = get_paper_by_id(paper1.id, session=session)
        updated_p2 = get_paper_by_id(paper2.id, session=session)

        assert updated_p1.status == PaperStatus.USABLE
        assert updated_p2.status == PaperStatus.DISCARD


def test_get_chat_model_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_chat_model(provider="unknown_provider")


def test_run_relevance_screening_pipeline_function(monkeypatch):
    from peptide_extractor.pipeline import run_relevance_screening

    mock_screen = MagicMock(return_value=([("paper1", "res1"), ("paper2", "res2")], [], 1234567890))
    mock_controller_cls = MagicMock()

    monkeypatch.setattr("peptide_extractor.relevance.screen_papers", mock_screen)
    monkeypatch.setattr("peptide_extractor.llm.LLMController", mock_controller_cls)
    monkeypatch.setattr("peptide_extractor.pipeline.init_db", lambda: None)

    (succesful_screens, _, _) = run_relevance_screening(provider="google_genai", model_name="gemma-4-31b-it", batch_size=5)
    assert len(succesful_screens) == 2
    mock_screen.assert_called_once()


def test_screen_papers_skips_failed_api_call(mock_llm):
    llm, structured_mock = mock_llm
    from peptide_extractor.relevance import screen_papers

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

        controller = LLMController(
            llm=llm,
            structured_output=RelevanceResult,
            system_prompt=SYSTEM_PROMPT_GOLDEN_BENCHMARK,
        )
        (succesful_screens, _, _) = screen_papers(batch_size=10, classifier=controller, session=session, delay_seconds=0)

        # Only paper2 should be in results
        assert len(succesful_screens) == 1
        assert succesful_screens[0][0].id == paper2.id

        # Paper 1 should remain NOT_PROCESSED for retry
        refreshed_p1 = session.get(PaperMetadata, paper1.id)
        refreshed_p2 = session.get(PaperMetadata, paper2.id)

        assert refreshed_p1.status == PaperStatus.NOT_PROCESSED
        assert refreshed_p2.status == PaperStatus.USABLE


def test_screen_papers(mock_llm):
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
        paper4 = PaperMetadata(
            title="Paper 4",
            abstract="Abstract 4.",
            is_open_access=True,
            status=PaperStatus.USABLE,
            dataset_subset=DatasetSubset.GOLDEN_BENCHMARK
        )
        session.add_all([paper1, paper2, paper3, paper4])
        session.commit()

        res = RelevanceResult(
            detected_clues=["nanotube"],
            explanation="Nanotube assembly.",
            is_relevant=True,
            confidence=0.9,
        )
        structured_mock.invoke.return_value = res
        controller = LLMController(
            llm=llm,
            structured_output=RelevanceResult,
            system_prompt=SYSTEM_PROMPT_GOLDEN_BENCHMARK,
        )

        # By default screen all papers
        (succesful_screens, _, _) = screen_papers(
            classifier=controller, session=session, delay_seconds=0
        )
        assert len(succesful_screens) == 4
        assert succesful_screens[0][0].id == paper1.id

        (screened_golden_dataset, _, _) = screen_papers(
            classifier=controller, session=session, delay_seconds=0, golden_dataset=True
        )
        assert len(screened_golden_dataset) == 1
        assert screened_golden_dataset[0][0].id == paper4.id

        
def test_get_paper_relevances_query():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        rec1 = save_paper_relevance(
            paper_id=1,
            detected_clues="['fiber']",
            explanation="Forms fibers",
            is_relevant=True,
            confidence=0.95,
            provider="google_genai",
            run_identifier=1234567890,
            model_name="gemma-4-31b-it",
            session=session,
        )
        rec2 = save_paper_relevance(
            paper_id=1,
            detected_clues="[]",
            explanation="Not relevant",
            is_relevant=False,
            confidence=0.8,
            provider="ollama",
            run_identifier=1234567890,
            model_name="llama3",
            session=session,
        )
        rec3 = save_paper_relevance(
            paper_id=2,
            detected_clues="['hydrogel']",
            explanation="Forms hydrogel",
            is_relevant=True,
            confidence=0.99,
            provider="google_genai",
            run_identifier=1234567890,
            model_name="gemma-4-31b-it",
            session=session,
        )

        all_for_p1 = get_paper_relevances(paper_id=1, session=session)
        assert len(all_for_p1) == 2

        relevant_p1 = get_paper_relevances(paper_id=1, is_relevant=True, session=session)
        assert len(relevant_p1) == 1
        assert relevant_p1[0].provider == "google_genai"

        google_recs = get_paper_relevances(provider="google_genai", session=session)
        assert len(google_recs) == 2




