from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from sqlmodel import Session, SQLModel, create_engine

from peptide_extractor.cli import cli
from peptide_extractor.db.extraction_evaluation import ExtractionEvaluation
from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.db.paper_metadata import (
    DatasetSubset,
    PaperMetadata,
    PaperStatus,
)
from peptide_extractor.db.paper_relevance import PaperRelevance
from peptide_extractor.db.peptide_entry import ExtractionMode, PeptideEntry
from peptide_extractor.pipeline import (
    ALL_PIPELINE_STEPS,
    PipelineResult,
    _normalize_pipeline_steps,
    run_full_pipeline,
)
from peptide_extractor.schemas import RelevanceResult


@pytest.fixture
def test_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_normalize_pipeline_steps():
    assert _normalize_pipeline_steps(None) == list(ALL_PIPELINE_STEPS)
    assert _normalize_pipeline_steps("all") == list(ALL_PIPELINE_STEPS)
    assert _normalize_pipeline_steps(["all"]) == list(ALL_PIPELINE_STEPS)
    assert _normalize_pipeline_steps(["fetch", "extraction"]) == [
        "fetch_fulltext",
        "extraction",
    ]
    with pytest.raises(ValueError, match="Unknown pipeline step"):
        _normalize_pipeline_steps(["not_a_step"])


def test_run_full_pipeline_end_to_end_chaining(test_session, tmp_path, monkeypatch):
    """Verify outputs flow automatically from search -> deduplicate -> clean -> relevance -> fetch -> extraction -> evaluation -> visualization."""
    # Pre-existing unrelated paper & entry in DB that MUST NOT be touched by this run
    old_paper = PaperMetadata(
        title="Old Paper",
        abstract="Old abstract",
        doi="10.1000/old",
        pmid="111",
        is_open_access=True,
        status=PaperStatus.NOT_PROCESSED,
        dataset_subset=DatasetSubset.SCRAPED,
        run_identifier=100,
    )
    test_session.add(old_paper)
    test_session.commit()
    test_session.refresh(old_paper)

    old_entry = PeptideEntry(
        sequence="KLVFF",
        length=5,
        pmid="111",
        paper_id=old_paper.id,
        extraction_mode=ExtractionMode.LLM,
        run_identifier=100,
    )
    test_session.add(old_entry)
    test_session.commit()

    shared_run_id = 777001
    recorded_calls = {}

    def fake_search(ncbi_provider, db="pmc", query_list=None, run_identifier=None, log_to_file=None):
        recorded_calls["search"] = (db, list(query_list), run_identifier)
        p1 = PaperMetadata(
            title="Relevant Peptide Paper",
            abstract="FF forms nanotubes in water.",
            doi="10.1000/p1",
            pmid="201",
            is_open_access=True,
            run_identifier=run_identifier,
        )
        p_dup = PaperMetadata(
            title="Duplicate Paper",
            abstract="Duplicate abstract",
            doi="10.1000/p1",
            pmid="202",
            is_open_access=True,
            run_identifier=run_identifier,
        )
        p_empty = PaperMetadata(
            title="Empty Abstract Paper",
            abstract="",
            doi="10.1000/p3",
            pmid="203",
            is_open_access=True,
            run_identifier=run_identifier,
        )
        p_irrel = PaperMetadata(
            title="Irrelevant Review",
            abstract="A review of proteins.",
            doi="10.1000/p4",
            pmid="204",
            is_open_access=True,
            run_identifier=run_identifier,
        )
        test_session.add_all([p1, p_dup, p_empty, p_irrel])
        test_session.commit()
        return run_identifier

    def fake_find_abstracts(api_key, session=None, **filters):
        recorded_calls["find_abstracts"] = filters
        return 0

    def fake_relevance(
        model_name, provider, delay_seconds=3, batch_size=50, run_identifier=None, log_to_file=None, session=None, **kwargs
    ):
        recorded_calls["relevance"] = kwargs
        papers = test_session.query(PaperMetadata).filter(PaperMetadata.id.in_(kwargs["id"])).all()
        results = []
        for p in papers:
            is_rel = p.pmid == "201"
            p.status = PaperStatus.USABLE if is_rel else PaperStatus.DISCARD
            test_session.add(p)
            rel_res = RelevanceResult(
                detected_clues=["nanotubes"] if is_rel else [],
                explanation="ok",
                is_relevant=is_rel,
                confidence=0.9 if is_rel else 0.1,
            )
            test_session.add(
                PaperRelevance(
                    paper_id=p.id,
                    is_relevant=is_rel,
                    confidence=rel_res.confidence,
                    provider=provider,
                    model_name=model_name,
                    run_identifier=run_identifier,
                )
            )
            results.append((p.id, rel_res))
        test_session.commit()
        return results, [], run_identifier

    def fake_fetch(session=None, ncbi_provider=None, open_alex_provider=None, skip_existing_content=True, run_identifier=None, log_to_file=None, **kwargs):
        recorded_calls["fetch"] = kwargs
        for pid in kwargs["id"]:
            test_session.add(
                PaperContent(
                    paper_id=pid,
                    format="xml",
                    blob_url=f"https://example.com/{pid}.xml",
                    run_identifier=run_identifier,
                )
            )
        test_session.commit()
        return len(kwargs["id"]), 0, run_identifier

    def fake_extract(session=None, provider=None, model_name=None, delay_seconds=2, run_identifier=None, log_to_file=None, **kwargs):
        recorded_calls["extraction"] = kwargs
        for pid in kwargs["id"]:
            test_session.add(
                PeptideEntry(
                    sequence="FF",
                    length=2,
                    type="tube",
                    pmid="201",
                    paper_id=pid,
                    extraction_mode=ExtractionMode.LLM,
                    model=model_name,
                    run_identifier=run_identifier,
                )
            )
        test_session.commit()
        return [], run_identifier

    def fake_evaluate(session=None, run_identifier=None, log_to_file=None, **kwargs):
        recorded_calls["evaluation"] = kwargs
        entries = (
            test_session.query(PeptideEntry)
            .filter(PeptideEntry.id.in_(kwargs["id"]))
            .all()
        )
        for e in entries:
            test_session.add(
                ExtractionEvaluation(
                    peptide_entry_id=e.id,
                    groundness_rate=1.0,
                    faithfulness_rate=1.0,
                    completeness_rate=0.8,
                    run_identifier=run_identifier,
                )
            )
        test_session.commit()
        return [], run_identifier

    out_dir = tmp_path / "diagrams"
    result = run_full_pipeline(
        queries=["peptide self-assembly"],
        run_identifier=shared_run_id,
        log_to_file=False,
        config={"visualization": {"output_dir": str(out_dir)}},
        ncbi_provider=MagicMock(),
        open_alex_provider=MagicMock(),
        session=test_session,
        _step_fns={
            "search": fake_search,
            "find_abstracts": fake_find_abstracts,
            "relevance_bulk": fake_relevance,
            "fetch_fulltext": fake_fetch,
            "extraction": fake_extract,
            "evaluation": fake_evaluate,
        },
    )

    assert isinstance(result, PipelineResult)
    assert result.run_identifier == shared_run_id
    assert result.steps_executed == list(ALL_PIPELINE_STEPS)
    assert result.search_count == 4
    assert result.deduplicated_count == 1
    assert result.cleaned_empty_count == 1

    # Only p1 and p4 survived deduplicate + clean_empty, and old_paper was NOT included
    assert len(recorded_calls["relevance"]["id"]) == 2
    assert old_paper.id not in recorded_calls["relevance"]["id"]

    # Only p1 (USABLE) reached fetch and extraction
    assert len(recorded_calls["fetch"]["id"]) == 1
    assert recorded_calls["extraction"]["id"] == recorded_calls["fetch"]["id"]
    assert result.extraction_entry_count == 1

    # Evaluation only evaluated this run's extracted PeptideEntry IDs (not old_entry)
    assert len(recorded_calls["evaluation"]["id"]) == 1
    assert old_entry.id not in recorded_calls["evaluation"]["id"]

    # Visualizations were auto-generated for the run
    assert len(result.visualization_outputs) > 0
    assert (out_dir / "peptide_column_completion.png").exists()
    assert (out_dir / "evaluation_boxplot.png").exists()
    assert (out_dir / "relevance_outcome_counts.png").exists()


def test_run_full_pipeline_short_circuits_when_no_usable_papers(test_session):
    """If relevance marks all papers DISCARD, fetch/extraction/evaluation are skipped."""
    p = PaperMetadata(
        title="Review",
        abstract="Just a review.",
        doi="10.1000/rev",
        pmid="301",
        is_open_access=True,
        status=PaperStatus.NOT_PROCESSED,
        dataset_subset=DatasetSubset.SCRAPED,
    )
    test_session.add(p)
    test_session.commit()
    test_session.refresh(p)

    fetch_mock = MagicMock()
    extract_mock = MagicMock()
    eval_mock = MagicMock()

    def fake_relevance(model_name, provider, **kwargs):
        return [
            (
                p.id,
                RelevanceResult(
                    detected_clues=[],
                    explanation="review",
                    is_relevant=False,
                    confidence=0.95,
                ),
            )
        ], [], kwargs["run_identifier"]

    result = run_full_pipeline(
        steps=["relevance", "fetch_fulltext", "extraction", "evaluation"],
        paper_ids=[p.id],
        run_identifier=888001,
        log_to_file=False,
        session=test_session,
        _step_fns={
            "relevance_bulk": fake_relevance,
            "fetch_fulltext": fetch_mock,
            "extraction": extract_mock,
            "evaluation": eval_mock,
        },
    )

    assert result.relevance_usable_ids == []
    assert result.relevance_discarded_count == 1
    fetch_mock.assert_not_called()
    extract_mock.assert_not_called()
    eval_mock.assert_not_called()


def test_run_full_pipeline_explicit_kwargs_override_auto_scope(test_session):
    """Per-step kwargs in config still override automatic pipeline scope for dev use."""
    extract_mock = MagicMock(return_value=([], 999))
    result = run_full_pipeline(
        steps=["extraction"],
        config={"extraction": {"kwargs": {"id": [42, 43]}}},
        run_identifier=999,
        log_to_file=False,
        session=test_session,
        _step_fns={"extraction": extract_mock},
    )
    assert result.steps_executed == ["extraction"]
    assert extract_mock.call_args.kwargs["id"] == [42, 43]


def test_cli_run_pipeline_without_config_uses_cli_flags(monkeypatch):
    """CLI run_pipeline works directly with -q / --limit / --step without needing a YAML file."""
    captured = {}

    def fake_run_full_pipeline(*, config=None, log_to_file=None, _step_fns=None):
        captured["config"] = config
        captured["log_to_file"] = log_to_file
        return PipelineResult(run_identifier=12345, steps_executed=["search", "relevance"])

    monkeypatch.setattr("peptide_extractor.cli.run_full_pipeline", fake_run_full_pipeline)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "run_pipeline",
            "-q",
            "peptide self-assembly",
            "--limit",
            "5",
            "--provider",
            "ollama",
            "--model",
            "gemma4:26b",
            "--no-log-to-file",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["config"]["queries"] == ["peptide self-assembly"]
    assert captured["config"]["limit"] == 5
    assert captured["config"]["provider"] == "ollama"
    assert captured["config"]["model_name"] == "gemma4:26b"
    assert "Pipeline Run 12345" in res.output
