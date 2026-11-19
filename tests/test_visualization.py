import matplotlib.pyplot as plt
import pytest
import yaml
from click.testing import CliRunner
from matplotlib.figure import Figure
from sqlmodel import Session, SQLModel, create_engine

from peptide_extractor.cli import cli
from peptide_extractor.db import (
    DatasetSubset,
    ExtractionEvaluation,
    ExtractionMode,
    PaperMetadata,
    PaperRelevance,
    PaperStatus,
    PeptideEntry,
    get_records,
)
from peptide_extractor.pipeline import run_visualization
from peptide_extractor.visualization import (
    VisualizationCompatibilityError,
    VisualizationOperation,
    get_supported_operations,
    validate_operations_for_model,
    visualize,
)


@pytest.fixture
def viz_session():
    """In-memory SQLite database seeded with multi-run records for all models."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        p1 = PaperMetadata(
            id=1,
            title="Paper 1",
            abstract="Abstract 1",
            pmid="1001",
            status=PaperStatus.USABLE,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
            run_identifier=10,
        )
        p2 = PaperMetadata(
            id=2,
            title="Paper 2",
            abstract="Abstract 2",
            pmid="1002",
            status=PaperStatus.DISCARD,
            is_open_access=False,
            dataset_subset=DatasetSubset.GOLDEN_BENCHMARK,
            run_identifier=10,
        )
        p3 = PaperMetadata(
            id=3,
            title="Paper 3",
            abstract="Abstract 3",
            pmid="1003",
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
            run_identifier=20,
        )
        session.add_all([p1, p2, p3])

        e1 = PeptideEntry(
            id=1,
            paper_id=1,
            sequence="FF",
            length=2,
            type="nanotube",
            n_terminal="Fmoc",
            c_terminal="Free",
            ph=7.0,
            pmid="1001",
            extraction_mode=ExtractionMode.LLM,
            model="gemini-3.5-flash-lite",
            run_identifier=100,
        )
        e2 = PeptideEntry(
            id=2,
            paper_id=1,
            sequence="KLVFF",
            length=5,
            type="fiber",
            n_terminal="Acetylation",
            pmid="1001",
            extraction_mode=ExtractionMode.LLM,
            model="gemini-3.5-flash-lite",
            run_identifier=100,
        )
        e3 = PeptideEntry(
            id=3,
            paper_id=2,
            sequence="FKFEFKFE",
            length=8,
            type="hydrogel",
            ph=7.4,
            temperature=25.0,
            pmid="1002",
            extraction_mode=ExtractionMode.LLM,
            model="gemma-4-31b-it",
            run_identifier=200,
        )
        session.add_all([e1, e2, e3])

        ev1 = ExtractionEvaluation(
            id=1,
            peptide_entry_id=1,
            groundness_rate=0.9,
            faithfulness_rate=0.95,
            completeness_rate=0.8,
            run_identifier=300,
        )
        ev2 = ExtractionEvaluation(
            id=2,
            peptide_entry_id=2,
            groundness_rate=0.75,
            faithfulness_rate=0.85,
            completeness_rate=0.7,
            run_identifier=300,
        )
        ev3 = ExtractionEvaluation(
            id=3,
            peptide_entry_id=3,
            groundness_rate=0.88,
            faithfulness_rate=0.92,
            completeness_rate=0.65,
            run_identifier=301,
        )
        session.add_all([ev1, ev2, ev3])

        r1 = PaperRelevance(
            id=1,
            paper_id=1,
            is_relevant=True,
            confidence=0.95,
            provider="google_genai",
            model_name="gemma-4-31b-it",
            run_identifier=400,
        )
        r2 = PaperRelevance(
            id=2,
            paper_id=2,
            is_relevant=False,
            confidence=0.85,
            provider="google_genai",
            model_name="gemma-4-31b-it",
            run_identifier=400,
        )
        r3 = PaperRelevance(
            id=3,
            paper_id=3,
            is_relevant=True,
            confidence=0.72,
            provider="ollama",
            model_name="llama4",
            run_identifier=401,
        )
        session.add_all([r1, r2, r3])
        session.commit()

        yield session

    plt.close("all")


def test_get_records_supports_extraction_evaluation(viz_session):
    evals = get_records("extraction_evaluation", session=viz_session, run_identifier=300)
    assert len(evals) == 2
    assert all(isinstance(ev, ExtractionEvaluation) for ev in evals)


def test_get_supported_operations():
    all_ops = get_supported_operations()
    assert len(all_ops) == len(VisualizationOperation)

    peptide_ops = get_supported_operations(PeptideEntry)
    assert VisualizationOperation.PEPTIDE_COLUMN_COMPLETION in peptide_ops
    assert VisualizationOperation.EVALUATION_BOXPLOT not in peptide_ops

    eval_ops = get_supported_operations("extraction_evaluation")
    assert VisualizationOperation.EVALUATION_BOXPLOT in eval_ops
    assert VisualizationOperation.RELEVANCE_CONFIDENCE_HIST not in eval_ops


def test_incompatible_operation_raises_error(viz_session, tmp_path):
    with pytest.raises(VisualizationCompatibilityError):
        visualize(
            operations=[VisualizationOperation.EVALUATION_BOXPLOT],
            model=PeptideEntry,
            session=viz_session,
            output_dir=tmp_path,
            run_identifier=100,
        )

    with pytest.raises(VisualizationCompatibilityError):
        visualize(
            operations=["peptide_length_distribution"],
            model="paper_relevance",
            session=viz_session,
            output_dir=tmp_path,
        )

    with pytest.raises(ValueError):
        validate_operations_for_model(["non_existent_op"], PeptideEntry)


def test_mismatched_records_and_model_raises_error(viz_session, tmp_path):
    peptides = get_records(PeptideEntry, session=viz_session)
    with pytest.raises(VisualizationCompatibilityError):
        visualize(
            operations=["evaluation_boxplot"],
            model=ExtractionEvaluation,
            records=peptides,
            output_dir=tmp_path,
        )


def test_visualize_peptide_entries_multi_run(viz_session, tmp_path):
    ops = [
        VisualizationOperation.PEPTIDE_COLUMN_COMPLETION,
        VisualizationOperation.PEPTIDE_LENGTH_DISTRIBUTION,
        VisualizationOperation.EXTRACTIONS_PER_PAPER,
        VisualizationOperation.PEPTIDE_MORPHOLOGY_DISTRIBUTION,
        VisualizationOperation.PEPTIDE_RUN_COMPARISON,
    ]
    results = visualize(
        operations=ops,
        model="peptide_entry",
        session=viz_session,
        output_dir=tmp_path,
        save=True,
        run_identifier=[100, 200],
    )

    assert set(results.keys()) == set(ops)
    for op, res in results.items():
        assert res.record_count == 3
        assert isinstance(res.figure, Figure)
        assert res.output_path is not None and res.output_path.exists()


def test_visualize_extraction_evaluation(viz_session, tmp_path):
    ops = [
        "evaluation_boxplot",
        "evaluation_summary_range",
        "evaluation_distribution",
    ]
    results = visualize(
        operations=ops,
        model=ExtractionEvaluation,
        session=viz_session,
        output_dir=tmp_path,
        run_identifier=[300, 301],
    )

    assert len(results) == 3
    for res in results.values():
        assert res.record_count == 3
        assert res.output_path is not None and res.output_path.exists()


def test_visualize_paper_relevance_and_metadata(viz_session, tmp_path):
    rel_results = visualize(
        operations=["relevance_confidence_hist", "relevance_outcome_counts"],
        model=PaperRelevance,
        session=viz_session,
        output_dir=tmp_path / "rel",
        run_identifier=[400, 401],
    )
    assert len(rel_results) == 2
    assert all(r.record_count == 3 and r.output_path.exists() for r in rel_results.values())

    meta_results = visualize(
        operations=[VisualizationOperation.PAPER_STATUS_COUNTS],
        model=PaperMetadata,
        session=viz_session,
        output_dir=tmp_path / "meta",
    )
    assert len(meta_results) == 1
    assert meta_results[VisualizationOperation.PAPER_STATUS_COUNTS].record_count == 3
    assert meta_results[VisualizationOperation.PAPER_STATUS_COUNTS].output_path.exists()


def test_visualize_empty_query_returns_empty_dict(viz_session, tmp_path):
    results = visualize(
        operations=["peptide_length_distribution"],
        model=PeptideEntry,
        session=viz_session,
        output_dir=tmp_path,
        run_identifier=999999,
    )
    assert results == {}


def test_run_visualization_pipeline_step(viz_session, tmp_path):
    results, viz_id = run_visualization(
        operations=["peptide_column_completion", "peptide_run_comparison"],
        model="peptide_entry",
        session=viz_session,
        output_dir=tmp_path,
        viz_run_identifier=777,
        log_to_file=False,
        run_identifier=[100, 200],
    )
    assert viz_id == 777
    assert len(results) == 2
    for res in results.values():
        assert res.output_path is not None and res.output_path.exists()


def test_cli_run_pipeline_visualization_step(viz_session, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "peptide_extractor.pipeline.get_session_context",
        lambda *args, **kwargs: _session_ctx(viz_session),
    )
    monkeypatch.setattr("peptide_extractor.pipeline.init_db", lambda *args, **kwargs: None)

    cfg_path = tmp_path / "viz_config.yaml"
    out_dir = tmp_path / "diagrams"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "steps": ["visualization"],
                "visualization": {
                    "run_identifier": 888,
                    "model": "peptide_entry",
                    "operations": ["peptide_column_completion"],
                    "output_dir": str(out_dir),
                    "kwargs": {"run_identifier": [100, 200]},
                },
            }
        )
    )

    runner = CliRunner()
    res = runner.invoke(cli, ["run_pipeline", "-c", str(cfg_path), "--no-log-to-file"])
    assert res.exit_code == 0, res.output
    assert (out_dir / "peptide_column_completion.png").exists()


from contextlib import contextmanager


@contextmanager
def _session_ctx(session):
    yield session
