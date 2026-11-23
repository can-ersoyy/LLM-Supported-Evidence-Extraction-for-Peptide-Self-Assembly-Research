from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from sqlmodel import Session, SQLModel, create_engine, select

from peptide_extractor.cli import cli
from peptide_extractor.db.extraction_evaluation import ExtractionEvaluation
from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.db.paper_metadata import (
    DatasetSubset,
    PaperMetadata,
    PaperStatus,
)
from peptide_extractor.db.peptide_entry import ExtractionMode, PeptideEntry
from peptide_extractor.pipeline import run_full_pipeline
from peptide_extractor.schemas import RelevanceResult


@pytest.fixture
def test_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_run_full_pipeline_end_to_end_chaining(test_session, monkeypatch):
    """Outputs flow automatically from search -> deduplicate -> clean -> relevance -> fetch -> extraction -> evaluation."""
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
    recorded = {}

    def fake_search(ncbi_provider, db="pmc", query_list=None, run_identifier=None, log_to_file=None):
        recorded["search"] = (db, list(query_list), run_identifier)
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

    def fake_relevance(
        model_name, provider, delay_seconds=3, batch_size=50, run_identifier=None, log_to_file=None, session=None, **kwargs
    ):
        recorded["relevance"] = kwargs
        papers = test_session.exec(
            select(PaperMetadata).where(PaperMetadata.id.in_(kwargs["id"]))
        ).all()
        results = []
        for p in papers:
            is_rel = p.pmid == "201"
            p.status = PaperStatus.USABLE if is_rel else PaperStatus.DISCARD
            test_session.add(p)
            results.append(
                (
                    p.id,
                    RelevanceResult(
                        detected_clues=["nanotubes"] if is_rel else [],
                        explanation="ok",
                        is_relevant=is_rel,
                        confidence=0.9 if is_rel else 0.1,
                    ),
                )
            )
        test_session.commit()
        return results, [], run_identifier

    def fake_fetch(session=None, ncbi_provider=None, open_alex_provider=None, run_identifier=None, log_to_file=None, **kwargs):
        recorded["fetch"] = kwargs
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
        recorded["extraction"] = kwargs
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
        recorded["evaluation"] = kwargs
        entries = test_session.exec(
            select(PeptideEntry).where(PeptideEntry.id.in_(kwargs["id"]))
        ).all()
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

    monkeypatch.setattr("peptide_extractor.pipeline.retrive_ncbi_by_query", fake_search)
    monkeypatch.setattr("peptide_extractor.pipeline.find_abstracts_by_doi", lambda *a, **kw: 0)
    monkeypatch.setattr("peptide_extractor.pipeline.run_bulk_relevance_screening", fake_relevance)
    monkeypatch.setattr("peptide_extractor.pipeline.fetch_paper_fulltexts", fake_fetch)
    monkeypatch.setattr("peptide_extractor.pipeline.run_peptide_extraction", fake_extract)
    monkeypatch.setattr("peptide_extractor.pipeline.run_evaluation", fake_evaluate)

    summary = run_full_pipeline(
        queries=["peptide self-assembly"],
        run_identifier=shared_run_id,
        log_to_file=False,
        skip_visualization=True,
        ncbi_provider=MagicMock(),
        open_alex_provider=MagicMock(),
        session=test_session,
    )

    assert summary["run_identifier"] == shared_run_id
    assert summary["search_count"] == 4
    assert summary["deduplicated"] == 1
    assert summary["cleaned_empty"] == 1
    assert summary["relevance_usable"] == 1
    assert summary["extraction_entries"] == 1

    assert len(recorded["relevance"]["id"]) == 2
    assert old_paper.id not in recorded["relevance"]["id"]
    assert recorded["fetch"]["id"] == recorded["extraction"]["id"]
    assert len(recorded["evaluation"]["id"]) == 1
    assert old_entry.id not in recorded["evaluation"]["id"]


def test_run_full_pipeline_short_circuits_when_no_usable_papers(test_session, monkeypatch):
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

    monkeypatch.setattr("peptide_extractor.pipeline.find_abstracts_by_doi", lambda *a, **kw: 0)
    monkeypatch.setattr(
        "peptide_extractor.pipeline.run_bulk_relevance_screening",
        lambda **kw: (
            [
                (
                    p.id,
                    RelevanceResult(
                        detected_clues=[],
                        explanation="review",
                        is_relevant=False,
                        confidence=0.95,
                    ),
                )
            ],
            [],
            kw["run_identifier"],
        ),
    )
    monkeypatch.setattr("peptide_extractor.pipeline.fetch_paper_fulltexts", fetch_mock)
    monkeypatch.setattr("peptide_extractor.pipeline.run_peptide_extraction", extract_mock)
    monkeypatch.setattr("peptide_extractor.pipeline.run_evaluation", eval_mock)

    summary = run_full_pipeline(
        paper_ids=[p.id],
        run_identifier=888001,
        log_to_file=False,
        session=test_session,
    )

    assert summary["relevance_usable"] == 0
    fetch_mock.assert_not_called()
    extract_mock.assert_not_called()
    eval_mock.assert_not_called()


def test_cli_run_full_pipeline_command(monkeypatch):
    """Alternative CLI command `run_full_pipeline` delegates cleanly to pipeline.run_full_pipeline."""
    captured = {}

    def fake_run_full_pipeline(**kwargs):
        captured.update(kwargs)
        return {"run_identifier": 12345, "extraction_entries": 2}

    monkeypatch.setattr("peptide_extractor.cli.run_full_pipeline", fake_run_full_pipeline)

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "run_full_pipeline",
            "-q",
            "peptide self-assembly",
            "--limit",
            "5",
            "--provider",
            "ollama",
            "--model",
            "gemma4:26b",
            "--skip-visualization",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["queries"] == ["peptide self-assembly"]
    assert captured["limit"] == 5
    assert captured["provider"] == "ollama"
    assert captured["model_name"] == "gemma4:26b"
    assert captured["skip_visualization"] is True
