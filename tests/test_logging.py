import logging
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from click.testing import CliRunner
from sqlmodel import SQLModel, Session, create_engine

from peptide_extractor.logging_utils import (
    LOG_TO_FILE,
    is_file_logging_enabled,
    normalize_step_name,
    step_logger,
)
from peptide_extractor.db.paper_metadata import PaperMetadata, PaperStatus, DatasetSubset
from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.cli import cli
import peptide_extractor.logging_utils as logging_utils
import peptide_extractor.pipeline as pipeline


@pytest.fixture
def memory_db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    return engine


def test_normalize_step_name():
    assert normalize_step_name("search") == "search"
    assert normalize_step_name("fetch_fulltext") == "fetch"
    assert normalize_step_name("fetch") == "fetch"
    assert normalize_step_name("extraction") == "extraction"
    assert normalize_step_name("relevance") == "relevance"
    assert normalize_step_name("evaluation") == "evaluation"
    assert normalize_step_name("FETCH_FULLTEXT") == "fetch"
    assert normalize_step_name("custom_step") == "custom_step"


def test_is_file_logging_enabled(monkeypatch):
    # Test override
    assert is_file_logging_enabled(override=True) is True
    assert is_file_logging_enabled(override=False) is False

    # Test config dictionary (global)
    assert is_file_logging_enabled(config={"log_to_file": True}) is True
    assert is_file_logging_enabled(config={"save_logs": True}) is True
    assert is_file_logging_enabled(config={"file_logging": True}) is True
    assert is_file_logging_enabled(config={"log_to_file": False}) is False

    # Test config dictionary (per-step)
    assert is_file_logging_enabled(step="search", config={"search": {"log_to_file": True}}) is True
    assert is_file_logging_enabled(step="fetch", config={"search": {"log_to_file": True}}) is False

    # Test module variable
    monkeypatch.setattr(logging_utils, "LOG_TO_FILE", True)
    assert is_file_logging_enabled() is True
    monkeypatch.setattr(logging_utils, "LOG_TO_FILE", False)
    assert is_file_logging_enabled() is False

    # Test environment variable
    monkeypatch.setenv("LOG_TO_FILE", "true")
    assert is_file_logging_enabled() is True
    monkeypatch.delenv("LOG_TO_FILE", raising=False)


def test_step_logger_writes_to_file(tmp_path):
    log_dir = tmp_path / "docs" / "logs"
    run_id = 999111
    test_logger = logging.getLogger("test_step_logger")

    with step_logger("extraction", run_id, enabled=True, log_dir=log_dir):
        test_logger.info("This is an info log message for extraction")
        test_logger.warning("This is a warning log message for extraction")

    expected_file = log_dir / "extraction" / f"{run_id}.log"
    assert expected_file.exists()

    content = expected_file.read_text(encoding="utf-8")
    assert "This is an info log message for extraction" in content
    assert "This is a warning log message for extraction" in content
    assert "[INFO]" in content
    assert "[WARNING]" in content


def test_step_logger_disabled_creates_no_file(tmp_path):
    log_dir = tmp_path / "docs" / "logs"
    run_id = 111222
    test_logger = logging.getLogger("test_disabled_logger")

    with step_logger("search", run_id, enabled=False, log_dir=log_dir):
        test_logger.info("Should not be saved to disk")

    expected_file = log_dir / "search" / f"{run_id}.log"
    assert not expected_file.exists()


def test_step_logger_reentrant(tmp_path):
    log_dir = tmp_path / "docs" / "logs"
    run_id = 888777
    test_logger = logging.getLogger("test_reentrant")

    with step_logger("fetch_fulltext", run_id, enabled=True, log_dir=log_dir):
        test_logger.info("Outer start")
        with step_logger("fetch", run_id, enabled=True, log_dir=log_dir):
            test_logger.info("Inner log")
        test_logger.info("Outer end")

    expected_file = log_dir / "fetch" / f"{run_id}.log"
    assert expected_file.exists()

    content = expected_file.read_text(encoding="utf-8")
    assert "Outer start" in content
    assert "Inner log" in content
    assert "Outer end" in content
    # Verify no duplicate entries for the inner log
    assert content.count("Inner log") == 1


def test_paper_metadata_and_content_run_identifier(memory_db):
    with Session(memory_db) as session:
        meta = PaperMetadata(
            title="Peptide Study",
            is_open_access=True,
            status=PaperStatus.USABLE,
            dataset_subset=DatasetSubset.SCRAPED,
            run_identifier=123456,
        )
        session.add(meta)
        session.commit()
        session.refresh(meta)

        assert meta.run_identifier == 123456

        content = PaperContent(
            paper_id=meta.id,
            format="xml",
            blob_url="https://blob.storage/paper.xml",
            run_identifier=654321,
        )
        session.add(content)
        session.commit()
        session.refresh(content)

        assert content.run_identifier == 654321


def test_pipeline_functions_with_logging(memory_db, tmp_path, monkeypatch):
    log_dir = tmp_path / "docs" / "logs"
    monkeypatch.setenv("PEPTIDE_EXTRACTOR_LOG_DIR", str(log_dir))

    # Test retrive_ncbi_by_query with mock provider
    from peptide_extractor.db.raw_provider_record import RawProviderRecord
    mock_raw_record = RawProviderRecord(id=1, provider="ncbi", raw_data={"key": "val"})
    with Session(memory_db) as s:
        s.add(mock_raw_record)
        s.commit()

    mock_ncbi = MagicMock()
    mock_ncbi.search_by_query.return_value = [mock_raw_record]
    mock_record_meta = PaperMetadata(
        title="NCBI Paper",
        is_open_access=True,
        status=PaperStatus.NOT_PROCESSED,
        dataset_subset=DatasetSubset.SCRAPED,
    )
    mock_ncbi.parse_raw_search_records.return_value = [mock_record_meta]


    monkeypatch.setattr(pipeline, "init_db", lambda: None)
    monkeypatch.setattr(pipeline, "get_session_context", lambda: Session(memory_db))

    search_run_id = 777111
    returned_id = pipeline.retrive_ncbi_by_query(
        mock_ncbi,
        query_list=["test query"],
        save=True,
        run_identifier=search_run_id,
        log_to_file=True,
    )

    assert returned_id == search_run_id
    search_log = log_dir / "search" / f"{search_run_id}.log"
    assert search_log.exists()
    content = search_log.read_text(encoding="utf-8")
    assert "Query: test query" in content

    # Test fetch_paper_fulltexts with mock
    with Session(memory_db) as s:
        p = PaperMetadata(
            id=101,
            title="Fetch Paper",
            is_open_access=True,
            status=PaperStatus.USABLE,
            dataset_subset=DatasetSubset.SCRAPED,
        )
        s.add(p)
        s.commit()

    mock_ncbi_fetch = MagicMock()
    mock_content = PaperContent(paper_id=101, format="xml", blob_url="http://blob/101.xml")
    mock_ncbi_fetch.fetch.return_value = ([mock_content], [], [])

    fetch_run_id = 777222
    success, fail, returned_fetch_id = pipeline.fetch_paper_fulltexts(
        session=Session(memory_db),
        ncbi_provider=mock_ncbi_fetch,
        open_alex_provider=MagicMock(),
        run_identifier=fetch_run_id,
        log_to_file=True,
        id=101,
    )

    assert returned_fetch_id == fetch_run_id
    assert success == 1
    fetch_log = log_dir / "fetch" / f"{fetch_run_id}.log"
    assert fetch_log.exists()
    assert "Success: 1, Fail: 0" in fetch_log.read_text(encoding="utf-8")


def test_cli_run_pipeline_file_logging(tmp_path, monkeypatch):
    log_dir = tmp_path / "docs" / "logs"
    monkeypatch.setenv("PEPTIDE_EXTRACTOR_LOG_DIR", str(log_dir))

    # Mock pipeline steps in cli
    mock_relevance = MagicMock(return_value=([], [], 555444))
    monkeypatch.setattr("peptide_extractor.cli.run_bulk_relevance_screening", mock_relevance)

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        """
log_to_file: true
steps:
  - relevance
relevance:
  run_identifier: 555444
"""
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["run_pipeline", "-c", str(yaml_file)])
    assert result.exit_code == 0

    log_file = log_dir / "relevance" / "555444.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Running relevance screening..." in content


def test_extraction_records_and_logs_failed_papers(memory_db, tmp_path, monkeypatch):
    from peptide_extractor.db.peptide_entry import PeptideExtractionResult, PeptideLLMEntry

    log_dir = tmp_path / "logs"
    monkeypatch.setenv("PEPTIDE_EXTRACTOR_LOG_DIR", str(log_dir))

    with Session(memory_db) as s:
        s.add(
            PaperMetadata(
                id=1,
                title="Fail Paper",
                pmid="111",
                is_open_access=True,
                status=PaperStatus.USABLE,
                dataset_subset=DatasetSubset.SCRAPED,
            )
        )
        s.add(
            PaperMetadata(
                id=2,
                title="Pass Paper",
                pmid="222",
                is_open_access=True,
                status=PaperStatus.USABLE,
                dataset_subset=DatasetSubset.SCRAPED,
            )
        )
        s.add(PaperContent(paper_id=1, format="txt", raw_content="bad paper text"))
        s.add(PaperContent(paper_id=2, format="txt", raw_content="good paper text"))
        s.commit()

    def mock_get_content(self):
        return "bad paper text" if self.paper_id == 1 else "good paper text"

    monkeypatch.setattr(PaperContent, "get_content", mock_get_content)

    def mock_run_extraction(text, *args, **kwargs):
        if "bad paper text" in text:
            raise RuntimeError("LLM extraction error")
        return PeptideExtractionResult(entries=[PeptideLLMEntry(sequence="KLVFF")])

    monkeypatch.setattr(pipeline, "run_extraction", mock_run_extraction)
    monkeypatch.setattr(pipeline, "to_extraction_ready_string", lambda x: x)

    with Session(memory_db) as s:
        failed_ids, run_id = pipeline.run_peptide_extraction(
            session=s,
            save=True,
            delay_seconds=0,
            run_identifier=101010,
            log_to_file=True,
        )

    assert failed_ids == [1]
    assert run_id == 101010

    log_file = log_dir / "extraction" / "101010.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Failed to extract peptides for paper 1: LLM extraction error" in content
    assert "Extraction failed for 1 paper(s) with IDs: [1]" in content


def test_evaluation_records_and_logs_failed_entries(memory_db, tmp_path, monkeypatch):
    from peptide_extractor.db.peptide_entry import PeptideEntry
    from peptide_extractor.db.extraction_evaluation import ExtractionEvaluation

    log_dir = tmp_path / "logs"
    monkeypatch.setenv("PEPTIDE_EXTRACTOR_LOG_DIR", str(log_dir))

    with Session(memory_db) as s:
        s.add(
            PaperMetadata(
                id=10,
                title="Eval Paper",
                pmid="333",
                is_open_access=True,
                status=PaperStatus.USABLE,
                dataset_subset=DatasetSubset.SCRAPED,
            )
        )
        s.add(PaperContent(paper_id=10, format="xml", raw_content="<article/>"))
        s.add(PeptideEntry(id=100, paper_id=999, pmid="333", sequence="FF", length=2))  # missing PaperContent
        s.add(PeptideEntry(id=101, paper_id=10, pmid="333", sequence="KLVFF", length=5))
        s.commit()

    def mock_evaluate(extraction, fulltext):
        return ExtractionEvaluation(groundness_rate=1.0, completeness_rate=1.0)

    monkeypatch.setattr(PaperContent, "get_content", lambda self: "<article/>")
    monkeypatch.setattr(pipeline, "evaluate_extraction", mock_evaluate)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)

    with Session(memory_db) as s:
        failed_ids, run_id = pipeline.run_evaluation(
            session=s,
            save=True,
            run_identifier=202020,
            log_to_file=True,
        )

    assert failed_ids == [100]
    assert run_id == 202020

    log_file = log_dir / "evaluation" / "202020.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Failed to evaluate extraction for peptide 100" in content
    assert "Evaluation failed for 1 peptide entry(ies) with IDs: [100]" in content


def test_bulk_relevance_records_and_logs_failed_batches_and_omissions(
    memory_db, tmp_path, monkeypatch
):
    from peptide_extractor.relevance import bulk_screen_papers
    from peptide_extractor.schemas import PerPaperResult, BulkRelevanceResults, RelevanceResult

    log_dir = tmp_path / "logs"
    monkeypatch.setenv("PEPTIDE_EXTRACTOR_LOG_DIR", str(log_dir))

    with Session(memory_db) as s:
        for pid in (1, 2, 3, 4):
            s.add(
                PaperMetadata(
                    id=pid,
                    title=f"Paper {pid}",
                    abstract=f"Abstract {pid}",
                    is_open_access=True,
                    status=PaperStatus.NOT_PROCESSED,
                    dataset_subset=DatasetSubset.SCRAPED,
                )
            )
        s.commit()

    mock_classifier = MagicMock()
    # Batch 1 ([1, 2]) raises; Batch 2 ([3, 4]) omits paper 4
    mock_classifier.bulk_classify_paper.side_effect = [
        RuntimeError("Batch API timeout"),
        BulkRelevanceResults(
            results=[
                PerPaperResult(
                    paper_id=3,
                    relevance=RelevanceResult(
                        detected_clues=["TEM"],
                        explanation="Relevant",
                        is_relevant=True,
                        confidence=0.9,
                    ),
                )
            ]
        ),
    ]

    with Session(memory_db) as s:
        with step_logger("relevance", 303030, enabled=True, log_dir=log_dir):
            results, failed_ids = bulk_screen_papers(
                batch_size=2,
                classifier=mock_classifier,
                session=s,
                delay_seconds=0,
                run_identifier=303030,
            )

    assert len(results) == 1
    assert failed_ids == [1, 2, 4]

    log_file = log_dir / "relevance" / "303030.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Skipping papers [1, 2] due to error: Batch API timeout" in content
    assert "Bulk relevance response omitted paper IDs [4]" in content
    assert "Bulk relevance screening failed for 3 paper(s) with IDs: [1, 2, 4]" in content


def test_fetch_fulltexts_records_and_logs_failed_papers(memory_db, tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("PEPTIDE_EXTRACTOR_LOG_DIR", str(log_dir))

    with Session(memory_db) as s:
        s.add(
            PaperMetadata(
                id=50,
                title="Unfetchable Paper",
                doi="10.1000/fail",
                is_open_access=True,
                status=PaperStatus.USABLE,
                dataset_subset=DatasetSubset.SCRAPED,
            )
        )
        s.commit()

    mock_ncbi = MagicMock()
    mock_ncbi.fetch.side_effect = RuntimeError("NCBI down")
    mock_openalex = MagicMock()
    mock_openalex.fetch.return_value = None
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)

    with Session(memory_db) as s:
        success, fail, run_id = pipeline.fetch_paper_fulltexts(
            session=s,
            ncbi_provider=mock_ncbi,
            open_alex_provider=mock_openalex,
            run_identifier=404040,
            log_to_file=True,
            id=50,
        )

    assert success == 0
    assert fail == 1
    assert run_id == 404040

    log_file = log_dir / "fetch" / "404040.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Failed to retrieve fulltext for paper 50 (10.1000/fail)" in content
    assert "Fulltext fetch failed for 1 paper(s) with IDs: [50]" in content

