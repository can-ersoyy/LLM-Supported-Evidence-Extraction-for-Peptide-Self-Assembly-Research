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
