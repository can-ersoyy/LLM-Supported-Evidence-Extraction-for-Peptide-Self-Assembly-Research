from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from sqlmodel import SQLModel, Session, create_engine, select

from peptide_extractor import pipeline
from peptide_extractor.cli import cli
from peptide_extractor.db.paper_content import (
    PaperContent,
    get_paper_content_by_paper_id,
)
from peptide_extractor.db.paper_metadata import (
    DatasetSubset,
    PaperMetadata,
    PaperStatus,
)
from peptide_extractor.db.peptide_entry import (
    PeptideExtractionResult,
    PeptideLLMEntry,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _seed_paper(session: Session, paper_id: int = 1) -> PaperMetadata:
    paper = PaperMetadata(
        id=paper_id,
        pmcid=f"PMC{1000 + paper_id}",
        pmid=str(2000 + paper_id),
        doi=f"10.1000/paper.{paper_id}",
        title=f"Paper {paper_id}",
        is_open_access=True,
        status=PaperStatus.USABLE,
        dataset_subset=DatasetSubset.SCRAPED,
    )
    session.add(paper)
    session.commit()
    session.refresh(paper)
    return paper


def test_get_paper_content_by_paper_id_filters_by_format(db_session: Session):
    _seed_paper(db_session, 1)
    xml_row = PaperContent(
        paper_id=1, format="xml", source="pmc", blob_url="https://blob/1.xml"
    )
    pdf_row = PaperContent(
        paper_id=1, format="pdf", source="pmc", blob_url="https://blob/1.pdf"
    )
    db_session.add_all([xml_row, pdf_row])
    db_session.commit()

    assert get_paper_content_by_paper_id(1, db_session).format == "xml"
    assert get_paper_content_by_paper_id(1, db_session, format="xml").format == "xml"
    assert get_paper_content_by_paper_id(1, db_session, format="pdf").format == "pdf"


def test_fetch_paper_fulltexts_pdf_does_not_skip_when_only_xml_exists(
    db_session: Session, monkeypatch
):
    _seed_paper(db_session, 1)
    db_session.add(
        PaperContent(
            paper_id=1, format="xml", source="pmc", blob_url="https://blob/1.xml"
        )
    )
    db_session.commit()

    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)

    mock_ncbi = MagicMock()
    mock_ncbi.fetch.return_value = (
        [
            PaperContent(
                paper_id=1,
                format="pdf",
                source="pmc",
                blob_url="https://blob/1.pdf",
            )
        ],
        [1],
        [],
    )

    success, fail, _ = pipeline.fetch_paper_fulltexts(
        session=db_session,
        ncbi_provider=mock_ncbi,
        open_alex_provider=None,
        skip_existing_content=True,
        content_format="pdf",
        log_to_file=False,
        id=1,
    )

    assert success == 1
    assert fail == 0
    mock_ncbi.fetch.assert_called_once()
    assert mock_ncbi.fetch.call_args.kwargs["format"] == "pdf"

    rows = db_session.exec(select(PaperContent).where(PaperContent.paper_id == 1)).all()
    assert sorted(r.format for r in rows) == ["pdf", "xml"]


def test_run_peptide_extraction_selects_requested_format(
    db_session: Session, monkeypatch
):
    _seed_paper(db_session, 1)
    db_session.add_all(
        [
            PaperContent(
                paper_id=1, format="xml", source="pmc", blob_url="https://blob/1.xml"
            ),
            PaperContent(
                paper_id=1, format="pdf", source="pmc", blob_url="https://blob/1.pdf"
            ),
        ]
    )
    db_session.commit()

    monkeypatch.setattr(
        PaperContent,
        "get_content",
        lambda self: b"%PDF-1.4 bytes" if self.format == "pdf" else "<article/>",
    )

    captured = []

    def fake_run_extraction(content, *args, **kwargs):
        captured.append(content)
        return PeptideExtractionResult(entries=[PeptideLLMEntry(sequence="KLVFF")])

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction)
    monkeypatch.setattr(
        pipeline,
        "extraction_ready_string_from_pdf",
        lambda pdf_bytes: f"PARSED_PDF:{len(pdf_bytes)}",
    )

    failed_ids, _ = pipeline.run_peptide_extraction(
        session=db_session,
        save=False,
        delay_seconds=0,
        content_format="pdf",
        log_to_file=False,
        id=1,
    )

    assert failed_ids == []
    assert captured == ["PARSED_PDF:14"]


def test_run_peptide_extraction_pass_pdf_directly_defaults_to_pdf_format(
    db_session: Session, monkeypatch
):
    _seed_paper(db_session, 1)
    _seed_paper(db_session, 2)
    # Paper 1 has both XML (first) and PDF; Paper 2 has only XML
    db_session.add_all(
        [
            PaperContent(
                paper_id=1, format="xml", source="pmc", blob_url="https://blob/1.xml"
            ),
            PaperContent(
                paper_id=1, format="pdf", source="pmc", blob_url="https://blob/1.pdf"
            ),
            PaperContent(
                paper_id=2, format="xml", source="pmc", blob_url="https://blob/2.xml"
            ),
        ]
    )
    db_session.commit()

    monkeypatch.setattr(
        PaperContent,
        "get_content",
        lambda self: b"%PDF-1.4 raw" if self.format == "pdf" else "<article/>",
    )

    captured = []

    def fake_run_extraction(content, *args, **kwargs):
        captured.append((content, kwargs.get("filename")))
        return PeptideExtractionResult(entries=[PeptideLLMEntry(sequence="KLVFF")])

    monkeypatch.setattr(pipeline, "run_extraction", fake_run_extraction)

    failed_ids, _ = pipeline.run_peptide_extraction(
        session=db_session,
        save=False,
        delay_seconds=0,
        pass_pdf_directly=True,
        log_to_file=False,
    )

    assert failed_ids == []
    # Paper 1 used PDF bytes; Paper 2 (XML-only) was skipped
    assert captured == [(b"%PDF-1.4 raw", "paper_1.pdf")]


def test_run_peptide_extraction_rejects_xml_with_pass_pdf_directly(db_session: Session):
    with pytest.raises(ValueError, match="pass_pdf_directly=True"):
        pipeline.run_peptide_extraction(
            session=db_session,
            content_format="xml",
            pass_pdf_directly=True,
            log_to_file=False,
        )


def test_cli_forwards_content_format_for_fetch_and_extraction(tmp_path, monkeypatch):
    mock_fetch = MagicMock(return_value=(1, 0, 111))
    mock_extract = MagicMock(return_value=([], 222))
    monkeypatch.setattr("peptide_extractor.cli.fetch_paper_fulltexts", mock_fetch)
    monkeypatch.setattr("peptide_extractor.cli.run_peptide_extraction", mock_extract)

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        """
steps:
  - fetch
  - extraction
fetch:
  run_identifier: 111
  content_format: pdf
extraction:
  run_identifier: 222
  content_format: pdf
  pass_pdf_directly: true
"""
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["run_pipeline", "-c", str(yaml_file)])
    assert result.exit_code == 0
    assert mock_fetch.call_args.kwargs["content_format"] == "pdf"
    assert mock_extract.call_args.kwargs["content_format"] == "pdf"
    assert mock_extract.call_args.kwargs["pass_pdf_directly"] is True
