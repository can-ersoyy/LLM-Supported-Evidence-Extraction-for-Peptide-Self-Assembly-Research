from pathlib import Path
from unittest.mock import patch

import pytest
from sqlmodel import SQLModel, Session, create_engine, select

from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.db.paper_metadata import DatasetSubset, PaperMetadata, PaperStatus
from peptide_extractor.pipeline import ingest_manual_pdfs


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _add_paper(
    session: Session,
    *,
    title: str = "Test Paper",
    doi: str | None = None,
    pmid: str | None = None,
    status: PaperStatus = PaperStatus.NOT_PROCESSED,
) -> PaperMetadata:
    paper = PaperMetadata(
        title=title,
        abstract="Some abstract",
        doi=doi,
        pmid=pmid,
        status=status,
        is_open_access=True,
        dataset_subset=DatasetSubset.SCRAPED,
    )
    session.add(paper)
    session.commit()
    session.refresh(paper)
    return paper


def test_ingest_manual_pdfs_matches_by_pmid_only(db_session: Session, tmp_path: Path):
    p_with_doi = _add_paper(
        db_session, title="Paper With DOI", pmid="99887766", doi="10.1021/jacs.1c01234"
    )
    p_no_doi = _add_paper(db_session, title="Paper No DOI", pmid="11223344", doi=None)

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    cache_dir = tmp_path / "cache"

    (pdf_dir / "99887766.pdf").write_bytes(b"%PDF-1.4 with doi")
    (pdf_dir / "11223344.pdf").write_bytes(b"%PDF-1.4 no doi")
    # Non-PMID filenames should be unmatched
    (pdf_dir / f"{p_with_doi.id}.pdf").write_bytes(b"%PDF-1.4 by id")
    (pdf_dir / "10_1021_jacs_1c01234.pdf").write_bytes(b"%PDF-1.4 by doi")

    def fake_upload_pdf(pdf_content: bytes, doi: str, **kwargs) -> str:
        return f"https://example.blob.core.windows.net/papers/pdf/{doi}.pdf"

    with patch("peptide_extractor.pipeline.upload_pdf", side_effect=fake_upload_pdf):
        result = ingest_manual_pdfs(
            pdf_dir=pdf_dir,
            session=db_session,
            cache_dir=cache_dir,
            run_identifier=42,
            log_to_file=False,
        )

    assert sorted(result["uploaded"]) == sorted([p_with_doi.id, p_no_doi.id])
    assert result["skipped"] == []
    assert sorted(result["unmatched"]) == sorted(
        [f"{p_with_doi.id}.pdf", "10_1021_jacs_1c01234.pdf"]
    )
    assert result["failed"] == []
    assert result["run_identifier"] == 42

    contents = db_session.exec(select(PaperContent)).all()
    assert len(contents) == 2
    for content in contents:
        assert content.format == "pdf"
        assert content.source == "manual"
        assert content.run_identifier == 42
        assert content.blob_url.endswith(".pdf")

    assert (cache_dir / "pdf" / "10_1021_jacs_1c01234.pdf").read_bytes() == b"%PDF-1.4 with doi"
    assert (cache_dir / "pdf" / f"paper_{p_no_doi.id}.pdf").read_bytes() == b"%PDF-1.4 no doi"

    db_session.refresh(p_with_doi)


def test_ingest_manual_pdfs_skip_and_overwrite_existing(
    db_session: Session, tmp_path: Path
):
    paper = _add_paper(
        db_session, title="Existing Content", pmid="55667788", doi="10.1000/existing"
    )
    existing = PaperContent(
        paper_id=paper.id,
        format="pdf",
        source="openalex",
        blob_url="https://example.blob.core.windows.net/papers/pdf/old.pdf",
    )
    db_session.add(existing)
    db_session.commit()

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "55667788.pdf").write_bytes(b"%PDF-1.4 new")

    with patch(
        "peptide_extractor.pipeline.upload_pdf",
        return_value="https://example.blob.core.windows.net/papers/pdf/10_1000_existing.pdf",
    ) as mock_upload:
        res_skip = ingest_manual_pdfs(
            pdf_dir=pdf_dir,
            session=db_session,
            skip_existing_content=True,
            cache_locally=False,
            log_to_file=False,
        )
        assert res_skip["uploaded"] == []
        assert res_skip["skipped"] == [paper.id]
        mock_upload.assert_not_called()

        res_overwrite = ingest_manual_pdfs(
            pdf_dir=pdf_dir,
            session=db_session,
            skip_existing_content=False,
            cache_locally=False,
            run_identifier=99,
            log_to_file=False,
        )
        assert res_overwrite["uploaded"] == [paper.id]
        mock_upload.assert_called_once()

    rows = db_session.exec(select(PaperContent)).all()
    assert len(rows) == 1
    assert rows[0].source == "manual"
    assert rows[0].run_identifier == 99
    assert rows[0].blob_url.endswith("10_1000_existing.pdf")


def test_ingest_manual_pdfs_empty_file_recorded_as_failed(
    db_session: Session, tmp_path: Path
):
    _add_paper(db_session, title="Empty PDF", pmid="44556677")
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "44556677.pdf").write_bytes(b"")

    with patch("peptide_extractor.pipeline.upload_pdf") as mock_upload:
        result = ingest_manual_pdfs(
            pdf_dir=pdf_dir,
            session=db_session,
            cache_locally=False,
            log_to_file=False,
        )

    assert result["uploaded"] == []
    assert result["failed"] == ["44556677.pdf"]
    mock_upload.assert_not_called()
