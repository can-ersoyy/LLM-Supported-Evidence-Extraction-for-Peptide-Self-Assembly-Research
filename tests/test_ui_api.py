"""Tests for the ui/ backend, FastAPI server, and PipelineClient."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from sqlmodel import create_engine  # noqa: E402

from peptide_extractor.db import PaperMetadata, PaperStatus, init_db  # noqa: E402
from peptide_extractor.db.database import get_session_context  # noqa: E402
from ui import api  # noqa: E402
from ui.client import PipelineClient  # noqa: E402


@pytest.fixture()
def temp_db(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "test_ui.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    test_engine = create_engine(db_url, connect_args={"check_same_thread": False})
    monkeypatch.setattr("peptide_extractor.db.database.engine", test_engine)
    monkeypatch.setattr("peptide_extractor.db.database.DATABASE_URL", db_url)
    monkeypatch.setattr("ui.backend.DATABASE_URL", db_url)
    init_db(test_engine)

    with get_session_context(db_engine=test_engine) as s:
        s.add(
            PaperMetadata(
                doi="10.1000/test1",
                pmid="123456",
                title="Test Peptide Paper",
                abstract="Fmoc-FF forms hydrogels.",
                status=PaperStatus.USABLE,
                is_open_access=True,
            )
        )
        s.commit()

    return db_file, test_engine


def test_api_health_and_db_stats_and_records(temp_db):
    client = TestClient(api.app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}

    r_stats = client.get("/db/stats")
    assert r_stats.status_code == 200
    stats = r_stats.json()
    assert stats["tables"]["paper_metadata"]["total"] == 1
    assert stats["tables"]["paper_metadata"]["by_status"]["usable"] == 1

    r_rows = client.post(
        "/db/records/paper_metadata",
        json={"limit": 10, "offset": 0, "filters": {"pmid": "123456"}},
    )
    assert r_rows.status_code == 200
    data = r_rows.json()
    assert data["count"] == 1
    assert data["rows"][0]["title"] == "Test Peptide Paper"


def test_api_key_enforcement(monkeypatch, temp_db):
    monkeypatch.setenv("PEPTIDE_API_KEY", "secret-token")
    client = TestClient(api.app)

    assert client.get("/db/stats").status_code == 401
    assert (
        client.get("/db/stats", headers={"X-API-Key": "wrong"}).status_code == 401
    )
    assert (
        client.get("/db/stats", headers={"X-API-Key": "secret-token"}).status_code
        == 200
    )


def test_job_runner_captures_logs(temp_db):
    client = TestClient(api.app)
    resp = client.post("/jobs/step/deduplicate", json={"params": {"normalize": True}})
    assert resp.status_code == 200
    job_id = resp.json()["id"]

    for _ in range(30):
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert j["status"] == "completed"
    assert j["result"]["step"] == "deduplicate"
    assert any("Started 'step:deduplicate'" in line for line in j["logs"])


def test_pdf_ingestion_endpoint(temp_db):
    client = TestClient(api.app)
    with patch(
        "peptide_extractor.pipeline.ingest_manual_pdfs",
        return_value={
            "uploaded": [1],
            "skipped": [],
            "unmatched": [],
            "failed": [],
            "run_identifier": 999,
        },
    ) as mock_ingest:
        resp = client.post(
            "/pdfs/ingest",
            files=[("files", ("123456.pdf", b"%PDF-1.4 dummy", "application/pdf"))],
            data={"skip_existing_content": "true", "cache_locally": "false", "source": "manual"},
        )
        assert resp.status_code == 200
        job_id = resp.json()["id"]

        for _ in range(30):
            j = client.get(f"/jobs/{job_id}").json()
            if j["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)

        assert j["status"] == "completed"
        assert j["result"]["uploaded"] == [1]
        mock_ingest.assert_called_once()


def test_sync_db_subprocess_job(tmp_path: Path):
    script = tmp_path / "sync_db.py"
    script.write_text('print("syncing database ok")\n', encoding="utf-8")

    client = TestClient(api.app)
    resp = client.post("/sync-db/run", json={"script_path": str(script)})
    assert resp.status_code == 200
    job_id = resp.json()["id"]

    for _ in range(30):
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert j["status"] == "completed"
    assert any("syncing database ok" in line for line in j["logs"])


def test_db_download_and_upload(temp_db):
    db_file, _ = temp_db
    client = TestClient(api.app)

    dl = client.get("/db/download")
    assert dl.status_code == 200
    assert dl.content.startswith(b"SQLite format 3\x00")

    bad_up = client.post(
        "/db/upload",
        files={"file": ("bad.db", b"not-sqlite", "application/octet-stream")},
    )
    assert bad_up.status_code == 400

    good_up = client.post(
        "/db/upload",
        files={"file": ("backup.db", dl.content, "application/octet-stream")},
    )
    assert good_up.status_code == 200
    assert good_up.json()["status"] == "replaced"
    assert good_up.json()["backup_path"] is not None


def test_local_pipeline_client(temp_db):
    local_client = PipelineClient(mode="local")
    assert local_client.health()["status"] == "ok"
    stats = local_client.get_db_stats()
    assert stats["tables"]["paper_metadata"]["total"] == 1
    res = local_client.query_records("paper_metadata", limit=5)
    assert res["count"] == 1


def test_azure_db_snapshot_sync(tmp_path: Path, monkeypatch, temp_db):
    from types import SimpleNamespace
    from ui import backend

    src_db_file, _ = temp_db
    valid_bytes = src_db_file.read_bytes()

    target_db = tmp_path / "azure_synced.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{target_db}")
    monkeypatch.setattr("ui.backend.DATABASE_URL", f"sqlite:///{target_db}")
    monkeypatch.setenv("AZURE_DB_SAS_URL", "https://example.blob.core.windows.net/db.sqlite?sig=abc")
    monkeypatch.setenv("AZURE_DB_REFRESH_TTL_SECONDS", "3600")

    fake_resp = SimpleNamespace(content=valid_bytes, raise_for_status=lambda: None)
    with patch("ui.backend.requests.get", return_value=fake_resp) as mock_get:
        res1 = backend.sync_db_from_azure(force=False)
        assert res1["status"] == "downloaded"
        assert target_db.exists()
        assert mock_get.call_count == 1

        # Second call within TTL uses cached file
        res2 = backend.sync_db_from_azure(force=False)
        assert res2["status"] == "cached"
        assert mock_get.call_count == 1

        # Force=True re-downloads
        res3 = backend.sync_db_from_azure(force=True)
        assert res3["status"] == "downloaded"
        assert mock_get.call_count == 2


def test_azure_db_snapshot_sync_tar_gz(tmp_path: Path, monkeypatch, temp_db):
    import io
    import tarfile
    from types import SimpleNamespace
    from ui import backend

    src_db_file, _ = temp_db
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src_db_file, arcname="peptide-extractor.db")
    tar_gz_bytes = buf.getvalue()

    target_db = tmp_path / "peptide-extractor.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{target_db}")
    monkeypatch.setattr("ui.backend.DATABASE_URL", f"sqlite:///{target_db}")
    monkeypatch.setenv(
        "AZURE_DB_SAS_URL",
        "https://example.blob.core.windows.net/db-backups/peptide-extractor.db.tar.gz?sig=abc",
    )

    fake_resp = SimpleNamespace(content=tar_gz_bytes, raise_for_status=lambda: None)
    with patch("ui.backend.requests.get", return_value=fake_resp):
        res = backend.sync_db_from_azure(force=True)
        assert res["status"] == "downloaded"
        assert target_db.exists()
        assert target_db.read_bytes().startswith(b"SQLite format 3\x00")


def test_session_isolated_db_query(tmp_path: Path, temp_db):
    other_db = tmp_path / "session_other.db"
    other_engine = create_engine(
        f"sqlite:///{other_db}", connect_args={"check_same_thread": False}
    )
    init_db(other_engine)
    with get_session_context(db_engine=other_engine) as s:
        s.add(
            PaperMetadata(
                doi="10.1000/session-only",
                pmid="999999",
                title="Session Isolated Paper",
                abstract="Isolated DB",
                status=PaperStatus.DISCARD,
                is_open_access=False,
            )
        )
        s.commit()

    local_client = PipelineClient(mode="local")
    default_res = local_client.query_records("paper_metadata", limit=10)
    assert default_res["rows"][0]["pmid"] == "123456"

    session_res = local_client.query_records(
        "paper_metadata", limit=10, db_path=str(other_db)
    )
    assert session_res["count"] == 1
    assert session_res["rows"][0]["pmid"] == "999999"

    session_stats = local_client.get_db_stats(db_path=str(other_db))
    assert session_stats["tables"]["paper_metadata"]["by_status"]["discard"] == 1


def test_session_isolated_visualizations(tmp_path: Path, monkeypatch, temp_db):
    from ui import backend

    monkeypatch.setattr("ui.backend.get_project_root", lambda: tmp_path)
    local_client = PipelineClient(mode="local")

    job_a = local_client.run_step(
        "visualization",
        {
            "model": "paper_metadata",
            "operations": ["paper_status_counts"],
            "session_id": "sessA",
            "log_to_file": False,
        },
    )
    for _ in range(30):
        j = local_client.get_job(job_a["id"])
        if j["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert j["status"] == "completed"
    items_a = local_client.list_visualizations(session_id="sessA")
    items_b = local_client.list_visualizations(session_id="sessB")
    assert len(items_a) == 1
    assert len(items_b) == 0
    assert backend.read_visualization_image(items_a[0]["relative_path"]).startswith(
        b"\x89PNG"
    )


def test_search_peptides_and_links(temp_db):
    from peptide_extractor.db import ExtractionMode, PeptideEntry

    _, test_engine = temp_db
    with get_session_context(db_engine=test_engine) as s:
        s.add(
            PeptideEntry(
                paper_id=1,
                sequence="KLVFF",
                length=5,
                type="fiber",
                n_terminal="Fmoc",
                ph=7.4,
                temperature=25.0,
                solvent="PBS",
                detection_method="TEM; ThT",
                pmid="123456",
                extraction_mode=ExtractionMode.LLM,
                run_identifier=42,
            )
        )
        s.commit()

    client = TestClient(api.app)
    resp = client.post(
        "/db/peptides/search",
        json={"sequence": "klv", "morphology": "fiber", "solvent": "pbs"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    row = data["rows"][0]
    assert row["sequence"] == "KLVFF"
    assert row["pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/123456/"
    assert row["doi_url"] == "https://doi.org/10.1000/test1"


def test_paper_search_and_detail(temp_db):
    from peptide_extractor.db import ExtractionEvaluation, ExtractionMode, PaperRelevance, PeptideEntry

    _, test_engine = temp_db
    with get_session_context(db_engine=test_engine) as s:
        s.add(
            PaperRelevance(
                paper_id=1,
                is_relevant=True,
                confidence=0.95,
                explanation="Experimental hydrogel reported",
                provider="google",
                model_name="gemini-3.5-flash-lite",
                run_identifier=10,
            )
        )
        entry = PeptideEntry(
            paper_id=1,
            sequence="FF",
            length=2,
            type="hydrogel",
            evidence_span='{"sequence": "[sec-1/p-1]"}',
            pmid="123456",
            extraction_mode=ExtractionMode.LLM,
            run_identifier=42,
        )
        s.add(entry)
        s.commit()
        s.refresh(entry)
        s.add(
            ExtractionEvaluation(
                peptide_entry_id=entry.id,
                groundness_rate=1.0,
                faithfulness_rate=0.9,
                completeness_rate=0.8,
                run_identifier=42,
            )
        )
        s.commit()

    client = TestClient(api.app)
    s_resp = client.post("/db/papers/search", json={"query": "123456"})
    assert s_resp.status_code == 200
    assert s_resp.json()["count"] == 1

    d_resp = client.get("/db/papers/detail", params={"pmid": "123456"})
    assert d_resp.status_code == 200
    detail = d_resp.json()
    assert detail["paper"]["title"] == "Test Peptide Paper"
    assert len(detail["relevance"]) == 1
    assert len(detail["peptide_entries"]) == 1
    assert detail["peptide_entries"][0]["evidence_span_parsed"] == {
        "sequence": "[sec-1/p-1]"
    }
    assert len(detail["evaluations"]) == 1


def test_run_comparison_and_benchmark(temp_db):
    from peptide_extractor.db import ExtractionMode, PeptideEntry

    _, test_engine = temp_db
    with get_session_context(db_engine=test_engine) as s:
        s.add(
            PeptideEntry(
                paper_id=1,
                sequence="FF",
                length=2,
                type="hydrogel",
                pmid="123456",
                extraction_mode=ExtractionMode.MANUAL,
                run_identifier=0,
            )
        )
        s.add(
            PeptideEntry(
                paper_id=1,
                sequence="FF",
                length=2,
                type="hydrogel",
                pmid="123456",
                extraction_mode=ExtractionMode.LLM,
                model="gemini-3.5-flash-lite",
                run_identifier=42,
            )
        )
        s.commit()

    client = TestClient(api.app)
    s_resp = client.get("/db/runs/summary")
    assert s_resp.status_code == 200
    assert s_resp.json()["count"] == 2

    b_resp = client.get(
        "/db/runs/benchmark",
        params={"llm_run_identifier": 42, "manual_run_identifier": 0},
    )
    assert b_resp.status_code == 200
    bm = b_resp.json()
    assert bm["sequence_f1"] == 1.0


def test_overview_charts_data(temp_db):
    from peptide_extractor.db import ExtractionMode, PaperRelevance, PeptideEntry

    _, test_engine = temp_db
    with get_session_context(db_engine=test_engine) as s:
        s.add(
            PeptideEntry(
                paper_id=1,
                sequence="FF",
                length=2,
                type="hydrogel",
                pmid="123456",
                extraction_mode=ExtractionMode.LLM,
                run_identifier=42,
            )
        )
        s.add(
            PaperRelevance(
                paper_id=1,
                is_relevant=True,
                confidence=0.92,
                provider="google",
                model_name="gemini-3.5-flash-lite",
                run_identifier=10,
            )
        )
        s.commit()

    client = TestClient(api.app)
    resp = client.get("/db/charts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["morphology_counts"][0]["morphology"] == "hydrogel"
    assert data["length_counts"][0]["length"] == 2
    assert data["relevance_confidence"][0]["outcome"] == "relevant"


def test_pipeline_demo_end_to_end(tmp_path: Path):
    from ui import backend
    from ui.demo import init_demo_db, reset_demo_db, submit_demo_step

    demo_db = str(tmp_path / "demo.db")
    init_demo_db(demo_db)

    job = submit_demo_step("all", demo_db)
    for _ in range(50):
        j = backend.job_manager.get_job(job["id"])
        if j["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert j["status"] == "completed"
    stats = backend.get_db_stats(db_path=demo_db)
    tables = stats["tables"]
    assert tables["paper_metadata"]["total"] == 4
    assert tables["paper_metadata"]["by_status"]["usable"] == 3
    assert tables["paper_metadata"]["by_status"]["discard"] == 1
    assert tables["paper_relevance"]["total"] == 4
    assert tables["paper_content"]["total"] == 3
    assert tables["peptide_entry"]["total"] == 4  # 3 LLM + 1 golden manual
    assert tables["extraction_evaluation"]["total"] == 3

    reset_demo_db(demo_db)
    stats_after = backend.get_db_stats(db_path=demo_db)
    assert stats_after["tables"]["paper_metadata"]["total"] == 0






