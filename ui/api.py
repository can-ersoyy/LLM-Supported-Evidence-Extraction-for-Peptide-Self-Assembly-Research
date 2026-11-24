"""FastAPI server exposing peptide_extractor pipeline, DB, PDF ingestion, and logs."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ui import backend

app = FastAPI(
    title="Peptide Extractor API",
    version="0.1.0",
    description="Remote API server for running peptide_extractor pipelines, PDF ingestion, DB sync, and viewing results/logs.",
)


def _verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("PEPTIDE_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class StepRunRequest(BaseModel):
    params: Dict[str, Any] = {}


class ConfigPipelineRequest(BaseModel):
    config: Dict[str, Any]


class SyncDbRequest(BaseModel):
    script_path: Optional[str] = None
    extra_args: Optional[List[str]] = None


class QueryRecordsRequest(BaseModel):
    limit: int = 100
    offset: int = 0
    filters: Dict[str, Any] = {}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/db/stats", dependencies=[Depends(_verify_api_key)])
def db_stats() -> Dict[str, Any]:
    return backend.get_db_stats()


@app.post("/db/records/{table_name}", dependencies=[Depends(_verify_api_key)])
def db_records(table_name: str, req: QueryRecordsRequest) -> Dict[str, Any]:
    try:
        return backend.query_records(
            table_name=table_name,
            limit=req.limit,
            offset=req.offset,
            filters=req.filters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/db/peptides/search", dependencies=[Depends(_verify_api_key)])
def db_peptides_search(req: Dict[str, Any]) -> Dict[str, Any]:
    return backend.search_peptides(**req)


@app.post("/db/papers/search", dependencies=[Depends(_verify_api_key)])
def db_papers_search(req: Dict[str, Any]) -> Dict[str, Any]:
    return backend.search_papers(**req)


@app.get("/db/papers/detail", dependencies=[Depends(_verify_api_key)])
def db_papers_detail(
    paper_id: Optional[int] = Query(default=None),
    pmid: Optional[str] = Query(default=None),
    doi: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    try:
        return backend.get_paper_detail(paper_id=paper_id, pmid=pmid, doi=doi)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/db/runs/summary", dependencies=[Depends(_verify_api_key)])
def db_runs_summary() -> Dict[str, Any]:
    return backend.get_run_comparison()


@app.get("/db/runs/benchmark", dependencies=[Depends(_verify_api_key)])
def db_runs_benchmark(
    llm_run_identifier: int = Query(...),
    manual_run_identifier: int = Query(0),
) -> Dict[str, Any]:
    return backend.run_golden_benchmark(
        llm_run_identifier=llm_run_identifier,
        manual_run_identifier=manual_run_identifier,
    )


@app.get("/db/charts", dependencies=[Depends(_verify_api_key)])
def db_charts() -> Dict[str, Any]:
    return backend.get_overview_charts_data()


@app.post("/db/peptides/phase", dependencies=[Depends(_verify_api_key)])
def db_peptides_phase(req: Dict[str, Any]) -> Dict[str, Any]:
    return backend.get_condition_phase_data(**req)


@app.get("/db/runs/disagreements", dependencies=[Depends(_verify_api_key)])
def db_runs_disagreements(
    llm_run_identifier: int = Query(...),
    manual_run_identifier: int = Query(0),
) -> Dict[str, Any]:
    return backend.get_golden_disagreements(
        llm_run_identifier=llm_run_identifier,
        manual_run_identifier=manual_run_identifier,
    )


@app.get("/db/download", dependencies=[Depends(_verify_api_key)])
def db_download() -> FileResponse:
    db_path = backend.get_db_file_path()
    if db_path is None or not db_path.exists():
        raise HTTPException(status_code=404, detail="SQLite database file not found.")
    return FileResponse(
        path=str(db_path),
        filename=db_path.name,
        media_type="application/octet-stream",
    )


@app.post("/db/upload", dependencies=[Depends(_verify_api_key)])
async def db_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    raw_bytes = await file.read()
    try:
        return backend.replace_db_file(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/db/init", dependencies=[Depends(_verify_api_key)])
def db_init() -> Dict[str, Any]:
    return backend.run_init_db()


@app.post("/db/alembic-upgrade", dependencies=[Depends(_verify_api_key)])
def db_alembic_upgrade() -> Dict[str, Any]:
    return backend.run_alembic_upgrade_job()


@app.get("/sync-db/scripts", dependencies=[Depends(_verify_api_key)])
def list_sync_db_scripts() -> Dict[str, List[str]]:
    return {"scripts": backend.find_sync_db_scripts()}


@app.post("/sync-db/run", dependencies=[Depends(_verify_api_key)])
def run_sync_db(req: SyncDbRequest) -> Dict[str, Any]:
    try:
        return backend.run_sync_db_job(
            script_path=req.script_path,
            extra_args=req.extra_args,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/jobs/step/{step_name}", dependencies=[Depends(_verify_api_key)])
def run_step(step_name: str, req: StepRunRequest) -> Dict[str, Any]:
    try:
        return backend.run_step_job(step_name, req.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jobs/pipeline", dependencies=[Depends(_verify_api_key)])
def run_pipeline(req: ConfigPipelineRequest) -> Dict[str, Any]:
    return backend.run_config_pipeline_job(req.config)


@app.post("/pdfs/ingest", dependencies=[Depends(_verify_api_key)])
async def ingest_pdfs(
    files: List[UploadFile] = File(...),
    skip_existing_content: bool = Form(True),
    cache_locally: bool = Form(True),
    source: str = Form("manual"),
) -> Dict[str, Any]:
    payload = []
    for f in files:
        content = await f.read()
        payload.append((f.filename or "upload.pdf", content))
    try:
        return backend.ingest_uploaded_pdfs_job(
            files=payload,
            skip_existing_content=skip_existing_content,
            cache_locally=cache_locally,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs", dependencies=[Depends(_verify_api_key)])
def list_jobs() -> Dict[str, List[Dict[str, Any]]]:
    return {"jobs": backend.job_manager.list_jobs()}


@app.get("/jobs/{job_id}", dependencies=[Depends(_verify_api_key)])
def get_job(
    job_id: str,
    log_offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    job = backend.job_manager.get_job(job_id, log_offset=log_offset)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/logs/files", dependencies=[Depends(_verify_api_key)])
def list_log_files() -> Dict[str, List[Dict[str, Any]]]:
    return {"files": backend.list_log_files()}


@app.get("/logs/files/{step}/{filename}", dependencies=[Depends(_verify_api_key)])
def read_log_file(
    step: str,
    filename: str,
    tail: int = Query(500, ge=0),
) -> Dict[str, Any]:
    try:
        return backend.read_log_file(step=step, filename=filename, tail=tail)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/visualizations/catalog", dependencies=[Depends(_verify_api_key)])
def visualization_catalog() -> Dict[str, Any]:
    return backend.get_visualization_catalog()


@app.get("/visualizations", dependencies=[Depends(_verify_api_key)])
def list_visualizations(
    session_id: Optional[str] = Query(default=None),
) -> Dict[str, List[Dict[str, Any]]]:
    return {"items": backend.list_visualizations(session_id=session_id)}


@app.get("/visualizations/image", dependencies=[Depends(_verify_api_key)])
def get_visualization_image(path: str = Query(...)) -> Response:
    try:
        data = backend.read_visualization_image(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
    return Response(content=data, media_type=media_type)
