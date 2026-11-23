"""Backend service layer and job runner for the Peptide Extractor UI & API."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests
from sqlalchemy import func
from sqlmodel import select

from peptide_extractor.db import (
    DATABASE_URL,
    DatasetSubset,
    ExtractionEvaluation,
    ExtractionMode,
    PaperContent,
    PaperMetadata,
    PaperRelevance,
    PaperStatus,
    PeptideEntry,
    get_engine,
    get_records,
    get_session_context,
    init_db,
)
from peptide_extractor.db.database import DEFAULT_DB_PATH, _resolve_model
from peptide_extractor.env import load_env
from peptide_extractor.logging_utils import get_default_log_dir, get_project_root
from peptide_extractor.visualization import (
    VISUALIZATION_REGISTRY,
    VisualizationOperation,
)

load_env()

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRecord:
    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, line: str) -> None:
        with self._lock:
            self.logs.append(line)

    def to_dict(self, log_offset: int = 0) -> Dict[str, Any]:
        with self._lock:
            sliced_logs = self.logs[log_offset:]
            total_logs = len(self.logs)
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "logs": sliced_logs,
            "log_offset": log_offset,
            "total_logs": total_logs,
            "result": _serialize_value(self.result),
            "error": self.error,
        }


class _ThreadLogHandler(logging.Handler):
    """Logging handler that captures log records emitted by a specific thread."""

    def __init__(self, thread_id: int, job: JobRecord) -> None:
        super().__init__(level=logging.INFO)
        self.thread_id = thread_id
        self.job = job
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            msg = self.format(record)
            self.job.append_log(msg)
        except Exception:
            self.handleError(record)


class JobManager:
    """In-memory background job runner with per-job log capture."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        job = JobRecord(id=job_id, name=name, status=JobStatus.PENDING)
        with self._lock:
            self._jobs[job_id] = job

        def _worker() -> None:
            tid = threading.get_ident()
            handler = _ThreadLogHandler(tid, job)
            root_logger = logging.getLogger()
            if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
                root_logger.setLevel(logging.INFO)
            root_logger.addHandler(handler)

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.append_log(f"[job:{job_id}] Started '{name}'")
            try:
                res = fn(*args, **kwargs)
                job.result = res
                job.status = JobStatus.COMPLETED
                job.append_log(f"[job:{job_id}] Completed successfully.")
            except Exception as exc:
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = JobStatus.FAILED
                logger.exception(f"Job '{name}' failed: {exc}")
                job.append_log(f"[job:{job_id}] FAILED: {job.error}")
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()
                root_logger.removeHandler(handler)

        thread = threading.Thread(target=_worker, name=f"job-{job_id}", daemon=True)
        thread.start()
        return job

    def submit_subprocess(
        self,
        name: str,
        cmd: Sequence[str],
        cwd: Optional[Path] = None,
    ) -> JobRecord:
        def _run_proc() -> Dict[str, Any]:
            target_cwd = str(cwd or get_project_root())
            logger.info(f"Running command: {' '.join(cmd)} (cwd={target_cwd})")
            proc = subprocess.Popen(
                list(cmd),
                cwd=target_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                logger.info(line.rstrip("\n"))
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"Command exited with status {returncode}")
            return {"returncode": returncode, "command": list(cmd)}

        return self.submit(name, _run_proc)

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [
            {
                "id": j.id,
                "name": j.name,
                "status": j.status.value,
                "created_at": j.created_at,
                "started_at": j.started_at,
                "finished_at": j.finished_at,
                "total_logs": len(j.logs),
                "error": j.error,
            }
            for j in jobs
        ]

    def get_job(self, job_id: str, log_offset: int = 0) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.to_dict(log_offset=log_offset)


job_manager = JobManager()


def _serialize_value(val: Any) -> Any:
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_serialize_value(v) for v in val]
    if hasattr(val, "model_dump"):
        return _serialize_value(val.model_dump())
    return str(val)


def get_db_file_path() -> Optional[Path]:
    """Return the local SQLite database path if DATABASE_URL points to SQLite."""
    url = os.getenv("DATABASE_URL", DATABASE_URL)
    if url.startswith("sqlite:///"):
        raw = url.replace("sqlite:///", "", 1)
        if raw and raw != ":memory:":
            return Path(raw).resolve()
    return DEFAULT_DB_PATH.resolve()


def _get_target_engine(db_path: Optional[str] = None):
    if db_path:
        return get_engine(f"sqlite:///{Path(db_path).resolve()}")
    return None


def is_azure_db_configured() -> bool:
    """Return True if Azure DB download env vars are configured."""
    return bool(
        os.getenv("AZURE_DB_SAS_URL")
        or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )


def sync_db_from_azure(force: bool = False) -> Dict[str, Any]:
    """Download the SQLite database snapshot from Azure Blob Storage if needed.

    Matches data/sync_db.py conventions:
    - Container: AZURE_CONTAINER_NAME (or AZURE_STORAGE_CONTAINER, default 'db-backups')
    - Blob: AZURE_BLOB_NAME (or AZURE_DB_BLOB_NAME, default 'peptide-extractor.db.tar.gz')
    - Or direct HTTPS read-only SAS URL via AZURE_DB_SAS_URL.
    Automatically extracts .tar.gz archives or writes raw SQLite .db files.
    """
    import tarfile

    sas_url = os.getenv("AZURE_DB_SAS_URL")
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not sas_url and not conn_str:
        return {"status": "skipped", "reason": "Azure DB source not configured"}

    db_path = get_db_file_path()
    if db_path is None:
        raise ValueError("Azure DB snapshot sync requires a SQLite DATABASE_URL.")

    ttl = int(os.getenv("AZURE_DB_REFRESH_TTL_SECONDS", "3600"))
    if not force and db_path.exists() and db_path.stat().st_size > 0:
        age = time.time() - db_path.stat().st_mtime
        if ttl <= 0 or age < ttl:
            return {
                "status": "cached",
                "db_path": str(db_path),
                "age_seconds": round(age, 1),
            }

    if sas_url:
        resp = requests.get(sas_url, timeout=120)
        resp.raise_for_status()
        raw_bytes = resp.content
        source_desc = "AZURE_DB_SAS_URL"
        blob_name = sas_url.split("?", 1)[0].rsplit("/", 1)[-1]
    else:
        from peptide_extractor.azure_storage import get_blob_service_client

        container = (
            os.getenv("AZURE_CONTAINER_NAME")
            or os.getenv("AZURE_STORAGE_CONTAINER")
            or "db-backups"
        )
        blob_name = (
            os.getenv("AZURE_BLOB_NAME")
            or os.getenv("AZURE_DB_BLOB_NAME")
            or f"{db_path.name}.tar.gz"
        )
        service = get_blob_service_client(conn_str)
        blob_client = service.get_container_client(container).get_blob_client(
            blob_name
        )
        raw_bytes = blob_client.download_blob().readall()
        source_desc = f"{container}/{blob_name}"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".db.azure_tmp")

    # Handle .tar.gz archives (sync_db.py format) or raw SQLite .db bytes
    is_gzip = raw_bytes[:2] == b"\x1f\x8b" or blob_name.endswith(".tar.gz")
    if is_gzip:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "db.tar.gz"
            archive_path.write_bytes(raw_bytes)
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            extracted = [p for p in extract_dir.rglob("*") if p.is_file()]
            if not extracted:
                raise ValueError(f"Archive from {source_desc} contains no files.")
            target_file = next(
                (p for p in extracted if p.name == db_path.name),
                extracted[0],
            )
            db_bytes = target_file.read_bytes()
    else:
        db_bytes = raw_bytes

    if not db_bytes.startswith(b"SQLite format 3\x00"):
        raise ValueError(
            f"Downloaded file from {source_desc} is not a valid SQLite 3 database."
        )

    tmp_path.write_bytes(db_bytes)
    os.replace(tmp_path, db_path)

    from peptide_extractor.db import database as db_mod

    db_mod.engine.dispose()

    return {
        "status": "downloaded",
        "source": source_desc,
        "db_path": str(db_path),
        "size_bytes": db_path.stat().st_size,
    }


def get_db_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Return summary counts across all database tables."""
    target_path = Path(db_path).resolve() if db_path else get_db_file_path()
    stats: Dict[str, Any] = {
        "database_url": (
            f"sqlite:///{target_path}"
            if db_path
            else os.getenv("DATABASE_URL", DATABASE_URL)
        ),
        "db_path": str(target_path) if target_path else None,
        "db_exists": bool(target_path and target_path.exists()),
        "db_size_bytes": (
            target_path.stat().st_size
            if target_path and target_path.exists()
            else 0
        ),
        "tables": {},
    }

    target_engine = _get_target_engine(db_path)
    init_db(target_engine)
    with get_session_context(db_engine=target_engine) as session:
        # PaperMetadata
        pm_total = session.exec(select(func.count()).select_from(PaperMetadata)).one()
        pm_status = {
            s.value: session.exec(
                select(func.count())
                .select_from(PaperMetadata)
                .where(PaperMetadata.status == s)
            ).one()
            for s in PaperStatus
        }
        pm_subset = {
            sub.value: session.exec(
                select(func.count())
                .select_from(PaperMetadata)
                .where(PaperMetadata.dataset_subset == sub)
            ).one()
            for sub in DatasetSubset
        }
        stats["tables"]["paper_metadata"] = {
            "total": pm_total,
            "by_status": pm_status,
            "by_subset": pm_subset,
        }

        # PaperContent
        pc_total = session.exec(select(func.count()).select_from(PaperContent)).one()
        pc_xml = session.exec(
            select(func.count())
            .select_from(PaperContent)
            .where(PaperContent.format == "xml")
        ).one()
        pc_pdf = session.exec(
            select(func.count())
            .select_from(PaperContent)
            .where(PaperContent.format == "pdf")
        ).one()
        stats["tables"]["paper_content"] = {
            "total": pc_total,
            "by_format": {"xml": pc_xml, "pdf": pc_pdf},
        }

        # PaperRelevance
        pr_total = session.exec(select(func.count()).select_from(PaperRelevance)).one()
        stats["tables"]["paper_relevance"] = {"total": pr_total}

        # PeptideEntry
        pe_total = session.exec(select(func.count()).select_from(PeptideEntry)).one()
        pe_mode = {
            m.value: session.exec(
                select(func.count())
                .select_from(PeptideEntry)
                .where(PeptideEntry.extraction_mode == m)
            ).one()
            for m in ExtractionMode
        }
        stats["tables"]["peptide_entry"] = {
            "total": pe_total,
            "by_extraction_mode": pe_mode,
        }

        # ExtractionEvaluation
        ee_total = session.exec(
            select(func.count()).select_from(ExtractionEvaluation)
        ).one()
        stats["tables"]["extraction_evaluation"] = {"total": ee_total}

    return stats


def query_records(
    table_name: str,
    limit: int = 100,
    offset: int = 0,
    filters: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Query records from a DB table and serialize them for UI/API consumption."""
    target_engine = _get_target_engine(db_path)
    init_db(target_engine)
    model = _resolve_model(table_name)
    clean_filters: Dict[str, Any] = {}
    for k, v in (filters or {}).items():
        if v is None or v == "":
            continue
        # Coerce integer columns when passed as strings
        if k in {"id", "paper_id", "run_identifier", "peptide_entry_id", "length"}:
            if isinstance(v, str):
                if "," in v:
                    clean_filters[k] = [int(x.strip()) for x in v.split(",") if x.strip()]
                else:
                    clean_filters[k] = int(v)
            else:
                clean_filters[k] = v
        else:
            clean_filters[k] = v

    with get_session_context(db_engine=target_engine) as session:
        records = get_records(
            model,
            session=session,
            limit=limit,
            offset=offset,
            **clean_filters,
        )
        rows = [_serialize_value(r.model_dump()) for r in records]
    return {
        "table": table_name,
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "rows": rows,
    }


def _parse_id_list(val: Any) -> Any:
    if val is None or val == "":
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        parts = [int(x.strip()) for x in val.split(",") if x.strip()]
        if len(parts) == 1:
            return parts[0]
        return parts or None
    return val


def _build_filter_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if params.get("limit") not in (None, "", 0):
        kwargs["limit"] = int(params["limit"])
    if params.get("offset") not in (None, ""):
        off = int(params["offset"])
        if off > 0:
            kwargs["offset"] = off
    if params.get("status"):
        kwargs["status"] = params["status"]
    if params.get("dataset_subset"):
        kwargs["dataset_subset"] = params["dataset_subset"]
    if params.get("run_identifier") not in (None, ""):
        kwargs["run_identifier"] = int(params["run_identifier"])
    if params.get("paper_id") not in (None, ""):
        kwargs["paper_id"] = _parse_id_list(params["paper_id"])
    if params.get("id") not in (None, ""):
        parsed = _parse_id_list(params["id"])
        if parsed is not None:
            kwargs["id"] = parsed
    return kwargs


def _execute_step(step_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    step = step_name.strip().lower()
    run_id = params.get("run_identifier")
    run_identifier = int(run_id) if run_id not in (None, "") else int(time.time())
    log_to_file = bool(params.get("log_to_file", True))

    if step == "visualization":
        from peptide_extractor.logging_utils import step_logger
        from peptide_extractor.visualization import visualize

        filter_kwargs = _build_filter_kwargs(params)
        ops = params.get("operations") or []
        db_path = params.get("db_path")
        target_engine = _get_target_engine(db_path)
        output_dir = (
            Path(params["output_dir"])
            if params.get("output_dir")
            else get_project_root() / "data" / "diagrams" / str(run_identifier)
        )
        limit = filter_kwargs.pop("limit", None)
        offset = filter_kwargs.pop("offset", 0)

        with step_logger("visualization", run_identifier, enabled=log_to_file):
            init_db(target_engine)
            with get_session_context(db_engine=target_engine) as session:
                results = visualize(
                    operations=ops,
                    model=params.get("model", "peptide_entry"),
                    session=session,
                    limit=limit,
                    offset=offset,
                    output_dir=output_dir,
                    save=bool(params.get("save", True)),
                    show=False,
                    file_format=params.get("file_format", "png"),
                    dpi=int(params.get("dpi", 150)),
                    **filter_kwargs,
                )
        return {
            "step": "visualization",
            "run_identifier": run_identifier,
            "diagrams": [
                {
                    "operation": op.value,
                    "record_count": res.record_count,
                    "output_path": str(res.output_path) if res.output_path else None,
                }
                for op, res in results.items()
            ],
        }

    from peptide_extractor.pipeline import (
        NCBI_SEARCH_QUERIES_V3,
        NCBI_SEARCH_QUERY_V1,
        NCBI_SEARCH_QUERY_V2,
        SCOPUS_SEARCH_QUERY_V1,
        deduplicate_by_doi,
        fetch_paper_fulltexts,
        find_abstracts_by_doi,
        remove_empty_abstract_title_papers,
        retrieve_scopus_by_query,
        retrive_ncbi_by_query,
        run_bulk_relevance_screening,
        run_evaluation,
        run_peptide_extraction,
    )
    from peptide_extractor.providers import (
        NCBIProvider,
        OpenAlexProvider,
        ScopusProvider,
    )

    ncbi_api_key = os.getenv("NCBI_API_KEY")
    open_alex_api_key = os.getenv("OPEN_ALEX_API_KEY")
    scopus_api_key = os.getenv("SCOPUS_API_KEY")

    if step == "search":
        source = params.get("source", "ncbi").lower()
        if source == "scopus":
            provider = ScopusProvider(api_key=scopus_api_key)
            query = params.get("query") or SCOPUS_SEARCH_QUERY_V1
            rid = retrieve_scopus_by_query(
                provider,
                query=query,
                save=bool(params.get("save", True)),
                run_identifier=run_identifier,
                log_to_file=log_to_file,
            )
            return {"step": "search", "source": "scopus", "run_identifier": rid}

        preset = params.get("preset", "custom")
        queries = params.get("queries") or []
        if preset == "V1":
            queries = [NCBI_SEARCH_QUERY_V1]
        elif preset == "V2":
            queries = [NCBI_SEARCH_QUERY_V2]
        elif preset == "V3":
            queries = list(NCBI_SEARCH_QUERIES_V3)
        elif isinstance(queries, str):
            queries = [q.strip() for q in queries.splitlines() if q.strip()]
        if not queries:
            queries = [NCBI_SEARCH_QUERY_V1]

        ncbi_provider = NCBIProvider(
            api_key=ncbi_api_key,
            batch_size=int(params.get("batch_size", 100)),
        )
        rid = retrive_ncbi_by_query(
            ncbi_provider,
            db=params.get("db", "pmc"),
            query_list=queries,
            save=bool(params.get("save", True)),
            run_identifier=run_identifier,
            log_to_file=log_to_file,
        )
        return {"step": "search", "source": "ncbi", "run_identifier": rid}

    if step == "deduplicate":
        removed = deduplicate_by_doi(
            normalize=bool(params.get("normalize", True)),
            remove_empty=bool(params.get("remove_empty", True)),
        )
        return {"step": "deduplicate", "removed_count": removed}

    if step == "find_abstracts":
        find_abstracts_by_doi(open_alex_api_key)
        return {"step": "find_abstracts", "status": "done"}

    if step == "clean_empty":
        removed = remove_empty_abstract_title_papers()
        return {"step": "clean_empty", "removed_count": removed}

    if step == "relevance":
        filter_kwargs = _build_filter_kwargs(params)
        success, fails, rid = run_bulk_relevance_screening(
            model_name=params.get("model_name", "gemini-3.5-flash-lite"),
            provider=params.get("provider", "google"),
            batch_size=int(params.get("batch_size", 10)),
            delay_seconds=float(params.get("delay_seconds", 2.0)),
            run_identifier=run_identifier,
            log_to_file=log_to_file,
            max_retries=int(params.get("max_retries", 0)),
            retry_delay=float(params.get("retry_delay", 2.0)),
            **filter_kwargs,
        )
        return {
            "step": "relevance",
            "success_count": len(success),
            "failed_ids": fails,
            "run_identifier": rid,
        }

    if step in {"fetch", "fetch_fulltext"}:
        filter_kwargs = _build_filter_kwargs(params)
        ncbi_provider = NCBIProvider(
            api_key=ncbi_api_key,
            batch_size=int(params.get("batch_size", 100)),
        )
        open_alex_provider = OpenAlexProvider(api_key=open_alex_api_key)
        success_count, fail_count, rid = fetch_paper_fulltexts(
            ncbi_provider=ncbi_provider,
            open_alex_provider=open_alex_provider,
            skip_existing_content=bool(params.get("skip_existing_content", True)),
            content_format=params.get("content_format", "xml"),
            run_identifier=run_identifier,
            log_to_file=log_to_file,
            max_retries=int(params.get("max_retries", 0)),
            retry_delay=float(params.get("retry_delay", 2.0)),
            **filter_kwargs,
        )
        return {
            "step": "fetch_fulltext",
            "success_count": success_count,
            "fail_count": fail_count,
            "run_identifier": rid,
        }

    if step == "extraction":
        filter_kwargs = _build_filter_kwargs(params)
        content_format = params.get("content_format") or None
        table_provider = params.get("table_provider") or None
        table_model_name = params.get("table_model_name") or None
        fails, rid = run_peptide_extraction(
            save=bool(params.get("save", True)),
            provider=params.get("provider", "google"),
            model_name=params.get("model_name", "gemini-3.5-flash-lite"),
            delay_seconds=float(params.get("delay_seconds", 2.0)),
            run_identifier=run_identifier,
            log_to_file=log_to_file,
            content_format=content_format,
            pass_pdf_directly=bool(params.get("pass_pdf_directly", False)),
            tables_as_images=bool(params.get("tables_as_images", False)),
            images_as_images=bool(params.get("images_as_images", False)),
            table_image_layout=params.get("table_image_layout", "interleaved"),
            factored_extraction=bool(params.get("factored_extraction", True)),
            table_provider=table_provider,
            table_model_name=table_model_name,
            max_retries=int(params.get("max_retries", 0)),
            retry_delay=float(params.get("retry_delay", 2.0)),
            **filter_kwargs,
        )
        return {
            "step": "extraction",
            "failed_paper_ids": fails,
            "run_identifier": rid,
        }

    if step == "evaluation":
        filter_kwargs = _build_filter_kwargs(params)
        fails, rid = run_evaluation(
            save=bool(params.get("save", True)),
            run_identifier=run_identifier,
            log_to_file=log_to_file,
            max_retries=int(params.get("max_retries", 0)),
            retry_delay=float(params.get("retry_delay", 2.0)),
            **filter_kwargs,
        )
        return {
            "step": "evaluation",
            "failed_entry_ids": fails,
            "run_identifier": rid,
        }

    raise ValueError(f"Unknown pipeline step: '{step_name}'")


def run_step_job(step_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Submit a single pipeline step as a background job."""
    job = job_manager.submit(
        f"step:{step_name}",
        _execute_step,
        step_name,
        params or {},
    )
    return job.to_dict()


def _execute_config_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    steps = cfg.get("steps", [])
    global_log = bool(
        cfg.get("log_to_file", cfg.get("save_logs", cfg.get("file_logging", True)))
    )
    results: List[Dict[str, Any]] = []
    for step_name in steps:
        step_cfg = dict(cfg.get(step_name, {}) or {})
        kwargs_cfg = step_cfg.pop("kwargs", {}) or {}
        merged = {**step_cfg, **kwargs_cfg}
        merged.setdefault("log_to_file", global_log)
        if "batch_size" not in merged and "ncbi_batch_size" in cfg:
            merged["batch_size"] = cfg["ncbi_batch_size"]
        res = _execute_step(step_name, merged)
        results.append(res)
    return {"steps_executed": steps, "results": results}


def run_config_pipeline_job(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a full YAML config pipeline run as a background job."""
    job = job_manager.submit("pipeline:config", _execute_config_pipeline, cfg)
    return job.to_dict()


def ingest_uploaded_pdfs_job(
    files: List[Tuple[str, bytes]],
    skip_existing_content: bool = True,
    cache_locally: bool = True,
    source: str = "manual",
) -> Dict[str, Any]:
    """Save uploaded PDFs to a temporary folder and run ingest_manual_pdfs."""
    if not files:
        raise ValueError("No PDF files provided.")

    def _run_ingest() -> Dict[str, Any]:
        from peptide_extractor.pipeline import ingest_manual_pdfs

        with tempfile.TemporaryDirectory(prefix="peptide_pdfs_") as tmpdir:
            tmp_path = Path(tmpdir)
            for filename, content in files:
                safe_name = Path(filename).name
                (tmp_path / safe_name).write_bytes(content)
            return ingest_manual_pdfs(
                pdf_dir=tmp_path,
                skip_existing_content=skip_existing_content,
                cache_locally=cache_locally,
                source=source,
                log_to_file=True,
            )

    job = job_manager.submit(
        f"pdf_ingest ({len(files)} files)",
        _run_ingest,
    )
    return job.to_dict()


def find_sync_db_scripts() -> List[str]:
    """Discover sync_db.py in common project locations."""
    root = get_project_root()
    candidates = [
        root / "sync_db.py",
        root / "data" / "sync_db.py",
        root / "scripts" / "sync_db.py",
        root / "src" / "peptide_extractor" / "sync_db.py",
    ]
    return [str(p) for p in candidates if p.is_file()]


def run_sync_db_job(
    script_path: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run sync_db.py as a background subprocess job."""
    root = get_project_root()
    target = Path(script_path) if script_path else None
    if target is None:
        found = find_sync_db_scripts()
        if not found:
            raise FileNotFoundError(
                "No sync_db.py script found in repo root, data/, or scripts/. "
                "Provide an explicit path or use the built-in DB Download/Upload tools."
            )
        target = Path(found[0])
    elif not target.is_absolute():
        target = (root / target).resolve()

    if not target.is_file():
        raise FileNotFoundError(f"sync_db script not found: {target}")

    cmd = [sys.executable, str(target), *(extra_args or [])]
    job = job_manager.submit_subprocess(f"sync_db:{target.name}", cmd, cwd=root)
    return job.to_dict()


def run_init_db() -> Dict[str, Any]:
    init_db()
    return {"status": "ok", "db_path": str(get_db_file_path())}


def run_alembic_upgrade_job() -> Dict[str, Any]:
    root = get_project_root()
    cmd = [sys.executable, "-m", "alembic", "upgrade", "head"]
    job = job_manager.submit_subprocess("alembic:upgrade_head", cmd, cwd=root)
    return job.to_dict()


def replace_db_file(raw_bytes: bytes) -> Dict[str, Any]:
    """Backup and replace the local SQLite database file."""
    db_path = get_db_file_path()
    if db_path is None:
        raise ValueError("Database replacement is only supported for SQLite databases.")
    if not raw_bytes.startswith(b"SQLite format 3\x00"):
        raise ValueError("Uploaded file does not appear to be a valid SQLite 3 database.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Optional[Path] = None
    if db_path.exists():
        backup_path = db_path.with_suffix(f".db.bak.{int(time.time())}")
        shutil.copy2(db_path, backup_path)

    tmp_path = db_path.with_suffix(".db.tmp")
    tmp_path.write_bytes(raw_bytes)
    os.replace(tmp_path, db_path)

    from peptide_extractor.db import database as db_mod

    db_mod.engine.dispose()

    return {
        "status": "replaced",
        "db_path": str(db_path),
        "backup_path": str(backup_path) if backup_path else None,
        "size_bytes": db_path.stat().st_size,
    }


def list_log_files() -> List[Dict[str, Any]]:
    """List persisted step log files in the log directory."""
    log_dir = get_default_log_dir()
    if not log_dir.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for path in sorted(log_dir.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        rel = path.relative_to(log_dir)
        step = rel.parts[0] if len(rel.parts) > 1 else "root"
        entries.append(
            {
                "step": step,
                "filename": path.name,
                "relative_path": str(rel),
                "size_bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return entries


def read_log_file(step: str, filename: str, tail: int = 500) -> Dict[str, Any]:
    log_dir = get_default_log_dir().resolve()
    target = (log_dir / step / filename).resolve()
    if not str(target).startswith(str(log_dir)) or not target.is_file():
        raise FileNotFoundError(f"Log file not found: {step}/{filename}")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    if tail > 0:
        lines = lines[-tail:]
    return {"step": step, "filename": filename, "lines": lines}


def get_visualization_catalog() -> Dict[str, Any]:
    """Return available visualization operations and their supported DB models."""
    catalog: List[Dict[str, Any]] = []
    for op in VisualizationOperation:
        spec = VISUALIZATION_REGISTRY[op]
        catalog.append(
            {
                "operation": op.value,
                "models": [m.__tablename__ for m in spec.supported_models],
                "description": spec.description,
            }
        )
    return {"operations": catalog}


def list_visualizations() -> List[Dict[str, Any]]:
    diagrams_dir = get_project_root() / "data" / "diagrams"
    if not diagrams_dir.exists():
        return []
    items: List[Dict[str, Any]] = []
    for path in sorted(
        diagrams_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            rel = path.relative_to(diagrams_dir)
            items.append(
                {
                    "relative_path": str(rel),
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
    return items


def read_visualization_image(rel_path: str) -> bytes:
    diagrams_dir = (get_project_root() / "data" / "diagrams").resolve()
    target = (diagrams_dir / rel_path).resolve()
    if not str(target).startswith(str(diagrams_dir)) or not target.is_file():
        raise FileNotFoundError(f"Visualization not found: {rel_path}")
    return target.read_bytes()
