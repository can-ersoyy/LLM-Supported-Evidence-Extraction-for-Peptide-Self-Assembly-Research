"""Unified client for local (in-process) or remote (HTTP API) pipeline execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import requests

from ui import backend


class PipelineClient:
    """Talks either to local backend functions or a remote FastAPI server."""

    def __init__(
        self,
        mode: str = "local",
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.mode = mode.lower().strip()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        headers = {**self._headers(), **kwargs.pop("headers", {})}
        resp = requests.request(
            method,
            url,
            headers=headers,
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"API error {resp.status_code}: {detail}")
        return resp

    def health(self) -> Dict[str, Any]:
        if self.mode == "local":
            return {"status": "ok", "mode": "local"}
        data = self._request("GET", "/health").json()
        data["mode"] = "remote"
        return data

    def get_db_stats(self, db_path: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.get_db_stats(db_path=db_path)
        return self._request("GET", "/db/stats").json()

    def sync_db_from_azure(self, force: bool = False) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.sync_db_from_azure(force=force)
        return {"status": "skipped", "reason": "Handled by remote server"}

    def query_records(
        self,
        table_name: str,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.query_records(
                table_name=table_name,
                limit=limit,
                offset=offset,
                filters=filters,
                db_path=db_path,
            )
        return self._request(
            "POST",
            f"/db/records/{table_name}",
            json={"limit": limit, "offset": offset, "filters": filters or {}},
        ).json()

    def download_db(self) -> bytes:
        if self.mode == "local":
            db_path = backend.get_db_file_path()
            if db_path is None or not db_path.exists():
                raise FileNotFoundError("SQLite database file not found.")
            return db_path.read_bytes()
        return self._request("GET", "/db/download", timeout=120.0).content

    def upload_db(self, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.replace_db_file(raw_bytes)
        return self._request(
            "POST",
            "/db/upload",
            files={"file": (filename, raw_bytes, "application/octet-stream")},
            timeout=120.0,
        ).json()

    def init_db(self) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.run_init_db()
        return self._request("POST", "/db/init").json()

    def alembic_upgrade(self) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.run_alembic_upgrade_job()
        return self._request("POST", "/db/alembic-upgrade").json()

    def find_sync_db_scripts(self) -> List[str]:
        if self.mode == "local":
            return backend.find_sync_db_scripts()
        return self._request("GET", "/sync-db/scripts").json().get("scripts", [])

    def run_sync_db(
        self,
        script_path: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.run_sync_db_job(
                script_path=script_path,
                extra_args=extra_args,
            )
        return self._request(
            "POST",
            "/sync-db/run",
            json={"script_path": script_path, "extra_args": extra_args},
        ).json()

    def run_step(
        self,
        step_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.run_step_job(step_name, params)
        return self._request(
            "POST",
            f"/jobs/step/{step_name}",
            json={"params": params or {}},
        ).json()

    def run_config_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.run_config_pipeline_job(config)
        return self._request(
            "POST",
            "/jobs/pipeline",
            json={"config": config},
        ).json()

    def ingest_pdfs(
        self,
        files: List[Tuple[str, bytes]],
        skip_existing_content: bool = True,
        cache_locally: bool = True,
        source: str = "manual",
    ) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.ingest_uploaded_pdfs_job(
                files=files,
                skip_existing_content=skip_existing_content,
                cache_locally=cache_locally,
                source=source,
            )
        multipart = [
            ("files", (name, data, "application/pdf")) for name, data in files
        ]
        return self._request(
            "POST",
            "/pdfs/ingest",
            files=multipart,
            data={
                "skip_existing_content": str(skip_existing_content).lower(),
                "cache_locally": str(cache_locally).lower(),
                "source": source,
            },
            timeout=120.0,
        ).json()

    def list_jobs(self) -> List[Dict[str, Any]]:
        if self.mode == "local":
            return backend.job_manager.list_jobs()
        return self._request("GET", "/jobs").json().get("jobs", [])

    def get_job(self, job_id: str, log_offset: int = 0) -> Optional[Dict[str, Any]]:
        if self.mode == "local":
            return backend.job_manager.get_job(job_id, log_offset=log_offset)
        return self._request(
            "GET",
            f"/jobs/{job_id}",
            params={"log_offset": log_offset},
        ).json()

    def list_log_files(self) -> List[Dict[str, Any]]:
        if self.mode == "local":
            return backend.list_log_files()
        return self._request("GET", "/logs/files").json().get("files", [])

    def read_log_file(
        self, step: str, filename: str, tail: int = 500
    ) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.read_log_file(step=step, filename=filename, tail=tail)
        return self._request(
            "GET",
            f"/logs/files/{step}/{filename}",
            params={"tail": tail},
        ).json()

    def get_visualization_catalog(self) -> Dict[str, Any]:
        if self.mode == "local":
            return backend.get_visualization_catalog()
        return self._request("GET", "/visualizations/catalog").json()

    def list_visualizations(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self.mode == "local":
            return backend.list_visualizations(session_id=session_id)
        params = {"session_id": session_id} if session_id else {}
        return self._request("GET", "/visualizations", params=params).json().get(
            "items", []
        )

    def get_visualization_image(self, rel_path: str) -> bytes:
        if self.mode == "local":
            return backend.read_visualization_image(rel_path)
        return self._request(
            "GET",
            "/visualizations/image",
            params={"path": rel_path},
        ).content
