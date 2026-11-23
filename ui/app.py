"""Simple Streamlit GUI for peptide_extractor."""

from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import yaml

# Ensure repo root is importable when running `streamlit run ui/app.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ui.client import PipelineClient  # noqa: E402

st.set_page_config(
    page_title="Peptide Extractor",
    page_icon="🧬",
    layout="wide",
)


def _load_streamlit_secrets() -> None:
    """Copy top-level scalar secrets from st.secrets into os.environ if present."""
    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ.setdefault(str(k), str(v))
    except Exception:
        pass


def _is_read_only() -> bool:
    return os.getenv("PEPTIDE_READ_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ensure_azure_db_synced(client: PipelineClient, force: bool = False) -> None:
    if client.mode != "local":
        return
    if not force and st.session_state.get("_azure_sync_attempted"):
        return
    st.session_state["_azure_sync_attempted"] = True
    try:
        res = client.sync_db_from_azure(force=force)
        st.session_state["_azure_sync_status"] = res
    except Exception as exc:
        st.session_state["_azure_sync_status"] = {"status": "error", "error": str(exc)}


def _get_client() -> PipelineClient:
    mode = st.session_state.get("conn_mode", "Local (In-Process)")
    if mode == "Remote API Server":
        return PipelineClient(
            mode="remote",
            base_url=st.session_state.get("api_url", "http://localhost:8000"),
            api_key=st.session_state.get("api_key", ""),
        )
    return PipelineClient(mode="local")


def _submit_and_track(job: Dict[str, Any]) -> None:
    st.session_state["active_job_id"] = job["id"]
    st.success(f"Job submitted: `{job['name']}` (ID: `{job['id']}`)")


def _render_job_log_window(
    client: PipelineClient,
    job_id: Optional[str],
    key_prefix: str = "log",
) -> None:
    if not job_id:
        st.info("No job selected yet.")
        return

    try:
        job = client.get_job(job_id)
    except Exception as exc:
        st.error(f"Failed to fetch job `{job_id}`: {exc}")
        return

    if not job:
        st.warning(f"Job `{job_id}` not found.")
        return

    status = job.get("status", "unknown")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.markdown(f"**Job:** `{job['name']}` (`{job['id']}`)")
    col2.markdown(f"**Status:** `{status}`")
    col3.markdown(f"**Lines:** `{job.get('total_logs', 0)}`")
    auto_refresh = col4.checkbox(
        "Auto-refresh",
        value=(status in {"pending", "running"}),
        key=f"{key_prefix}_auto_{job_id}",
    )

    if job.get("error"):
        st.error(job["error"])
    elif status == "completed" and job.get("result") is not None:
        with st.expander("Job Result", expanded=True):
            st.json(job["result"])

    logs = job.get("logs", [])
    log_text = "\n".join(logs) if logs else "(waiting for logs...)"
    st.code(log_text, language="log")

    if st.button("Refresh Logs", key=f"{key_prefix}_refresh_{job_id}"):
        st.rerun()

    if auto_refresh and status in {"pending", "running"}:
        time.sleep(1.5)
        st.rerun()


def _render_sidebar(client: PipelineClient) -> str:
    st.sidebar.title("Peptide Extractor")

    read_only = _is_read_only()
    if read_only:
        st.sidebar.info("Read-Only Mode")

    default_mode = (
        "Remote API Server" if os.getenv("PEPTIDE_API_URL") else "Local (In-Process)"
    )
    st.sidebar.radio(
        "Backend Mode",
        ["Local (In-Process)", "Remote API Server"],
        index=0 if default_mode == "Local (In-Process)" else 1,
        key="conn_mode",
    )

    if st.session_state["conn_mode"] == "Remote API Server":
        st.sidebar.text_input(
            "API Server URL",
            value=os.getenv("PEPTIDE_API_URL", "http://localhost:8000"),
            key="api_url",
        )
        st.sidebar.text_input(
            "API Key (optional)",
            value=os.getenv("PEPTIDE_API_KEY", ""),
            type="password",
            key="api_key",
        )
        if st.sidebar.button("Test Connection"):
            try:
                res = _get_client().health()
                st.sidebar.success(f"Connected ({res.get('status')})")
            except Exception as exc:
                st.sidebar.error(f"Connection failed: {exc}")

    if st.session_state["conn_mode"] == "Local (In-Process)":
        with st.sidebar.expander("Database Source (Session)", expanded=False):
            if st.button("Refresh DB from Azure", key="sidebar_refresh_azure"):
                _ensure_azure_db_synced(client, force=True)
                st.rerun()
            az_status = st.session_state.get("_azure_sync_status")
            if az_status and az_status.get("status") != "skipped":
                st.caption(f"Azure sync: `{az_status.get('status')}`")
                if az_status.get("error"):
                    st.error(az_status["error"])

            session_upload = st.file_uploader(
                "Load local .db (this session only)",
                type=["db", "sqlite", "sqlite3"],
                key="session_db_uploader",
            )
            if session_upload is not None:
                raw = session_upload.read()
                if not raw.startswith(b"SQLite format 3\x00"):
                    st.error("Invalid SQLite 3 file.")
                elif st.session_state.get("_session_db_name") != session_upload.name:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
                    tmp.write(raw)
                    tmp.close()
                    st.session_state["session_db_path"] = tmp.name
                    st.session_state["_session_db_name"] = session_upload.name
                    st.success(f"Using session DB: `{session_upload.name}`")

            if st.session_state.get("session_db_path"):
                st.caption(f"Active: `{st.session_state.get('_session_db_name')}`")
                if st.button("Reset to Default / Azure DB"):
                    st.session_state.pop("session_db_path", None)
                    st.session_state.pop("_session_db_name", None)
                    st.rerun()

    st.sidebar.divider()
    if read_only and st.session_state["conn_mode"] == "Local (In-Process)":
        nav_options = ["Overview & DB", "Peptide Search", "Paper Inspector", "Results & Plots"]
    else:
        nav_options = [
            "Overview & DB",
            "Peptide Search",
            "Paper Inspector",
            "Run Pipeline",
            "PDF Ingestion",
            "DB Sync & Admin",
            "Results & Plots",
            "Logs & Jobs",
        ]
    if st.session_state.get("nav_page") not in nav_options:
        st.session_state["nav_page"] = nav_options[0]
    page = st.sidebar.radio("Navigation", nav_options, key="nav_page")

    st.sidebar.divider()
    active_job_id = st.session_state.get("active_job_id")
    if active_job_id:
        try:
            job = client.get_job(active_job_id)
            if job:
                st.sidebar.caption(
                    f"Latest Job: `{job['name']}`\nStatus: **{job['status']}**"
                )
        except Exception:
            pass

    return page


def page_overview(client: PipelineClient) -> None:
    st.header("Database Overview")
    b1, b2 = st.columns([1, 2])
    if b1.button("Refresh Stats"):
        st.rerun()
    if client.mode == "local" and b2.button("Refresh DB from Azure", key="ov_refresh_azure"):
        _ensure_azure_db_synced(client, force=True)
        st.rerun()

    session_db = st.session_state.get("session_db_path")
    try:
        stats = client.get_db_stats(db_path=session_db)
    except Exception as exc:
        st.error(f"Could not load database stats: {exc}")
        return

    st.caption(
        f"Database: `{stats.get('db_path') or stats.get('database_url')}` "
        f"| Exists: `{stats.get('db_exists')}` "
        f"| Size: `{stats.get('db_size_bytes', 0) / (1024 * 1024):.2f} MB`"
    )

    tables = stats.get("tables", {})
    pm = tables.get("paper_metadata", {})
    pc = tables.get("paper_content", {})
    pr = tables.get("paper_relevance", {})
    pe = tables.get("peptide_entry", {})
    ee = tables.get("extraction_evaluation", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Papers (Metadata)", pm.get("total", 0))
    c2.metric("Fulltexts (Content)", pc.get("total", 0))
    c3.metric("Relevance Checks", pr.get("total", 0))
    c4.metric("Peptide Entries", pe.get("total", 0))
    c5.metric("Evaluations", ee.get("total", 0))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Paper Status")
        st.json(pm.get("by_status", {}))
    with col_b:
        st.subheader("Content Formats & Subsets")
        st.json(
            {
                "content_by_format": pc.get("by_format", {}),
                "papers_by_subset": pm.get("by_subset", {}),
            }
        )
    with col_c:
        st.subheader("Peptide Entries by Mode")
        st.json(pe.get("by_extraction_mode", {}))

    try:
        import altair as alt

        chart_data = client.get_overview_charts_data(db_path=session_db)
        st.divider()
        st.subheader("Interactive Distributions")
        ch1, ch2, ch3 = st.columns(3)

        morph_rows = chart_data.get("morphology_counts", [])
        if morph_rows:
            with ch1:
                st.caption("Morphology Distribution")
                c_morph = (
                    alt.Chart(pd.DataFrame(morph_rows))
                    .mark_bar()
                    .encode(
                        x=alt.X("morphology:N", sort="-y", title="Morphology"),
                        y=alt.Y("count:Q", title="Entries"),
                        tooltip=["morphology:N", "count:Q"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(c_morph, use_container_width=True)

        len_rows = chart_data.get("length_counts", [])
        if len_rows:
            with ch2:
                st.caption("Peptide Length Distribution")
                c_len = (
                    alt.Chart(pd.DataFrame(len_rows))
                    .mark_bar()
                    .encode(
                        x=alt.X("length:O", title="Length (aa)"),
                        y=alt.Y("count:Q", title="Entries"),
                        tooltip=["length:O", "count:Q"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(c_len, use_container_width=True)

        conf_rows = chart_data.get("relevance_confidence", [])
        if conf_rows:
            with ch3:
                st.caption("Relevance Confidence")
                c_conf = (
                    alt.Chart(pd.DataFrame(conf_rows))
                    .mark_bar()
                    .encode(
                        x=alt.X("confidence_bin:O", title="Confidence Bin"),
                        y=alt.Y("count:Q", title="Papers"),
                        color="outcome:N",
                        tooltip=["confidence_bin:O", "outcome:N", "count:Q"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(c_conf, use_container_width=True)
    except Exception:
        pass


def page_run_pipeline(client: PipelineClient) -> None:
    st.header("Run Pipeline Steps")

    tabs = st.tabs(
        [
            "1. Search",
            "2. Clean & Deduplicate",
            "3. Relevance",
            "4. Fetch Fulltext",
            "5. Extraction",
            "6. Evaluation",
            "7. Full YAML Config",
        ]
    )

    with tabs[0]:
        st.subheader("Literature Search")
        source = st.selectbox("Source", ["ncbi", "scopus"])
        if source == "ncbi":
            db = st.selectbox("NCBI Database", ["pmc", "pubmed"])
            preset = st.selectbox(
                "Query Preset",
                ["V1", "V2", "V3", "custom"],
                help="V1/V2/V3 map to built-in queries in pipeline.py",
            )
            custom_queries = ""
            if preset == "custom":
                custom_queries = st.text_area(
                    "Custom Queries (one per line)",
                    value='peptide "self-assembly"',
                )
            batch_size = st.number_input("Batch Size", min_value=1, value=100)
            run_id = st.text_input("Run Identifier (optional)", key="search_rid")
            if st.button("Run Search", type="primary"):
                try:
                    job = client.run_step(
                        "search",
                        {
                            "source": "ncbi",
                            "db": db,
                            "preset": preset,
                            "queries": custom_queries,
                            "batch_size": batch_size,
                            "run_identifier": run_id or None,
                        },
                    )
                    _submit_and_track(job)
                except Exception as exc:
                    st.error(str(exc))
        else:
            scopus_q = st.text_area("Scopus Query (leave empty for SCOPUS_SEARCH_QUERY_V1)")
            run_id = st.text_input("Run Identifier (optional)", key="scopus_rid")
            if st.button("Run Scopus Search", type="primary"):
                try:
                    job = client.run_step(
                        "search",
                        {
                            "source": "scopus",
                            "query": scopus_q or None,
                            "run_identifier": run_id or None,
                        },
                    )
                    _submit_and_track(job)
                except Exception as exc:
                    st.error(str(exc))

    with tabs[1]:
        st.subheader("Deduplicate & Clean Metadata")
        c1, c2, c3 = st.columns(3)
        with c1:
            normalize = st.checkbox("Normalize DOIs", value=True)
            remove_empty = st.checkbox("Remove Empty DOIs", value=True)
            if st.button("Deduplicate by DOI", type="primary"):
                try:
                    _submit_and_track(
                        client.run_step(
                            "deduplicate",
                            {"normalize": normalize, "remove_empty": remove_empty},
                        )
                    )
                except Exception as exc:
                    st.error(str(exc))
        with c2:
            st.write("Fetch missing abstracts via Semantic Scholar & OpenAlex.")
            if st.button("Find Missing Abstracts"):
                try:
                    _submit_and_track(client.run_step("find_abstracts", {}))
                except Exception as exc:
                    st.error(str(exc))
        with c3:
            st.write("Remove papers missing both/either title or abstract.")
            if st.button("Remove Empty Abstract/Title"):
                try:
                    _submit_and_track(client.run_step("clean_empty", {}))
                except Exception as exc:
                    st.error(str(exc))

    with tabs[2]:
        st.subheader("Abstract Relevance Screening")
        c1, c2 = st.columns(2)
        provider = c1.text_input("Provider", value="google", key="rel_provider")
        model_name = c2.text_input(
            "Model Name", value="gemini-3.5-flash-lite", key="rel_model"
        )
        c3, c4, c5 = st.columns(3)
        batch_size = c3.number_input("Batch Size", min_value=1, value=10, key="rel_bs")
        delay_seconds = c4.number_input(
            "Delay (s)", min_value=0.0, value=2.0, step=0.5, key="rel_delay"
        )
        max_retries = c5.number_input(
            "Max Retries", min_value=0, value=0, key="rel_retries"
        )
        c6, c7, c8 = st.columns(3)
        status_filter = c6.selectbox(
            "Status Filter", ["", "not_processed", "usable", "discard"], key="rel_status"
        )
        limit = c7.number_input("Limit (0 = all)", min_value=0, value=0, key="rel_limit")
        ids_str = c8.text_input("Paper IDs (comma-separated)", key="rel_ids")

        if st.button("Run Relevance Screening", type="primary"):
            try:
                _submit_and_track(
                    client.run_step(
                        "relevance",
                        {
                            "provider": provider,
                            "model_name": model_name,
                            "batch_size": batch_size,
                            "delay_seconds": delay_seconds,
                            "max_retries": max_retries,
                            "status": status_filter or None,
                            "limit": limit or None,
                            "id": ids_str or None,
                        },
                    )
                )
            except Exception as exc:
                st.error(str(exc))

    with tabs[3]:
        st.subheader("Fetch Paper Fulltexts")
        c1, c2, c3 = st.columns(3)
        content_format = c1.selectbox("Format", ["xml", "pdf"], key="ft_fmt")
        skip_existing = c2.checkbox(
            "Skip Existing Content", value=True, key="ft_skip"
        )
        max_retries = c3.number_input(
            "Max Retries", min_value=0, value=0, key="ft_retries"
        )
        c4, c5, c6 = st.columns(3)
        status_filter = c4.selectbox(
            "Status Filter", ["usable", "", "not_processed"], key="ft_status"
        )
        limit = c5.number_input("Limit (0 = all)", min_value=0, value=0, key="ft_limit")
        ids_str = c6.text_input("Paper IDs (comma-separated)", key="ft_ids")

        if st.button("Fetch Fulltexts", type="primary"):
            try:
                _submit_and_track(
                    client.run_step(
                        "fetch_fulltext",
                        {
                            "content_format": content_format,
                            "skip_existing_content": skip_existing,
                            "max_retries": max_retries,
                            "status": status_filter or None,
                            "limit": limit or None,
                            "id": ids_str or None,
                        },
                    )
                )
            except Exception as exc:
                st.error(str(exc))

    with tabs[4]:
        st.subheader("Peptide Extraction")
        c1, c2, c3 = st.columns(3)
        provider = c1.text_input("Provider", value="google", key="ext_provider")
        model_name = c2.text_input(
            "Model Name", value="gemini-3.5-flash-lite", key="ext_model"
        )
        content_format = c3.selectbox(
            "Content Format", ["", "xml", "pdf"], key="ext_fmt"
        )

        c4, c5, c6 = st.columns(3)
        delay_seconds = c4.number_input(
            "Delay (s)", min_value=0.0, value=4.0, step=0.5, key="ext_delay"
        )
        limit = c5.number_input("Limit (0 = all)", min_value=0, value=0, key="ext_limit")
        ids_str = c6.text_input("Paper IDs (comma-separated)", key="ext_ids")

        c7, c8, c9 = st.columns(3)
        pass_pdf = c7.checkbox("Pass PDF Directly", value=False)
        tables_as_images = c8.checkbox("Tables as Images", value=False)
        images_as_images = c9.checkbox("Figures as Images", value=False)

        with st.expander("Advanced Extraction Options"):
            a1, a2, a3 = st.columns(3)
            factored = a1.checkbox("Factored (2-pass) Extraction", value=True)
            layout = a2.selectbox(
                "Table/Image Layout", ["interleaved", "appendix"]
            )
            max_retries = a3.number_input(
                "Max Retries", min_value=0, value=0, key="ext_retries"
            )
            b1, b2 = st.columns(2)
            table_provider = b1.text_input("Table Provider (optional)")
            table_model = b2.text_input("Table Model Name (optional)")

        if st.button("Run Extraction", type="primary"):
            try:
                _submit_and_track(
                    client.run_step(
                        "extraction",
                        {
                            "provider": provider,
                            "model_name": model_name,
                            "content_format": content_format or None,
                            "delay_seconds": delay_seconds,
                            "limit": limit or None,
                            "id": ids_str or None,
                            "pass_pdf_directly": pass_pdf,
                            "tables_as_images": tables_as_images,
                            "images_as_images": images_as_images,
                            "factored_extraction": factored,
                            "table_image_layout": layout,
                            "table_provider": table_provider or None,
                            "table_model_name": table_model or None,
                            "max_retries": max_retries,
                        },
                    )
                )
            except Exception as exc:
                st.error(str(exc))

    with tabs[5]:
        st.subheader("Extraction Evaluation")
        c1, c2, c3 = st.columns(3)
        run_id_filter = c1.text_input("Filter PeptideEntry run_identifier", key="ev_rid")
        paper_id_filter = c2.text_input("Filter paper_id (comma-separated)", key="ev_pid")
        limit = c3.number_input("Limit (0 = all)", min_value=0, value=0, key="ev_limit")
        max_retries = st.number_input("Max Retries", min_value=0, value=0, key="ev_ret")
        if st.button("Run Evaluation", type="primary"):
            try:
                _submit_and_track(
                    client.run_step(
                        "evaluation",
                        {
                            "run_identifier": run_id_filter or None,
                            "paper_id": paper_id_filter or None,
                            "limit": limit or None,
                            "max_retries": max_retries,
                        },
                    )
                )
            except Exception as exc:
                st.error(str(exc))

    with tabs[6]:
        st.subheader("Run Full YAML Config")
        default_yaml_path = _REPO_ROOT / "default_config.yaml"
        default_yaml = (
            default_yaml_path.read_text(encoding="utf-8")
            if default_yaml_path.exists()
            else "steps:\n  - extraction\n"
        )
        uploaded_cfg = st.file_uploader("Upload YAML Config (optional)", type=["yaml", "yml"])
        if uploaded_cfg is not None:
            default_yaml = uploaded_cfg.read().decode("utf-8")

        yaml_text = st.text_area("Config YAML", value=default_yaml, height=280)
        if st.button("Run Config Pipeline", type="primary"):
            try:
                cfg = yaml.safe_load(yaml_text) or {}
                _submit_and_track(client.run_config_pipeline(cfg))
            except Exception as exc:
                st.error(f"Failed to start config pipeline: {exc}")

    st.divider()
    st.subheader("Active Job Log Window")
    _render_job_log_window(
        client, st.session_state.get("active_job_id"), key_prefix="pipe_log"
    )


def page_pdf_ingestion(client: PipelineClient) -> None:
    st.header("Manual PDF Ingestion")
    st.write(
        "Upload one or more PDF files named `<pmid>.pdf` (e.g. `18052148.pdf`). "
        "Each file is matched against `PaperMetadata.pmid`, uploaded to Azure Blob Storage, "
        "optionally cached locally, and registered in `PaperContent`."
    )

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    c1, c2, c3 = st.columns(3)
    skip_existing = c1.checkbox("Skip Existing Content", value=True)
    cache_locally = c2.checkbox("Cache Locally", value=True)
    source = c3.text_input("Source Tag", value="manual")

    if st.button("Upload & Ingest PDFs", type="primary", disabled=not uploaded_files):
        try:
            payload = [(f.name, f.read()) for f in uploaded_files]
            job = client.ingest_pdfs(
                files=payload,
                skip_existing_content=skip_existing,
                cache_locally=cache_locally,
                source=source,
            )
            _submit_and_track(job)
        except Exception as exc:
            st.error(f"PDF ingestion failed: {exc}")

    st.divider()
    _render_job_log_window(
        client, st.session_state.get("active_job_id"), key_prefix="pdf_log"
    )


def page_db_sync(client: PipelineClient) -> None:
    st.header("DB Sync & Administration")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Run sync_db.py")
        try:
            found_scripts = client.find_sync_db_scripts()
        except Exception as exc:
            found_scripts = []
            st.warning(f"Could not check sync_db scripts: {exc}")

        if found_scripts:
            st.caption(f"Detected scripts: `{found_scripts}`")
        script_path = st.text_input(
            "Script path (leave blank to auto-detect)",
            value=found_scripts[0] if found_scripts else "",
        )
        extra_args_str = st.text_input("Extra CLI arguments (optional)", value="")
        if st.button("Run sync_db.py", type="primary"):
            try:
                args = shlex.split(extra_args_str) if extra_args_str.strip() else None
                job = client.run_sync_db(
                    script_path=script_path.strip() or None,
                    extra_args=args,
                )
                _submit_and_track(job)
            except Exception as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("Schema & Migrations")
        b1, b2 = st.columns(2)
        if b1.button("Initialize DB (init_db)"):
            try:
                res = client.init_db()
                st.success(f"Initialized DB: {res}")
            except Exception as exc:
                st.error(str(exc))
        if b2.button("Run Alembic Upgrade Head"):
            try:
                _submit_and_track(client.alembic_upgrade())
            except Exception as exc:
                st.error(str(exc))

    with col2:
        st.subheader("Download / Upload SQLite DB")
        st.write("Useful for syncing `peptide-extractor.db` with a remote server.")
        if st.button("Prepare DB Download"):
            try:
                db_bytes = client.download_db()
                st.download_button(
                    label="Download peptide-extractor.db",
                    data=db_bytes,
                    file_name="peptide-extractor.db",
                    mime="application/octet-stream",
                )
            except Exception as exc:
                st.error(f"Download failed: {exc}")

        st.divider()
        uploaded_db = st.file_uploader("Upload SQLite DB (.db)", type=["db", "sqlite", "sqlite3"])
        if st.button("Replace Server Database", disabled=uploaded_db is None):
            try:
                res = client.upload_db(uploaded_db.name, uploaded_db.read())
                st.success(f"Database replaced: {res}")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    _render_job_log_window(
        client, st.session_state.get("active_job_id"), key_prefix="sync_log"
    )


def page_results(client: PipelineClient) -> None:
    st.header("Results & Visualizations")
    tab_data, tab_viz, tab_cmp = st.tabs(
        ["Data Explorer", "Visualizations", "Run Comparison"]
    )

    with tab_data:
        c1, c2, c3, c4 = st.columns(4)
        table = c1.selectbox(
            "Table",
            [
                "peptide_entry",
                "paper_metadata",
                "paper_relevance",
                "paper_content",
                "extraction_evaluation",
            ],
        )
        run_id = c2.text_input("Filter run_identifier", key="res_rid")
        paper_id = c3.text_input("Filter paper_id", key="res_pid")
        limit = c4.number_input("Limit", min_value=1, max_value=2000, value=100)

        filters: Dict[str, Any] = {}
        if run_id.strip():
            filters["run_identifier"] = run_id.strip()
        if paper_id.strip():
            filters["paper_id"] = paper_id.strip()

        if table == "paper_metadata":
            status_f = st.selectbox("Status", ["", "not_processed", "usable", "discard"])
            if status_f:
                filters["status"] = status_f
        elif table == "peptide_entry":
            mode_f = st.selectbox("Extraction Mode", ["", "llm", "manual"])
            if mode_f:
                filters["extraction_mode"] = mode_f

        session_db = st.session_state.get("session_db_path")
        try:
            data = client.query_records(
                table,
                limit=int(limit),
                offset=0,
                filters=filters,
                db_path=session_db,
            )
            rows = data.get("rows", [])
            st.caption(f"Showing {len(rows)} row(s) from `{table}`")
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                dl1, dl2 = st.columns(2)
                dl1.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{table}.csv",
                    mime="text/csv",
                )
                dl2.download_button(
                    "Download JSON",
                    data=json.dumps(rows, indent=2).encode("utf-8"),
                    file_name=f"{table}.json",
                    mime="application/json",
                )

                with st.expander("Inspect Row Details"):
                    idx = st.number_input(
                        "Row index", min_value=0, max_value=len(rows) - 1, value=0
                    )
                    row = rows[int(idx)]
                    if isinstance(row.get("evidence_span"), str):
                        try:
                            row = {
                                **row,
                                "evidence_span_parsed": json.loads(row["evidence_span"]),
                            }
                        except Exception:
                            pass
                    st.json(row)
            else:
                st.info("No matching records found.")
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    with tab_viz:
        st.subheader("Generate Plots")
        try:
            catalog = client.get_visualization_catalog().get("operations", [])
        except Exception as exc:
            st.error(f"Could not load visualization catalog: {exc}")
            catalog = []

        if catalog:
            viz_model = st.selectbox(
                "Target Table",
                [
                    "peptide_entry",
                    "extraction_evaluation",
                    "paper_relevance",
                    "paper_metadata",
                ],
            )
            compat_ops = [
                item["operation"]
                for item in catalog
                if viz_model in item.get("models", [])
            ]
            selected_ops = st.multiselect(
                "Operations", compat_ops, default=compat_ops[:1]
            )
            v1, v2 = st.columns(2)
            viz_run_filter = v1.text_input("Filter run_identifier (optional)", key="viz_rid")
            viz_limit = v2.number_input("Limit (0 = all)", min_value=0, value=0, key="viz_lim")
            if st.button("Generate Plots", type="primary", disabled=not selected_ops):
                try:
                    job = client.run_step(
                        "visualization",
                        {
                            "model": viz_model,
                            "operations": selected_ops,
                            "run_identifier": viz_run_filter or None,
                            "limit": viz_limit or None,
                            "db_path": st.session_state.get("session_db_path"),
                            "session_id": st.session_state.get("session_id"),
                        },
                    )
                    _submit_and_track(job)
                except Exception as exc:
                    st.error(str(exc))

        if st.session_state.get("active_job_id"):
            with st.expander("Plot Generation Logs", expanded=True):
                _render_job_log_window(
                    client, st.session_state.get("active_job_id"), key_prefix="viz_log"
                )

        st.divider()
        st.subheader("Saved Diagrams")
        if st.button("Refresh Diagrams"):
            st.rerun()
        try:
            items = client.list_visualizations(
                session_id=st.session_state.get("session_id")
            )
            if not items:
                st.info("No diagrams found in `data/diagrams/`.")
            else:
                choices = [item["relative_path"] for item in items]
                selected = st.selectbox("Select Diagram", choices)
                if selected:
                    img_bytes = client.get_visualization_image(selected)
                    st.image(img_bytes, caption=selected)
        except Exception as exc:
            st.error(f"Could not load diagrams: {exc}")

    with tab_cmp:
        st.subheader("Extraction Runs & Evaluation Summary")
        session_db = st.session_state.get("session_db_path")
        try:
            cmp_data = client.get_run_comparison(db_path=session_db)
            cmp_rows = cmp_data.get("rows", [])
        except Exception as exc:
            st.error(f"Failed to load run summary: {exc}")
            cmp_rows = []

        if not cmp_rows:
            st.info("No PeptideEntry runs found in the database.")
        else:
            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True)

            llm_run_ids = sorted(
                {
                    int(r["run_identifier"])
                    for r in cmp_rows
                    if r.get("extraction_mode") == "llm"
                    and r.get("run_identifier") is not None
                },
                reverse=True,
            )
            manual_run_ids = sorted(
                {
                    int(r["run_identifier"])
                    for r in cmp_rows
                    if r.get("extraction_mode") == "manual"
                    and r.get("run_identifier") is not None
                }
            )
            if llm_run_ids and manual_run_ids:
                st.divider()
                st.subheader("Golden Dataset Benchmark")
                b1, b2 = st.columns(2)
                sel_llm = b1.selectbox("LLM run_identifier", llm_run_ids)
                sel_manual = b2.selectbox("Manual run_identifier", manual_run_ids)
                if st.button("Compute Golden Benchmark", type="primary"):
                    try:
                        bm = client.run_golden_benchmark(
                            llm_run_identifier=int(sel_llm),
                            manual_run_identifier=int(sel_manual),
                            db_path=session_db,
                        )
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Seq Recall", f"{bm['sequence_recall']:.1%}")
                        m2.metric("Seq Precision", f"{bm['sequence_precision']:.1%}")
                        m3.metric("Seq F1", f"{bm['sequence_f1']:.1%}")
                        m4.metric(
                            "Condition Prec", f"{bm['condition_precision']:.1%}"
                        )
                        st.code(bm["formatted_report"], language="text")
                    except Exception as exc:
                        st.error(f"Benchmark failed: {exc}")


def page_logs_and_jobs(client: PipelineClient) -> None:
    st.header("Logs & Background Jobs")
    tab_jobs, tab_files = st.tabs(["Live Jobs", "Saved Log Files"])

    with tab_jobs:
        if st.button("Refresh Job List"):
            st.rerun()
        try:
            jobs = client.list_jobs()
        except Exception as exc:
            st.error(f"Could not fetch jobs: {exc}")
            jobs = []

        if not jobs:
            st.info("No jobs have been run in this session yet.")
        else:
            st.dataframe(pd.DataFrame(jobs), use_container_width=True)
            job_ids = [j["id"] for j in jobs]
            default_idx = (
                job_ids.index(st.session_state["active_job_id"])
                if st.session_state.get("active_job_id") in job_ids
                else 0
            )
            selected_id = st.selectbox("Inspect Job", job_ids, index=default_idx)
            _render_job_log_window(client, selected_id, key_prefix="jobs_tab")

    with tab_files:
        try:
            files = client.list_log_files()
        except Exception as exc:
            st.error(f"Could not list log files: {exc}")
            files = []

        if not files:
            st.info("No saved log files found.")
        else:
            labels = [f"{f['step']}/{f['filename']}" for f in files]
            choice = st.selectbox("Select Log File", labels)
            tail = st.number_input("Tail lines", min_value=50, max_value=5000, value=300)
            if choice:
                step, fname = choice.split("/", 1)
                try:
                    res = client.read_log_file(step, fname, tail=int(tail))
                    st.code("\n".join(res.get("lines", [])), language="log")
                except Exception as exc:
                    st.error(f"Failed to read log file: {exc}")


def page_peptide_search(client: PipelineClient) -> None:
    st.header("Peptide & Condition Search")

    c1, c2, c3, c4 = st.columns(4)
    seq = c1.text_input(
        "Sequence (substring)",
        value=st.session_state.get("search_seq_prefill", ""),
        placeholder="e.g. FF, KLVFF",
    )
    length_range = c2.slider("Length (residues)", min_value=1, max_value=50, value=(1, 30))
    morphology = c3.selectbox(
        "Morphology",
        [
            "",
            "hydrogel",
            "fiber",
            "tube",
            "nanosphere",
            "ribbon",
            "vesicle",
            "particle",
            "organogel",
            "amyloid",
            "none",
        ],
    )
    chirality = c4.selectbox(
        "Chirality",
        ["", "l", "d", "heterochiral", "racemic", "achiral"],
    )

    c5, c6, c7, c8 = st.columns(4)
    n_term = c5.text_input("N-terminal", placeholder="e.g. Fmoc, Acetylation")
    c_term = c6.text_input("C-terminal", placeholder="e.g. Amidation, Free")
    solvent = c7.text_input("Solvent / Solution", placeholder="e.g. DMSO, PBS, H2O")
    detection = c8.text_input("Detection Method", placeholder="e.g. TEM, ThT, CD")

    with st.expander("Numeric & Run Filters"):
        n1, n2, n3, n4, n5, n6 = st.columns(6)
        ph_min = n1.text_input("pH min")
        ph_max = n2.text_input("pH max")
        t_min = n3.text_input("Temp min (°C)")
        t_max = n4.text_input("Temp max (°C)")
        mode_f = n5.selectbox("Mode", ["", "llm", "manual"], key="ps_mode")
        run_id_f = n6.text_input("run_identifier", key="ps_rid")
        limit = st.number_input(
            "Max results", min_value=10, max_value=2000, value=200, key="ps_limit"
        )

    def _opt_float(val: str) -> Optional[float]:
        return float(val.strip()) if val.strip() else None

    try:
        res = client.search_peptides(
            db_path=st.session_state.get("session_db_path"),
            sequence=seq or None,
            min_length=length_range[0],
            max_length=length_range[1],
            morphology=morphology or None,
            chirality=chirality or None,
            n_terminal=n_term or None,
            c_terminal=c_term or None,
            ph_min=_opt_float(ph_min),
            ph_max=_opt_float(ph_max),
            temp_min=_opt_float(t_min),
            temp_max=_opt_float(t_max),
            solvent=solvent or None,
            detection_method=detection or None,
            extraction_mode=mode_f or None,
            run_identifier=int(run_id_f.strip()) if run_id_f.strip() else None,
            limit=int(limit),
        )
    except Exception as exc:
        st.error(f"Peptide search failed: {exc}")
        return

    rows = res.get("rows", [])
    st.caption(f"Found {len(rows)} matching peptide entry(ies).")
    if not rows:
        st.info("No matching peptides found.")
        return

    df = pd.DataFrame(rows)
    preferred_cols = [
        "sequence",
        "length",
        "type",
        "n_terminal",
        "c_terminal",
        "chirality",
        "ph",
        "temperature",
        "peptide_concentration",
        "solution",
        "solvent",
        "detection_method",
        "extraction_mode",
        "model",
        "run_identifier",
        "pmid",
        "pubmed_url",
        "doi_url",
    ]
    ordered_cols = [c for c in preferred_cols if c in df.columns] + [
        c for c in df.columns if c not in preferred_cols
    ]
    st.dataframe(
        df[ordered_cols],
        use_container_width=True,
        column_config={
            "pubmed_url": st.column_config.LinkColumn("PubMed", display_text="PubMed"),
            "doi_url": st.column_config.LinkColumn("DOI", display_text="DOI"),
        },
    )

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="peptide_search.csv",
        mime="text/csv",
    )
    dl2.download_button(
        "Download JSON",
        data=json.dumps(rows, indent=2).encode("utf-8"),
        file_name="peptide_search.json",
        mime="application/json",
    )


def page_paper_inspector(client: PipelineClient) -> None:
    st.header("Paper Inspector")
    session_db = st.session_state.get("session_db_path")

    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input(
        "Search by PMID, DOI, or Title",
        value=st.session_state.get("inspector_query_prefill", ""),
        placeholder="e.g. 18052148, 10.1021/..., hydrogel",
    )
    status_f = c2.selectbox("Status", ["", "usable", "not_processed", "discard"], key="pi_status")
    subset_f = c3.selectbox("Subset", ["", "golden_benchmark", "scraped"], key="pi_subset")

    try:
        search_res = client.search_papers(
            query=query or None,
            status=status_f or None,
            dataset_subset=subset_f or None,
            limit=100,
            db_path=session_db,
        )
    except Exception as exc:
        st.error(f"Paper search failed: {exc}")
        return

    papers = search_res.get("rows", [])
    if not papers:
        st.info("No matching papers found.")
        return

    options = {
        f"[{p['id']}] PMID:{p.get('pmid') or '-'} | {p.get('title', '')[:80]}": p["id"]
        for p in papers
    }
    labels = list(options.keys())
    default_idx = 0
    prefill_pid = st.session_state.get("inspector_paper_id_prefill")
    if prefill_pid is not None:
        for i, lbl in enumerate(labels):
            if options[lbl] == prefill_pid:
                default_idx = i
                break

    selected_label = st.selectbox("Select Paper", labels, index=default_idx)
    selected_id = options[selected_label]

    try:
        detail = client.get_paper_detail(paper_id=selected_id, db_path=session_db)
    except Exception as exc:
        st.error(f"Could not load paper details: {exc}")
        return

    paper = detail["paper"]
    st.subheader(paper.get("title") or "(No title)")
    meta_parts = [
        f"ID: `{paper.get('id')}`",
        f"Status: `{paper.get('status')}`",
        f"Subset: `{paper.get('dataset_subset')}`",
    ]
    if paper.get("pmid"):
        meta_parts.append(f"[PubMed ({paper['pmid']})]({paper['pubmed_url']})")
    if paper.get("doi"):
        meta_parts.append(f"[DOI ({paper['doi']})]({paper['doi_url']})")
    st.markdown(" | ".join(meta_parts))

    with st.expander("Abstract", expanded=True):
        st.write(paper.get("abstract") or "_No abstract available._")

    relevances = detail.get("relevance", [])
    if relevances:
        with st.expander(f"Relevance Screening ({len(relevances)})", expanded=True):
            for rel in relevances:
                r1, r2, r3 = st.columns(3)
                r1.metric("Relevant", str(rel.get("is_relevant")))
                r2.metric(
                    "Confidence",
                    f"{rel['confidence']:.2f}" if rel.get("confidence") is not None else "-",
                )
                r3.metric("Model", f"{rel.get('provider')}/{rel.get('model_name')}")
                if rel.get("explanation"):
                    st.write(rel["explanation"])

    entries = detail.get("peptide_entries", [])
    st.subheader(f"Extracted Peptide Entries ({len(entries)})")
    if entries:
        manual_entries = [e for e in entries if e.get("extraction_mode") == "manual"]
        llm_entries = [e for e in entries if e.get("extraction_mode") == "llm"]

        if manual_entries and llm_entries:
            col_m, col_l = st.columns(2)
            with col_m:
                st.markdown("**Golden / Manual Entries**")
                st.dataframe(pd.DataFrame(manual_entries), use_container_width=True)
            with col_l:
                st.markdown("**LLM Extracted Entries**")
                st.dataframe(pd.DataFrame(llm_entries), use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(entries), use_container_width=True)

        evals = detail.get("evaluations", [])
        evals_by_entry: Dict[int, list] = {}
        for ev in evals:
            eid = ev.get("peptide_entry_id")
            if eid is not None:
                evals_by_entry.setdefault(eid, []).append(ev)

        with st.expander("Evidence Spans & Evaluation Scores"):
            for e in entries:
                st.markdown(
                    f"**Entry #{e.get('id')}** — `{e.get('sequence')}` "
                    f"(`{e.get('extraction_mode')}`, model=`{e.get('model')}`, run=`{e.get('run_identifier')}`)"
                )
                if e.get("evidence_span_parsed"):
                    st.json(e["evidence_span_parsed"])
                elif e.get("evidence_span"):
                    st.code(str(e["evidence_span"]))
                entry_evals = evals_by_entry.get(e.get("id"), [])
                if entry_evals:
                    st.json(entry_evals)
    else:
        st.info("No peptide entries recorded for this paper.")


def _apply_query_params() -> None:
    """Initialize navigation and search prefills from URL query parameters once."""
    if st.session_state.get("_qp_applied"):
        return
    st.session_state["_qp_applied"] = True
    try:
        qp = st.query_params
    except Exception:
        return

    page_param = qp.get("page")
    seq_param = qp.get("sequence")
    pmid_param = qp.get("pmid")
    doi_param = qp.get("doi")
    pid_param = qp.get("paper_id")

    if seq_param:
        st.session_state["search_seq_prefill"] = str(seq_param)
        if not page_param:
            page_param = "Peptide Search"

    if pmid_param or doi_param or pid_param:
        if pmid_param:
            st.session_state["inspector_query_prefill"] = str(pmid_param)
        elif doi_param:
            st.session_state["inspector_query_prefill"] = str(doi_param)
        if pid_param:
            try:
                st.session_state["inspector_paper_id_prefill"] = int(pid_param)
            except ValueError:
                pass
        if not page_param:
            page_param = "Paper Inspector"

    if page_param:
        st.session_state["nav_page"] = str(page_param)


def main() -> None:
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])
    _load_streamlit_secrets()
    _apply_query_params()
    client = _get_client()
    _ensure_azure_db_synced(client)
    page = _render_sidebar(client)
    try:
        st.query_params["page"] = page
    except Exception:
        pass

    if page == "Overview & DB":
        page_overview(client)
    elif page == "Peptide Search":
        page_peptide_search(client)
    elif page == "Paper Inspector":
        page_paper_inspector(client)
    elif page == "Run Pipeline":
        page_run_pipeline(client)
    elif page == "PDF Ingestion":
        page_pdf_ingestion(client)
    elif page == "DB Sync & Admin":
        page_db_sync(client)
    elif page == "Results & Plots":
        page_results(client)
    elif page == "Logs & Jobs":
        page_logs_and_jobs(client)


if __name__ == "__main__":
    main()
