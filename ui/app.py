"""Simple Streamlit GUI for peptide_extractor."""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
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

    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview & DB",
            "Run Pipeline",
            "PDF Ingestion",
            "DB Sync & Admin",
            "Results & Plots",
            "Logs & Jobs",
        ],
    )

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
    if st.button("Refresh Stats"):
        st.rerun()

    try:
        stats = client.get_db_stats()
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
    tab_data, tab_viz = st.tabs(["Data Explorer", "Visualizations"])

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

        try:
            data = client.query_records(table, limit=int(limit), offset=0, filters=filters)
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
                        },
                    )
                    _submit_and_track(job)
                except Exception as exc:
                    st.error(str(exc))

        st.divider()
        st.subheader("Saved Diagrams")
        if st.button("Refresh Diagrams"):
            st.rerun()
        try:
            items = client.list_visualizations()
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


def main() -> None:
    client = _get_client()
    page = _render_sidebar(client)

    if page == "Overview & DB":
        page_overview(client)
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
