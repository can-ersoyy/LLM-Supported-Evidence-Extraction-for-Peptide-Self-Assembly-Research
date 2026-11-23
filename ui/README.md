# Peptide Extractor UI

Simple Streamlit UI and optional FastAPI server for `peptide_extractor`.

## Installation

From the repository root using `uv`:

```bash
uv sync --extra ui
```

Or with `pip`:

```bash
pip install -r ui/requirements.txt
```

## Mode 1: Local (In-Process)

Run Streamlit directly on the same machine where the repo and database live:

```bash
uv run streamlit run ui/app.py
```

In the sidebar, keep **Backend Mode** set to `Local (In-Process)`.

## Mode 2: Separate Pipeline API Server + Remote UI

1. **On the server running the pipeline & database:**

```bash
# Optional: protect the API with an API key
export PEPTIDE_API_KEY="your-secret-key"

uv run uvicorn ui.api:app --host 0.0.0.0 --port 8000
```

2. **On your local machine (or anywhere running the UI):**

```bash
# Optional: pre-configure connection via environment variables
export PEPTIDE_API_URL="http://your-server:8000"
export PEPTIDE_API_KEY="your-secret-key"

uv run streamlit run ui/app.py
```

In the sidebar, select `Remote API Server`, enter the server URL (and API key if configured), and click **Test Connection**.

## Mode 3: Streamlit Community Cloud (Always-On Read-Only + Azure DB Snapshot)

1. In [Streamlit Community Cloud](https://share.streamlit.io), create a new app pointing to this repository with **Main file path** set to `ui/app.py`.
2. In **App Settings → Sharing**, set the app to **Private** if you want to restrict access to invited emails/accounts.
3. In **App Settings → Secrets**, add your read-only configuration:

```toml
PEPTIDE_READ_ONLY = "true"
AZURE_DB_SAS_URL = "https://<account>.blob.core.windows.net/db-backups/peptide-extractor.db.tar.gz?<sas_token>"
# Optional: refresh interval in seconds (default: 3600)
AZURE_DB_REFRESH_TTL_SECONDS = "3600"
```

*(Alternatively, instead of `AZURE_DB_SAS_URL`, you can set `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_CONTAINER_NAME` [default: `db-backups`], and `AZURE_BLOB_NAME` [default: `peptide-extractor.db.tar.gz`]. Both `.tar.gz` archives and raw `.db` files are supported.)*

On startup, the app downloads the latest SQLite snapshot from Azure. Visitors can also upload their own `.db` file in the sidebar's **Database Source (Session)** expander to inspect it in isolation without affecting other users.

