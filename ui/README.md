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
