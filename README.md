# LLM-Assisted Evidence Extraction for 4–20mer Peptide Self-Assembly Literature

`peptide_extractor` is an end-to-end scientific literature curation pipeline that discovers publications on short-peptide self-assembly, screens abstracts for experimental relevance, fetches and parses full-text JATS XML and PDFs into a unified citation-grounded node tree, extracts structured `PeptideEntry` records via LLMs, and evaluates both reference-free grounding and golden-dataset accuracy.

---
#### Streamlit (private) URL: https://gdj4m9vp3rbvzqbsbezscm.streamlit.app/

## Documentation Index

### System & Technical Guides
- **[System Design & Pipeline Architecture](docs/project_notes/architecture.md)**:
  - [End-to-End Pipeline Flow](docs/project_notes/architecture.md#1-end-to-end-pipeline-flow)
  - [Pipeline Lifecycle & Data Models](docs/project_notes/architecture.md#2-pipeline-lifecycle--data-models)
  - [CLI & YAML Configuration](docs/project_notes/architecture.md#3-cli--yaml-configuration)
  - [Module Map](docs/project_notes/architecture.md#4-module-map-srcpeptide_extractor)
- **[Document Parsing & Multimodal Construction](docs/project_notes/parsing.md)**:
  - [Supported Input Formats & Conversion Flow](docs/project_notes/parsing.md#1-supported-input-formats--conversion-flow)
  - [Unified Node Object Tree (`sec`/`p`/`s`/`table`/`fig`)](docs/project_notes/parsing.md#2-the-unified-node-object-tree)
  - [PDF Table & Figure Cropping](docs/project_notes/parsing.md#3-pdf-table--figure-cropping-table_rendererpy)
  - [Serialization Formats for LLMs (`to_llm_string` & `to_llm_blocks`)](docs/project_notes/parsing.md#4-serialization-formats-for-llms)
- **[LLM Workflows, Prompts & Extraction Modes](docs/project_notes/llm_methods.md)**:
  - [Provider & Structured Output Layer](docs/project_notes/llm_methods.md#1-provider--structured-output-layer-llmpy)
  - [Stage 1: Abstract Relevance Screening](docs/project_notes/llm_methods.md#2-stage-1-abstract-relevance-screening-relevancepy)
  - [Stage 2: Peptide Extraction (Direct PDF, Factored Two-Pass, Single-Pass Multimodal, Text-Only)](docs/project_notes/llm_methods.md#3-stage-2-peptide-extraction-extractionpy--pipelinepy)
  - [Stage 3: Evaluation & Golden Benchmarking](docs/project_notes/llm_methods.md#4-stage-3-evaluation--benchmarking)
- **[Literature Providers, Storage & Visualization](docs/project_notes/providers_and_data.md)**:
  - [Literature Providers & Full-Text Fetching](docs/project_notes/providers_and_data.md#1-literature-providers-providerspy)
  - [Azure Blob Storage & Caching](docs/project_notes/providers_and_data.md#2-azure-blob-storage--local-caching-azure_storagepy)
  - [Visualization Framework](docs/project_notes/providers_and_data.md#3-visualization-framework-visualizationpy)

### Setup Guides (`docs/project_notes/`)
- **[GROBID Setup & PDF Preprocessing Guide](docs/project_notes/grobid_setup.md)**: Docker/standalone setup for converting scholarly PDFs into TEI XML.
- **[Alembic Database Migration Guide](docs/project_notes/alembic_tutorial.md)**: Creating and applying SQLModel schema migrations.

---

## Quick Start (CLI)

### 1. Run Pipeline Steps via YAML Config

Create a `config.yaml` (or use `default_config.yaml`):

```yaml
log_level: "INFO"
log_to_file: true
ncbi_batch_size: 100

steps:
  - search
  - deduplicate
  - find_abstracts
  - clean_empty
  - relevance
  - fetch_fulltext
  - extraction
  - evaluation

search:
  db: "pmc"
  queries:
    - "peptide self-assembly"

relevance:
  model_name: "gemini-3.5-flash-lite"
  provider: "google"
  batch_size: 50
  delay_seconds: 3

extraction:
  provider: "google"
  model_name: "gemini-3.5-flash-lite"
  delay_seconds: 4
```

Execute the pipeline:

```bash
uv run src/peptide_extractor/cli.py run_pipeline --config config.yaml
```

### 2. Generate Visualizations

```bash
uv run src/peptide_extractor/cli.py visualize \
  --model peptide_entry \
  --operation peptide_column_completion \
  --format png
```

### 3. Run Tests

Development commands run inside the `idp-dev` Distrobox container:

```bash
distrobox enter idp-dev -- uv run pytest
```
