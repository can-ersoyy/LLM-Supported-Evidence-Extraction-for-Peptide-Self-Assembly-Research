# LLM-Assisted Evidence Extraction for 4–20mer Peptide Self-Assembly Literature

This is the main development repository for the IDP project.

## CLI Usage

The pipeline can be run using the provided CLI and a YAML configuration file.

### Configuration

Create a `config.yaml` file with the desired pipeline steps and parameters:

```yaml
log_level: "INFO"
ncbi_batch_size: 100

# List of steps to run in order
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

### Running

Execute the CLI by passing the config file path:

```bash
uv run src/peptide_extractor/cli.py run_pipeline --config config.yaml
```

## Full-Text Preprocessing & GROBID Setup

The pipeline supports both **JATS XML** and **scholarly PDFs**:
- **JATS XML**: Processed directly via PubMed Central into the unified object tree.
- **PDFs**: Processed into TEI XML using [GROBID](https://grobid.readthedocs.io/), or a lightweight zero-daemon fallback (`pdftotext`/`pypdf`). Both XML formats produce a unified tree of citation-grounded nodes (`sec-0/p-0/s-0`, `sec-1/table-0/row-0`).

For instructions on running GROBID via Docker and configuring it in the pipeline, see [docs/grobid_setup.md](docs/grobid_setup.md).

