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
