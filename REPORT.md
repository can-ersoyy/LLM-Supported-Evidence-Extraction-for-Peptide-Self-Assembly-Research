# Peptide Extractor - End-User Usability Report & Improvement Areas

## 1. Overview
The Peptide Extractor tool is a complex pipeline designed to mine scientific literature for experimental self-assembly data of 4–20mer peptides. The current state is highly developer-oriented. To serve an end-user audience (like biologists/researchers) who do not want to interact with Python code or debugging, there are significant improvements needed, specifically around tooling, pipeline invocation, environment setup, and data visibility.

## 2. Execution Pipeline and CLI Improvements

**Current Weakness:**
The primary execution method relies on a commented-out section under `if __name__ == "__main__":` in `src/peptide_extractor/pipeline.py`. To run different stages (search, abstract fetching, screening, full-text extraction, etc.), users must uncomment Python code and edit arguments manually. This is a dealbreaker for a non-technical end user.

```python
# snippet from src/peptide_extractor/pipeline.py
if __name__ == "__main__":
    # ...
    # 5. Run relevance screening
    #start_time = time.time()
    #success, fails, identifier = run_bulk_relevance_screening("mistral-small3.2:latest", "ollama", delay_seconds=0, id=papers_to_relevance_check, batch_size=10)

    # 7. Run extraction
    #fails, identifier = run_peptide_extraction(provider="google", model_name="gemini-3.5-flash-lite", delay_seconds=4, id=papers_to_extract[1490:1500])
```

**Recommendations:**
- **Create a Robust CLI Tool:** Implement a Command Line Interface (CLI) using `Typer` or `Click`. This allows users to execute pipeline steps sequentially or independently via standard bash commands.
- **Example Commands:**
  - `peptide-extractor search --query "peptide assembly" --provider ncbi`
  - `peptide-extractor screen --model gemma-4-31b-it --batch-size 10`
  - `peptide-extractor extract --model gemma-4-31b-it`
  - `peptide-extractor pipeline-run --all` (Executes the whole pipeline sequentially)
- **Configuration File:** Instead of directly editing variables in `pipeline.py`, the CLI should accept a YAML/TOML configuration file (`config.yaml`) where the user can define API keys, models, queries, batch sizes, and thresholds.

## 3. Environment and Setup

**Current Weakness:**
The current instructions in `AGENTS.md` require the user to run inside a Distrobox container using `uv run`. Managing Distrobox images, understanding the `uv` package manager, and keeping dependencies updated can be daunting. There are also hardcoded local LLM endpoints which will fail for a standard user.

```python
# snippet from src/peptide_extractor/relevance.py
def bulk_screen_papers(...):
    if classifier is None:
        classifier = LLMController(
            model_name=model_name,
            provider=provider,
            structured_output=BulkRelevanceResults,
            system_prompt=SYSTEM_PROMPT_BULK,
            base_url="http://localhost:11435" # HARDCODED URL
        )
```

Furthermore, we see similar hardcoded URLs in `evaluation.py`:
```python
# snippet from src/peptide_extractor/evaluation.py
def evaluate_extraction(extraction: PeptideEntry, fulltext: str) -> ExtractionEvaluation:
    model = LLMController(provider="ollama", model_name="mistral-small3.2:latest", system_prompt=GROUNDNESS_CHECK_SYSTEM_PROMPT, structured_output=GroundnessVerdict, base_url="http://localhost:11435")
```

**Recommendations:**
- **Simplified Installation:** Provide a standard `pip install` mechanism or a pre-built Docker image.
  - A Docker setup (`docker run -v ./data:/app/data peptide-extractor run-pipeline`) is often preferred because it isolates dependencies entirely without the user needing to understand `uv` or `distrobox`.
- **API and Model Configuration:** Make sure all endpoints (like Ollama) are configurable via the `.env` file or `config.yaml`. Remove hardcoded URLs from the source code. Pass these configuration options all the way down to the LLM initialization layers in `relevance.py`, `extraction.py`, and `evaluation.py`.

## 4. Pipeline State, Error Handling, and Logging

**Current Weakness:**
The pipeline relies heavily on standard `print()` statements. When processing hundreds of papers with rate limits and timeouts, the terminal output will be noisy and lacks a clear indicator of overall progress or time remaining.

```python
# snippet from src/peptide_extractor/pipeline.py -> run_peptide_extraction
        for paper in papers_to_extract:
            try:
                # ... extraction logic ...
            except Exception as e:
                failed_paper_ids.append(paper.id)
                print(f"Failed to extract peptides for paper {paper.id}: {e}")
```
This broad `except Exception as e` swallows errors without a traceback, making debugging hard if something unexpected (like a database schema error) occurs.

**Recommendations:**
- **Rich Progress Bars:** Implement `tqdm` or the `rich` library to show progress bars for long-running batch jobs (e.g., screening abstracts, downloading PDFs, running LLM extraction).
- **Structured Logging:** Replace all `print()` statements with a proper Python `logging` setup (`import logging`). Allow the user to specify log levels (e.g., `--verbose`) so they only see warnings/errors on the terminal while detailed debug info goes into a `pipeline.log` file.
- **Graceful Failures:** If an API key is missing or invalid, the pipeline should fail fast and inform the user immediately, rather than attempting requests and failing inside a loop.

## 5. Data Input, Output, and Database Interactions

**Current Weakness:**
All extracted data goes into a SQLite database (`data/papers.db`) via `SQLModel` and `Alembic`. Biologists and end-users generally expect tabular data (CSV, Excel) to review the results, perform statistical analysis, or import into other software. Telling a biologist to run an SQL query or write a Python script to get their data is a major usability flaw.

Furthermore, we see data loading from specific hardcoded CSV files instead of user-configurable paths:
```python
# snippet from src/peptide_extractor/golden_dataset_utils.py
GOLDEN_DATASET_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "golden_dataset_final.csv"
)
```

**Recommendations:**
- **Export Functionality:** Add an explicit export command to the CLI.
  - `peptide-extractor export --format csv --output results.csv`
  - This should dump the contents of the `PeptideEntry` table (and joined `PaperMetadata`, evaluation metrics, etc.) into a clean Excel or CSV file (using `pandas`).
- **Input Lists:** Allow users to provide a text file or CSV of DOIs/PMIDs they want to process, rather than relying exclusively on the search API. Pass the CSV path via CLI argument instead of hardcoding `GOLDEN_DATASET_CSV_PATH`.

## 6. Model Evaluation Extensibility

**Current Weakness:**
Evaluation models are tightly coupled to the logic. If a researcher wants to evaluate extractions using `gemini` instead of `ollama`, they must manually edit Python code.

```python
# snippet from src/peptide_extractor/evaluation.py
    #model = LLMController(provider="google-genai", model_name="gemini-3.5-flash-lite", system_prompt=JUDGE_SYSTEM_PROMPT, structured_output=FaithfulnessVerdict)
    model = LLMController(provider="ollama", model_name="mistral-small3.2:latest", system_prompt=GROUNDNESS_CHECK_SYSTEM_PROMPT, structured_output=GroundnessVerdict, base_url="http://localhost:11435")
```

**Recommendations:**
- **Decoupled Evaluators:** Read the evaluator model, provider, and base URL from the global configuration. Allow end-users to specify evaluator properties from the CLI (e.g. `peptide-extractor evaluate --model gemini-3.5-flash`).

## 7. Testing & Database Initialization

**Current Weakness:**
There is an ongoing issue where some tests (e.g., `test_ncbi_provider.py`) fail locally because the database schema (`raw_provider_records`) is not properly initialized in the test setup.

```bash
# Output from pytest
Error processing batch 0: (sqlite3.OperationalError) no such table: raw_provider_records
[SQL: INSERT INTO raw_provider_records (provider, search_query, raw_data) VALUES (?, ?, ?)]
```

**Recommendations:**
- **Test Database Initialization:** Fix the test environment so that tests initialize the SQLite in-memory database correctly (making sure `init_db()` is called and models are registered before `NCBIProvider` tries to insert into `raw_provider_records`). This ensures that developers can confidently maintain the tool without breaking the user experience.

## Summary of Priority Actions for End-User Adoption:
1. **Critical:** Build a Typer/Click CLI to replace `pipeline.py` script editing.
2. **Critical:** Add a CSV/Excel export function so users can actually see and use the extracted data without SQL.
3. **High:** Implement a unified config file (`config.yaml`) for search queries, model selection, evaluator configuration, and API keys. Remove all hardcoded `localhost` endpoints.
4. **Medium:** Add progress bars (`rich` or `tqdm`) and remove raw `print()` statements. Make error logs informative instead of silently catching and skipping exceptions.
5. **Medium:** Fix failing tests by ensuring the database tables are created in the pytest fixtures.
