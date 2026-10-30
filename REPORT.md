# Peptide Extractor - End-User Usability Report & Improvement Areas

## 1. Overview
The Peptide Extractor tool is a complex pipeline designed to mine scientific literature for experimental self-assembly data of 4–20mer peptides. The current state is highly developer-oriented. To serve an end-user audience (like biologists/researchers) who do not want to interact with Python code or debugging, there are significant improvements needed, specifically around tooling, pipeline invocation, environment setup, and data visibility.

## 2. Execution Pipeline and CLI Improvements

**Current Weakness:**
The primary execution method relies on a commented-out section under `if __name__ == "__main__":` in `src/peptide_extractor/pipeline.py`. To run different stages (search, abstract fetching, screening, full-text extraction, etc.), users must uncomment Python code and edit arguments manually. This is a dealbreaker for a non-technical end user.

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
The current instructions require the user to run inside a Distrobox container using `uv run`. Managing Distrobox images, understanding the `uv` package manager, and keeping dependencies updated can be daunting. There are also hardcoded local LLM endpoints (like `http://localhost:11435` in `relevance.py`) which will fail for a standard user.

**Recommendations:**
- **Simplified Installation:** Provide a standard `pip install` mechanism or a pre-built Docker image.
  - A Docker setup (`docker run -v ./data:/app/data peptide-extractor run-pipeline`) is often preferred because it isolates dependencies entirely without the user needing to understand `uv` or `distrobox`.
- **API and Model Configuration:** Make sure all endpoints (like Ollama) are configurable via the `.env` file or `config.yaml`. Remove hardcoded URLs like `http://localhost:11435` from the source code.

## 4. Pipeline State and User Feedback (Logging)

**Current Weakness:**
The pipeline relies heavily on standard `print()` statements and sporadic `logger.info()` calls. When processing hundreds of papers with rate limits and timeouts (e.g., `time.sleep(0.2)` or `time.sleep(2.0)`), the terminal output will be noisy, hard to parse, and lacks a clear indicator of overall progress or time remaining.

**Recommendations:**
- **Rich Progress Bars:** Implement `tqdm` or the `rich` library to show progress bars for long-running batch jobs (e.g., screening abstracts, downloading PDFs, running LLM extraction).
- **Structured Logging:** Replace all `print()` statements with a proper Python `logging` setup. Allow the user to specify log levels (e.g., `--verbose`) so they only see warnings/errors on the terminal while detailed debug info goes into a `pipeline.log` file.
- **Resilience and State Checkpoints:** The user needs to know where the pipeline stopped if it crashes. Currently, failures just print exceptions and skip papers. The CLI should provide a summary at the end of a run (e.g., "Extracted 45 papers successfully. 5 failed. See `errors.log`").

## 5. Data Input, Output, and Database Interactions

**Current Weakness:**
All extracted data goes into a SQLite database (`data/papers.db`) via `SQLModel` and `Alembic`. Biologists and end-users generally expect tabular data (CSV, Excel) to review the results, perform statistical analysis, or import into other software. Telling a biologist to run an SQL query or write a Python script to get their data is a major usability flaw.

**Recommendations:**
- **Export Functionality:** Add an explicit export command to the CLI.
  - `peptide-extractor export --format csv --output results.csv`
  - This should dump the contents of the `PeptideEntry` table (and joined `PaperMetadata`) into a clean Excel or CSV file.
- **Input Lists:** Allow users to provide a text file or CSV of DOIs/PMIDs they want to process, rather than relying exclusively on the search API.

## 6. Testing & Error Handling

**Current Weakness:**
The current `pipeline.py` wraps critical LLM steps and API requests in broad `try...except Exception as e:` blocks. It just prints the error and continues. There is also an ongoing issue where some tests (e.g., `test_ncbi_provider.py`) fail locally because the database schema (`raw_provider_records`) is not properly initialized in the test setup.

**Recommendations:**
- **Graceful Failures:** If an API key is missing or invalid, the pipeline should fail fast and inform the user immediately, rather than attempting requests and failing inside a loop.
- **Test Database Initialization:** Fix the test environment so that tests initialize the SQLite in-memory database correctly (making sure `init_db()` is called and models are registered before `NCBIProvider` tries to insert into `raw_provider_records`).

## Summary of Priority Actions for End-User Adoption:
1. **Critical:** Build a Typer/Click CLI to replace `pipeline.py` script editing.
2. **Critical:** Add a CSV/Excel export function so users can actually see and use the extracted data without SQL.
3. **High:** Implement a unified config file (`config.yaml`) for search queries, model selection, and API keys.
4. **Medium:** Add progress bars (`rich` or `tqdm`) and remove raw `print()` statements.
5. **Medium:** Fix failing tests by ensuring the database tables are created in the pytest fixtures.
