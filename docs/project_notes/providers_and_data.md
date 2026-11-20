# Literature Providers, Storage & Visualization

This document covers data acquisition providers, full-text storage/caching, and the unified visualization framework.

---

## 1. Literature Providers (`providers.py`)

All retrieval providers live in `src/peptide_extractor/providers.py` and normalize external API responses into `PaperMetadata` records (and `RawProviderRecord` archives).

| Provider Class              | API / Source                               | Capabilities                                                                                    | Notes                                                                                                                                                              |
| :-------------------------- | :----------------------------------------- | :---------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NCBIProvider`            | NCBI E-Utilities (`esearch`, `efetch`) | `search_by_query`, `search_by_ids`, `fetch` (JATS XML)                                    | Supports `db="pmc"` and `db="pubmed"`. Uses `NCBI_API_KEY`, batches requests (`batch_size=100`–`150`), and sleeps between calls to respect rate limits. |
| `OpenAlexProvider`        | OpenAlex Works REST API                    | Abstract retrieval by DOI (`find_abstracts_by_doi`), Open Access PDF URL discovery & download | Uses `OPEN_ALEX_API_KEY` (or polite pool). Used as a fallback when PMC JATS XML is unavailable.                                                                  |
| `ScopusProvider`          | Elsevier Scopus Search API                 | `search_by_query` (title/DOI metadata)                                                        | Uses `SCOPUS_API_KEY`. Metadata-only (no full text).                                                                                                             |
| `SemanticScholarProvider` | Semantic Scholar API                       | Metadata / abstract lookup                                                                      | Auxiliary metadata provider.                                                                                                                                       |

### Metadata Cleaning & Full-Text Fetching (`pipeline.py`)

1. **`deduplicate_by_doi(normalize=True, remove_empty=True)`**: Normalizes DOIs (lowercasing, stripping URL prefixes) and removes duplicate `PaperMetadata` rows.
2. **`find_abstracts_by_doi(open_alex_api_key)`**: Backfills missing abstracts via OpenAlex.
3. **`remove_empty_abstract_title_papers()`**: Deletes papers that still lack a title or abstract before LLM screening.
4. **`fetch_paper_fulltexts()`**: Targets `PaperStatus.USABLE` papers without existing `PaperContent`. Attempts **PMC JATS XML** first via `NCBIProvider`; if unavailable, falls back to **PDF** download via `OpenAlexProvider` / `pdf_url`.
5. **`ingest_manual_pdfs()`**: Ingests manually downloaded PDFs from a local directory (e.g., for paywalled golden-dataset papers) and links them to `PaperMetadata`.

---

## 2. Azure Blob Storage & Local Caching (`azure_storage.py`)

Full-text JATS XML strings and PDF bytes are **not** stored inside SQLite. Instead:

- `upload_xml()` / `upload_pdf()` store content in Azure Blob Storage and save the resulting `blob_url` on `PaperContent`.
- `PaperContent.get_content(use_cache=True)` calls `download_xml()` or `download_pdf()`, checking a local cache directory first before fetching from Azure Blob Storage.

---

## 3. Visualization Framework (`visualization.py`)

`src/peptide_extractor/visualization.py` provides a unified `visualize()` function and CLI command (`peptide_extractor visualize` or the `visualization` pipeline step) that validates `(model, operation)` compatibility, queries records via `get_records()`, and exports PNG/SVG figures.

### Supported Operations (`VisualizationOperation`)

| Target Model             | Operation                           | Description                                                                                   |
| :----------------------- | :---------------------------------- | :-------------------------------------------------------------------------------------------- |
| `PeptideEntry`         | `peptide_column_completion`       | Non-null completion percentage across all extracted columns.                                  |
| `PeptideEntry`         | `peptide_length_distribution`     | Histogram/bar chart of peptide sequence lengths.                                              |
| `PeptideEntry`         | `extractions_per_paper`           | Distribution of extracted entry counts per paper.                                             |
| `PeptideEntry`         | `peptide_morphology_distribution` | Counts of `type` morphology classes (`hydrogel`, `fiber`, `amyloid`, `none`, etc.). |
| `PeptideEntry`         | `peptide_run_comparison`          | Side-by-side comparison across multiple `run_identifier` values.                            |
| `ExtractionEvaluation` | `evaluation_boxplot`              | Boxplots of `completeness_rate` and `groundness_rate`.                                    |
| `ExtractionEvaluation` | `evaluation_summary_range`        | Mean/range summary plot of evaluation metrics.                                                |
| `ExtractionEvaluation` | `evaluation_distribution`         | Distribution histograms of evaluation scores.                                                 |
| `PaperRelevance`       | `relevance_confidence_hist`       | Histogram of abstract screening confidence scores.                                            |
| `PaperRelevance`       | `relevance_outcome_counts`        | Counts of relevant vs. discarded papers.                                                      |
| `PaperMetadata`        | `paper_status_counts`             | Breakdown of `not_processed`, `usable`, and `discard` papers.                           |
