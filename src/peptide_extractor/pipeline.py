from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, Union
from sqlmodel import SQLModel, select
from peptide_extractor.visualization import (
    VisualizationOperation,
    VisualizationResult,
    visualize,
)
from peptide_extractor.logging_utils import (
    step_logger,
    is_file_logging_enabled,
    get_project_root,
    LOG_TO_FILE as _DEFAULT_LOG_TO_FILE,
)
from peptide_extractor.azure_storage import (
    DEFAULT_CACHE_DIR,
    build_blob_name,
    upload_pdf,
)
from peptide_extractor.db import PaperStatus
from peptide_extractor.db import DatasetSubset
from peptide_extractor.schemas import RelevanceResult
from peptide_extractor.evaluation import evaluate_extraction
from peptide_extractor.db.peptide_entry import PeptideEntry
from peptide_extractor.db import peptide_llm_entry_to_peptide_entry
from peptide_extractor.extraction import run_extraction
from peptide_extractor.parsers import (
    extraction_ready_blocks_from_pdf,
    extraction_ready_blocks_from_xml,
    extraction_ready_string_from_pdf,
    extraction_ready_string_from_xml,
    to_extraction_ready_blocks,
    to_extraction_ready_string,
)
from peptide_extractor.db.paper_content import get_paper_content_by_paper_id
from peptide_extractor.db import PaperContent
from peptide_extractor.utils import doi_to_filename, normalize_database_dois
from peptide_extractor.relevance import screen_papers
from peptide_extractor.providers import NCBIProvider
from peptide_extractor.db import get_records
from peptide_extractor.db import PaperMetadata
from peptide_extractor.providers import OpenAlexProvider
import time
from peptide_extractor.providers import SemanticScholarProvider
from peptide_extractor.db import get_paper_metadatas
from sqlmodel import Session
from peptide_extractor.db import get_session_context, init_db

import logging
from peptide_extractor.logging_utils import (
    step_logger,
    is_file_logging_enabled,
    LOG_TO_FILE as _DEFAULT_LOG_TO_FILE,
)

logger = logging.getLogger(__name__)

# Set this variable to True (or pass log_to_file=True) to store logs in docs/logs/[step]/{run_identifier}.log
LOG_TO_FILE: bool = _DEFAULT_LOG_TO_FILE


def _is_logging_active(step: str, log_to_file: Optional[bool] = None) -> bool:
    if log_to_file is not None:
        return bool(log_to_file)
    if LOG_TO_FILE:
        return True
    return is_file_logging_enabled(step=step)


# maps peptide length to structure name or self-assembly keyword, includes open-access, english and non review article filters, only title and abstract
NCBI_SEARCH_QUERY_V1 = """((tetrapeptide[tiab] OR pentapeptide[tiab] OR hexapeptide[tiab] OR heptapeptide[tiab] OR octapeptide[tiab] OR nonapeptide[tiab] OR decapeptide[tiab] OR undecapeptide[tiab] OR dodecapeptide[tiab] OR tridecapeptide[tiab] OR tetradecapeptide[tiab] OR pentadecapeptide[tiab] OR hexadecapeptide[tiab] OR heptadecapeptide[tiab] OR octadecapeptide[tiab] OR nonadecapeptide[tiab] OR eicosapeptide[tiab] OR icosapeptide[tiab]) AND (nanostructure[tiab] OR nanotube[tiab] OR nanofiber[tiab] OR nanofibre[tiab] OR nanorod[tiab] OR hydrogel[tiab] OR nanosphere[tiab] OR nanoparticle[tiab] OR "self-assembly"[tiab])) AND (english[la]) AND ("free full text"[filter]) NOT (review[pt])"""
NCBI_SEARCH_QUERY_V2 = """peptide self-assembly"""

# this query on pmc returns 55k results (pmc only returns the first 10k per query)
NCBI_SEARCH_QUERY_V3_TEMPLATE_INTERVAL = """("peptid"[All Fields] OR "peptidal"[All Fields] OR "peptide s"[All Fields] OR "peptides"[Supplementary Concept] OR "peptides"[All Fields] OR "peptide"[All Fields] OR "peptides"[MeSH Terms] OR "peptidic"[All Fields]) AND "self-assembly"[All Fields] AND (open_access[Filter]) AND ({}:{}[pdat]) AND (pmc_public[Filter])"""
NCBI_SEARCH_QUERY_V3_TEMPLATE_YEAR = """("peptid"[All Fields] OR "peptidal"[All Fields] OR "peptide s"[All Fields] OR "peptides"[Supplementary Concept] OR "peptides"[All Fields] OR "peptide"[All Fields] OR "peptides"[MeSH Terms] OR "peptidic"[All Fields]) AND "self-assembly"[All Fields] AND (open_access[Filter]) AND ({}[pdat]) AND (pmc_public[Filter])"""
NCBI_SEARCH_QUERIES_V3 = [NCBI_SEARCH_QUERY_V3_TEMPLATE_INTERVAL.format("1900", "2009")]
NCBI_SEARCH_QUERIES_V3.extend(
    [NCBI_SEARCH_QUERY_V3_TEMPLATE_YEAR.format(year) for year in range(2010, 2028)]
)

SCOPUS_SEARCH_QUERY_V1 = """TITLE-ABS-KEY(tetrapeptide OR pentapeptide OR hexapeptide OR heptapeptide OR octapeptide OR nonapeptide OR decapeptide OR undecapeptide OR dodecapeptide OR tridecapeptide OR tetradecapeptide OR pentadecapeptide OR hexadecapeptide OR heptadecapeptide OR octadecapeptide OR nonadecapeptide OR eicosapeptide OR icosapeptide) AND TITLE-ABS-KEY(nanostructure OR nanotube OR nanofiber OR nanofibre OR nanorod OR hydrogel OR nanosphere OR nanoparticle OR "self-assembly") AND LANGUAGE(english) AND DOCTYPE(ar) AND OA(all)"""


def retrive_ncbi_by_query(
    ncbi_provider: NCBIProvider,
    query_list=[NCBI_SEARCH_QUERY_V1],
    save=True,
    db="pubmed",
    delete_raw_records=True,
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
):
    """Wrapper method for doing query search on pubmed/pmc from ncbi provider."""
    if run_identifier is None:
        run_identifier = int(time.time())

    with step_logger(
        "search",
        run_identifier,
        enabled=_is_logging_active("search", log_to_file),
    ):
        raw_records = []
        for query in query_list:
            logger.info(f"Query: {query}")
            raw_records.extend(ncbi_provider.search_by_query(query, db=db))
        ncbi_metadatas = ncbi_provider.parse_raw_search_records(raw_records)

        for metadata in ncbi_metadatas:
            metadata.run_identifier = run_identifier

        if save:
            init_db()
            with get_session_context() as session:
                session.add_all(ncbi_metadatas)
                session.commit()

            if (
                delete_raw_records
            ):  # cleanup since raw records are only needed before storing metadata
                init_db()
                with get_session_context() as session:
                    for record in raw_records:
                        session.delete(record)
                    session.commit()
            return run_identifier
        else:
            return ncbi_metadatas


def retrieve_scopus_by_query(
    scopus_provider,
    query=SCOPUS_SEARCH_QUERY_V1,
    save=True,
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
):
    if run_identifier is None:
        run_identifier = int(time.time())

    with step_logger(
        "search",
        run_identifier,
        enabled=_is_logging_active("search", log_to_file),
    ):
        raw_records = scopus_provider.search_by_query(query)
        scopus_metadatas = scopus_provider.parse_raw_search_records(raw_records)

        for metadata in scopus_metadatas:
            metadata.run_identifier = run_identifier

        if save:
            init_db()
            with get_session_context() as session:
                session.add_all(scopus_metadatas)
                session.commit()
            return run_identifier
        else:
            return scopus_metadatas


def deduplicate_by_doi(
    session: Optional[Session] = None, normalize: bool = True, remove_empty: bool = True
) -> int:
    """Updated to remove papers with duplicate doi. Also supports normalization and removal of papers with empty doi
    Returns deleted count
    """

    def _deduplicate(s: Session):
        scraped_papers = get_records(PaperMetadata, session=s)
        seen_papers = []
        deleted_count = 0
        logger.info(f"Found {len(scraped_papers)} scraped papers.")
        for paper in scraped_papers:
            if paper.doi is None:
                if remove_empty:
                    s.delete(paper)
                    deleted_count += 1
                continue
            if paper.doi in seen_papers:
                s.delete(paper)
                deleted_count += 1
            else:
                seen_papers.append(paper.doi)
        s.commit()
        return deleted_count

    if session is not None:
        if normalize:
            normalize_database_dois(session)
        return _deduplicate(session)
    init_db()
    with get_session_context() as s:
        if normalize:
            normalize_database_dois(s)
        return _deduplicate(s)


def find_abstracts_by_doi(
    open_alex_api_key: Optional[str] = None,
    session: Optional[Session] = None,
    **filters: Any,
) -> int:
    if filters.get("id") == []:
        return 0

    def _find(s: Session) -> int:
        empty_str_papers = get_paper_metadatas(abstract="", session=s, **filters)
        none_papers = get_paper_metadatas(abstract=None, session=s, **filters)
        seen_ids = set()
        non_abstract_metadas: List[PaperMetadata] = []
        for paper in empty_str_papers + none_papers:
            if paper.id not in seen_ids:
                seen_ids.add(paper.id)
                non_abstract_metadas.append(paper)

        if not non_abstract_metadas:
            return 0

        found_abstract_count = 0
        semantic_scholar = SemanticScholarProvider()
        open_alex = OpenAlexProvider(api_key=open_alex_api_key)

        for metadata in non_abstract_metadas:
            if metadata.doi is None:
                logger.info(f"Skipping paper {metadata.id}: no DOI")
                continue
            try:
                abstract = semantic_scholar.get_abstract_by_doi(metadata.doi)

                logger.info(abstract)

                if abstract is not None:
                    found_abstract_count += 1
                    metadata.abstract = abstract
                    s.add(metadata)
                    s.commit()
                else:
                    try:
                        abstract = open_alex.get_abstract_by_doi(metadata.doi)
                        time.sleep(0.5)
                        if abstract is not None:
                            found_abstract_count += 1
                            metadata.abstract = abstract
                            s.add(metadata)
                            s.commit()
                    except Exception as e:
                        logger.info(
                            f"Failed to retrieve abstract for {metadata.doi}: {e}"
                        )
                logger.info(f"Retrieved abstract for {metadata.doi}: {abstract}")
            except Exception as e:
                if "429" in str(e):
                    try:
                        abstract = open_alex.get_abstract_by_doi(metadata.doi)
                        time.sleep(0.5)
                        if abstract is not None:
                            found_abstract_count += 1
                            metadata.abstract = abstract
                            s.add(metadata)
                            s.commit()
                        logger.info(
                            f"Retrieved abstract for {metadata.doi}: {abstract}"
                        )
                    except Exception as e:
                        logger.info(
                            f"Failed to retrieve abstract for {metadata.doi}: {e}"
                        )
                logger.info(f"Failed to retrieve abstract for {metadata.doi}: {e}")

            time.sleep(1)

        logger.info(
            f"Retrieved {found_abstract_count} abstracts out of {len(non_abstract_metadas)}"
        )
        return found_abstract_count

    if session is not None:
        return _find(session)
    init_db()
    with get_session_context() as s:
        return _find(s)


def remove_empty_abstract_title_papers(
    session: Optional[Session] = None,
    **filters: Any,
) -> int:
    """Returns removed count"""
    if filters.get("id") == []:
        return 0

    def _remove(s: Session) -> int:
        empty_abstract = get_paper_metadatas(abstract="", session=s, **filters)
        none_abstract = get_paper_metadatas(abstract=None, session=s, **filters)
        empty_title = get_paper_metadatas(title="", session=s, **filters)

        seen_ids = set()
        empty_papers: List[PaperMetadata] = []
        for paper in empty_abstract + none_abstract + empty_title:
            if paper.id not in seen_ids:
                seen_ids.add(paper.id)
                empty_papers.append(paper)

        deleted_count = 0
        for paper in empty_papers:
            s.delete(paper)
            deleted_count += 1
        s.commit()
        return deleted_count

    if session is not None:
        return _remove(session)
    init_db()
    with get_session_context() as s:
        return _remove(s)


def run_relevance_screening(
    model_name: str,
    provider: str,
    session: Optional[Session] = None,
    batch_size: Optional[int] = None,
    delay_seconds: float = 2.0,
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    **kwargs,
) -> tuple[List[tuple[PaperMetadata, RelevanceResult]], List[int], int]:
    """Run abstract relevance screening on paper metadata records in the database.

    Evaluates papers using the relevance classifier, persisting PaperRelevance records
    and updating paper status to USABLE or DISCARD.

    Returns the succesful paper/relevance pairs, failed_paper_ids and a run_identifier
    """
    if run_identifier is None:
        run_identifier = int(time.time())

    def _run(session: Session):
        screened_results, failed_paper_ids = screen_papers(
            model_name=model_name,
            provider=provider,
            session=session,
            batch_size=batch_size,
            delay_seconds=delay_seconds,
            run_identifier=run_identifier,
            **kwargs,
        )
        return screened_results, failed_paper_ids, run_identifier

    with step_logger(
        "relevance",
        run_identifier,
        enabled=_is_logging_active("relevance", log_to_file),
    ):
        if session is not None:
            return _run(session)
        init_db()
        with get_session_context() as s:
            return _run(s)


def run_bulk_relevance_screening(
    model_name: str,
    provider: str,
    batch_size: Optional[int] = 10,
    delay_seconds: float = 2.0,
    session: Optional[Session] = None,
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    **kwargs,
):
    """Run abstract relevance screening on paper metadata records in the database in bulk.

    Evaluates papers using the relevance classifier, persisting PaperRelevance records
    and updating paper status to USABLE or DISCARD.

    Returns the succesful paper/relevance pairs, failed_paper_ids and a run_identifier
    """
    from peptide_extractor.relevance import bulk_screen_papers

    if run_identifier is None:
        run_identifier = int(time.time())

    def _run(session: Session):
        screened_results, failed_paper_ids = bulk_screen_papers(
            model_name=model_name,
            provider=provider,
            session=session,
            batch_size=batch_size,
            delay_seconds=delay_seconds,
            run_identifier=run_identifier,
            **kwargs,
        )
        return screened_results, failed_paper_ids, run_identifier

    with step_logger(
        "relevance",
        run_identifier,
        enabled=_is_logging_active("relevance", log_to_file),
    ):
        if session is not None:
            return _run(session)
        init_db()
        with get_session_context() as s:
            return _run(s)


def fetch_paper_fulltexts(
    session: Optional[Session] = None,
    ncbi_provider: NCBIProvider = None,
    open_alex_provider: OpenAlexProvider = None,
    skip_existing_content: bool = True,
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    **kwargs,
):
    """Retrieve fulltexts according to db query"""
    if run_identifier is None:
        run_identifier = int(time.time())

    def _fetch(session: Session):
        papers_to_retrieve = get_paper_metadatas(**kwargs, session=session)
        logger.info(papers_to_retrieve)

        success_count = 0
        fail_count = 0

        for paper in papers_to_retrieve:
            paper_content: Optional[PaperContent] = get_paper_content_by_paper_id(
                paper.id, session
            )

            if skip_existing_content and paper_content is not None:
                logger.info(f"Paper {paper.id} already has content")
                continue

            # normally it is better to batch retrieve from pubmed but for now just singular fetch
            try:
                paper_contents, _, _ = ncbi_provider.fetch([paper])
                paper_content = paper_contents[0]
            except Exception as e:
                logger.info(f"Failed to retrieve fulltext for {paper.doi}: {e}")
                logger.info("Trying pdf retrieval")
                try:
                    paper_content = open_alex_provider.fetch(paper)

                except Exception as e:
                    logger.info(f"Failed to retrieve fulltext for {paper.doi}: {e}")

            if paper_content is not None:
                paper_content.run_identifier = run_identifier
                session.add(paper_content)
                session.commit()
                success_count += 1
            else:
                fail_count += 1

            time.sleep(1)

        logger.info(f"Success: {success_count}, Fail: {fail_count}")
        return success_count, fail_count, run_identifier

    with step_logger(
        "fetch",
        run_identifier,
        enabled=_is_logging_active("fetch", log_to_file),
    ):
        if session is not None:
            return _fetch(session)
        init_db()
        with get_session_context() as s:
            return _fetch(s)


def ingest_manual_pdfs(
    pdf_dir: Union[str, Path],
    session: Optional[Session] = None,
    skip_existing_content: bool = True,
    cache_locally: bool = True,
    cache_dir: Optional[Path] = None,
    source: str = "manual",
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
) -> Dict[str, object]:
    """Upload manually downloaded PDFs to Azure Blob Storage and register PaperContent rows.

    Each ``<pmid>.pdf`` file in ``pdf_dir`` is matched against existing
    ``PaperMetadata.pmid`` records.
    """
    target_dir = Path(pdf_dir)
    if not target_dir.is_dir():
        raise FileNotFoundError(f"PDF directory does not exist: {target_dir}")

    if run_identifier is None:
        run_identifier = int(time.time())

    def _ingest(s: Session) -> Dict[str, object]:
        pdf_files = sorted(
            p
            for p in target_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        )
        uploaded: List[int] = []
        skipped: List[int] = []
        unmatched: List[str] = []
        failed: List[str] = []

        for pdf_path in pdf_files:
            pmid = pdf_path.stem.strip()
            matches = get_paper_metadatas(session=s, pmid=pmid) if pmid else []
            paper = matches[0] if matches else None
            if paper is None or paper.id is None:
                logger.warning(
                    f"No matching PaperMetadata found for PMID '{pmid}' ('{pdf_path.name}')"
                )
                unmatched.append(pdf_path.name)
                continue

            existing_content = get_paper_content_by_paper_id(paper.id, s)
            if existing_content is not None and skip_existing_content:
                logger.info(
                    f"Skipping paper {paper.id} ({pdf_path.name}): content already exists"
                )
                skipped.append(paper.id)
                continue

            try:
                pdf_bytes = pdf_path.read_bytes()
                if not pdf_bytes:
                    raise ValueError(f"PDF file is empty: {pdf_path.name}")

                identifier = (
                    doi_to_filename(paper.doi) if paper.doi else f"paper_{paper.id}"
                )
                blob_url = upload_pdf(pdf_content=pdf_bytes, doi=identifier)

                if cache_locally:
                    target_cache_dir = cache_dir or DEFAULT_CACHE_DIR
                    blob_name = build_blob_name(
                        identifier=identifier, format="pdf", compress=False
                    )
                    cached_file = target_cache_dir / blob_name
                    cached_file.parent.mkdir(parents=True, exist_ok=True)
                    cached_file.write_bytes(pdf_bytes)

                if existing_content is not None:
                    existing_content.blob_url = blob_url
                    existing_content.format = "pdf"
                    existing_content.source = source
                    existing_content.run_identifier = run_identifier
                    s.add(existing_content)
                else:
                    s.add(
                        PaperContent(
                            paper_id=paper.id,
                            format="pdf",
                            source=source,
                            blob_url=blob_url,
                            run_identifier=run_identifier,
                        )
                    )

                s.commit()
                uploaded.append(paper.id)
                logger.info(
                    f"Ingested manual PDF '{pdf_path.name}' for paper {paper.id}"
                )
            except Exception as e:
                s.rollback()
                logger.error(
                    f"Failed to ingest '{pdf_path.name}' for paper {paper.id}: {e}"
                )
                failed.append(pdf_path.name)

        return {
            "uploaded": uploaded,
            "skipped": skipped,
            "unmatched": unmatched,
            "failed": failed,
            "run_identifier": run_identifier,
        }

    with step_logger(
        "pdf_manual_upload",
        run_identifier,
        enabled=_is_logging_active("pdf_manual_upload", log_to_file),
    ):
        if session is not None:
            return _ingest(session)
        init_db()
        with get_session_context() as s:
            return _ingest(s)


def run_peptide_extraction(
    session: Optional[Session] = None,
    save: bool = True,
    provider: str = "google_genai",
    model_name: str = "gemma-4-31b-it",
    delay_seconds: float = 2.0,
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    pass_pdf_directly: bool = False,
    tables_as_images: bool = False,
    images_as_images: bool = False,
    table_image_layout: str = "interleaved",
    factored_extraction: bool = True,
    table_provider: Optional[str] = None,
    table_model_name: Optional[str] = None,
    **kwargs,
) -> tuple[List[int], int]:
    """Run peptide extraction"""
    if run_identifier is None:
        run_identifier = int(time.time())

    def _run(session: Session):

        papers_to_extract = get_paper_metadatas(session, **kwargs)
        failed_paper_ids = []

        for paper in papers_to_extract:
            try:
                paper_content = get_paper_content_by_paper_id(paper.id, session)

                if paper_content is None:
                    logger.info(f"Skipping paper {paper.id}: no content")
                    continue

                paper_fulltext = paper_content.get_content()

                if paper_content.format == "pdf" and pass_pdf_directly:
                    logger.info(
                        f"Passing raw PDF ({len(paper_fulltext)} bytes) directly to LLM for paper {paper.id}"
                    )
                    entries = run_extraction(
                        paper_fulltext,
                        provider,
                        model_name,
                        filename=f"paper_{paper.id}.pdf",
                    ).entries
                elif tables_as_images or images_as_images:
                    if paper_content.format == "pdf":
                        extraction_payload = extraction_ready_blocks_from_pdf(
                            paper_fulltext,
                            layout=table_image_layout,
                            tables_as_images=tables_as_images,
                            images_as_images=images_as_images,
                        )
                    elif paper_content.format == "xml":
                        extraction_payload = extraction_ready_blocks_from_xml(
                            paper_fulltext,
                            layout=table_image_layout,
                            tables_as_images=tables_as_images,
                            images_as_images=images_as_images,
                        )
                    else:
                        extraction_payload = to_extraction_ready_blocks(
                            paper_fulltext,
                            layout=table_image_layout,
                            tables_as_images=tables_as_images,
                            images_as_images=images_as_images,
                        )
                    logger.info(
                        f"Extraction Ready Blocks ({len(extraction_payload)} blocks, layout={table_image_layout})"
                    )
                    entries = run_extraction(
                        extraction_payload,
                        provider,
                        model_name,
                        factored=factored_extraction,
                        table_provider=table_provider,
                        table_model_name=table_model_name,
                    ).entries
                else:
                    if paper_content.format == "pdf":
                        extraction_ready_string = extraction_ready_string_from_pdf(
                            paper_fulltext
                        )
                    elif paper_content.format == "xml":
                        extraction_ready_string = extraction_ready_string_from_xml(
                            paper_fulltext
                        )
                    else:
                        extraction_ready_string = to_extraction_ready_string(
                            paper_fulltext
                        )
                    logger.info(f"Extraction Ready String: {extraction_ready_string}")
                    entries = run_extraction(
                        extraction_ready_string, provider, model_name
                    ).entries

                db_entries = [
                    peptide_llm_entry_to_peptide_entry(
                        entry,
                        paper.id,
                        paper.pmid,
                        model=model_name,
                        run_identifier=run_identifier,
                    )
                    for entry in entries
                ]
                if save:
                    session.add_all(db_entries)
                    session.commit()
                time.sleep(delay_seconds)
            except Exception as e:
                failed_paper_ids.append(paper.id)
                logger.info(f"Failed to extract peptides for paper {paper.id}: {e}")

        return failed_paper_ids, run_identifier

    with step_logger(
        "extraction",
        run_identifier,
        enabled=_is_logging_active("extraction", log_to_file),
    ):
        if session is not None:
            return _run(session)
        init_db()
        with get_session_context() as s:
            return _run(s)


def run_evaluation(
    session: Optional[Session] = None,
    save: bool = True,
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    **kwargs,
):
    if run_identifier is None:
        run_identifier = int(time.time())

    def _evaluate(session: Session):

        extractions_to_evaluate = get_records(PeptideEntry, session=session, **kwargs)
        failed_entries = []

        for extraction in extractions_to_evaluate:
            try:
                paper_for_extraction = extraction.paper_id
                paper_content = get_paper_content_by_paper_id(
                    paper_for_extraction, session
                )

                if paper_content is None:
                    raise ValueError(
                        f"No paper content found for peptide {extraction.id}"
                    )

                paper_fulltext = paper_content.get_content()

                evaluation = evaluate_extraction(extraction, paper_fulltext)
                evaluation.peptide_entry_id = extraction.id
                evaluation.run_identifier = run_identifier
                if save:
                    session.add(evaluation)
                    session.commit()
            except Exception as e:
                logger.info(
                    f"Failed to evaluate extraction for peptide {extraction.id}: {e}"
                )
                failed_entries.append(extraction.id)
            time.sleep(1)

        return failed_entries, run_identifier

    with step_logger(
        "evaluation",
        run_identifier,
        enabled=_is_logging_active("evaluation", log_to_file),
    ):
        if session is not None:
            return _evaluate(session)
        init_db()
        with get_session_context() as s:
            return _evaluate(s)


def run_visualization(
    operations: Union[
        VisualizationOperation,
        str,
        Sequence[Union[VisualizationOperation, str]],
    ],
    model: Union[Type[SQLModel], str],
    *,
    session: Optional[Session] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
    show: bool = False,
    file_format: str = "png",
    dpi: int = 150,
    limit: Optional[int] = None,
    offset: int = 0,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    viz_run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    **kwargs: Any,
) -> tuple[Dict[VisualizationOperation, VisualizationResult], int]:
    """Run visualization as a pipeline step for any supported DB model and filters."""
    if viz_run_identifier is None:
        viz_run_identifier = int(time.time())

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_project_root() / "data" / "diagrams" / str(viz_run_identifier)
    )

    def _visualize(s: Session):
        results = visualize(
            operations=operations,
            model=model,
            session=s,
            limit=limit,
            offset=offset,
            output_dir=target_dir,
            save=save,
            show=show,
            file_format=file_format,
            dpi=dpi,
            plot_kwargs=plot_kwargs,
            **kwargs,
        )
        return results, viz_run_identifier

    with step_logger(
        "visualization",
        viz_run_identifier,
        enabled=_is_logging_active("visualization", log_to_file),
    ):
        if session is not None:
            return _visualize(session)
        init_db()
        with get_session_context() as s:
            return _visualize(s)


ALL_PIPELINE_STEPS: tuple[str, ...] = (
    "search",
    "deduplicate",
    "find_abstracts",
    "clean_empty",
    "relevance",
    "fetch_fulltext",
    "extraction",
    "evaluation",
    "visualization",
)

_STEP_ALIASES: Dict[str, str] = {
    "fetch": "fetch_fulltext",
}


def _normalize_pipeline_steps(
    steps: Optional[Union[str, Sequence[str]]],
) -> List[str]:
    """Normalize step names, expanding 'all' or None into ALL_PIPELINE_STEPS."""
    if steps is None:
        return list(ALL_PIPELINE_STEPS)
    if isinstance(steps, str):
        raw_steps = [steps]
    else:
        raw_steps = list(steps)

    if not raw_steps or any(str(s).strip().lower() == "all" for s in raw_steps):
        return list(ALL_PIPELINE_STEPS)

    normalized: List[str] = []
    for raw in raw_steps:
        name = _STEP_ALIASES.get(str(raw).strip().lower(), str(raw).strip().lower())
        if name not in ALL_PIPELINE_STEPS:
            raise ValueError(
                f"Unknown pipeline step '{raw}'. Valid steps: {list(ALL_PIPELINE_STEPS)} or 'all'."
            )
        if name not in normalized:
            normalized.append(name)
    return normalized


@dataclass
class PipelineResult:
    """Summary of an end-to-end `run_full_pipeline` execution."""

    run_identifier: int
    steps_executed: List[str] = field(default_factory=list)
    paper_ids: Optional[List[int]] = None
    search_count: int = 0
    deduplicated_count: int = 0
    abstracts_found_count: int = 0
    cleaned_empty_count: int = 0
    relevance_run_identifier: Optional[int] = None
    relevance_usable_ids: List[int] = field(default_factory=list)
    relevance_discarded_count: int = 0
    relevance_failed_ids: List[int] = field(default_factory=list)
    fetch_run_identifier: Optional[int] = None
    fetch_success_count: int = 0
    fetch_fail_count: int = 0
    extraction_run_identifier: Optional[int] = None
    extraction_entry_count: int = 0
    extraction_failed_ids: List[int] = field(default_factory=list)
    evaluation_run_identifier: Optional[int] = None
    evaluation_failed_ids: List[int] = field(default_factory=list)
    visualization_outputs: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Pipeline Run {self.run_identifier}",
            f"  Steps executed: {', '.join(self.steps_executed) or 'none'}",
            f"  Search papers added: {self.search_count}",
            f"  Deduplicated removed: {self.deduplicated_count}",
            f"  Abstracts found: {self.abstracts_found_count}",
            f"  Empty title/abstract removed: {self.cleaned_empty_count}",
            f"  Relevance usable: {len(self.relevance_usable_ids)} (discarded: {self.relevance_discarded_count}, failed: {len(self.relevance_failed_ids)})",
            f"  Fulltext fetched: {self.fetch_success_count} (failed: {self.fetch_fail_count})",
            f"  Extraction entries: {self.extraction_entry_count} (failed papers: {len(self.extraction_failed_ids)})",
            f"  Evaluation failed entries: {len(self.evaluation_failed_ids)}",
            f"  Visualizations generated: {len(self.visualization_outputs)}",
        ]
        return "\n".join(lines)


def _refresh_existing_paper_ids(
    paper_ids: Sequence[int],
    session: Optional[Session] = None,
) -> List[int]:
    """Return subset of `paper_ids` that still exist in `paper_metadata`."""
    if not paper_ids:
        return []
    papers = get_paper_metadatas(session=session, id=list(paper_ids))
    existing = {p.id for p in papers if p.id is not None}
    return [pid for pid in paper_ids if pid in existing]


def run_full_pipeline(
    *,
    steps: Optional[Union[str, Sequence[str]]] = None,
    queries: Optional[Sequence[str]] = None,
    db: str = "pmc",
    paper_ids: Optional[Sequence[int]] = None,
    pmids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    provider: str = "google",
    model_name: str = "gemini-3.5-flash-lite",
    delay_seconds: float = 3.0,
    run_identifier: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    config: Optional[Dict[str, Any]] = None,
    ncbi_provider: Optional[NCBIProvider] = None,
    open_alex_provider: Optional[OpenAlexProvider] = None,
    ncbi_api_key: Optional[str] = None,
    open_alex_api_key: Optional[str] = None,
    session: Optional[Session] = None,
    _step_fns: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """Run the end-to-end peptide extraction pipeline with automatic inter-step data flow.

    Instead of manually wiring `id` / `run_identifier` filters into every step, each
    step automatically passes its output scope (surviving paper IDs, `USABLE` papers,
    and `run_identifier`) to downstream steps. Explicit per-step `kwargs` in `config`
    still override the automatic scope when needed.
    """
    import os

    cfg = dict(config or {})
    pipeline_run_id = (
        run_identifier
        if run_identifier is not None
        else cfg.get("run_identifier") or int(time.time())
    )

    if steps is None:
        steps = cfg.get("steps")
    steps_to_run = _normalize_pipeline_steps(steps)

    # Resolve top-level defaults from config + function args
    global_provider = cfg.get("provider", provider)
    global_model = cfg.get("model_name", model_name)
    global_delay = cfg.get("delay_seconds", delay_seconds)
    global_limit = limit if limit is not None else cfg.get("limit")
    global_queries = list(queries) if queries is not None else cfg.get("queries")
    global_db = cfg.get("db", db)

    if log_to_file is None:
        global_log_to_file = bool(
            cfg.get("log_to_file", cfg.get("save_logs", cfg.get("file_logging", False)))
        )
    else:
        global_log_to_file = bool(log_to_file)

    def _is_step_log_enabled(step_name: str) -> bool:
        step_cfg = cfg.get(step_name, {})
        if isinstance(step_cfg, dict):
            for k in ("log_to_file", "save_logs", "file_logging"):
                if k in step_cfg:
                    return bool(step_cfg[k])
        if step_name == "fetch_fulltext":
            alt_cfg = cfg.get("fetch", {})
            if isinstance(alt_cfg, dict):
                for k in ("log_to_file", "save_logs", "file_logging"):
                    if k in alt_cfg:
                        return bool(alt_cfg[k])
        return global_log_to_file

    def _get_step_cfg(step_name: str) -> Dict[str, Any]:
        if step_name == "fetch_fulltext":
            return dict(cfg.get("fetch") or cfg.get("fetch_fulltext") or {})
        return dict(cfg.get(step_name) or {})

    fns = {
        "search": retrive_ncbi_by_query,
        "deduplicate": deduplicate_by_doi,
        "find_abstracts": find_abstracts_by_doi,
        "clean_empty": remove_empty_abstract_title_papers,
        "relevance_bulk": run_bulk_relevance_screening,
        "relevance_single": run_relevance_screening,
        "fetch_fulltext": fetch_paper_fulltexts,
        "extraction": run_peptide_extraction,
        "evaluation": run_evaluation,
        "visualization": run_visualization,
    }
    if _step_fns:
        fns.update(_step_fns)

    resolved_ncbi_key = ncbi_api_key if ncbi_api_key is not None else os.getenv("NCBI_API_KEY")
    resolved_oa_key = (
        open_alex_api_key
        if open_alex_api_key is not None
        else os.getenv("OPEN_ALEX_API_KEY")
    )

    def _get_ncbi_provider() -> NCBIProvider:
        nonlocal ncbi_provider
        if ncbi_provider is None:
            ncbi_provider = NCBIProvider(
                api_key=resolved_ncbi_key, batch_size=cfg.get("ncbi_batch_size", 100)
            )
        return ncbi_provider

    def _get_oa_provider() -> OpenAlexProvider:
        nonlocal open_alex_provider
        if open_alex_provider is None:
            open_alex_provider = OpenAlexProvider(api_key=resolved_oa_key)
        return open_alex_provider

    def _with_session_if_needed(fn):
        if session is not None:
            return fn(session)
        init_db()
        with get_session_context() as s:
            return fn(s)

    result = PipelineResult(run_identifier=pipeline_run_id)

    # Seed initial paper scope if explicit paper_ids or pmids were provided
    seed_paper_ids = (
        list(paper_ids)
        if paper_ids is not None
        else (list(cfg["paper_ids"]) if cfg.get("paper_ids") is not None else None)
    )
    seed_pmids = (
        list(pmids)
        if pmids is not None
        else (list(cfg["pmids"]) if cfg.get("pmids") is not None else None)
    )

    current_paper_ids: Optional[List[int]] = None
    if seed_paper_ids is not None:
        current_paper_ids = seed_paper_ids
    elif seed_pmids is not None:
        matched = _with_session_if_needed(
            lambda s: get_paper_metadatas(session=s, pmid=[str(p) for p in seed_pmids])
        )
        current_paper_ids = [p.id for p in matched if p.id is not None]

    # 1. Search
    if "search" in steps_to_run:
        search_cfg = _get_step_cfg("search")
        search_queries = search_cfg.get("queries") or global_queries
        if not search_queries:
            logger.info("No search queries provided; skipping search step.")
        else:
            step_run_id = search_cfg.get("run_identifier") or pipeline_run_id
            step_enabled = _is_step_log_enabled("search")
            with step_logger("search", step_run_id, enabled=step_enabled):
                logger.info("Running query search...")
                step_db = search_cfg.get("db", global_db)
                fns["search"](
                    _get_ncbi_provider(),
                    db=step_db,
                    query_list=list(search_queries),
                    run_identifier=step_run_id,
                    log_to_file=step_enabled,
                )
                new_papers = _with_session_if_needed(
                    lambda s: get_paper_metadatas(session=s, run_identifier=step_run_id)
                )
                new_ids = [p.id for p in new_papers if p.id is not None]
                result.search_count = len(new_ids)
                if current_paper_ids is None:
                    current_paper_ids = new_ids
                else:
                    existing_set = set(current_paper_ids)
                    current_paper_ids.extend(pid for pid in new_ids if pid not in existing_set)
                result.steps_executed.append("search")

    # 2. Deduplicate
    if "deduplicate" in steps_to_run:
        dd_cfg = _get_step_cfg("deduplicate")
        step_run_id = dd_cfg.get("run_identifier") or pipeline_run_id
        step_enabled = _is_step_log_enabled("deduplicate")
        with step_logger("deduplicate", step_run_id, enabled=step_enabled):
            logger.info("Deduplicating by DOI...")
            normalize = dd_cfg.get("normalize", True)
            remove_empty = dd_cfg.get("remove_empty", True)
            removed = fns["deduplicate"](
                session=session, normalize=normalize, remove_empty=remove_empty
            )
            result.deduplicated_count = removed
            logger.info(f"Removed {removed} papers with duplicate doi")
            if current_paper_ids is not None:
                current_paper_ids = _with_session_if_needed(
                    lambda s: _refresh_existing_paper_ids(current_paper_ids, session=s)
                )
            result.steps_executed.append("deduplicate")

    # 3. Find missing abstracts
    if "find_abstracts" in steps_to_run:
        fa_cfg = _get_step_cfg("find_abstracts")
        step_run_id = fa_cfg.get("run_identifier") or pipeline_run_id
        step_enabled = _is_step_log_enabled("find_abstracts")
        if current_paper_ids is not None and len(current_paper_ids) == 0:
            logger.info("No papers in pipeline scope; skipping find_abstracts.")
        else:
            with step_logger("find_abstracts", step_run_id, enabled=step_enabled):
                logger.info("Finding missing abstracts...")
                fa_filters = dict(fa_cfg.get("kwargs", {}))
                if not fa_filters and current_paper_ids is not None:
                    fa_filters["id"] = current_paper_ids
                found = fns["find_abstracts"](
                    resolved_oa_key, session=session, **fa_filters
                )
                result.abstracts_found_count = found or 0
                result.steps_executed.append("find_abstracts")

    # 4. Clean empty abstract / title
    if "clean_empty" in steps_to_run:
        ce_cfg = _get_step_cfg("clean_empty")
        step_run_id = ce_cfg.get("run_identifier") or pipeline_run_id
        step_enabled = _is_step_log_enabled("clean_empty")
        if current_paper_ids is not None and len(current_paper_ids) == 0:
            logger.info("No papers in pipeline scope; skipping clean_empty.")
        else:
            with step_logger("clean_empty", step_run_id, enabled=step_enabled):
                logger.info("Removing papers with empty abstract or title...")
                ce_filters = dict(ce_cfg.get("kwargs", {}))
                if not ce_filters and current_paper_ids is not None:
                    ce_filters["id"] = current_paper_ids
                removed = fns["clean_empty"](session=session, **ce_filters)
                result.cleaned_empty_count = removed
                logger.info(f"Removed {removed} papers with empty abstract or title")
                if current_paper_ids is not None:
                    current_paper_ids = _with_session_if_needed(
                        lambda s: _refresh_existing_paper_ids(current_paper_ids, session=s)
                    )
                result.steps_executed.append("clean_empty")

    # 5. Relevance screening
    if "relevance" in steps_to_run:
        rel_cfg = _get_step_cfg("relevance")
        step_run_id = rel_cfg.get("run_identifier") or pipeline_run_id
        step_enabled = _is_step_log_enabled("relevance")
        rel_kwargs = dict(rel_cfg.get("kwargs", {}))

        skip_relevance = False
        if not rel_kwargs:
            if current_paper_ids is not None:
                if len(current_paper_ids) == 0:
                    logger.info("No papers in pipeline scope; skipping relevance.")
                    skip_relevance = True
                else:
                    rel_kwargs = {
                        "id": current_paper_ids,
                        "status": PaperStatus.NOT_PROCESSED,
                    }
            else:
                rel_kwargs = {
                    "status": PaperStatus.NOT_PROCESSED,
                    "dataset_subset": DatasetSubset.SCRAPED,
                }
                if global_limit is not None:
                    rel_kwargs["limit"] = global_limit

        if not skip_relevance:
            with step_logger("relevance", step_run_id, enabled=step_enabled):
                logger.info("Running relevance screening...")
                start_time = time.time()
                use_bulk = rel_cfg.get("bulk", True)
                rel_fn = fns["relevance_bulk"] if use_bulk else fns["relevance_single"]
                screened_results, fails, identifier = rel_fn(
                    model_name=rel_cfg.get("model_name", global_model),
                    provider=rel_cfg.get("provider", global_provider),
                    delay_seconds=rel_cfg.get("delay_seconds", global_delay),
                    batch_size=rel_cfg.get("batch_size", 50),
                    run_identifier=step_run_id,
                    log_to_file=step_enabled,
                    session=session,
                    **rel_kwargs,
                )
                end_time = time.time()
                logger.info(
                    f"Relevance Success: {screened_results}, Fails: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
                )

                usable_ids: List[int] = []
                discarded_count = 0
                for item, rel_res in screened_results:
                    pid = item.id if hasattr(item, "id") else item
                    if getattr(rel_res, "is_relevant", False):
                        if pid is not None:
                            usable_ids.append(pid)
                    else:
                        discarded_count += 1

                # If user seeded specific paper_ids that were already USABLE before relevance, keep them too
                if current_paper_ids is not None and (
                    seed_paper_ids is not None or seed_pmids is not None
                ):
                    already_usable = _with_session_if_needed(
                        lambda s: [
                            p.id
                            for p in get_paper_metadatas(
                                session=s,
                                id=current_paper_ids,
                                status=PaperStatus.USABLE,
                            )
                            if p.id is not None
                        ]
                    )
                    for pid in already_usable:
                        if pid not in usable_ids:
                            usable_ids.append(pid)

                result.relevance_run_identifier = identifier
                result.relevance_usable_ids = usable_ids
                result.relevance_discarded_count = discarded_count
                result.relevance_failed_ids = list(fails)
                current_paper_ids = usable_ids
                result.steps_executed.append("relevance")

    # 6. Fetch fulltexts
    if "fetch_fulltext" in steps_to_run:
        ft_cfg = _get_step_cfg("fetch_fulltext")
        step_run_id = ft_cfg.get("run_identifier") or pipeline_run_id
        step_enabled = _is_step_log_enabled("fetch_fulltext")
        ft_kwargs = dict(ft_cfg.get("kwargs", {}))

        skip_fetch = False
        if not ft_kwargs:
            if current_paper_ids is not None:
                if len(current_paper_ids) == 0:
                    logger.info("No usable papers in pipeline scope; skipping fetch_fulltext.")
                    skip_fetch = True
                else:
                    ft_kwargs = {"id": current_paper_ids, "status": PaperStatus.USABLE}
            else:
                ft_kwargs = {"status": PaperStatus.USABLE}
                if global_limit is not None:
                    ft_kwargs["limit"] = global_limit

        if not skip_fetch:
            with step_logger("fetch", step_run_id, enabled=step_enabled):
                logger.info("Fetching fulltexts...")
                success, fails, identifier = fns["fetch_fulltext"](
                    session=session,
                    ncbi_provider=_get_ncbi_provider(),
                    open_alex_provider=_get_oa_provider(),
                    skip_existing_content=ft_cfg.get("skip_existing_content", True),
                    run_identifier=step_run_id,
                    log_to_file=step_enabled,
                    **ft_kwargs,
                )
                logger.info(
                    f"Fetch Fulltext Success: {success}, Fails: {fails}, Identifier: {identifier}"
                )
                result.fetch_run_identifier = identifier
                result.fetch_success_count = success
                result.fetch_fail_count = fails
                # If pipeline started at fetch_fulltext without prior paper_ids, carry the usable IDs forward
                if current_paper_ids is None and not ft_cfg.get("kwargs"):
                    usable_papers = _with_session_if_needed(
                        lambda s: get_paper_metadatas(session=s, **ft_kwargs)
                    )
                    current_paper_ids = [p.id for p in usable_papers if p.id is not None]
                result.steps_executed.append("fetch_fulltext")

    # 7. Peptide extraction
    if "extraction" in steps_to_run:
        ext_cfg = _get_step_cfg("extraction")
        step_run_id = ext_cfg.get("run_identifier") or pipeline_run_id
        step_enabled = _is_step_log_enabled("extraction")
        ext_kwargs = dict(ext_cfg.get("kwargs", {}))

        skip_extraction = False
        if not ext_kwargs:
            if current_paper_ids is not None:
                if len(current_paper_ids) == 0:
                    logger.info("No papers in pipeline scope; skipping extraction.")
                    skip_extraction = True
                else:
                    ext_kwargs = {"id": current_paper_ids, "status": PaperStatus.USABLE}
            else:
                def _find_extractable_ids(s: Session) -> List[int]:
                    content_paper_ids = {
                        c.paper_id
                        for c in get_records(PaperContent, session=s)
                        if c.paper_id is not None
                    }
                    if ext_cfg.get("skip_existing_extractions", True):
                        extracted_ids = {
                            e.paper_id
                            for e in get_records(PeptideEntry, session=s)
                            if e.paper_id is not None
                        }
                    else:
                        extracted_ids = set()
                    usable_papers = get_paper_metadatas(
                        session=s, status=PaperStatus.USABLE
                    )
                    candidates = [
                        p.id
                        for p in usable_papers
                        if p.id is not None
                        and p.id in content_paper_ids
                        and p.id not in extracted_ids
                    ]
                    if global_limit is not None:
                        candidates = candidates[:global_limit]
                    return candidates

                candidate_ids = _with_session_if_needed(_find_extractable_ids)
                if not candidate_ids:
                    logger.info("No unextracted usable papers with content found; skipping extraction.")
                    skip_extraction = True
                else:
                    ext_kwargs = {"id": candidate_ids}

        if not skip_extraction:
            with step_logger("extraction", step_run_id, enabled=step_enabled):
                logger.info("Running extraction...")
                start_time = time.time()
                extra_ext_opts = {
                    k: ext_cfg[k]
                    for k in (
                        "save",
                        "pass_pdf_directly",
                        "tables_as_images",
                        "images_as_images",
                        "table_image_layout",
                        "factored_extraction",
                        "table_provider",
                        "table_model_name",
                    )
                    if k in ext_cfg
                }
                fails, identifier = fns["extraction"](
                    session=session,
                    provider=ext_cfg.get("provider", global_provider),
                    model_name=ext_cfg.get("model_name", global_model),
                    delay_seconds=ext_cfg.get("delay_seconds", global_delay),
                    run_identifier=step_run_id,
                    log_to_file=step_enabled,
                    **extra_ext_opts,
                    **ext_kwargs,
                )
                end_time = time.time()
                logger.info(
                    f"Extraction failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
                )
                result.extraction_run_identifier = identifier
                result.extraction_failed_ids = list(fails)
                extracted_entries = _with_session_if_needed(
                    lambda s: get_records(PeptideEntry, session=s, run_identifier=identifier)
                )
                result.extraction_entry_count = len(extracted_entries)
                result.steps_executed.append("extraction")

    # 8. Evaluation
    if "evaluation" in steps_to_run:
        eval_cfg = _get_step_cfg("evaluation")
        step_run_id = eval_cfg.get("run_identifier") or pipeline_run_id
        step_enabled = _is_step_log_enabled("evaluation")
        eval_kwargs = dict(eval_cfg.get("kwargs", {}))

        skip_evaluation = False
        if not eval_kwargs:
            if current_paper_ids is not None and len(current_paper_ids) == 0:
                logger.info("No papers in pipeline scope; skipping evaluation.")
                skip_evaluation = True
            elif result.extraction_run_identifier is not None:
                if result.extraction_entry_count == 0:
                    logger.info("No PeptideEntry records extracted in this run; skipping evaluation.")
                    skip_evaluation = True
                else:
                    entry_ids = _with_session_if_needed(
                        lambda s: [
                            e.id
                            for e in get_records(
                                PeptideEntry,
                                session=s,
                                run_identifier=result.extraction_run_identifier,
                            )
                            if e.id is not None
                        ]
                    )
                    if not entry_ids:
                        skip_evaluation = True
                    else:
                        eval_kwargs = {"id": entry_ids}
            elif run_identifier is not None or cfg.get("run_identifier") is not None:
                entry_ids = _with_session_if_needed(
                    lambda s: [
                        e.id
                        for e in get_records(
                            PeptideEntry, session=s, run_identifier=pipeline_run_id
                        )
                        if e.id is not None
                    ]
                )
                if not entry_ids:
                    logger.info(
                        f"No PeptideEntry records found for run_identifier={pipeline_run_id}; skipping evaluation."
                    )
                    skip_evaluation = True
                else:
                    eval_kwargs = {"id": entry_ids}
            else:
                from peptide_extractor.db.extraction_evaluation import (
                    ExtractionEvaluation,
                )

                def _find_unevaluated_entry_ids(s: Session) -> List[int]:
                    evaluated_ids = {
                        ev.peptide_entry_id
                        for ev in get_records(ExtractionEvaluation, session=s)
                        if ev.peptide_entry_id is not None
                    }
                    entries = get_records(PeptideEntry, session=s)
                    pending = [
                        e.id
                        for e in entries
                        if e.id is not None and e.id not in evaluated_ids
                    ]
                    if global_limit is not None:
                        pending = pending[:global_limit]
                    return pending

                pending_entry_ids = _with_session_if_needed(_find_unevaluated_entry_ids)
                if not pending_entry_ids:
                    logger.info("No unevaluated PeptideEntry records found; skipping evaluation.")
                    skip_evaluation = True
                else:
                    eval_kwargs = {"id": pending_entry_ids}
        elif "run_identifier" in eval_kwargs:
            target_ext_run_id = eval_kwargs.pop("run_identifier")
            entry_ids = _with_session_if_needed(
                lambda s: [
                    e.id
                    for e in get_records(
                        PeptideEntry, session=s, run_identifier=target_ext_run_id
                    )
                    if e.id is not None
                ]
            )
            if not entry_ids:
                logger.info(
                    f"No PeptideEntry records found for run_identifier={target_ext_run_id}; skipping evaluation."
                )
                skip_evaluation = True
            else:
                eval_kwargs["id"] = entry_ids

        if not skip_evaluation:
            with step_logger("evaluation", step_run_id, enabled=step_enabled):
                logger.info("Running evaluation...")
                start_time = time.time()
                fails, identifier = fns["evaluation"](
                    session=session,
                    run_identifier=step_run_id,
                    log_to_file=step_enabled,
                    **eval_kwargs,
                )
                end_time = time.time()
                logger.info(
                    f"Evaluation failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
                )
                result.evaluation_run_identifier = identifier
                result.evaluation_failed_ids = list(fails)
                result.steps_executed.append("evaluation")

    # 9. Visualization
    if "visualization" in steps_to_run:
        viz_cfg = _get_step_cfg("visualization")
        viz_run_id = viz_cfg.get("run_identifier") or pipeline_run_id
        step_enabled = _is_step_log_enabled("visualization")
        explicit_model = viz_cfg.get("model") or viz_cfg.get("data_type")
        explicit_ops = viz_cfg.get("operations")

        with step_logger("visualization", viz_run_id, enabled=step_enabled):
            logger.info("Running visualization...")
            start_time = time.time()
            if explicit_model and explicit_ops:
                viz_kwargs = dict(viz_cfg.get("kwargs", {}))
                if "run_identifier" not in viz_kwargs:
                    norm_model = str(explicit_model).strip().lower()
                    if norm_model == "peptide_entry" and result.extraction_run_identifier is not None:
                        viz_kwargs["run_identifier"] = result.extraction_run_identifier
                    elif norm_model == "extraction_evaluation" and result.evaluation_run_identifier is not None:
                        viz_kwargs["run_identifier"] = result.evaluation_run_identifier
                    elif norm_model == "paper_relevance" and result.relevance_run_identifier is not None:
                        viz_kwargs["run_identifier"] = result.relevance_run_identifier

                viz_results, identifier = fns["visualization"](
                    session=session,
                    operations=explicit_ops,
                    model=explicit_model,
                    output_dir=viz_cfg.get("output_dir"),
                    save=viz_cfg.get("save", True),
                    show=viz_cfg.get("show", False),
                    file_format=viz_cfg.get("file_format", "png"),
                    dpi=viz_cfg.get("dpi", 150),
                    limit=viz_cfg.get("limit"),
                    offset=viz_cfg.get("offset", 0),
                    plot_kwargs=viz_cfg.get("plot_kwargs"),
                    viz_run_identifier=viz_run_id,
                    log_to_file=step_enabled,
                    **viz_kwargs,
                )
                for op, res_obj in viz_results.items():
                    result.visualization_outputs[op.value] = res_obj.output_path
                logger.info(
                    f"Visualization generated {len(viz_results)} diagrams, Identifier: {identifier}, Time: {time.time() - start_time}"
                )
                result.steps_executed.append("visualization")
            else:
                # Auto-visualize outputs generated in this pipeline run
                auto_specs: List[tuple[str, List[str], Dict[str, Any]]] = []
                if result.extraction_entry_count > 0 and result.extraction_run_identifier is not None:
                    auto_specs.append(
                        (
                            "peptide_entry",
                            [
                                "peptide_column_completion",
                                "peptide_length_distribution",
                                "peptide_morphology_distribution",
                                "extractions_per_paper",
                            ],
                            {"run_identifier": result.extraction_run_identifier},
                        )
                    )
                if result.evaluation_run_identifier is not None:
                    auto_specs.append(
                        (
                            "extraction_evaluation",
                            ["evaluation_boxplot", "evaluation_summary_range"],
                            {"run_identifier": result.evaluation_run_identifier},
                        )
                    )
                if result.relevance_run_identifier is not None:
                    auto_specs.append(
                        (
                            "paper_relevance",
                            ["relevance_outcome_counts", "relevance_confidence_hist"],
                            {"run_identifier": result.relevance_run_identifier},
                        )
                    )

                total_generated = 0
                for auto_model, auto_ops, auto_kwargs in auto_specs:
                    viz_results, _ = fns["visualization"](
                        session=session,
                        operations=auto_ops,
                        model=auto_model,
                        output_dir=viz_cfg.get("output_dir"),
                        save=viz_cfg.get("save", True),
                        show=viz_cfg.get("show", False),
                        file_format=viz_cfg.get("file_format", "png"),
                        dpi=viz_cfg.get("dpi", 150),
                        viz_run_identifier=viz_run_id,
                        log_to_file=step_enabled,
                        **auto_kwargs,
                    )
                    total_generated += len(viz_results)
                    for op, res_obj in viz_results.items():
                        result.visualization_outputs[op.value] = res_obj.output_path

                logger.info(
                    f"Visualization generated {total_generated} diagrams, Identifier: {viz_run_id}, Time: {time.time() - start_time}"
                )
                if total_generated > 0:
                    result.steps_executed.append("visualization")

    result.paper_ids = current_paper_ids
    return result


# Not removing this for development/testing purposes
if __name__ == "__main__":
    from peptide_extractor.env import load_env
    import os

    load_env()

    # Variable to enable storing logs in docs/logs/[step]/{run_identifier}.log
    # Set this to True to automatically store step logs in files.
    log_to_file = True
    LOG_TO_FILE = log_to_file

    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    ncbi_api_key = os.getenv("NCBI_API_KEY")

    scopus_api_key = os.getenv("SCOPUS_API_KEY")

    open_alex_api_key = os.getenv("OPEN_ALEX_API_KEY")

    ncbi_provider = NCBIProvider(api_key=ncbi_api_key, batch_size=100)

    open_alex_provider = OpenAlexProvider(api_key=open_alex_api_key)

    # upload sapdb/peptideminer manually downloded pdfs to azure
    # result = ingest_manual_pdfs(
    #    pdf_dir="manual_pdfs",
    #    skip_existing_content=True,
    #    cache_locally=True,
    # )
    #
    # print(result)
    #
    ##
    # paper_content = get_paper_content_by_paper_id(9)
    # fulltext = to_extraction_ready_string(paper_content.get_content(), format="pdf")
    # print(fulltext)

    #

    # 1. Run query search

    # retrive_ncbi_by_query(ncbi_provider, db="pmc", query_list=NCBI_SEARCH_QUERIES_V3)

    # 2. Deduplicate by doi (+ normalize doi and remove empty)

    # removed = deduplicate_by_doi()

    # print(f"Removed {removed} papers with duplicate doi")

    # 3. Find abstracts by doi

    # find_abstracts_by_doi(open_alex_api_key)

    # 4. Remove papers with empty abstract or title

    # removed = remove_empty_abstract_title_papers()

    # print(f"Removed {removed} papers with empty abstract or title")

    # 5. Run relevance screening

    # start_time = time.time()

    #

    ## doing relevance in batches

    # success, fails, identifier = run_bulk_relevance_screening("gemma4:26b", "ollama", delay_seconds=1, batch_size=3, id=papers_to_relevance_check)

    #

    ##success, fails, identifier = run_bulk_relevance_screening("llama4:16x17b", "ollama", delay_seconds=1, batch_size=10, id=9533, limit=5000, offset=0)

    #

    # success, fails, identifier = run_bulk_relevance_screening("gemini-3.5-flash-lite", "google", delay_seconds=4, batch_size=10, search_query=NCBI_SEARCH_QUERIES_V3, limit=5000, offset=5000)

    #
    # success, fails, identifier = run_bulk_relevance_screening("gemini-3.1-flash-lite", "google", delay_seconds=4, batch_size=10, search_query=papers_with_relevance_conflict, limit=400, offset=0)

    # end_time = time.time()

    # print(f"Relevance Success: {success}, Fails: {fails}, Identifier: {identifier}, Time: {end_time - start_time}")

    # 6. Fetch fulltexts

    # fetch_paper_fulltexts(
    #    ncbi_provider=ncbi_provider,
    #    open_alex_provider=open_alex_provider,
    #    id=papers_with_relevance_conflict,
    # )

    # from peptide_extractor.scratchpad import papers_to_extract_2

    # 7. Run extraction

    # papers_with_content = [p.paper_id for p in get_records(PaperContent)]
    # papers_with_extraction = [p.paper_id for p in get_records(PeptideEntry)]
    # relevant_papers_without_extraction = [
    #    p.id
    #    for p in get_paper_metadatas(dataset_subset=DatasetSubset.SCRAPED)
    #    if p.id not in papers_with_extraction
    #    and p.id in papers_with_content
    #    and p.status == PaperStatus.USABLE
    # ]
    #

    ## print(len(papers_with_extraction))
    # print(len(relevant_papers_without_extraction))
    #
    # start_time = time.time()
    #
    # fails, identifier = run_peptide_extraction(
    #    provider="google",
    #    model_name="gemini-3.5-flash-lite",
    #    delay_seconds=4,
    #    id=sap_db,
    # )
    #
    # end_time = time.time()
    # print(
    #    f"Extraction failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
    # )
    #
    ## 8. Run evaluation
    #
    # start_time = time.time()

    # fails, identifier = run_evaluation(run_identifier=123)

    # end_time = time.time()

    # print(f"Evaluation failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}")

    # fails, identifier = run_peptide_extraction(
    #    save=True,
    #    provider="google_genai",
    #    model_name="gemini-3.5-flash-lite",
    #    pass_pdf_directly=True,
    #    id=9,
    # )
    # print(
    #    f"Extraction failed for papers with IDs: {fails}, Run identifier: {identifier}"
    # )

    # golden_papers_ids = [r.paper_id for r in get_records(PeptideEntry, run_identifier=0)]

    from peptide_extractor.golden_benchmark import (
        DEV_BENCHMARK_PMIDS,
        benchmark_run_against_golden,
    )

    fails, identifier = run_peptide_extraction(
        # provider="ollama",
        # model_name="gemma4:26b",
        provider="google",
        model_name="gemini-3.5-flash-lite",
        delay_seconds=4,
        # id=golden_papers_ids,
        pmid=[20460796],
        tables_as_images=True,
        factored_extraction=True,
        table_provider="ollama",
        table_model_name="gemma4:26b",
        # table_provider="google",
        # table_model_name="gemini-3.5-flash-lite",
    )
    print(f"Fails: {fails}, Run identifier: {identifier}")

    print(
        benchmark_run_against_golden(
            llm_run_identifier=identifier, pmids=[20460796]
        ).format_report()
    )
