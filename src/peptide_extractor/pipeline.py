from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, Union
from sqlmodel import SQLModel
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


def find_abstracts_by_doi(open_alex_api_key):
    init_db()
    non_abstract_metadas = get_paper_metadatas(abstract="")
    found_abstract_count = 0

    semantic_scholar = SemanticScholarProvider()
    open_alex = OpenAlexProvider(api_key=open_alex_api_key)

    with get_session_context() as session:
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
                    session.add(metadata)
                    session.commit()
                else:
                    try:
                        abstract = open_alex.get_abstract_by_doi(metadata.doi)
                        time.sleep(0.5)
                        if abstract is not None:
                            found_abstract_count += 1
                            metadata.abstract = abstract
                            session.add(metadata)
                            session.commit()
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
                            session.add(metadata)
                            session.commit()
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


def remove_empty_abstract_title_papers(session: Optional[Session] = None) -> int:
    """Returns removed count"""

    def _remove(session: Session):
        empty_abstract = get_paper_metadatas(abstract="", session=session)
        empty_title = get_paper_metadatas(title="", session=session)
        empty_papers = [paper for paper in empty_abstract + empty_title]

        deleted_count = 0
        for paper in empty_papers:
            session.delete(paper)
            deleted_count += 1
        session.commit()
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
