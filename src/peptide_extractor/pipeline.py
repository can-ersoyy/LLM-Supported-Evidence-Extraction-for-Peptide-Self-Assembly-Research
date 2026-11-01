from peptide_extractor.db.paper_metadata import PaperStatus
from peptide_extractor.schemas import RelevanceResult
from typing import List
from langchain_core.language_models import BaseChatModel
from peptide_extractor.evaluation import evaluate_extraction
from peptide_extractor.db.peptide_entry import PeptideEntry
from peptide_extractor.db import peptide_llm_entry_to_peptide_entry
from peptide_extractor.extraction import run_extraction
from peptide_extractor.parsers import extraction_ready_string_from_xml
from peptide_extractor.db.paper_content import get_paper_content_by_paper_id
from peptide_extractor.db import PaperContent
from peptide_extractor.utils import doi_to_filename
from peptide_extractor import azure_storage
from peptide_extractor.db import get_session
from peptide_extractor.utils import normalize_database_dois
from peptide_extractor.llm import LLMController
from typing import Optional
from peptide_extractor.relevance import screen_papers
from peptide_extractor.providers import NCBIProvider
from sqlmodel import select
from peptide_extractor.utils import normalize_doi
from peptide_extractor.db import get_records
from peptide_extractor.db import DatasetSubset
from peptide_extractor.db import PaperMetadata
from peptide_extractor.providers import OpenAlexProvider
import time
from peptide_extractor.providers import SemanticScholarProvider
from peptide_extractor.db import get_paper_metadatas
from dotenv import find_dotenv, load_dotenv
from sqlmodel import Session
from peptide_extractor.db import get_session_context, init_db
import os

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# maps peptide length to structure name or self-assembly keyword, includes open-access, english and non review article filters, only title and abstract
NCBI_SEARCH_QUERY_V1 = """((tetrapeptide[tiab] OR pentapeptide[tiab] OR hexapeptide[tiab] OR heptapeptide[tiab] OR octapeptide[tiab] OR nonapeptide[tiab] OR decapeptide[tiab] OR undecapeptide[tiab] OR dodecapeptide[tiab] OR tridecapeptide[tiab] OR tetradecapeptide[tiab] OR pentadecapeptide[tiab] OR hexadecapeptide[tiab] OR heptadecapeptide[tiab] OR octadecapeptide[tiab] OR nonadecapeptide[tiab] OR eicosapeptide[tiab] OR icosapeptide[tiab]) AND (nanostructure[tiab] OR nanotube[tiab] OR nanofiber[tiab] OR nanofibre[tiab] OR nanorod[tiab] OR hydrogel[tiab] OR nanosphere[tiab] OR nanoparticle[tiab] OR "self-assembly"[tiab])) AND (english[la]) AND ("free full text"[filter]) NOT (review[pt])"""
NCBI_SEARCH_QUERY_V2 = """peptide self-assembly"""

# this query on pmc returns 55k results (pmc only returns the first 10k per query)
NCBI_SEARCH_QUERY_V3_TEMPLATE_INTERVAL = """("peptid"[All Fields] OR "peptidal"[All Fields] OR "peptide s"[All Fields] OR "peptides"[Supplementary Concept] OR "peptides"[All Fields] OR "peptide"[All Fields] OR "peptides"[MeSH Terms] OR "peptidic"[All Fields]) AND "self-assembly"[All Fields] AND (open_access[Filter]) AND ({}:{}[pdat]) AND (pmc_public[Filter])"""
NCBI_SEARCH_QUERY_V3_TEMPLATE_YEAR = """("peptid"[All Fields] OR "peptidal"[All Fields] OR "peptide s"[All Fields] OR "peptides"[Supplementary Concept] OR "peptides"[All Fields] OR "peptide"[All Fields] OR "peptides"[MeSH Terms] OR "peptidic"[All Fields]) AND "self-assembly"[All Fields] AND (open_access[Filter]) AND ({}[pdat]) AND (pmc_public[Filter])"""
NCBI_SEARCH_QUERIES_V3 = [NCBI_SEARCH_QUERY_V3_TEMPLATE_INTERVAL.format("1900", "2009")]
NCBI_SEARCH_QUERIES_V3.extend([NCBI_SEARCH_QUERY_V3_TEMPLATE_YEAR.format(year) for year in range(2010, 2028)])

SCOPUS_SEARCH_QUERY_V1 = """TITLE-ABS-KEY(tetrapeptide OR pentapeptide OR hexapeptide OR heptapeptide OR octapeptide OR nonapeptide OR decapeptide OR undecapeptide OR dodecapeptide OR tridecapeptide OR tetradecapeptide OR pentadecapeptide OR hexadecapeptide OR heptadecapeptide OR octadecapeptide OR nonadecapeptide OR eicosapeptide OR icosapeptide) AND TITLE-ABS-KEY(nanostructure OR nanotube OR nanofiber OR nanofibre OR nanorod OR hydrogel OR nanosphere OR nanoparticle OR "self-assembly") AND LANGUAGE(english) AND DOCTYPE(ar) AND OA(all)"""


def retrive_ncbi_by_query(ncbi_provider: NCBIProvider, query_list=[NCBI_SEARCH_QUERY_V1], save=True, db="pubmed", delete_raw_records=True):
    """Wrapper method for doing query search on pubmed/pmc from ncbi provider."""
    raw_records = []
    for query in query_list:
        print(f"Query: {query}")
        raw_records.extend(ncbi_provider.search_by_query(query, db=db))
    ncbi_metadatas = ncbi_provider.parse_raw_search_records(raw_records)


    if save:
        init_db()
        with get_session_context() as session:
            session.add_all(ncbi_metadatas)
            session.commit()

        if delete_raw_records: # cleanup since raw records are only needed before storing metadata
            init_db()
            with get_session_context() as session:
                for record in raw_records:
                    session.delete(record)
                session.commit()
    
    else:
        return ncbi_metadatas

def retrieve_scopus_by_query(scopus_provider, query=SCOPUS_SEARCH_QUERY_V1, save=True):
    raw_records = scopus_provider.search_by_query(query)
    scopus_metadatas = scopus_provider.parse_raw_search_records(raw_records)

    if save:
        init_db()
        with get_session_context() as session:
            session.add_all(scopus_metadatas)
            session.commit()
    else:
        return scopus_metadatas


def deduplicate_by_doi(session: Optional[Session] = None, normalize: bool = True, remove_empty: bool = True) -> int:
    """Updated to remove papers with duplicate doi. Also supports normalization and removal of papers with empty doi
        Returns deleted count
    """
    def _deduplicate(s: Session):
        scraped_papers = get_records(PaperMetadata, session=s)
        seen_papers = []
        deleted_count = 0
        print(f"Found {len(scraped_papers)} scraped papers.")
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
                print(f"Skipping paper {metadata.id}: no DOI")
                continue
            try:
                abstract = semantic_scholar.get_abstract_by_doi(metadata.doi)

                print(abstract)

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
                        print(f"Failed to retrieve abstract for {metadata.doi}: {e}")
                print(f"Retrieved abstract for {metadata.doi}: {abstract}")
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
                        print(f"Retrieved abstract for {metadata.doi}: {abstract}")
                    except Exception as e:
                        print(f"Failed to retrieve abstract for {metadata.doi}: {e}")
                print(f"Failed to retrieve abstract for {metadata.doi}: {e}")
                
            time.sleep(1)

        print(f"Retrieved {found_abstract_count} abstracts out of {len(non_abstract_metadas)}")

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
    batch_size: Optional[int] = None,
    delay_seconds: float = 2.0,
    **kwargs
) -> tuple[List[tuple[PaperMetadata, RelevanceResult]], List[int], int]:
    """Run abstract relevance screening on paper metadata records in the database.

    Evaluates papers using the relevance classifier, persisting PaperRelevance records
    and updating paper status to USABLE or DISCARD.

    Returns the succesful paper/relevance pairs, failed_paper_ids and a run_identifier
    """
    from peptide_extractor.relevance import screen_papers

    run_identifier = int(time.time())

    init_db()
    screened_results, failed_paper_ids = screen_papers(
        model_name=model_name,
        provider=provider,
        batch_size=batch_size,
        delay_seconds=delay_seconds,
        run_identifier=run_identifier,
        **kwargs
    )
    return screened_results, failed_paper_ids, run_identifier

def run_bulk_relevance_screening(
    model_name: str,
    provider: str,
    batch_size: Optional[int] = 10,
    delay_seconds: float = 2.0,
    **kwargs
):
    """Run abstract relevance screening on paper metadata records in the database in bulk.

    Evaluates papers using the relevance classifier, persisting PaperRelevance records
    and updating paper status to USABLE or DISCARD.

    Returns the succesful paper/relevance pairs, failed_paper_ids and a run_identifier
    """
    from peptide_extractor.relevance import bulk_screen_papers

    run_identifier = int(time.time())

    init_db()
    screened_results, failed_paper_ids = bulk_screen_papers(
        model_name=model_name,
        provider=provider,
        batch_size=batch_size,
        delay_seconds=delay_seconds,
        run_identifier=run_identifier,
        **kwargs
    )
    return screened_results, failed_paper_ids, run_identifier

def fetch_paper_fulltexts(session: Optional[Session] = None, ncbi_provider: NCBIProvider = None, open_alex_provider: OpenAlexProvider = None, **kwargs):
    """Retrieve fulltexts according to db query"""
    def _fetch(session: Session):
        papers_to_retrieve = get_paper_metadatas(**kwargs, session=session)
        print(papers_to_retrieve)

        success_count = 0
        fail_count = 0

        for paper in papers_to_retrieve:
            paper_content: Optional[PaperContent] = None

            # normally it is better to batch retrieve from pubmed but for now just singular fetch
            try:
                paper_contents, _, _ = ncbi_provider.fetch([paper])
                paper_content = paper_contents[0]
            except Exception as e:
                print(f"Failed to retrieve fulltext for {paper.doi}: {e}")
                print("Trying pdf retrieval")
                try:
                    paper_content = open_alex_provider.fetch(paper)
                    
                except Exception as e:
                    print(f"Failed to retrieve fulltext for {paper.doi}: {e}")
                    
           
            if paper_content is not None:
                session.add(paper_content)
                session.commit()
                success_count += 1
            else:
                fail_count += 1
    
            time.sleep(1)
        
        print(f"Success: {success_count}, Fail: {fail_count}")
        return success_count, fail_count
        
    if session is not None:
        return _fetch(session)
    init_db()
    with get_session_context() as s:
        return _fetch(s)

def run_peptide_extraction(session: Optional[Session] = None, save: bool = True, provider: str = "google_genai", model_name: str = "gemma-4-31b-it", delay_seconds: float = 2.0, **kwargs) -> List[int]:
    """Run peptide extraction"""
    def _run(session: Session):
        
        papers_to_extract = get_paper_metadatas(session, **kwargs)
        failed_paper_ids = []
        run_identifier = int(time.time())

        for paper in papers_to_extract:
            try:
                paper_content = get_paper_content_by_paper_id(paper.id, session)

                if paper_content is None:
                    print(f"Skipping paper {paper.id}: no content")
                    continue
                
                paper_fulltext = paper_content.get_content()

                if paper_content.format == "pdf":
                    pass
                elif paper_content.format == "xml":
                    extraction_ready_string = extraction_ready_string_from_xml(paper_fulltext)
                    print(f"Extraction Ready String: {extraction_ready_string}")
                    entries = run_extraction(extraction_ready_string, provider, model_name).entries
                    db_entries = [peptide_llm_entry_to_peptide_entry(entry, paper.id, paper.pmid, model=model_name, run_identifier=run_identifier) for entry in entries]
                    if save:
                        session.add_all(db_entries)
                        session.commit()
                time.sleep(delay_seconds)
            except Exception as e:
                failed_paper_ids.append(paper.id)
                print(f"Failed to extract peptides for paper {paper.id}: {e}")
            

        return failed_paper_ids, run_identifier

    if session is not None:
        return _run(session)
    init_db()
    with get_session_context() as s:
        return _run(s)

def run_evaluation(session: Optional[Session] = None, save: bool = True, **kwargs):
    def _evaluate(session: Session):

        extractions_to_evaluate = get_records(PeptideEntry, session=session, **kwargs)
        failed_entries = []
        run_identifier = int(time.time())

        for extraction in extractions_to_evaluate:
            try:
                paper_for_extraction = extraction.paper_id
                paper_content = get_paper_content_by_paper_id(paper_for_extraction, session)

                if paper_content is None:
                    raise ValueError(f"No paper content found for peptide {extraction.id}")
                
                paper_fulltext = paper_content.get_content()

                evaluation = evaluate_extraction(extraction, paper_fulltext)
                evaluation.peptide_entry_id = extraction.id
                evaluation.run_identifier = run_identifier
                if save:
                    session.add(evaluation)
                    session.commit()
            except Exception as e:
                print(f"Failed to evaluate extraction for peptide {extraction.id}: {e}")
                failed_entries.append(extraction.id)
            time.sleep(1)

        return failed_entries, run_identifier

    if session is not None:
        return _evaluate(session)
    init_db()
    with get_session_context() as s:
        return _evaluate(s)

if __name__ == "__main__":
    find_dotenv("..")
    load_dotenv()

    ncbi_api_key = os.getenv("NCBI_API_KEY")
    scopus_api_key = os.getenv("SCOPUS_API_KEY")
    open_alex_api_key = os.getenv("OPEN_ALEX_API_KEY")

    ncbi_provider = NCBIProvider(api_key=ncbi_api_key, batch_size=100)
    open_alex_provider = OpenAlexProvider(api_key=open_alex_api_key)
    
    # 1. Run query search
    #retrive_ncbi_by_query(ncbi_provider, db="pmc", query_list=NCBI_SEARCH_QUERIES_V3)

    # 2. Deduplicate by doi (+ normalize doi and remove empty)
    #removed = deduplicate_by_doi()
    #print(f"Removed {removed} papers with duplicate doi")

    # 3. Find abstracts by doi
    #find_abstracts_by_doi(open_alex_api_key)

    # 4. Remove papers with empty abstract or title
    #removed = remove_empty_abstract_title_papers()
    #print(f"Removed {removed} papers with empty abstract or title")
    
    # 5. Run relevance screening
    #start_time = time.time()
    
    # doing relevance in batches
    #success, fails, identifier = run_bulk_relevance_screening("gemma4:26b", "ollama", delay_seconds=1, batch_size=1, search_query=NCBI_SEARCH_QUERIES_V3, limit=5000, offset=0)
    
    #success, fails, identifier = run_bulk_relevance_screening("llama4:16x17b", "ollama", delay_seconds=1, batch_size=10, id=9533, limit=5000, offset=0)
    
    #success, fails, identifier = run_bulk_relevance_screening("gemini-3.5-flash-lite", "google", delay_seconds=3, batch_size=50, search_query=NCBI_SEARCH_QUERIES_V3, limit=10000, offset=0)

    #end_time = time.time()
    #print(f"Relevance Success: {success}, Fails: {fails}, Identifier: {identifier}, Time: {end_time - start_time}")
    
    # 6. Fetch fulltexts
    #fetch_paper_fulltexts(ncbi_provider=ncbi_provider, open_alex_provider=open_alex_provider, search_query=NCBI_SEARCH_QUERIES_V3, status=PaperStatus.USABLE)
    from peptide_extractor.scratchpad import papers_to_extract_2

    # 7. Run extraction
    start_time = time.time()
    fails, identifier = run_peptide_extraction(provider="google", model_name="gemini-3.5-flash-lite", delay_seconds=4, id=papers_to_extract_2, limit=450, offset=0)
    end_time = time.time()
    print(f"Extraction failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}")

    # 8. Run evaluation
    #start_time = time.time()
    #fails, identifier = run_evaluation(run_identifier=123)
    #end_time = time.time()
    #print(f"Evaluation failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}")
