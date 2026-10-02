from peptide_extractor.azure_storage import download_xml
from peptide_extractor.db import PaperContent
from peptide_extractor.extraction import run_extraction
from peptide_extractor.utils import get_body_from_jats_xml
from peptide_extractor.relevance import screen_papers
from peptide_extractor.db.paper_metadata import update_paper_pmcid
from peptide_extractor.utils import get_pmcid_from_pmid
from peptide_extractor.providers import NCBIProvider
from sqlmodel import select
from peptide_extractor.utils import normalize_doi
from collections import defaultdict
from peptide_extractor.db import get_records
from peptide_extractor.db import DatasetSubset
from peptide_extractor.db import PaperMetadata
from peptide_extractor.providers import OpenAlexProvider
import time
from peptide_extractor.providers import SemanticScholarProvider
from peptide_extractor.db import get_paper_metadatas
import csv
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from sqlmodel import Session
from peptide_extractor.db import get_session_context, init_db
import os

GOLDEN_DATASET_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "golden_dataset_final.csv"
)

# maps peptide length to structure name or self-assembly keyword, includes open-access, english and non review article filters, only title and abstract
NCBI_SEARCH_QUERY_V1 = """((tetrapeptide[tiab] OR pentapeptide[tiab] OR hexapeptide[tiab] OR heptapeptide[tiab] OR octapeptide[tiab] OR nonapeptide[tiab] OR decapeptide[tiab] OR undecapeptide[tiab] OR dodecapeptide[tiab] OR tridecapeptide[tiab] OR tetradecapeptide[tiab] OR pentadecapeptide[tiab] OR hexadecapeptide[tiab] OR heptadecapeptide[tiab] OR octadecapeptide[tiab] OR nonadecapeptide[tiab] OR eicosapeptide[tiab] OR icosapeptide[tiab]) AND (nanostructure[tiab] OR nanotube[tiab] OR nanofiber[tiab] OR nanofibre[tiab] OR nanorod[tiab] OR hydrogel[tiab] OR nanosphere[tiab] OR nanoparticle[tiab] OR "self-assembly"[tiab])) AND (english[la]) AND ("free full text"[filter]) NOT (review[pt])"""

NCBI_SEARCH_QUERY_V2 = """peptide self-assembly"""

SCOPUS_SEARCH_QUERY_V1 = """TITLE-ABS-KEY(tetrapeptide OR pentapeptide OR hexapeptide OR heptapeptide OR octapeptide OR nonapeptide OR decapeptide OR undecapeptide OR dodecapeptide OR tridecapeptide OR tetradecapeptide OR pentadecapeptide OR hexadecapeptide OR heptadecapeptide OR octadecapeptide OR nonadecapeptide OR eicosapeptide OR icosapeptide) AND TITLE-ABS-KEY(nanostructure OR nanotube OR nanofiber OR nanofibre OR nanorod OR hydrogel OR nanosphere OR nanoparticle OR "self-assembly") AND LANGUAGE(english) AND DOCTYPE(ar) AND OA(all)"""

def get_golden_dataset_pmids(csv_path: Optional[Path | str] = None) -> List[str]:
    """Parse golden_dataset_initial.csv and return unique PubMed IDs."""
    path = Path(csv_path) if csv_path else GOLDEN_DATASET_CSV_PATH
    pmids: List[str] = []

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("pmid", "").strip()
            if pmid and pmid not in pmids:
                pmids.append(pmid)

    return pmids


# Pre-parsed list of golden dataset PubMed IDs
#golden_dataset_pmids: List[str] = get_golden_dataset_pmids()

def retrieve_golden_dataset(ncbi_provider, golden_dataset_pmids: List[str]):
    raw_records = ncbi_provider.search_by_ids(golden_dataset_pmids)
    golden_dataset_metadatas = ncbi_provider.parse_raw_search_records(raw_records)
    print(f"Retrieved {golden_dataset_metadatas}")

    init_db()
    with get_session_context() as session:
        session.add_all(golden_dataset_metadatas)
        session.commit()


def retrive_ncbi_by_query(ncbi_provider, query=NCBI_SEARCH_QUERY_V1, save=True):
    raw_records = ncbi_provider.search_by_query(query)
    ncbi_metadatas = ncbi_provider.parse_raw_search_records(raw_records)

    if save:
        init_db()
        with get_session_context() as session:
            session.add_all(ncbi_metadatas)
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

def find_abstracts_by_doi(open_alex_api_key):
    init_db()
    non_abstract_metadas = get_paper_metadatas(abstract="")
    abstract_found: list[PaperMetadata] = []
    
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
                    metadata.abstract = abstract
                    session.add(metadata)
                    session.commit()
                else:
                    try:
                        abstract = open_alex.get_abstract_by_doi(metadata.doi)
                        time.sleep(0.5)
                        if abstract is not None:
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
                            metadata.abstract = abstract
                            session.add(metadata)
                            session.commit()
                        print(f"Retrieved abstract for {metadata.doi}: {abstract}")
                    except Exception as e:
                        print(f"Failed to retrieve abstract for {metadata.doi}: {e}")
                print(f"Failed to retrieve abstract for {metadata.doi}: {e}")
                
            time.sleep(1)

        print(f"Retrieved {len(abstract_found)} abstracts out of {len(non_abstract_metadas)}")

def deduplicate_by_doi(session: Optional[Session] = None) -> int:
    """Removes scraped PaperMetadata records with duplicate DOIs according to priority rules:
    1. Has non-empty abstract over missing abstract
    2. Has non-empty PMCID over missing PMCID
    3. Provider 'ncbi' over other providers
    4. Lower database ID over higher database ID (keeps the older record, deletes the higher ID)

    Returns the number of deleted duplicate records.
    """
    def _deduplicate(s: Session) -> int:
        metadatas = get_records(PaperMetadata, dataset_subset=DatasetSubset.SCRAPED, session=s)
        
        # Group records by DOI (normalized/lowercased, ignoring records with no DOI)
        doi_groups = defaultdict(list)
        for m in metadatas:
            if m.doi and m.doi.strip():
                doi_groups[m.doi.strip().lower()].append(m)

        deleted_count = 0
        for doi, group in doi_groups.items():
            if len(group) <= 1:
                continue
            # Priority scoring tuple: higher values are kept
            # (has_abstract, has_pmcid, is_ncbi, -db_id)
            def candidate_score(paper: PaperMetadata):
                return (
                    1 if bool(paper.abstract and paper.abstract.strip()) else 0,
                    1 if bool(paper.pmcid and paper.pmcid.strip()) else 0,
                    1 if paper.provider == "ncbi" else 0,
                    -(paper.id or 0),  # Keep lower ID, delete higher ID
                )
            # Sort best candidate to the end
            sorted_candidates = sorted(group, key=candidate_score)
            winner = sorted_candidates[-1]
            duplicates_to_delete = sorted_candidates[:-1]
            for dup in duplicates_to_delete:
                s.delete(dup)
                deleted_count += 1
        s.commit()
        return deleted_count

    if session is not None:
        return _deduplicate(session)

    init_db()
    with get_session_context() as s:
        return _deduplicate(s)

def normalize_database_dois(session: Optional[Session] = None) -> int:
    """Normalizes all existing DOIs in the database to lowercase, prefix-free format."""
    def _normalize(s: Session) -> int:
        papers = s.exec(select(PaperMetadata).where(PaperMetadata.doi.is_not(None))).all()
        updated_count = 0
        for paper in papers:
            norm = normalize_doi(paper.doi)
            if norm != paper.doi:
                paper.doi = norm
                s.add(paper)
                updated_count += 1
        s.commit()
        return updated_count

    if session is not None:
        return _normalize(session)
    with get_session_context() as s:
        return _normalize(s)


def run_relevance_screening(
    provider: str = "google_genai",
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    api_key: Optional[str] = None,
    delay_seconds: float = 2.0,
    golden_dataset: bool = False,
    **kwargs
) -> int:
    """Run abstract relevance screening on paper metadata records in the database.

    Evaluates papers using the relevance classifier, persisting PaperRelevance records
    and updating paper status to USABLE or DISCARD.
     If batch_size is None, processes ALL eligible papers in the database.

    Returns the total number of paper records screened, failed_paper_ids and run_identifier
    """
    from peptide_extractor.llm import LLMController
    from peptide_extractor.relevance import (
        SYSTEM_PROMPT_GOLDEN_BENCHMARK,
        RelevanceResult,
        screen_papers,
    )

    init_db()
    classifier = LLMController(
        provider=provider,
        model_name=model_name,
        structured_output=RelevanceResult,
        system_prompt=SYSTEM_PROMPT_GOLDEN_BENCHMARK,
        api_key=api_key,
    )
    screened_results, failed_paper_ids, run_identifier = screen_papers(
        batch_size=batch_size,
        classifier=classifier,
        delay_seconds=delay_seconds,
        golden_dataset=golden_dataset,
        **kwargs
    )
    return len(screened_results), failed_paper_ids, run_identifier


if __name__ == "__main__":
    find_dotenv("..")
    load_dotenv()

    ncbi_api_key = os.getenv("NCBI_API_KEY")
    scopus_api_key = os.getenv("SCOPUS_API_KEY")
    open_alex_api_key = os.getenv("OPEN_ALEX_API_KEY")

    # 1. Search in different providers
    # 2. Complete abstracts and deduplicate by doi
    # 2.5 Do relevance check on deduplicated dois
    # 3. Fetch full texts for relevant papers

    #ncbi_provider = NCBIProvider(ncbi_api_key, batch_size=200)

    pmid_list = [
    18070345,
    25010703,
    22016623,
    25621167,
    22915849,
    23020140,
    23800286,
    24786493,
    33579941,
    28942918,
    ]

    # for testing purposes manually retrieve the paper content

    paper_ids = [paper_metadata.id for paper_metadata in get_records(PaperMetadata, pmid=pmid_list[0])]

    for id in paper_ids:
        paper_content = get_records(PaperContent, paper_id=id)[0]

        blob_url = paper_content.blob_url
        raw_paper_content = download_xml(blob_url, decompress=False)
        sample_body = get_body_from_jats_xml(raw_paper_content)
        print(sample_body)
        
        peptide_entries = run_extraction(sample_body)
        with get_session_context() as session:
            session.add_all(peptide_entries)
            session.commit()
            
    
    