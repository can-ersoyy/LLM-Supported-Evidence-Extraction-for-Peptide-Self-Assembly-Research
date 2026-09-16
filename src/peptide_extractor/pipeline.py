from peptide_extractor.db import PaperMetadata
from peptide_extractor.providers import OpenAlexProvider
import time
from peptide_extractor.providers import SemanticScholarProvider
from peptide_extractor.db import get_paper_metadatas
import csv
from pathlib import Path
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from peptide_extractor.db import get_session_context, init_db
import os

GOLDEN_DATASET_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "literature"
    / "golden_dataset_initial.csv"
)


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
golden_dataset_pmids: List[str] = get_golden_dataset_pmids()

def retrieve_golden_dataset(ncbi_provider):
    raw_records = ncbi_provider.search_by_ids(golden_dataset_pmids)
    golden_dataset_metadatas = ncbi_provider.parse_raw_search_records(raw_records)

    init_db()
    with get_session_context() as session:
        session.add_all(golden_dataset_metadatas)
        session.commit()


def retrive_ncbi_by_query(ncbi_provider):
    query = """((tetrapeptide[tiab] OR pentapeptide[tiab] OR hexapeptide[tiab] OR heptapeptide[tiab] OR octapeptide[tiab] OR nonapeptide[tiab] OR decapeptide[tiab] OR undecapeptide[tiab] OR dodecapeptide[tiab] OR tridecapeptide[tiab] OR tetradecapeptide[tiab] OR pentadecapeptide[tiab] OR hexadecapeptide[tiab] OR heptadecapeptide[tiab] OR octadecapeptide[tiab] OR nonadecapeptide[tiab] OR eicosapeptide[tiab] OR icosapeptide[tiab]) AND (nanostructure[tiab] OR nanotube[tiab] OR nanofiber[tiab] OR nanofibre[tiab] OR nanorod[tiab] OR hydrogel[tiab] OR nanosphere[tiab] OR nanoparticle[tiab] OR "self-assembly"[tiab])) AND (english[la]) AND ("free full text"[filter]) NOT (review[pt])"""
    raw_records = ncbi_provider.search_by_query(query)
    ncbi_metadatas = ncbi_provider.parse_raw_search_records(raw_records)

    init_db()
    with get_session_context() as session:
        session.add_all(ncbi_metadatas)
        session.commit()

def retrieve_scopus_by_query(scopus_provider):
    query = """TITLE-ABS-KEY(tetrapeptide OR pentapeptide OR hexapeptide OR heptapeptide OR octapeptide OR nonapeptide OR decapeptide OR undecapeptide OR dodecapeptide OR tridecapeptide OR tetradecapeptide OR pentadecapeptide OR hexadecapeptide OR heptadecapeptide OR octadecapeptide OR nonadecapeptide OR eicosapeptide OR icosapeptide) AND TITLE-ABS-KEY(nanostructure OR nanotube OR nanofiber OR nanofibre OR nanorod OR hydrogel OR nanosphere OR nanoparticle OR "self-assembly") AND LANGUAGE(english) AND DOCTYPE(ar) AND OA(all)"""
    raw_records = scopus_provider.search_by_query(query)
    scopus_metadatas = scopus_provider.parse_raw_search_records(raw_records)

    init_db()
    with get_session_context() as session:
        session.add_all(scopus_metadatas)
        session.commit()

def find_abstracts_by_doi(open_alex_api_key):
    init_db()
    non_abstract_metadas = get_paper_metadatas(abstract="")
    abstract_found: list[PaperMetadata] = []
    
    semantic_scholar = SemanticScholarProvider()
    open_alex = OpenAlexProvider(api_key=open_alex_api_key)

    
    
    with get_session_context() as session:
        for metadata in non_abstract_metadas:
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



def run_relevance_screening(
    provider: str = "google_genai",
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    api_key: Optional[str] = None,
    delay_seconds: float = 2.0,
    all_papers: bool = False,
    golden_dataset: bool = False,
) -> int:
    """Run abstract relevance screening on paper metadata records in the database.

    Evaluates papers using the relevance classifier, persisting PaperRelevance records
    and updating paper status to USABLE or DISCARD.
    If all_papers is False and ignore_status is False (default), evaluates papers with status NOT_PROCESSED.
    If all_papers is True or ignore_status is True, evaluates papers regardless of their status.
    If batch_size is None, processes ALL eligible papers in the database.

    Returns the total number of paper records screened in this execution.
    """
    from peptide_extractor.relevance import (
        PaperRelevanceClassifier,
        screen_papers,
    )

    init_db()
    classifier = PaperRelevanceClassifier(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
    )
    screened_results = screen_papers(
        batch_size=batch_size,
        classifier=classifier,
        delay_seconds=delay_seconds,
        all_papers=all_papers,
        golden_dataset=golden_dataset,
    )
    return len(screened_results)


if __name__ == "__main__":
    find_dotenv("..")
    load_dotenv()

    ncbi_api_key = os.getenv("NCBI_API_KEY")
    scopus_api_key = os.getenv("SCOPUS_API_KEY")

    # 1. Search in different providers
    # 2. Complete abstracts and deduplicate by doi
    # 2.5 Do relevance check on deduplicated dois
    # 3. Fetch full texts for relevant papers

    #ncbi_provider = NCBIProvider(ncbi_api_key, batch_size=150)
    #scopus_provider = ScopusProvider(scopus_api_key)

    

    # retrieve_golden_dataset(ncbi_provider)
    # retrieve_ncbi_by_query(ncbi_provider)
    # retrieve_scopus_by_query(scopus_provider)


    #open_alex_api_key = os.getenv("OPEN_ALEX_API_KEY")
    #find_abstracts_by_doi(open_alex_api_key)
    
    run_relevance_screening(provider="google_genai", model_name="gemma-4-31b-it", delay_seconds=0.5, golden_dataset=True)
    #run_relevance_screening(provider="google_genai", model_name="gemma-4-26b-a4b-it", delay_seconds=1, all_papers=True)
    


    