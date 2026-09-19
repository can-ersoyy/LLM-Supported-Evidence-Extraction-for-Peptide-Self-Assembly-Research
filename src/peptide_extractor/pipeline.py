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

    # 1. Search in different providers
    # 2. Complete abstracts and deduplicate by doi
    # 2.5 Do relevance check on deduplicated dois
    # 3. Fetch full texts for relevant papers

    ncbi_provider = NCBIProvider(ncbi_api_key, batch_size=150)
    #scopus_provider = ScopusProvider(scopus_api_key)

    

    # retrieve_golden_dataset(ncbi_provider)
    # retrieve_ncbi_by_query(ncbi_provider)
    # retrieve_scopus_by_query(scopus_provider)


    #open_alex_api_key = os.getenv("OPEN_ALEX_API_KEY")
    #find_abstracts_by_doi(open_alex_api_key)

    #run_relevance_screening(provider="google_genai", model_name="gemma-4-31b-it", delay_seconds=0.5, id=missed_paper_ids)
    #total_screened, failed_paper_ids, run_identifier = run_relevance_screening(provider="google_genai", model_name="gemini-3.1-flash-lite", delay_seconds=2, dataset_subset=DatasetSubset.SCRAPED, batch_size=50)
    
    #missed_ids = [110,111,112,113,114,115,116,117,118,119]
    #total_screened, failed_paper_ids, run_identifier = run_relevance_screening(provider="ollama", model_name="llama3.2:latest", delay_seconds=0.5, id=missed_ids)
   
    #print(f"Run identifier: {run_identifier}")
    #print(f"Total succesfully screened papers: {total_screened}")
    #print(f"Following papers failed to be screened: {failed_paper_ids}")

    #missing_ids = [73]
    #run_relevance_screening(provider="google_genai", model_name="gemma-4-31b-it", delay_seconds=0.5, id=missing_ids)
    
    #normalized = normalize_database_dois()
    #print(f"Normalized {normalized} DOIs.")
    #deleted = deduplicate_by_doi()
    #print(f"Deleted {deleted} duplicate records.")

    # last screened paper id: 369 by gemini3.5 flash lite

    # GOLDEN_DATASET paper fulltext retrieval - 1.via pmcid from ncbi

    #golden_papers = get_records(PaperMetadata, dataset_subset=DatasetSubset.GOLDEN_BENCHMARK)
    #fetched_papers, succesful_ids, failed_ids =  ncbi_provider.fetch(golden_papers)
#
    #print(f"Failed to fetch content for {len(failed_ids)} papers: {failed_ids}")
    #print(f"Succesfully fetched content for {len(succesful_ids)} papers: {succesful_ids}")
#
    #init_db()
    #with get_session_context() as s:
    #    s.add_all(fetched_papers)
    #    s.commit()

    """
    Failed to fetch content for 79 papers: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 66, 72, 75, 77, 79, 82, 85, 87, 88, 89, 90, 96, 97, 98, 99, 104, 109, 110, 112, 114, 115, 116, 117, 118, 119]
    
    """

    #scraped_papers = get_records(PaperMetadata, dataset_subset=DatasetSubset.SCRAPED)
    #fetched_papers, succesful_ids, failed_ids =  ncbi_provider.fetch(scraped_papers)
#
    #print(f"Failed to fetch content for {len(failed_ids)} papers: {failed_ids}")
    #print(f"Succesfully fetched content for {len(succesful_ids)} papers: {succesful_ids}")
##
    #init_db()
    #with get_session_context() as s:
    #    s.add_all(fetched_papers)
    #    s.commit()

    """
    Failed to fetch content for 426 papers: [121, 127, 132, 133, 139, 149, 150, 184, 206, 217, 230, 231, 241, 247, 262, 269, 276, 277, 282, 283, 284, 288, 289, 290, 291, 292, 293, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 315, 316, 317, 318, 320, 322, 323, 325, 329, 330, 334, 335, 336, 337, 338, 339, 342, 343, 344, 346, 347, 348, 350, 351, 352, 353, 356, 357, 359, 360, 361, 366, 368, 369, 371, 372, 374, 375, 376, 377, 380, 381, 383, 384, 385, 386, 387, 388, 390, 391, 392, 395, 397, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 418, 419, 420, 421, 423, 424, 426, 427, 429, 430, 431, 432, 433, 435, 436, 437, 438, 439, 440, 441, 442, 445, 446, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 464, 465, 466, 467, 469, 472, 473, 475, 476, 477, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 491, 492, 493, 494, 496, 498, 499, 500, 501, 502, 503, 505, 506, 507, 508, 509, 510, 512, 513, 514, 515, 517, 518, 519, 521, 522, 523, 524, 526, 527, 528, 529, 530, 531, 532, 534, 536, 538, 540, 541, 544, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 566, 567, 569, 570, 571, 572, 575, 577, 578, 579, 581, 584, 585, 586, 587, 588, 589, 590, 592, 593, 594, 595, 596, 598, 600, 603, 604, 605, 606, 607, 608, 609, 610, 611, 615, 616, 618, 620, 621, 622, 623, 624, 627, 628, 631, 632, 633, 635, 636, 637, 641, 642, 644, 646, 648, 649, 650, 653, 655, 656, 657, 660, 661, 662, 665, 666, 670, 672, 673, 674, 675, 676, 677, 678, 682, 683, 685, 686, 687, 690, 691, 693, 695, 696, 700, 704, 706, 707, 708, 709, 710, 711, 714, 715, 716, 718, 719, 720, 722, 723, 724, 725, 726, 728, 729, 730, 731, 733, 734, 738, 739, 741, 742, 743, 744, 745, 746, 747, 748, 750, 752, 754, 755, 756, 757, 760, 762, 763, 764, 766, 767, 769, 774, 776, 777, 778, 779, 780, 781, 782, 783, 786, 787, 788, 789, 790, 791, 792, 793, 795, 796, 797, 798, 799, 801, 802, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 830, 831, 835, 836, 838, 840, 841, 846, 848, 851, 852, 853, 854, 856, 858, 859, 863, 865, 866, 867]
    """