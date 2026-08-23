from abc import ABC, abstractmethod
import readline
from typing import List, Optional, Sequence, Tuple

import requests
import os
from dotenv import load_dotenv, find_dotenv
import json
import time

from sqlmodel import select
from peptide_extractor.db import get_session_context, init_db
from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.db.paper_metadata import DatasetSubset, PaperMetadata
from peptide_extractor.schemas import RawProviderRecord
from peptide_extractor.utils import get_pmcid_from_pmid, peptide_lengths_dict, process_duplicates_by_field

import xml.etree.ElementTree as ET


class BaseProvider(ABC):

    def __init__(self, api_key: Optional[str] = None, batch_size: int = 100):
        self.api_key = api_key
        self.batch_size = batch_size

    @abstractmethod
    def search_by_query(self, query: str, **kwargs) -> List[RawProviderRecord]:
        """Search provider using a text query string."""
        pass

    @abstractmethod
    def search_by_ids(self, id_list: List[str], **kwargs) -> List[RawProviderRecord]:
        """Search provider using a list of explicit IDs."""
        pass

    @abstractmethod
    def parse_raw_search_records(self, raw_records: List[RawProviderRecord]) -> List[PaperMetadata]:
        """Parse a list of RawProviderRecords into PaperMetadata instances."""
        pass

    @abstractmethod
    def fetch(self, paper_metadatas: List[PaperMetadata], **kwargs) -> List[PaperContent]:
        """Fetch full-text content for the given paper metadata instances."""
        pass


class NCBIProvider(BaseProvider):

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, api_key: Optional[str] = None, batch_size: int = 100):
        super().__init__(api_key, batch_size)

    def extract_match_count(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["count"]
    
    def extract_id_list(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["idlist"]
    

    def search_by_query(self, query: str, db="pmc", open_access: bool = True, not_review: bool = True) -> List[RawProviderRecord]:

        results: list[RawProviderRecord] = []

        if open_access:
            query = query + "AND \"open access\"[filter] "

        if not_review:
            query = query + "NOT review[publication type]"

        params = {
            "term":query,
            "retmax":1,
            "db": db,
            "retmode": "json",
            "api_key": self.api_key
        }
        
        # initial request to retrieve query match count
        response = requests.get(f"{self.BASE_URL}esearch.fcgi", params=params) 
        response.raise_for_status()
        
        matches = int(self.extract_match_count(response.content))
        time.sleep(0.1)

        for index_counter in range(0, matches, self.batch_size):

            params = {
                "term":query,
                "retmax":1,
                "db": db,
                "retmode": "json",
                "api_key": self.api_key
            }

            params["retstart"] = index_counter
            params["retmax"] = self.batch_size

            response = requests.get(f"{self.BASE_URL}esearch.fcgi", params=params)
            response.raise_for_status()
            time.sleep(0.2)
            id_list = self.extract_id_list(response.content)
            print(id_list)

    
            params = {
                "db":db,
                "retmode": "json",
                "api_key": self.api_key,
                "id": ",".join(id_list)
            }

            sum_res = requests.get(f"{self.BASE_URL}esummary.fcgi", params=params)
            sum_res.raise_for_status()

            time.sleep(0.2)
            result_dict = sum_res.json().get("result", {})
            uids = result_dict["uids"]

            for uid in uids:

                paper_metadata = result_dict[f"{uid}"]

                record = RawProviderRecord(
                    provider="pmc",
                    provider_id=uid,
                    search_query=query,
                    raw_data=paper_metadata
                )
                
                results.append(record)

        return results

    def search_by_ids(
        self,
        id_list: List[str],
        db: str = "pubmed",
        search_query: str = "id_list_search",
    ) -> List[RawProviderRecord]:
        results: list[RawProviderRecord] = []
        num_records = len(id_list)
    
        for i in range(0, num_records, self.batch_size):
            batch = id_list[i : i + self.batch_size]
            params = {
                "db": db,
                "retmode": "json",
                "api_key": self.api_key,
                "id": ",".join(str(x).strip() for x in batch if x),
            }
    
            sum_res = requests.get(f"{self.BASE_URL}esummary.fcgi", params=params)
            sum_res.raise_for_status()
            time.sleep(0.2)
    
            result_dict = sum_res.json().get("result", {})
            uids = result_dict.get("uids", [])
    
            for uid in uids:
                paper_metadata = result_dict.get(str(uid), {})
                record = RawProviderRecord(
                    provider=db,
                    provider_id=str(uid),
                    search_query=search_query,
                    raw_data=paper_metadata,
                )
                results.append(record)
    
        return results

    def parse_raw_search_records(self, raw_records: List[RawProviderRecord]) -> List[PaperMetadata]:
        paper_metadatas: List[PaperMetadata] = []

        for raw_record in raw_records:
            raw_json = raw_record.raw_data

            id_map = {
                item.get("idtype"): item.get("value") 
                for item in raw_json.get("articleids", [])
                if "idtype" in item
            }

            doi = id_map.get("doi")
            pmid = id_map.get("pmid") or id_map.get("pubmed") or (raw_record.provider_id if raw_record.provider == "pubmed" else None)
            pmcid = id_map.get("pmc") or id_map.get("pmcid")
            if pmcid:
                # Cleans "pmc-id: PMC6632094;" -> "PMC6632094"
                pmcid = pmcid.replace("pmc-id:", "").replace(";", "").strip()

            title = raw_json["title"]
            is_open_access = True # for now we just filter out non open access docs from ncbi

            paper_metadata = PaperMetadata(
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                is_open_access=is_open_access,
                title=title,
                search_query=raw_record.search_query,
            )

            paper_metadatas.append(paper_metadata)
            
        return paper_metadatas

    def fetch(self, paper_metadatas: Sequence[PaperMetadata], db = "pmc") -> List[Tuple[PaperMetadata, PaperContent]]:
        num_records = len(paper_metadatas)
        id_list = [paper.pmcid for paper in paper_metadatas]
        paper_tuples = []

        for i in range(0, num_records, self.batch_size):
            print("batch done")
            batch = id_list[i : i + self.batch_size]
            params = {
                "db": db,
                "retmode": "xml",
                "api_key": self.api_key,
                "id": ",".join(str(x).strip() for x in batch if x),
            }

            response = requests.get(f"{self.BASE_URL}efetch.fcgi", params=params)
            response.raise_for_status()
            time.sleep(0.2) 

            root = ET.fromstring(response.content)
            print("inside")

            for article_elem in root.findall("article"):

                body = article_elem.find("body")
                if body != None:
                    pmcid_node = article_elem.find(".//front/article-meta/article-id[@pub-id-type='pmcid']")
                    pmcid_text = pmcid_node.text.strip() if pmcid_node is not None and pmcid_node.text else None
                    if pmcid_text and not pmcid_text.upper().startswith("PMC"):
                        pmcid_text = f"PMC{pmcid_text}"

                    paper_metadata = next((p for p in paper_metadatas[i:i+self.batch_size] if p.pmcid == pmcid_text), None)
                    if paper_metadata:
                        xml_str = ET.tostring(article_elem, encoding="unicode")
                        print("this is xml str")
                        paper_content = PaperContent(
                            paper_id=paper_metadata.id,
                            paper=paper_metadata,
                            raw_content=xml_str,
                            format="xml"
                        )
                        paper_tuples.append((paper_metadata, paper_content))
        
        return paper_tuples

"""
class EuropePMCProvider(BaseProvider):

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/"

    def __init__(self, api_key: str | None = None, batch_size: int = 100):
        super().__init__(api_key, batch_size)

    def search(self):

        results: list[RawProviderRecord] = []

        params = {
            "format": "json",
            "pageSize": self.batch_size,
            "resultType": "lite"
        }

        for key, name in peptide_lengths_dict.items():
            cursor = "*"
            query = f'({name} AND "self-assembly" AND (HAS_FT:Y) AND (OPEN_ACCESS:y)) NOT (PUB_TYPE:"Review" OR PUB_TYPE:"review-article" OR PUB_TYPE:"Systematic Review")'
            params["query"] = query

            while cursor:
                params["cursorMark"] = cursor
                response = requests.get(f"{self.BASE_URL}search", params=params)
                response.raise_for_status()
                time.sleep(0.2)

                data = response.json()
                items = data.get("resultList", {}).get("result", [])
                next_cursor = data.get("nextCursorMark")

                if not items:
                    break

                for item in items:
                    provider_id = str(item.get("id"))
                    self.peptide_id_dict[key].append(provider_id)

                    record = RawProviderRecord(
                        provider="europe_pmc",
                        provider_id=provider_id,
                        search_query=query,
                        raw_data=item
                    )
                    results.append(record)

                if not next_cursor or next_cursor == cursor:
                    break

                cursor = next_cursor

        print(f"dedup count: {len(process_duplicates_by_field(results, 'provider_id'))}")

        return results
"""

if __name__ == "__main__":

    find_dotenv("..")
    load_dotenv()

    api_key = os.getenv("NCBI_API_KEY")

    ncbi_provider = NCBIProvider(api_key, batch_size=150)

    """
    ids = []
    with open("data/waltzdb/waltzdb_export.json", "r") as f:
        obj = json.load(f)
        for entry in obj:
            resource = entry["Reference"]
            ids.append(resource)

        print(ids)

    items = ncbi_provider.search_by_ids(ids)
    """
    """
    items2 = []
    for value in peptide_lengths_dict.values():
        query = f"({value}[tiab] AND self-assembly[tiab])"
        l = ncbi_provider.search_by_query(query, open_access=False)
        items2.extend(l)
    """
    #this is used for golden_benchmark papers (dataset from manually curated databases)

    import csv

    papers = []

    with open("data/literature/output/pmids_with_sources.csv", mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            papers.append({"pmid": row["pmid"], "databases": row["databases"]})


    #items3 = ncbi_provider.search_by_ids([paper["pmid"] for paper in papers])
    
    
    #clean_records = ncbi_provider.parse_raw_search_records(items)
    #clean_records2 = ncbi_provider.parse_raw_search_records(items2)
    #clean_records3 =  ncbi_provider.parse_raw_search_records(items3) # for golden_subset
    

    init_db()
    with get_session_context() as session:
        #session.add_all(clean_records)
        #session.add_all(clean_records2)
        #session.add_all(clean_records3)
        session.commit()

    with get_session_context() as session:
            # 1. Query papers that have a PMCID and don't already have PaperContent
            papers_to_fetch = session.exec(
                select(PaperMetadata).where(PaperMetadata.pmcid != None)
            ).all()
    
            # 2. Fetch full-text XML
            print(papers_to_fetch)
            all_records = ncbi_provider.fetch(papers_to_fetch)
            fetched_items = [content for (_, content) in all_records]
    
            # 3. Save paper content
            session.add_all(fetched_items)
            session.commit()



#
#    from sqlalchemy import text
#
#    with get_session_context() as session:
#        for paper in papers:
#            session.exec(
#                text("""
#                    UPDATE paper_metadata  # type: ignore
#                    SET dataset_subset = 'golden_benchmark', # type: ignore 
#                        source_databases = :databases # type: ignore
#                    WHERE pmid = :pmid # type: ignore 
#    """), # type: ignore
#                params={"databases": paper["databases"], "pmid": paper["pmid"]}
#            ) # type: ignore
#        session.commit()
#        print("Updated dataset_subset and source_databases successfully!")
#

    #europe_pmc_provider = EuropePMCProvider(api_key, batch_size=150)
    #items = europe_pmc_provider.search()
    #print(len(items))
    