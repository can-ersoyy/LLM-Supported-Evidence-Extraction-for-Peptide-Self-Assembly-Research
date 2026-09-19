from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import requests
import json
import time

from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.db.paper_metadata import PaperMetadata, PaperStatus
from peptide_extractor.schemas import RawProviderRecord

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
    

    def search_by_query(self, query: str, db="pubmed", open_access: bool = False, not_review: bool = False) -> List[RawProviderRecord]:

        print("executing search_by_query")

        results: list[RawProviderRecord] = []

        if open_access:
            oa_filter = '"free full text"[filter]' if db == "pubmed" else '"open access"[filter]'
            query = f"{query.strip()} AND {oa_filter} "

        if not_review:
            query = f"{query.strip()} NOT review[publication type] "

        params = {
            "term":query,
            "retmax":1,
            "db": db,
            "retmode": "json",
            "api_key": self.api_key
        }
        
        # initial request to retrieve query match count
        response = requests.get(f"{self.BASE_URL}esearch.fcgi", params=params) 
        print(response.content)
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

    
            if not id_list:
                continue

            params = {
                "db": db,
                "retmode": "xml",
                "api_key": self.api_key,
                "id": ",".join(id_list)
            }

            fetch_res = requests.get(f"{self.BASE_URL}efetch.fcgi", params=params)
            fetch_res.raise_for_status()

            time.sleep(0.2)
            root = ET.fromstring(fetch_res.content)

            for article_elem in root:
                xml_str = ET.tostring(article_elem, encoding="unicode")
                record = RawProviderRecord(
                    provider="ncbi",
                    search_query=query,
                    raw_data={"xml": xml_str}
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
            cleaned_batch = [str(x).strip() for x in batch if str(x).strip()]
            if not cleaned_batch:
                continue

            params = {
                "db": db,
                "retmode": "xml",
                "api_key": self.api_key,
                "id": ",".join(cleaned_batch),
            }
    
            fetch_res = requests.get(f"{self.BASE_URL}efetch.fcgi", params=params)
            fetch_res.raise_for_status()
            time.sleep(0.2)
    
            root = ET.fromstring(fetch_res.content)
            for article_elem in root:
                xml_str = ET.tostring(article_elem, encoding="unicode")
                record = RawProviderRecord(
                    provider=db,
                    search_query=search_query,
                    raw_data={"xml": xml_str},
                )
                results.append(record)
    
        return results

    def parse_xml(self, xml_content: str | ET.Element) -> dict:
        """Parse an article XML element or XML string to extract metadata, abstract, and IDs."""
        if isinstance(xml_content, str):
            elem = ET.fromstring(xml_content)
        else:
            elem = xml_content

        # 1. Title
        title_node = elem.find(".//MedlineCitation/Article/ArticleTitle")
        if title_node is None:
            title_node = elem.find(".//ArticleTitle") or elem.find(".//title-group/article-title")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        title = " ".join(title.split())

        # 2. Abstract (supports structured abstracts with Label attributes and handles nested tags)
        abstract_nodes = elem.findall(".//AbstractText") or elem.findall(".//abstract/p") or elem.findall(".//abstract/sec")
        abstract_parts = []
        for node in abstract_nodes:
            label = node.get("Label")
            text = "".join(node.itertext()).strip()
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)
        abstract = "\n\n".join(abstract_parts)

        # 3. Identifiers
        id_map = {}
        for aid in elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
            id_type = aid.get("IdType")
            if id_type and aid.text:
                id_map[id_type.lower()] = aid.text.strip()

        # PMC XML article-id fallback
        for aid in elem.findall(".//article-id"):
            id_type = aid.get("pub-id-type")
            if id_type and aid.text:
                id_map[id_type.lower()] = aid.text.strip()

        pmid = elem.findtext(".//MedlineCitation/PMID") or id_map.get("pubmed") or id_map.get("pmid")
        doi = id_map.get("doi")
        pmcid = id_map.get("pmc") or id_map.get("pmcid")
        if pmcid:
            pmcid = pmcid.replace("pmc-id:", "").replace(";", "").strip()
            if not pmcid.upper().startswith("PMC"):
                pmcid = f"PMC{pmcid}"

        return {
            "title": title,
            "abstract": abstract,
            "pmid": pmid.strip() if pmid else None,
            "doi": doi.strip() if doi else None,
            "pmcid": pmcid,
        }

    def parse_raw_search_records(self, raw_records: List[RawProviderRecord]) -> List[PaperMetadata]:
        paper_metadatas: List[PaperMetadata] = []

        for raw_record in raw_records:
            if "xml" in raw_record.raw_data:
                parsed = self.parse_xml(raw_record.raw_data["xml"])
                doi = parsed.get("doi")
                pmid = parsed.get("pmid")
                pmcid = parsed.get("pmcid")
                title = parsed.get("title", "")
                abstract = parsed.get("abstract", "")
            else:
                raw_json = raw_record.raw_data
                id_map = {
                    item.get("idtype"): item.get("value")
                    for item in raw_json.get("articleids", [])
                    if "idtype" in item
                }
                doi = id_map.get("doi")
                pmid = id_map.get("pmid") or id_map.get("pubmed")
                pmcid = id_map.get("pmc") or id_map.get("pmcid")
                if pmcid:
                    pmcid = pmcid.replace("pmc-id:", "").replace(";", "").strip()
                    if not pmcid.upper().startswith("PMC"):
                        pmcid = f"PMC{pmcid}"
                title = raw_json.get("title", "")
                abstract = raw_json.get("abstract", "")

            # Determine open access status:
            # - If PMCID is present, full text is deposited in PMC and freely accessible.
            # - Alternatively, check if the record originated from an explicit free/open access search query.
            is_open_access = bool(pmcid) or any(
                term in (raw_record.search_query or "").lower()
                for term in ['"free full text"', '"open access"']
            )

            paper_metadata = PaperMetadata(
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                title=title,
                abstract=abstract,
                status=PaperStatus.NOT_PROCESSED,
                is_open_access=is_open_access,
                search_query=raw_record.search_query,
                provider="ncbi"
            )

            paper_metadatas.append(paper_metadata)

        return paper_metadatas

    def fetch(self, paper_metadatas: Sequence[PaperMetadata], db = "pmc", batch_size=10) -> List[Tuple[PaperMetadata, PaperContent]]:
        num_records = len(paper_metadatas)
        pmcid_list = [paper.pmcid for paper in paper_metadatas]
        
        succesful_paper_ids: list[int] = []
        failed_paper_ids: list[int] = []
        paper_contents: List[PaperContent] = []

        for i in range(0, num_records, batch_size):
            print("batch done")
            batch_pmcids = pmcid_list[i : i + batch_size]
            
            params = {
                "db": db,
                "retmode": "xml",
                "api_key": self.api_key,
                "id": ",".join(str(x).strip() for x in batch_pmcids if x),
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

                    paper_metadata = next((p for p in paper_metadatas[i:i+batch_size] if p.pmcid == pmcid_text), None)
                    if paper_metadata:
                        xml_str = ET.tostring(article_elem, encoding="unicode")
                        print("this is xml str")
                        paper_content = PaperContent(
                            paper_id=paper_metadata.id,
                            paper=paper_metadata,
                            raw_content=xml_str,
                            format="xml",
                            source=db
                        )
                        paper_contents.append(paper_content)
                        succesful_paper_ids.append(paper_metadata.id)

        failed_paper_ids = [paper.id for paper in paper_metadatas if paper.id not in succesful_paper_ids]
        
        return paper_contents, succesful_paper_ids, failed_paper_ids

class ScopusProvider(BaseProvider):

    BASE_URL = "https://api.elsevier.com/content/search/scopus"

    def __init__(self, api_key: Optional[str] = None, batch_size: int = 25):
            super().__init__(api_key, batch_size)

    def search_by_query(self, query):
        
        results: List[RawProviderRecord] = []

        params = {
            "query" : query,
            "apiKey" : self.api_key,
            "contentType" : "json",
            "start": 0,
            "count": self.batch_size
        }

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        time.sleep(0.2)

        data = response.json()
        count = data.get("search-results", {}).get("opensearch:totalResults", 0)
        
        for page_number in range(0, int(count), self.batch_size):
            print(f"receving the {page_number}th to {page_number+self.batch_size} results")
            params["start"] = page_number
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            time.sleep(0.2)
            data = response.json()

            items = data.get("search-results", {}).get("entry", [])

            for item in items:
                results.append(
                    RawProviderRecord(
                        provider="scopus",
                        search_query=query,
                        raw_data=item
                    )
                )
        
        return results

    def search_by_ids():
        pass

    def parse_raw_search_records(self, raw_records: List[RawProviderRecord])-> List[PaperMetadata]:
        paper_metadatas = []

        for raw_record in raw_records:
            data = raw_record.raw_data
            
            doi = data.get("prism:doi", "")
            pmid = data.get("pubmed-id", "")
            title = data.get("dc:title", "")

            paper_metadata = PaperMetadata(
                doi=doi,
                pmid=pmid,
                pmcid=None,
                title=title,
                abstract="",
                status=PaperStatus.NOT_PROCESSED,
                is_open_access=True,
                search_query=raw_record.search_query,
                provider="scopus"
            )

            paper_metadatas.append(paper_metadata)

        return paper_metadatas

    def fetch():
        pass

class SemanticScholarProvider():
    """This is not a proper provider and is intended to be used for utility functions like finding abstracts of papers."""

    def __init__(self):
        pass

    def get_abstract_by_doi(self, doi: str) -> str | None:
        BASE_URL="https://api.semanticscholar.org/graph/v1/paper/DOI:"

        url = f"{BASE_URL}{doi}?fields=abstract"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        abstract = data.get("abstract", None)
        return None if abstract == "None" or abstract == "" else abstract
    
class OpenAlexProvider():
    """This is not a proper provider and is intended to be used for utility functions like finding abstracts of papers."""

    BASE_URL="https://api.openalex.org/works/"

    def __init__(self, api_key):
        self.api_key = api_key

    def reconstruct_abstract(self, inverted_index: dict[str, list[int]] | None) -> str:
        if inverted_index is None:
            return ""
        tokens = [
            (pos, word)
            for word, positions in inverted_index.items()
            for pos in positions
        ]

        tokens.sort(key=lambda item: item[0])
        return " ".join(word for _, word in tokens)

    def get_abstract_by_doi(self, doi:str) -> str | None:
        url = f"{self.BASE_URL}https://doi.org/{doi}?select=abstract_inverted_index"

        params = {
            "api_key": self.api_key
        }
        response = requests.get(url, params)
        response.raise_for_status()
        data = response.json()
        abstract_list = data.get("abstract_inverted_index", {})
        
        abstract = self.reconstruct_abstract(abstract_list)
        return None if abstract == "None" or abstract == "" else abstract

        

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