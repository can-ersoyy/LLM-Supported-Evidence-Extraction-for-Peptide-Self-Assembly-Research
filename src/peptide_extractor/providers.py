import logging
import os

logger = logging.getLogger(__name__)
from peptide_extractor.utils import (
    convert_pmc_to_pmid_batch,
    doi_to_filename,
    normalize_doi,
    retry_on_external_error,
)
from peptide_extractor.db.raw_provider_record import save_raw_provider_record
from peptide_extractor.db.raw_provider_record import RawProviderRecord
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import requests
import json
import time

from peptide_extractor.azure_storage import upload_xml, upload_pdf
from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.db.paper_metadata import PaperMetadata, PaperStatus

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
    def parse_raw_search_records(
        self, raw_records: List[RawProviderRecord]
    ) -> List[PaperMetadata]:
        """Parse a list of RawProviderRecords into PaperMetadata instances."""
        pass

    @abstractmethod
    def fetch(
        self, paper_metadatas: List[PaperMetadata], **kwargs
    ) -> List[PaperContent]:
        """Fetch full-text content for the given paper metadata instances."""
        pass


class NCBIProvider(BaseProvider):
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        rate_limit_delay: Optional[float] = None,
    ):
        super().__init__(api_key, batch_size)
        # NCBI allows 10 req/s with an API key (0.11s) and 3 req/s without (0.35s)
        self.rate_limit_delay = (
            rate_limit_delay
            if rate_limit_delay is not None
            else (0.11 if api_key else 0.35)
        )
        self._last_request_time: Optional[float] = None

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_time is not None and self.rate_limit_delay > 0:
            elapsed = time.monotonic() - self._last_request_time
            remaining = self.rate_limit_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_time = time.monotonic()

    def _get_with_retry(
        self,
        url: str,
        params: dict,
        max_retries: int = 0,
        retry_delay: float = 2.0,
        context: str = "",
    ) -> requests.Response:
        def _do_get() -> requests.Response:
            self._wait_for_rate_limit()
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response

        return retry_on_external_error(
            _do_get,
            max_retries=max_retries,
            retry_delay=retry_delay,
            context=context,
        )

    @staticmethod
    def _normalize_pmcid(pmcid: Optional[str]) -> Optional[str]:
        if not pmcid:
            return None
        cleaned = str(pmcid).replace("pmc-id:", "").replace(";", "").strip()
        if not cleaned:
            return None
        if not cleaned.upper().startswith("PMC"):
            return f"PMC{cleaned}"
        return f"PMC{cleaned[3:]}"

    def extract_match_count(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["count"]

    def extract_id_list(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["idlist"]

    def search_by_query(
        self,
        query: str,
        db: str = "pubmed",
        use_history: bool = False,
        max_retries: int = 0,
        retry_delay: float = 2.0,
    ) -> List[RawProviderRecord]:
        """
        The usage of efetch here is to retrieve the abstract and title of the
        papers for metadata object before retrieving the fulltext. Hence db=pubmed.
        When ``use_history=True`` and ``db=="pubmed"``, uses NCBI Entrez WebEnv/query_key
        to page ``efetch`` directly without per-batch ``esearch`` round-trips.
        """
        results: list[RawProviderRecord] = []

        if use_history and db == "pubmed":
            init_params = {
                "term": query,
                "retmax": 0,
                "db": db,
                "retmode": "json",
                "usehistory": "y",
                "api_key": self.api_key,
            }
            try:
                response = self._get_with_retry(
                    f"{self.BASE_URL}esearch.fcgi",
                    params=init_params,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    context="ncbi esearch history init",
                )
            except Exception as e:
                logger.info(f"Error: {e}")
                return []

            esearch_res = json.loads(response.content).get("esearchresult", {})
            matches = int(esearch_res.get("count", 0))
            web_env = esearch_res.get("webenv")
            query_key = esearch_res.get("querykey")

            if matches == 0:
                return []

            if web_env and query_key:
                for index_counter in range(0, matches, self.batch_size):
                    try:
                        fetch_params = {
                            "db": "pubmed",
                            "retmode": "xml",
                            "WebEnv": web_env,
                            "query_key": query_key,
                            "retstart": index_counter,
                            "retmax": self.batch_size,
                            "api_key": self.api_key,
                        }
                        fetch_res = self._get_with_retry(
                            f"{self.BASE_URL}efetch.fcgi",
                            params=fetch_params,
                            max_retries=max_retries,
                            retry_delay=retry_delay,
                            context=f"ncbi efetch history batch {index_counter}",
                        )
                        root = ET.fromstring(fetch_res.content)
                        for article_elem in root:
                            xml_str = ET.tostring(article_elem, encoding="unicode")
                            record = RawProviderRecord(
                                provider="ncbi",
                                search_query=query,
                                raw_data={"xml": xml_str},
                            )
                            save_raw_provider_record(record)
                            results.append(record)
                    except Exception as e:
                        logger.info(
                            f"Error processing batch {index_counter}: {e}. Skipping it."
                        )
                        continue
                return results

        params = {
            "term": query,
            "retmax": 1,
            "db": db,
            "retmode": "json",
            "api_key": self.api_key,
        }

        # initial request to retrieve query match count
        try:
            response = self._get_with_retry(
                f"{self.BASE_URL}esearch.fcgi",
                params=params,
                max_retries=max_retries,
                retry_delay=retry_delay,
                context="ncbi esearch count",
            )
        except Exception as e:
            logger.info(f"Error: {e}")
            return []

        matches = int(self.extract_match_count(response.content))

        for index_counter in range(0, matches, self.batch_size):
            try:
                batch_params = {
                    "term": query,
                    "retstart": index_counter,
                    "retmax": self.batch_size,
                    "db": db,
                    "retmode": "json",
                    "api_key": self.api_key,
                }

                try:
                    response = self._get_with_retry(
                        f"{self.BASE_URL}esearch.fcgi",
                        params=batch_params,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        context=f"ncbi esearch batch {index_counter}",
                    )
                except Exception as e:
                    logger.info(f"Error: {e}")
                    continue

                id_list = self.extract_id_list(response.content)
                logger.info(id_list)

                if not id_list:
                    continue

                if db == "pmc":
                    formatted_pmcids = [
                        f"PMC{i}" if not str(i).startswith("PMC") else str(i)
                        for i in id_list
                    ]

                    def _convert_ids():
                        self._wait_for_rate_limit()
                        return convert_pmc_to_pmid_batch(
                            email=os.getenv("NCBI_EMAIL", ""),
                            pmc_ids=formatted_pmcids,
                            api_key=self.api_key,
                        )

                    id_map = retry_on_external_error(
                        _convert_ids,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        context=f"ncbi idconv batch {index_counter}",
                    )

                    pmid_list = [
                        id_map[pid]["pmid"]
                        for pid in formatted_pmcids
                        if id_map[pid]["pmid"]
                    ]

                    if not pmid_list:
                        continue

                    id_list = pmid_list

                fetch_params = {
                    "db": "pubmed",  # we always want to search here from pubmed since pmc returns fulltext and not only abstract
                    "retmode": "xml",
                    "api_key": self.api_key,
                    "id": ",".join(id_list),
                }

                try:
                    fetch_res = self._get_with_retry(
                        f"{self.BASE_URL}efetch.fcgi",
                        params=fetch_params,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        context=f"ncbi efetch batch {index_counter}",
                    )
                except Exception as e:
                    logger.info(f"Error: {e}")
                    continue

                root = ET.fromstring(fetch_res.content)

                for article_elem in root:
                    xml_str = ET.tostring(article_elem, encoding="unicode")
                    record = RawProviderRecord(
                        provider="ncbi", search_query=query, raw_data={"xml": xml_str}
                    )
                    # we need to save the record to the database as a backup for long tasks
                    save_raw_provider_record(record)
                    results.append(record)

            except Exception as e:
                logger.info(
                    f"Error processing batch {index_counter}: {e}. Skipping it."
                )
                continue

        return results

    def search_by_ids(
        self,
        id_list: List[str],
        db: str = "pubmed",
        search_query: str = "id_list_search",
        max_retries: int = 0,
        retry_delay: float = 2.0,
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

            fetch_res = self._get_with_retry(
                f"{self.BASE_URL}efetch.fcgi",
                params=params,
                max_retries=max_retries,
                retry_delay=retry_delay,
                context=f"ncbi search_by_ids batch {i}",
            )

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

    def _parse_xml(self, xml_content: str | ET.Element) -> dict:
        """Parse an article XML element or XML string to extract metadata, abstract, and IDs."""
        if isinstance(xml_content, str):
            elem = ET.fromstring(xml_content)
        else:
            elem = xml_content

        # 1. Title
        title_node = elem.find(".//MedlineCitation/Article/ArticleTitle")
        if title_node is None:
            title_node = elem.find(".//ArticleTitle") or elem.find(
                ".//title-group/article-title"
            )
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        title = " ".join(title.split())

        # 2. Abstract (supports structured abstracts with Label attributes and handles nested tags)
        abstract_nodes = (
            elem.findall(".//AbstractText")
            or elem.findall(".//abstract/p")
            or elem.findall(".//abstract/sec")
        )
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

        pmid = (
            elem.findtext(".//MedlineCitation/PMID")
            or id_map.get("pubmed")
            or id_map.get("pmid")
        )
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

    def parse_raw_search_records(
        self, raw_records: List[RawProviderRecord]
    ) -> List[PaperMetadata]:
        paper_metadatas: List[PaperMetadata] = []

        for raw_record in raw_records:
            if "xml" in raw_record.raw_data:
                parsed = self._parse_xml(raw_record.raw_data["xml"])
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
                provider="ncbi",
            )

            paper_metadatas.append(paper_metadata)

        return paper_metadatas

    def fetch(
        self,
        paper_metadatas: Sequence[PaperMetadata],
        db: str = "pmc",
        batch_size: int = 20,
        max_retries: int = 0,
        retry_delay: float = 2.0,
    ) -> Tuple[List[PaperContent], List[int], List[int]]:
        papers_with_pmcid: list[tuple[PaperMetadata, str]] = []
        for paper in paper_metadatas:
            norm_pmcid = self._normalize_pmcid(paper.pmcid)
            if norm_pmcid:
                papers_with_pmcid.append((paper, norm_pmcid))

        succesful_paper_ids: list[int] = []
        paper_contents: List[PaperContent] = []

        for i in range(0, len(papers_with_pmcid), batch_size):
            batch = papers_with_pmcid[i : i + batch_size]
            valid_pmcids = [pmcid for _, pmcid in batch]

            params = {
                "db": db,
                "retmode": "xml",
                "api_key": self.api_key,
                "id": ",".join(valid_pmcids),
            }

            try:
                response = self._get_with_retry(
                    f"{self.BASE_URL}efetch.fcgi",
                    params=params,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    context=f"ncbi fetch batch {i // batch_size + 1}",
                )
                root = ET.fromstring(response.content)
            except Exception as e:
                logger.error(
                    f"NCBI fetch failed for batch {i // batch_size + 1} "
                    f"(PMCIDs={valid_pmcids}): {e}"
                )
                continue

            logger.info("inside")
            batch_by_pmcid = {pmcid: paper for paper, pmcid in batch}

            for article_elem in root.findall("article"):
                body = article_elem.find("body")
                if body is not None:
                    pmcid_node = article_elem.find(
                        ".//front/article-meta/article-id[@pub-id-type='pmcid']"
                    )
                    pmcid_text = self._normalize_pmcid(
                        pmcid_node.text
                        if pmcid_node is not None and pmcid_node.text
                        else None
                    )

                    paper_metadata = (
                        batch_by_pmcid.get(pmcid_text) if pmcid_text else None
                    )
                    if paper_metadata:
                        clean_doi = doi_to_filename(paper_metadata.doi)
                        xml_str = ET.tostring(article_elem, encoding="unicode")
                        blob_url = upload_xml(
                            xml_content=xml_str,
                            doi=clean_doi,
                            format="xml",
                        )
                        paper_content = PaperContent(
                            paper_id=paper_metadata.id,
                            paper=paper_metadata,
                            blob_url=blob_url,
                            format="xml",
                            source=db,
                        )
                        paper_contents.append(paper_content)
                        succesful_paper_ids.append(paper_metadata.id)

        successful_set = set(succesful_paper_ids)
        failed_paper_ids = [
            paper.id for paper in paper_metadatas if paper.id not in successful_set
        ]

        return paper_contents, succesful_paper_ids, failed_paper_ids


class ScopusProvider(BaseProvider):
    BASE_URL = "https://api.elsevier.com/content/search/scopus"

    def __init__(self, api_key: Optional[str] = None, batch_size: int = 25):
        super().__init__(api_key, batch_size)

    def search_by_query(self, query):

        results: List[RawProviderRecord] = []

        params = {
            "query": query,
            "apiKey": self.api_key,
            "contentType": "json",
            "start": 0,
            "count": self.batch_size,
        }

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        time.sleep(0.2)

        data = response.json()
        count = data.get("search-results", {}).get("opensearch:totalResults", 0)

        for page_number in range(0, int(count), self.batch_size):
            logger.info(
                f"receving the {page_number}th to {page_number + self.batch_size} results"
            )
            params["start"] = page_number
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            time.sleep(0.2)
            data = response.json()

            items = data.get("search-results", {}).get("entry", [])

            for item in items:
                results.append(
                    RawProviderRecord(
                        provider="scopus", search_query=query, raw_data=item
                    )
                )

        return results

    def search_by_ids():
        pass

    def parse_raw_search_records(
        self, raw_records: List[RawProviderRecord]
    ) -> List[PaperMetadata]:
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
                provider="scopus",
            )

            paper_metadatas.append(paper_metadata)

        return paper_metadatas

    def fetch():
        pass


class SemanticScholarProvider:
    """This is not a proper provider and is intended to be used for utility functions like finding abstracts of papers."""

    def __init__(self):
        pass

    def get_abstract_by_doi(self, doi: str) -> str | None:
        BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:"

        url = f"{BASE_URL}{doi}?fields=abstract"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        abstract = data.get("abstract", None)
        return None if abstract == "None" or abstract == "" else abstract


class OpenAlexProvider:
    """This is not a proper provider and is intended to be used for utility functions like finding abstracts of papers."""

    BASE_URL = "https://api.openalex.org/works/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        batch_size: int = 50,
        rate_limit_delay: Optional[float] = None,
    ):
        self.api_key = api_key
        self.batch_size = batch_size
        self.rate_limit_delay = (
            rate_limit_delay
            if rate_limit_delay is not None
            else (0.11 if api_key else 0.25)
        )
        self._last_request_time: Optional[float] = None

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_time is not None and self.rate_limit_delay > 0:
            elapsed = time.monotonic() - self._last_request_time
            remaining = self.rate_limit_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_time = time.monotonic()

    def _get_with_retry(
        self,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = 0,
        retry_delay: float = 2.0,
        context: str = "",
    ) -> requests.Response:
        def _do_get() -> requests.Response:
            self._wait_for_rate_limit()
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response

        return retry_on_external_error(
            _do_get,
            max_retries=max_retries,
            retry_delay=retry_delay,
            context=context,
        )

    def reconstruct_abstract(self, inverted_index: dict[str, list[int]] | None) -> str:
        """
        Abstract is returned as an inverted index from open alex so we need this function
        to build it as a proper text.
        """
        if inverted_index is None:
            return ""
        tokens = [
            (pos, word)
            for word, positions in inverted_index.items()
            for pos in positions
        ]

        tokens.sort(key=lambda item: item[0])
        return " ".join(word for _, word in tokens)

    def get_abstract_by_doi(self, doi: str) -> str | None:
        url = f"{self.BASE_URL}https://doi.org/{doi}?select=abstract_inverted_index"

        params = {"api_key": self.api_key} if self.api_key else None
        response = self._get_with_retry(
            url, params=params, context=f"openalex abstract {doi}"
        )
        data = response.json()
        abstract_list = data.get("abstract_inverted_index", {})

        abstract = self.reconstruct_abstract(abstract_list)
        return None if abstract == "None" or abstract == "" else abstract

    def fetch(
        self,
        paper_metadata: PaperMetadata,
        download_pdf: bool = True,
        max_retries: int = 0,
        retry_delay: float = 2.0,
    ) -> Optional[PaperContent]:
        if not paper_metadata.doi:
            return None

        url = f"{self.BASE_URL}https://doi.org/{paper_metadata.doi}"
        params = {"api_key": self.api_key} if self.api_key else None

        try:
            response = self._get_with_retry(
                url,
                params=params,
                max_retries=max_retries,
                retry_delay=retry_delay,
                context=f"openalex metadata {paper_metadata.doi}",
            )
            data = response.json()
        except Exception as e:
            logger.info(f"Failed to retrieve paper {paper_metadata.doi}: {e}")
            return None

        primary_location = data.get("primary_location") or {}
        pdf_url = primary_location.get("pdf_url", None)

        if pdf_url is None:
            return None

        if download_pdf:
            try:
                response = self._get_with_retry(
                    pdf_url,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    context=f"openalex pdf {paper_metadata.doi}",
                )
                pdf_content = response.content
            except Exception as e:
                logger.info(f"Failed to retrieve pdf {pdf_url}: {e}")
                return None

            clean_doi = doi_to_filename(paper_metadata.doi)

            blob_url = upload_pdf(
                pdf_content=pdf_content,
                doi=clean_doi,
            )
            paper_content = PaperContent(
                pdf_url=pdf_url,
                blob_url=blob_url,
                paper=paper_metadata,
                paper_id=paper_metadata.id,
                format="pdf",
                source="openalex",
            )
            return paper_content

        return None

    def fetch_batch(
        self,
        paper_metadatas: Sequence[PaperMetadata],
        download_pdf: bool = True,
        batch_size: Optional[int] = None,
        max_retries: int = 0,
        retry_delay: float = 2.0,
    ) -> Tuple[List[PaperContent], List[int], List[int]]:
        """Batch-resolve OpenAlex PDF URLs via ``filter=doi:...`` and download PDFs."""
        effective_batch_size = min(batch_size or self.batch_size, 50)
        papers_with_doi: list[tuple[PaperMetadata, str]] = []
        for paper in paper_metadatas:
            norm_doi = normalize_doi(paper.doi)
            if norm_doi:
                papers_with_doi.append((paper, norm_doi))

        pdf_url_by_doi: dict[str, str] = {}
        works_url = self.BASE_URL.rstrip("/")

        for i in range(0, len(papers_with_doi), effective_batch_size):
            batch = papers_with_doi[i : i + effective_batch_size]
            doi_filter = "|".join(f"https://doi.org/{d}" for _, d in batch)
            params: dict = {
                "filter": f"doi:{doi_filter}",
                "select": "doi,primary_location",
                "per-page": len(batch),
            }
            if self.api_key:
                params["api_key"] = self.api_key

            try:
                response = self._get_with_retry(
                    works_url,
                    params=params,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    context=f"openalex batch works {i // effective_batch_size + 1}",
                )
                results = response.json().get("results", [])
                for work in results:
                    work_doi = normalize_doi(work.get("doi"))
                    primary_location = work.get("primary_location") or {}
                    pdf_url = primary_location.get("pdf_url")
                    if work_doi and pdf_url:
                        pdf_url_by_doi[work_doi] = pdf_url
            except Exception as e:
                logger.error(
                    f"OpenAlex batch metadata lookup failed for batch {i // effective_batch_size + 1}: {e}"
                )
                continue

        paper_contents: List[PaperContent] = []
        successful_paper_ids: list[int] = []

        if download_pdf:
            for paper, norm_doi in papers_with_doi:
                pdf_url = pdf_url_by_doi.get(norm_doi)
                if not pdf_url:
                    continue
                try:
                    response = self._get_with_retry(
                        pdf_url,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        context=f"openalex pdf paper {paper.id}",
                    )
                    pdf_content = response.content
                except Exception as e:
                    logger.info(f"Failed to retrieve pdf {pdf_url}: {e}")
                    continue

                clean_doi = doi_to_filename(paper.doi)
                blob_url = upload_pdf(
                    pdf_content=pdf_content,
                    doi=clean_doi,
                )
                paper_content = PaperContent(
                    pdf_url=pdf_url,
                    blob_url=blob_url,
                    paper=paper,
                    paper_id=paper.id,
                    format="pdf",
                    source="openalex",
                )
                paper_contents.append(paper_content)
                successful_paper_ids.append(paper.id)

        successful_set = set(successful_paper_ids)
        failed_paper_ids = [
            paper.id for paper in paper_metadatas if paper.id not in successful_set
        ]
        return paper_contents, successful_paper_ids, failed_paper_ids


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

        logger.info(f"dedup count: {len(process_duplicates_by_field(results, 'provider_id'))}")

        return results
"""
