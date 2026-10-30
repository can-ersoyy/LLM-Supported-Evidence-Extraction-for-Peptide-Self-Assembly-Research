from peptide_extractor.db.raw_provider_record import RawProviderRecord
import json
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest
import requests

from peptide_extractor.db.paper_metadata import DatasetSubset, PaperMetadata, PaperStatus
from peptide_extractor.providers import NCBIProvider



class TestNCBIProviderHelpers:
    """Tests for helper extraction methods in NCBIProvider."""

    def test_extract_match_count_bytes(self, ncbi_provider):
        payload = b'{"esearchresult": {"count": "42", "retmax": "1"}}'
        assert ncbi_provider.extract_match_count(payload) == "42"

    def test_extract_match_count_str(self, ncbi_provider):
        payload = '{"esearchresult": {"count": "100", "retmax": "1"}}'
        assert ncbi_provider.extract_match_count(payload) == "100"

    def test_extract_id_list_valid(self, ncbi_provider):
        payload = b'{"esearchresult": {"idlist": ["12345", "67890"]}}'
        assert ncbi_provider.extract_id_list(payload) == ["12345", "67890"]

    def test_extract_id_list_empty(self, ncbi_provider):
        payload = b'{"esearchresult": {"idlist": []}}'
        assert ncbi_provider.extract_id_list(payload) == []


class TestNCBIProviderXMLParsing:
    """Tests for parse_xml method in NCBIProvider."""

    def test_parse_xml_from_string(self, ncbi_provider, sample_pubmed_xml):
        root = ET.fromstring(sample_pubmed_xml)
        article_elem = root.findall(".//PubmedArticle")[0]
        xml_str = ET.tostring(article_elem, encoding="unicode")

        parsed = ncbi_provider._parse_xml(xml_str)
        assert parsed["pmid"] == "31000001"
        assert parsed["doi"] == "10.1000/182"
        assert parsed["pmcid"] == "PMC70001"
        assert "Self-Assembly of Tetrapeptides: A Structural Analysis" in parsed["title"]
        assert "BACKGROUND: Short peptides are capable of forming diverse nanostructures." in parsed["abstract"]
        assert "CONCLUSIONS: Valuable for biomaterials applications." in parsed["abstract"]

    def test_parse_xml_from_element(self, ncbi_provider, sample_pubmed_xml):
        root = ET.fromstring(sample_pubmed_xml)
        article_elem = root.findall(".//PubmedArticle")[1]

        parsed = ncbi_provider._parse_xml(article_elem)
        assert parsed["pmid"] == "31000002"
        assert parsed["doi"] == "10.1000/183"
        assert parsed["pmcid"] == "PMC70002"
        assert parsed["title"] == "Hydrogel Formation in Diphenylalanine Derivatives"
        assert parsed["abstract"] == "Diphenylalanine motifs readily form hydrogels under physiological conditions."

    def test_parse_xml_pmc_format_fallback(self, ncbi_provider, sample_pmc_xml):
        root = ET.fromstring(sample_pmc_xml)
        article_elem = root.find(".//article")

        parsed = ncbi_provider._parse_xml(article_elem)
        assert parsed["pmid"] == "32000003"
        assert parsed["doi"] == "10.1000/184"
        assert parsed["pmcid"] == "PMC70003"
        assert parsed["title"] == "Designed self-assembling peptide hydrogels"
        assert parsed["abstract"] == "This is a paragraph-based abstract from PMC format."

    def test_parse_xml_pmcid_normalization_variants(self, ncbi_provider):
        xml_template = """
        <PubmedArticle>
          <MedlineCitation><PMID>1</PMID><Article><ArticleTitle>Test</ArticleTitle></Article></MedlineCitation>
          <PubmedData><ArticleIdList><ArticleId IdType="pmc">{pmc_val}</ArticleId></ArticleIdList></PubmedData>
        </PubmedArticle>
        """

        parsed_1 = ncbi_provider._parse_xml(xml_template.format(pmc_val="pmc-id: 12345;"))
        assert parsed_1["pmcid"] == "PMC12345"

        parsed_2 = ncbi_provider._parse_xml(xml_template.format(pmc_val="PMC99999"))
        assert parsed_2["pmcid"] == "PMC99999"

        parsed_3 = ncbi_provider._parse_xml(xml_template.format(pmc_val="88888"))
        assert parsed_3["pmcid"] == "PMC88888"

    def test_parse_xml_missing_fields(self, ncbi_provider, sample_minimal_article_xml):
        root = ET.fromstring(sample_minimal_article_xml)
        article_elem = root.findall(".//PubmedArticle")[0]

        parsed = ncbi_provider._parse_xml(article_elem)
        assert parsed["pmid"] == "33000004"
        assert parsed["doi"] is None
        assert parsed["pmcid"] is None
        assert parsed["title"] == "Minimal Article Without Abstract"
        assert parsed["abstract"] == ""

    def test_parse_xml_nested_formatting_tags(self, ncbi_provider):
        xml = """
        <PubmedArticle>
          <MedlineCitation>
            <PMID>55555</PMID>
            <Article>
              <ArticleTitle>Effect of <b>pH</b> and <i>temperature</i> on assembly</ArticleTitle>
              <Abstract>
                <AbstractText>The <sub>alpha</sub>-helix structure with <sup>14</sup>C marker.</AbstractText>
              </Abstract>
            </Article>
          </MedlineCitation>
        </PubmedArticle>
        """
        parsed = ncbi_provider._parse_xml(xml)
        assert parsed["title"] == "Effect of pH and temperature on assembly"
        assert parsed["abstract"] == "The alpha-helix structure with 14C marker."


class TestNCBIProviderParseRawSearchRecords:
    """Tests for parse_raw_search_records converting RawProviderRecord to PaperMetadata."""

    def test_parse_raw_search_records_xml(self, ncbi_provider, sample_pubmed_xml):
        root = ET.fromstring(sample_pubmed_xml)
        records = []
        for elem in root.findall(".//PubmedArticle"):
            xml_str = ET.tostring(elem, encoding="unicode")
            records.append(
                RawProviderRecord(
                    provider="ncbi",
                    search_query="peptide self-assembly",
                    raw_data={"xml": xml_str},
                )
            )

        metadatas = ncbi_provider.parse_raw_search_records(records)
        assert len(metadatas) == 2

        meta1 = metadatas[0]
        assert isinstance(meta1, PaperMetadata)
        assert meta1.pmid == "31000001"
        assert meta1.doi == "10.1000/182"
        assert meta1.pmcid == "PMC70001"
        assert "Self-Assembly of Tetrapeptides" in meta1.title
        assert meta1.status == PaperStatus.NOT_PROCESSED
        assert meta1.dataset_subset == DatasetSubset.SCRAPED
        assert meta1.is_open_access is True
        assert meta1.search_query == "peptide self-assembly"
        assert meta1.provider == "ncbi"

        meta2 = metadatas[1]
        assert meta2.pmid == "31000002"
        assert meta2.is_open_access is True
        assert meta2.provider == "ncbi"

    def test_parse_raw_search_records_json_fallback(self, ncbi_provider):
        record = RawProviderRecord(
            provider="ncbi",
            search_query="diphenylalanine",
            raw_data={
                "title": "Diphenylalanine Nanotubes",
                "abstract": "Nanotube self-assembly details.",
                "articleids": [
                    {"idtype": "pubmed", "value": "20000001"},
                    {"idtype": "doi", "value": "10.1000/999"},
                    {"idtype": "pmcid", "value": "pmc-id: 99999;"},
                ],
            },
        )

        metadatas = ncbi_provider.parse_raw_search_records([record])
        assert len(metadatas) == 1
        meta = metadatas[0]
        assert meta.pmid == "20000001"
        assert meta.doi == "10.1000/999"
        assert meta.pmcid == "PMC99999"
        assert meta.title == "Diphenylalanine Nanotubes"
        assert meta.abstract == "Nanotube self-assembly details."
        assert meta.is_open_access is True

    @pytest.mark.parametrize(
        "pmcid,search_query,expected_open_access",
        [
            ("PMC12345", "general query", True),
            (None, 'peptide AND "free full text"[filter]', True),
            (None, 'peptide AND "open access"[filter]', True),
            (None, "peptide self-assembly closed access", False),
        ],
    )
    def test_open_access_determination(
        self, ncbi_provider, pmcid, search_query, expected_open_access
    ):
        pmc_elem = f'<ArticleId IdType="pmc">{pmcid}</ArticleId>' if pmcid else ""
        xml = f"""
        <PubmedArticle>
          <MedlineCitation><PMID>123</PMID><Article><ArticleTitle>Test</ArticleTitle></Article></MedlineCitation>
          <PubmedData><ArticleIdList>{pmc_elem}</ArticleIdList></PubmedData>
        </PubmedArticle>
        """
        record = RawProviderRecord(
            provider="ncbi",
            search_query=search_query,
            raw_data={"xml": xml},
        )
        metadatas = ncbi_provider.parse_raw_search_records([record])
        assert len(metadatas) == 1
        assert metadatas[0].is_open_access is expected_open_access

    def test_parse_raw_search_records_empty_list(self, ncbi_provider):
        assert ncbi_provider.parse_raw_search_records([]) == []


class TestNCBIProviderSearchByQuery:
    """Tests for search_by_query method in NCBIProvider."""

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_query_success(
        self, mock_get, mock_sleep, ncbi_provider, sample_pubmed_xml
    ):
        # 1. Initial count request
        resp_count = MagicMock()
        resp_count.content = b'{"esearchresult": {"count": "2"}}'
        resp_count.raise_for_status = MagicMock()

        # 2. Paginated ID search request
        resp_ids = MagicMock()
        resp_ids.content = b'{"esearchresult": {"idlist": ["31000001", "31000002"]}}'
        resp_ids.raise_for_status = MagicMock()

        # 3. efetch XML request
        resp_fetch = MagicMock()
        resp_fetch.content = sample_pubmed_xml.encode("utf-8")
        resp_fetch.raise_for_status = MagicMock()

        mock_get.side_effect = [resp_count, resp_ids, resp_fetch]

        results = ncbi_provider.search_by_query("peptide assembly")

        assert len(results) == 2
        assert all(isinstance(r, RawProviderRecord) for r in results)
        assert results[0].provider == "ncbi"
        assert results[0].search_query == "peptide assembly"
        assert "xml" in results[0].raw_data

        # Verify calls
        assert mock_get.call_count == 3
        # First call checks count
        first_call_params = mock_get.call_args_list[0].kwargs["params"]
        assert first_call_params["term"] == "peptide assembly"
        assert first_call_params["retmax"] == 1
        assert first_call_params["api_key"] == "test_api_key"

        # Second call fetches IDs
        second_call_params = mock_get.call_args_list[1].kwargs["params"]
        assert second_call_params["retstart"] == 0
        assert second_call_params["retmax"] == 2

        # Third call fetches XML
        third_call_params = mock_get.call_args_list[2].kwargs["params"]
        assert third_call_params["id"] == "31000001,31000002"
        assert third_call_params["retmode"] == "xml"

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_query_zero_matches(self, mock_get, mock_sleep, ncbi_provider):
        resp_count = MagicMock()
        resp_count.content = b'{"esearchresult": {"count": "0"}}'
        resp_count.raise_for_status = MagicMock()

        mock_get.return_value = resp_count

        results = ncbi_provider.search_by_query("nonexistent query")
        assert results == []
        assert mock_get.call_count == 1

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_query_pagination_multiple_batches(
        self, mock_get, mock_sleep, sample_pubmed_xml
    ):
        provider = NCBIProvider(api_key="key", batch_size=2)

        # Total 4 matches -> 2 batches
        resp_count = MagicMock()
        resp_count.content = b'{"esearchresult": {"count": "4"}}'
        resp_count.raise_for_status = MagicMock()

        resp_ids_1 = MagicMock()
        resp_ids_1.content = b'{"esearchresult": {"idlist": ["101", "102"]}}'
        resp_ids_1.raise_for_status = MagicMock()

        resp_fetch_1 = MagicMock()
        resp_fetch_1.content = sample_pubmed_xml.encode("utf-8")
        resp_fetch_1.raise_for_status = MagicMock()

        resp_ids_2 = MagicMock()
        resp_ids_2.content = b'{"esearchresult": {"idlist": ["103", "104"]}}'
        resp_ids_2.raise_for_status = MagicMock()

        resp_fetch_2 = MagicMock()
        resp_fetch_2.content = sample_pubmed_xml.encode("utf-8")
        resp_fetch_2.raise_for_status = MagicMock()

        mock_get.side_effect = [
            resp_count,
            resp_ids_1,
            resp_fetch_1,
            resp_ids_2,
            resp_fetch_2,
        ]

        results = provider.search_by_query("test query")
        assert len(results) == 4
        assert mock_get.call_count == 5

        # Check retstart in batch calls
        assert mock_get.call_args_list[1].kwargs["params"]["retstart"] == 0
        assert mock_get.call_args_list[3].kwargs["params"]["retstart"] == 2

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_query_empty_id_list_skipped(
        self, mock_get, mock_sleep, ncbi_provider
    ):
        resp_count = MagicMock()
        resp_count.content = b'{"esearchresult": {"count": "2"}}'
        resp_count.raise_for_status = MagicMock()

        # Batch returns empty id list
        resp_ids = MagicMock()
        resp_ids.content = b'{"esearchresult": {"idlist": []}}'
        resp_ids.raise_for_status = MagicMock()

        mock_get.side_effect = [resp_count, resp_ids]

        results = ncbi_provider.search_by_query("test query")
        assert results == []
        # Count check + ID search, but NO efetch call
        assert mock_get.call_count == 2

class TestNCBIProviderSearchByIds:
    """Tests for search_by_ids method in NCBIProvider."""

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_ids_success(
        self, mock_get, mock_sleep, ncbi_provider, sample_pubmed_xml
    ):
        mock_fetch = MagicMock()
        mock_fetch.content = sample_pubmed_xml.encode("utf-8")
        mock_fetch.raise_for_status = MagicMock()
        mock_get.return_value = mock_fetch

        results = ncbi_provider.search_by_ids(
            ["31000001", "31000002"], db="pubmed", search_query="my_ids"
        )

        assert len(results) == 2
        assert results[0].provider == "pubmed"
        assert results[0].search_query == "my_ids"
        assert "xml" in results[0].raw_data

        params = mock_get.call_args.kwargs["params"]
        assert params["id"] == "31000001,31000002"
        assert params["retmode"] == "xml"
        assert params["api_key"] == "test_api_key"

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_ids_multiple_batches(
        self, mock_get, mock_sleep, sample_pubmed_xml
    ):
        provider = NCBIProvider(api_key="key", batch_size=2)

        resp1 = MagicMock()
        resp1.content = sample_pubmed_xml.encode("utf-8")
        resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.content = sample_pubmed_xml.encode("utf-8")
        resp2.raise_for_status = MagicMock()

        mock_get.side_effect = [resp1, resp2]

        results = provider.search_by_ids(["1", "2", "3"])
        # 2 in first batch, 2 in second batch (sample xml has 2 articles each)
        assert len(results) == 4
        assert mock_get.call_count == 2
        assert mock_get.call_args_list[0].kwargs["params"]["id"] == "1,2"
        assert mock_get.call_args_list[1].kwargs["params"]["id"] == "3"

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_ids_filters_whitespace_and_empty(
        self, mock_get, mock_sleep, sample_pubmed_xml
    ):
        provider = NCBIProvider(api_key="key", batch_size=10)
        mock_fetch = MagicMock()
        mock_fetch.content = sample_pubmed_xml.encode("utf-8")
        mock_fetch.raise_for_status = MagicMock()
        mock_get.return_value = mock_fetch

        results = provider.search_by_ids(["  12345  ", "", "  ", "67890"])
        assert mock_get.call_count == 1
        assert mock_get.call_args.kwargs["params"]["id"] == "12345,67890"

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_ids_skips_empty_batch(
        self, mock_get, mock_sleep, sample_pubmed_xml
    ):
        # batch_size 2: batch 1 is ["", "  "] (skipped), batch 2 is ["12345"] (processed)
        provider = NCBIProvider(api_key="key", batch_size=2)
        mock_fetch = MagicMock()
        mock_fetch.content = sample_pubmed_xml.encode("utf-8")
        mock_fetch.raise_for_status = MagicMock()
        mock_get.return_value = mock_fetch

        results = provider.search_by_ids(["", "  ", "12345"])
        assert mock_get.call_count == 1
        assert mock_get.call_args.kwargs["params"]["id"] == "12345"

    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_ids_empty_list(self, mock_get, ncbi_provider):
        results = ncbi_provider.search_by_ids([])
        assert results == []
        mock_get.assert_not_called()

    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_ids_all_whitespace_list(self, mock_get, ncbi_provider):
        results = ncbi_provider.search_by_ids(["", "   "])
        assert results == []
        mock_get.assert_not_called()

    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_ids_http_error(self, mock_get, ncbi_provider):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_resp

        with pytest.raises(requests.HTTPError):
            ncbi_provider.search_by_ids(["12345"])


class TestNCBIProviderFetch:
    """Tests for fetch method in NCBIProvider using Azure Blob Storage."""

    @patch("peptide_extractor.providers.upload_xml")
    @patch("peptide_extractor.providers.requests.get")
    def test_fetch_success(self, mock_get, mock_upload, ncbi_provider):
        pmc_xml_with_body = """<?xml version="1.0" ?>
        <pmc-articleset>
          <article>
            <front>
              <article-meta>
                <article-id pub-id-type="pmcid">PMC70003</article-id>
              </article-meta>
            </front>
            <body>
              <p>Full text self-assembly results.</p>
            </body>
          </article>
        </pmc-articleset>
        """
        mock_resp = MagicMock()
        mock_resp.content = pmc_xml_with_body.encode("utf-8")
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mock_upload.return_value = "https://myaccount.blob.core.windows.net/paper-xmls/xml/10_1021_acs_nano_1c01792.xml.gz"

        paper = PaperMetadata(
            id=42,
            pmcid="PMC70003",
            doi="10.1021/acsnano.1c01792",
            title="Designed self-assembling peptide hydrogels",
            is_open_access=True,
            status=PaperStatus.USABLE,
            dataset_subset=DatasetSubset.SCRAPED,
        )

        contents, successful_ids, failed_ids = ncbi_provider.fetch([paper])

        assert len(contents) == 1
        assert contents[0].paper_id == 42
        assert contents[0].blob_url == "https://myaccount.blob.core.windows.net/paper-xmls/xml/10_1021_acs_nano_1c01792.xml.gz"
        assert contents[0].format == "xml"
        assert successful_ids == [42]
        assert failed_ids == []

        mock_upload.assert_called_once()
        assert mock_upload.call_args.kwargs["doi"] == "10_1021_acsnano_1c01792"
        assert mock_upload.call_args.kwargs["format"] == "xml"

    @patch("peptide_extractor.providers.requests.get")
    def test_fetch_no_pmcid_fails(self, mock_get, ncbi_provider):
        paper_without_pmcid = PaperMetadata(
            id=99,
            pmcid=None,
            title="No PMC ID Paper",
            is_open_access=False,
            status=PaperStatus.USABLE,
            dataset_subset=DatasetSubset.SCRAPED,
        )

        contents, successful_ids, failed_ids = ncbi_provider.fetch([paper_without_pmcid])

        assert contents == []
        assert successful_ids == []
        assert failed_ids == [99]
        mock_get.assert_not_called()
