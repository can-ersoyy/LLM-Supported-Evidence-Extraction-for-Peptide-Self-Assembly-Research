import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest
import requests
from sqlmodel import Session, SQLModel, create_engine

from peptide_extractor import pipeline
from peptide_extractor.db.paper_content import PaperContent
from peptide_extractor.db.paper_metadata import (
    DatasetSubset,
    PaperMetadata,
    PaperStatus,
)
from peptide_extractor.providers import NCBIProvider, OpenAlexProvider


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


class TestNCBIRateLimitAndRetry:
    def test_rate_limit_defaults(self):
        with_key = NCBIProvider(api_key="my_key")
        without_key = NCBIProvider(api_key=None)
        assert with_key.rate_limit_delay == pytest.approx(0.11)
        assert without_key.rate_limit_delay == pytest.approx(0.35)

    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.time.monotonic")
    def test_wait_for_rate_limit_sleeps_only_remaining_interval(
        self, mock_monotonic, mock_sleep
    ):
        provider = NCBIProvider(api_key="key", rate_limit_delay=0.2)
        mock_monotonic.side_effect = [10.0, 10.05, 10.2, 10.5, 10.5]

        # First request: no previous timestamp -> no sleep
        provider._wait_for_rate_limit()
        mock_sleep.assert_not_called()

        # Second request 0.05s later -> sleeps remaining 0.15s
        provider._wait_for_rate_limit()
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] == pytest.approx(0.15)

        # Third request 0.30s later -> no additional sleep needed
        mock_sleep.reset_mock()
        provider._wait_for_rate_limit()
        mock_sleep.assert_not_called()

    @patch("peptide_extractor.utils.time.sleep")
    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_ids_retries_transient_429(
        self, mock_get, mock_prov_sleep, mock_util_sleep, sample_pubmed_xml
    ):
        provider = NCBIProvider(api_key="key", batch_size=10, rate_limit_delay=0)

        resp_429 = requests.Response()
        resp_429.status_code = 429
        err_429 = requests.HTTPError("429 Too Many Requests", response=resp_429)

        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = err_429

        mock_ok = MagicMock()
        mock_ok.content = sample_pubmed_xml.encode("utf-8")
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_fail, mock_ok]

        records = provider.search_by_ids(
            ["31000001", "31000002"], max_retries=2, retry_delay=0.01
        )
        assert len(records) == 2
        assert mock_get.call_count == 2


class TestNCBISearchWithWebHistory:
    @patch("peptide_extractor.providers.save_raw_provider_record")
    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_search_by_query_use_history_skips_paginated_esearch(
        self, mock_get, mock_sleep, mock_save, sample_pubmed_xml
    ):
        provider = NCBIProvider(api_key="key", batch_size=2, rate_limit_delay=0)

        resp_init = MagicMock()
        resp_init.content = (
            b'{"esearchresult": {"count": "4", "webenv": "NCID_1", "querykey": "1"}}'
        )
        resp_init.raise_for_status = MagicMock()

        resp_fetch_1 = MagicMock()
        resp_fetch_1.content = sample_pubmed_xml.encode("utf-8")
        resp_fetch_1.raise_for_status = MagicMock()

        resp_fetch_2 = MagicMock()
        resp_fetch_2.content = sample_pubmed_xml.encode("utf-8")
        resp_fetch_2.raise_for_status = MagicMock()

        mock_get.side_effect = [resp_init, resp_fetch_1, resp_fetch_2]

        records = provider.search_by_query("peptide", db="pubmed", use_history=True)
        # 1 esearch + 2 efetch = 3 HTTP calls instead of 5
        assert mock_get.call_count == 3
        assert len(records) == 4
        assert mock_get.call_args_list[0].kwargs["params"]["usehistory"] == "y"
        assert mock_get.call_args_list[1].kwargs["params"]["WebEnv"] == "NCID_1"
        assert mock_get.call_args_list[1].kwargs["params"]["retstart"] == 0
        assert mock_get.call_args_list[2].kwargs["params"]["retstart"] == 2


class TestNCBIBulkFetch:
    @patch("peptide_extractor.providers.upload_xml")
    @patch("peptide_extractor.utils.time.sleep")
    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_fetch_filters_missing_pmcid_and_isolates_batch_failures(
        self, mock_get, mock_prov_sleep, mock_util_sleep, mock_upload
    ):
        provider = NCBIProvider(api_key="key", rate_limit_delay=0)

        papers = [
            PaperMetadata(id=1, pmcid=None, doi="10.1000/no-pmc", title="No PMC"),
            PaperMetadata(id=2, pmcid="PMC101", doi="10.1000/101", title="P101"),
            PaperMetadata(id=3, pmcid="102", doi="10.1000/102", title="P102"),
            PaperMetadata(id=4, pmcid="PMC103", doi="10.1000/103", title="P103"),
        ]

        # Batch 1 (PMC101, PMC102) fails with 503 even after 1 retry; Batch 2 (PMC103) succeeds
        resp_503 = requests.Response()
        resp_503.status_code = 503
        err_503 = requests.HTTPError("503 Unavailable", response=resp_503)
        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = err_503

        pmc_xml_103 = """<?xml version="1.0" ?>
        <pmc-articleset>
          <article>
            <front><article-meta><article-id pub-id-type="pmcid">PMC103</article-id></article-meta></front>
            <body><p>Self-assembly body.</p></body>
          </article>
        </pmc-articleset>"""
        mock_ok = MagicMock()
        mock_ok.content = pmc_xml_103.encode("utf-8")
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_fail, mock_fail, mock_ok]
        mock_upload.return_value = "https://blob/10_1000_103.xml.gz"

        contents, ok_ids, fail_ids = provider.fetch(
            papers, batch_size=2, max_retries=1, retry_delay=0.01
        )

        # Paper 1 was filtered out before batching, so Batch 1 = [101, 102], Batch 2 = [103]
        assert mock_get.call_count == 3  # 2 attempts for batch 1, 1 for batch 2
        assert ok_ids == [4]
        assert fail_ids == [1, 2, 3]
        assert len(contents) == 1
        assert contents[0].paper_id == 4


class TestOpenAlexBulkFetch:
    @patch("peptide_extractor.providers.upload_pdf")
    @patch("peptide_extractor.utils.time.sleep")
    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_fetch_batch_resolves_dois_in_one_call_and_downloads_pdfs(
        self, mock_get, mock_prov_sleep, mock_util_sleep, mock_upload
    ):
        provider = OpenAlexProvider(api_key="oa_key", rate_limit_delay=0)
        papers = [
            PaperMetadata(id=10, doi="10.1000/a", title="A"),
            PaperMetadata(id=11, doi="https://doi.org/10.1000/B", title="B"),
            PaperMetadata(id=12, doi=None, title="No DOI"),
        ]

        works_resp = MagicMock()
        works_resp.raise_for_status = MagicMock()
        works_resp.json.return_value = {
            "results": [
                {
                    "doi": "https://doi.org/10.1000/a",
                    "primary_location": {"pdf_url": "https://pdf.example/a.pdf"},
                },
                {
                    "doi": "https://doi.org/10.1000/b",
                    "primary_location": {"pdf_url": None},
                },
            ]
        }

        pdf_resp = MagicMock()
        pdf_resp.raise_for_status = MagicMock()
        pdf_resp.content = b"%PDF-1.4..."

        mock_get.side_effect = [works_resp, pdf_resp]
        mock_upload.return_value = "https://blob/10_1000_a.pdf"

        contents, ok_ids, fail_ids = provider.fetch_batch(
            papers, batch_size=50, max_retries=1, retry_delay=0.01
        )

        # 1 batch works query + 1 PDF download = 2 HTTP calls
        assert mock_get.call_count == 2
        assert "filter" in mock_get.call_args_list[0].kwargs["params"]
        assert ok_ids == [10]
        assert fail_ids == [11, 12]
        assert len(contents) == 1
        assert contents[0].paper_id == 10
        assert contents[0].format == "pdf"


class TestPipelineBulkFetchIntegration:
    @patch("peptide_extractor.providers.upload_pdf")
    @patch("peptide_extractor.providers.upload_xml")
    @patch("peptide_extractor.providers.time.sleep")
    @patch("peptide_extractor.providers.requests.get")
    def test_fetch_paper_fulltexts_batches_ncbi_and_falls_back_to_openalex(
        self, mock_get, mock_sleep, mock_upload_xml, mock_upload_pdf, memory_db
    ):
        with Session(memory_db) as s:
            s.add(
                PaperMetadata(
                    id=1,
                    pmcid="PMC201",
                    doi="10.1000/p1",
                    title="PMC Paper",
                    is_open_access=True,
                    status=PaperStatus.USABLE,
                    dataset_subset=DatasetSubset.SCRAPED,
                )
            )
            s.add(
                PaperMetadata(
                    id=2,
                    pmcid="PMC202",
                    doi="10.1000/p2",
                    title="PMC Missing Body -> OpenAlex",
                    is_open_access=True,
                    status=PaperStatus.USABLE,
                    dataset_subset=DatasetSubset.SCRAPED,
                )
            )
            s.add(
                PaperMetadata(
                    id=3,
                    pmcid=None,
                    doi="10.1000/p3",
                    title="DOI Only -> OpenAlex",
                    is_open_access=True,
                    status=PaperStatus.USABLE,
                    dataset_subset=DatasetSubset.SCRAPED,
                )
            )
            s.commit()

        ncbi_xml = """<?xml version="1.0" ?>
        <pmc-articleset>
          <article>
            <front><article-meta><article-id pub-id-type="pmcid">PMC201</article-id></article-meta></front>
            <body><p>Full text 201.</p></body>
          </article>
          <article>
            <front><article-meta><article-id pub-id-type="pmcid">PMC202</article-id></article-meta></front>
          </article>
        </pmc-articleset>"""
        ncbi_resp = MagicMock()
        ncbi_resp.content = ncbi_xml.encode("utf-8")
        ncbi_resp.raise_for_status = MagicMock()

        oa_works_resp = MagicMock()
        oa_works_resp.raise_for_status = MagicMock()
        oa_works_resp.json.return_value = {
            "results": [
                {
                    "doi": "https://doi.org/10.1000/p2",
                    "primary_location": {"pdf_url": "https://pdf.example/p2.pdf"},
                }
            ]
        }

        oa_pdf_resp = MagicMock()
        oa_pdf_resp.raise_for_status = MagicMock()
        oa_pdf_resp.content = b"%PDF-1.4"

        mock_get.side_effect = [ncbi_resp, oa_works_resp, oa_pdf_resp]
        mock_upload_xml.return_value = "https://blob/p1.xml.gz"
        mock_upload_pdf.return_value = "https://blob/p2.pdf"

        ncbi = NCBIProvider(api_key="k", rate_limit_delay=0)
        oa = OpenAlexProvider(api_key="k", rate_limit_delay=0)

        with Session(memory_db) as s:
            success, fail, run_id = pipeline.fetch_paper_fulltexts(
                session=s,
                ncbi_provider=ncbi,
                open_alex_provider=oa,
                ncbi_batch_size=10,
                openalex_batch_size=50,
                run_identifier=999111,
                log_to_file=False,
            )

            assert success == 2
            assert fail == 1
            assert run_id == 999111
            # 1 NCBI efetch batch + 1 OpenAlex works batch + 1 PDF download = 3 calls total
            assert mock_get.call_count == 3

            saved_contents = s.query(PaperContent).all()
            assert len(saved_contents) == 2
            assert {c.paper_id for c in saved_contents} == {1, 2}
            assert all(c.run_identifier == 999111 for c in saved_contents)
