from peptide_extractor.db.raw_provider_record import RawProviderRecord
from unittest.mock import MagicMock, patch
import pytest

from peptide_extractor.db.paper_metadata import PaperMetadata, PaperStatus
from peptide_extractor.providers import ScopusProvider


@pytest.fixture
def scopus_provider():
    return ScopusProvider(api_key="test_scopus_key", batch_size=10)


def test_scopus_provider_init():
    provider = ScopusProvider(api_key="my_key", batch_size=25)
    assert provider.api_key == "my_key"
    assert provider.batch_size == 25
    assert provider.BASE_URL == "https://api.elsevier.com/content/search/scopus"


def test_scopus_parse_raw_search_records(scopus_provider):
    raw_records = [
        RawProviderRecord(
            provider="scopus",
            search_query="peptide self-assembly",
            raw_data={
                "dc:title": "Self-assembling peptide nanofibers",
                "prism:doi": "10.1016/j.biomaterials.2023.01.001",
                "pubmed-id": "36789012",
            },
        ),
        RawProviderRecord(
            provider="scopus",
            search_query="peptide self-assembly",
            raw_data={
                "dc:title": "Diphenylalanine hydrogels",
                "prism:doi": "10.1039/c3tb21234a",
                "pubmed-id": "36789013",
            },
        ),
    ]

    metadatas = scopus_provider.parse_raw_search_records(raw_records)
    assert len(metadatas) == 2

    m1 = metadatas[0]
    assert isinstance(m1, PaperMetadata)
    assert m1.title == "Self-assembling peptide nanofibers"
    assert m1.doi == "10.1016/j.biomaterials.2023.01.001"
    assert m1.pmid == "36789012"
    assert m1.pmcid is None
    assert m1.status == PaperStatus.NOT_PROCESSED
    assert m1.is_open_access is True
    assert m1.search_query == "peptide self-assembly"
    assert m1.provider == "scopus"

    m2 = metadatas[1]
    assert m2.title == "Diphenylalanine hydrogels"
    assert m2.doi == "10.1039/c3tb21234a"
    assert m2.pmid == "36789013"
    assert m2.provider == "scopus"


def test_scopus_parse_raw_search_records_empty_and_missing_fields(scopus_provider):
    assert scopus_provider.parse_raw_search_records([]) == []

    raw_record = RawProviderRecord(
        provider="scopus",
        search_query="test",
        raw_data={},
    )
    metadatas = scopus_provider.parse_raw_search_records([raw_record])
    assert len(metadatas) == 1
    assert metadatas[0].title == ""
    assert metadatas[0].doi == ""
    assert metadatas[0].pmid == ""


@patch("peptide_extractor.providers.time.sleep")
@patch("peptide_extractor.providers.requests.get")
def test_scopus_search_by_query_success(mock_get, mock_sleep, scopus_provider):
    # Initial count response
    resp_count = MagicMock()
    resp_count.json.return_value = {
        "search-results": {
            "opensearch:totalResults": "2",
            "entry": [],
        }
    }
    resp_count.raise_for_status = MagicMock()

    # Paginated results response
    resp_page = MagicMock()
    resp_page.json.return_value = {
        "search-results": {
            "entry": [
                {
                    "dc:title": "Paper 1",
                    "prism:doi": "10.1016/j.bio.2023.01",
                    "pubmed-id": "11111111",
                },
                {
                    "dc:title": "Paper 2",
                    "prism:doi": "10.1016/j.bio.2023.02",
                    "pubmed-id": "22222222",
                },
            ]
        }
    }
    resp_page.raise_for_status = MagicMock()

    mock_get.side_effect = [resp_count, resp_page]

    results = scopus_provider.search_by_query("peptide nanotubes")

    assert len(results) == 2
    assert all(isinstance(r, RawProviderRecord) for r in results)
    assert results[0].provider == "scopus"
    assert results[0].search_query == "peptide nanotubes"
    assert results[0].raw_data["dc:title"] == "Paper 1"
    assert results[1].raw_data["dc:title"] == "Paper 2"

    assert mock_get.call_count == 2
    first_call_params = mock_get.call_args_list[0].kwargs["params"]
    assert first_call_params["query"] == "peptide nanotubes"
    assert first_call_params["apiKey"] == "test_scopus_key"
    assert first_call_params["count"] == 10
