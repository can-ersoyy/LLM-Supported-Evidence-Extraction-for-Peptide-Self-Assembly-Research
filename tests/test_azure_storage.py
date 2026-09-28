"""Unit tests for Azure Blob Storage integration (azure_storage.py)."""

import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ResourceExistsError

from peptide_extractor.azure_storage import (
    build_blob_name,
    download_file,
    download_xml,
    extract_blob_name,
    get_blob_service_client,
    upload_file,
    upload_xml,
)


class TestAzureStorageHelpers:
    """Test helper functions in azure_storage."""

    def test_extract_blob_name_from_url(self):
        url = "https://myaccount.blob.core.windows.net/paper-xmls/xml/ncbi/PMC1234567.xml.gz"
        assert extract_blob_name(url) == "xml/ncbi/PMC1234567.xml.gz"

    def test_extract_blob_name_from_plain_name(self):
        name = "xml/ncbi/PMC1234567.xml.gz"
        assert extract_blob_name(name) == "xml/ncbi/PMC1234567.xml.gz"

    def test_extract_blob_name_with_leading_slash(self):
        name = "/papers/xml/ncbi/PMC1234567.xml.gz"
        assert extract_blob_name(name) == "xml/ncbi/PMC1234567.xml.gz"

    def test_extract_blob_name_with_custom_container(self):
        name = "/custom-container/xml/ncbi/PMC1234567.xml.gz"
        assert extract_blob_name(name, container_name="custom-container") == "xml/ncbi/PMC1234567.xml.gz"

    def test_build_blob_name_default_ncbi_xml(self):
        assert build_blob_name("PMC12345") == "xml/ncbi/PMC12345.xml.gz"

    def test_build_blob_name_europepmc(self):
        assert build_blob_name("PMC67890", provider="europepmc") == "xml/europepmc/PMC67890.xml.gz"

    def test_build_blob_name_pdf_uncompressed(self):
        assert (
            build_blob_name("article_123", provider="scopus", format="pdf", compress=False)
            == "pdf/scopus/article_123.pdf"
        )

    def test_build_blob_name_already_structured(self):
        assert build_blob_name("xml/ncbi/PMC12345.xml.gz") == "xml/ncbi/PMC12345.xml.gz"

    def test_get_blob_service_client_missing_env(self, monkeypatch):
        monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
        with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
            get_blob_service_client()

    def test_get_blob_service_client_with_conn_str(self):
        fake_conn = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fakekey==;EndpointSuffix=core.windows.net"
        with patch("peptide_extractor.azure_storage.BlobServiceClient.from_connection_string") as mock_from_conn:
            mock_client = MagicMock()
            mock_from_conn.return_value = mock_client
            client = get_blob_service_client(connection_string=fake_conn)
            mock_from_conn.assert_called_once_with(fake_conn)
            assert client == mock_client


class TestUploadXML:
    """Test upload_xml and upload_file functions with mocked Azure clients."""

    @pytest.fixture
    def mock_azure_clients(self):
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()

        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob
        mock_blob.url = "https://myaccount.blob.core.windows.net/paper-xmls/xml/ncbi/PMC12345.xml.gz"

        return mock_service, mock_container, mock_blob

    def test_upload_xml_compressed_ncbi_default(self, mock_azure_clients):
        mock_service, mock_container, mock_blob = mock_azure_clients
        sample_xml = "<article><title>Test Article</title></article>"

        url = upload_xml(
            xml_content=sample_xml,
            pmcid="PMC12345",
            container_name="paper-xmls",
            client=mock_service,
            compress=True,
        )

        assert url == "https://myaccount.blob.core.windows.net/paper-xmls/xml/ncbi/PMC12345.xml.gz"
        mock_container.create_container.assert_called_once()
        mock_container.get_blob_client.assert_called_once_with("xml/ncbi/PMC12345.xml.gz")

        mock_blob.upload_blob.assert_called_once()
        uploaded_bytes, kwargs = mock_blob.upload_blob.call_args
        data = uploaded_bytes[0]
        # Verify uploaded data is valid gzipped XML
        decompressed = gzip.decompress(data).decode("utf-8")
        assert decompressed == sample_xml
        assert kwargs["content_settings"].content_encoding == "gzip"
        assert kwargs["content_settings"].content_type == "application/xml"

    def test_upload_xml_europepmc_provider(self, mock_azure_clients):
        mock_service, mock_container, mock_blob = mock_azure_clients
        mock_blob.url = "https://myaccount.blob.core.windows.net/paper-xmls/xml/europepmc/PMC67890.xml.gz"
        sample_xml = "<article><title>EuropePMC Article</title></article>"

        url = upload_xml(
            xml_content=sample_xml,
            pmcid="PMC67890",
            provider="europepmc",
            client=mock_service,
        )

        assert "xml/europepmc/PMC67890.xml.gz" in url
        mock_container.get_blob_client.assert_called_once_with("xml/europepmc/PMC67890.xml.gz")

    def test_upload_file_pdf(self, mock_azure_clients):
        mock_service, mock_container, mock_blob = mock_azure_clients
        mock_blob.url = "https://myaccount.blob.core.windows.net/paper-xmls/pdf/scopus/paper.pdf"
        dummy_pdf = b"%PDF-1.4 dummy content"

        url = upload_file(
            content=dummy_pdf,
            filename_or_id="paper.pdf",
            provider="scopus",
            format="pdf",
            compress=False,
            client=mock_service,
        )

        assert "pdf/scopus/paper.pdf" in url
        mock_container.get_blob_client.assert_called_once_with("pdf/scopus/paper.pdf")
        uploaded_bytes, kwargs = mock_blob.upload_blob.call_args
        assert uploaded_bytes[0] == dummy_pdf
        assert kwargs["content_settings"].content_type == "application/pdf"
        assert kwargs["content_settings"].content_encoding is None

    def test_upload_xml_container_already_exists(self, mock_azure_clients):
        mock_service, mock_container, _ = mock_azure_clients
        mock_container.create_container.side_effect = ResourceExistsError("Container already exists")

        # Should not raise exception
        upload_xml(
            xml_content="<xml/>",
            pmcid="PMC99999",
            client=mock_service,
        )


class TestDownloadXML:
    """Test download_xml and download_file functions with mocked Azure clients and local cache."""

    def test_download_xml_from_cache(self, tmp_path):
        sample_xml = "<article><title>Cached Paper</title></article>"
        gzipped_data = gzip.compress(sample_xml.encode("utf-8"))

        cached_file = tmp_path / "xml" / "ncbi" / "PMC12345.xml.gz"
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_bytes(gzipped_data)

        # Mock service client that should NOT be called since cache hit occurs
        mock_service = MagicMock()

        result = download_xml(
            blob_url_or_name="xml/ncbi/PMC12345.xml.gz",
            client=mock_service,
            use_cache=True,
            cache_dir=tmp_path,
        )

        assert result == sample_xml
        mock_service.get_container_client.assert_not_called()

    def test_download_xml_from_azure_and_caches_in_subfolder(self, tmp_path):
        sample_xml = "<article><title>Remote Paper</title></article>"
        gzipped_data = gzip.compress(sample_xml.encode("utf-8"))

        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()
        mock_download_stream = MagicMock()

        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob
        mock_blob.download_blob.return_value = mock_download_stream
        mock_download_stream.readall.return_value = gzipped_data

        blob_url = "https://myaccount.blob.core.windows.net/paper-xmls/xml/ncbi/PMC77777.xml.gz"

        result = download_xml(
            blob_url_or_name=blob_url,
            client=mock_service,
            use_cache=True,
            cache_dir=tmp_path,
        )

        assert result == sample_xml
        mock_container.get_blob_client.assert_called_once_with("xml/ncbi/PMC77777.xml.gz")

        # Verify it was written to cache under xml/ncbi/
        cached_file = tmp_path / "xml" / "ncbi" / "PMC77777.xml.gz"
        assert cached_file.exists()
        assert gzip.decompress(cached_file.read_bytes()).decode("utf-8") == sample_xml

    def test_download_file_binary(self, tmp_path):
        dummy_pdf = b"%PDF-1.4 raw binary"

        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()
        mock_download_stream = MagicMock()

        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob
        mock_blob.download_blob.return_value = mock_download_stream
        mock_download_stream.readall.return_value = dummy_pdf

        result = download_file(
            blob_url_or_name="pdf/whatever/file.pdf",
            client=mock_service,
            use_cache=True,
            cache_dir=tmp_path,
        )

        assert result == dummy_pdf
        cached_file = tmp_path / "pdf" / "whatever" / "file.pdf"
        assert cached_file.exists()
        assert cached_file.read_bytes() == dummy_pdf


class TestPaperContentGetContent:
    """Tests for PaperContent.get_content helper method."""

    @patch("peptide_extractor.azure_storage.download_xml")
    def test_get_content_success(self, mock_download):
        from peptide_extractor.db.paper_content import PaperContent

        mock_download.return_value = "<article><title>Success</title></article>"
        content = PaperContent(
            id=1,
            paper_id=10,
            format="xml",
            blob_url="https://myaccount.blob.core.windows.net/paper-xmls/xml/ncbi/PMC10.xml.gz",
        )

        result = content.get_content()
        assert result == "<article><title>Success</title></article>"
        mock_download.assert_called_once_with(
            "https://myaccount.blob.core.windows.net/paper-xmls/xml/ncbi/PMC10.xml.gz",
            use_cache=True,
        )

    def test_get_content_missing_blob_url(self):
        from peptide_extractor.db.paper_content import PaperContent

        content = PaperContent(id=2, paper_id=10, format="xml", blob_url=None)
        with pytest.raises(ValueError, match="has no blob_url"):
            content.get_content()


class TestBackfillToAzure:
    """Tests for backfill_paper_contents_to_azure."""

    @patch("peptide_extractor.azure_storage.upload_xml")
    def test_backfill_success(self, mock_upload, tmp_path):
        import sqlite3
        from peptide_extractor.azure_storage import backfill_paper_contents_to_azure

        mock_upload.return_value = "https://myaccount.blob.core.windows.net/paper-xmls/xml/ncbi/PMC55555.xml.gz"

        # Create source db
        src_db = tmp_path / "source.db"
        with sqlite3.connect(src_db) as conn:
            conn.execute("CREATE TABLE paper_metadata (id INTEGER PRIMARY KEY, pmcid TEXT);")
            conn.execute("CREATE TABLE paper_content (id INTEGER PRIMARY KEY, paper_id INTEGER, raw_content TEXT);")
            conn.execute("INSERT INTO paper_metadata VALUES (1, 'PMC55555');")
            conn.execute("INSERT INTO paper_content VALUES (10, 1, '<xml>old raw</xml>');")

        # Create target db
        tgt_db = tmp_path / "target.db"
        with sqlite3.connect(tgt_db) as conn:
            conn.execute("CREATE TABLE paper_content (id INTEGER PRIMARY KEY, blob_url TEXT);")
            conn.execute("INSERT INTO paper_content VALUES (10, NULL);")

        mock_client = MagicMock()
        count = backfill_paper_contents_to_azure(
            source_db=src_db,
            target_db=tgt_db,
            client=mock_client,
        )

        assert count == 1
        mock_upload.assert_called_once_with(
            xml_content="<xml>old raw</xml>",
            pmcid="PMC55555",
            provider="ncbi",
            format="xml",
            container_name=None,
            client=mock_client,
        )

        with sqlite3.connect(tgt_db) as conn:
            cur = conn.cursor()
            cur.execute("SELECT blob_url FROM paper_content WHERE id = 10")
            row = cur.fetchone()
            assert row[0] == "https://myaccount.blob.core.windows.net/paper-xmls/xml/ncbi/PMC55555.xml.gz"
