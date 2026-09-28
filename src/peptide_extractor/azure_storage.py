"""Azure Blob Storage integration for peptide_extractor.

Handles uploading and downloading of full-text scientific paper XMLs,
transparently applying gzip compression to minimize bandwidth and storage costs.
"""

import gzip
import os
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import find_dotenv, load_dotenv

# Load environment variables from .env
load_dotenv(find_dotenv(usecwd=True))

DEFAULT_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER", "papers")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def get_blob_service_client(
    connection_string: Optional[str] = None,
) -> BlobServiceClient:
    """Create and return an Azure BlobServiceClient instance."""
    conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError(
            "Azure Storage connection string is missing. "
            "Please set the AZURE_STORAGE_CONNECTION_STRING environment variable."
        )
    return BlobServiceClient.from_connection_string(conn_str)


def extract_blob_name(
    blob_url_or_name: str, container_name: Optional[str] = None
) -> str:
    """Extract blob name from either a full URL or a relative blob name/path."""
    cleaned = blob_url_or_name.strip()
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        parsed = urlparse(cleaned)
        path = parsed.path.lstrip("/")
        parts = path.split("/", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    target_container = container_name or DEFAULT_CONTAINER_NAME
    cleaned = cleaned.lstrip("/")
    if target_container and cleaned.startswith(f"{target_container}/"):
        cleaned = cleaned[len(target_container) + 1 :]
    return cleaned


def build_blob_name(
    identifier: str,
    provider: str = "ncbi",
    format: str = "xml",
    compress: bool = True,
) -> str:
    """Construct a structured blob path: {format}/{provider}/{filename}."""
    clean_id = identifier.strip().lstrip("/")
    # If already prefixed with format/provider, preserve it
    if clean_id.startswith(f"{format}/{provider}/"):
        return clean_id

    ext = f".{format}"
    gz_ext = f".{format}.gz"
    if clean_id.endswith(gz_ext) or clean_id.endswith(ext):
        filename = clean_id
    elif clean_id.endswith(".gz"):
        filename = clean_id
    else:
        filename = f"{clean_id}{gz_ext}" if compress else f"{clean_id}{ext}"

    return f"{format}/{provider}/{filename}"


def upload_file(
    content: Union[str, bytes],
    filename_or_id: str,
    provider: str = "ncbi",
    format: str = "xml",
    container_name: Optional[str] = None,
    client: Optional[BlobServiceClient] = None,
    compress: bool = True,
    content_type: Optional[str] = None,
) -> str:
    """Upload content to Azure Blob Storage under {format}/{provider}/{filename}.

    Args:
        content: Text string or raw bytes to upload.
        filename_or_id: File name or entity ID (e.g. 'PMC12345').
        provider: Source provider (e.g. 'ncbi', 'europepmc', 'crossref').
        format: Content format directory (e.g. 'xml', 'pdf').
        container_name: Target Azure container name (default: from env or 'paper-xmls').
        client: Optional existing BlobServiceClient.
        compress: Whether to gzip-compress (default True for text/xml).
        content_type: MIME type (auto-detected if None).

    Returns:
        The URL of the uploaded blob.
    """
    blob_client_service = client or get_blob_service_client()
    target_container = container_name or DEFAULT_CONTAINER_NAME

    container_client = blob_client_service.get_container_client(target_container)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    blob_name = build_blob_name(
        identifier=filename_or_id,
        provider=provider,
        format=format,
        compress=compress,
    )
    blob_client = container_client.get_blob_client(blob_name)

    if isinstance(content, str):
        raw_bytes = content.encode("utf-8")
    else:
        raw_bytes = content

    if compress:
        data_to_upload = gzip.compress(raw_bytes)
        encoding = "gzip"
    else:
        data_to_upload = raw_bytes
        encoding = None

    mime_type = content_type or (
        "application/xml"
        if format == "xml"
        else ("application/pdf" if format == "pdf" else "application/octet-stream")
    )
    content_settings = ContentSettings(
        content_type=mime_type,
        content_encoding=encoding,
    )

    blob_client.upload_blob(
        data_to_upload,
        overwrite=True,
        content_settings=content_settings,
    )

    return blob_client.url


def upload_xml(
    xml_content: str,
    pmcid: str,
    provider: str = "ncbi",
    format: str = "xml",
    container_name: Optional[str] = None,
    client: Optional[BlobServiceClient] = None,
    compress: bool = True,
) -> str:
    """Compress and upload a paper XML to Azure Blob Storage under {format}/{provider}/{filename}.

    Args:
        xml_content: Raw XML string content.
        pmcid: PubMed Central ID or filename (e.g., 'PMC1234567').
        provider: Provider name (e.g., 'ncbi', 'europepmc'). Default: 'ncbi'.
        format: Format category (default: 'xml').
        container_name: Target Azure container name (default: from env or 'paper-xmls').
        client: Optional existing BlobServiceClient.
        compress: Whether to compress with gzip before uploading (default: True).

    Returns:
        The URL of the uploaded blob.
    """
    return upload_file(
        content=xml_content,
        filename_or_id=pmcid,
        provider=provider,
        format=format,
        container_name=container_name,
        client=client,
        compress=compress,
        content_type="application/xml",
    )


def download_file(
    blob_url_or_name: str,
    container_name: Optional[str] = None,
    client: Optional[BlobServiceClient] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> bytes:
    """Download raw bytes from Azure Blob Storage or local cache.

    Args:
        blob_url_or_name: The full blob URL or the relative blob name/path.
        container_name: Azure container name (default: from env or 'paper-xmls').
        client: Optional existing BlobServiceClient.
        use_cache: If True, reads from / writes to local disk cache.
        cache_dir: Local base directory for cache (default: data/).

    Returns:
        Raw downloaded bytes.
    """
    blob_name = extract_blob_name(blob_url_or_name, container_name=container_name)
    target_cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # 1. Try reading from local cache if enabled
    if use_cache:
        cached_file = target_cache_dir / blob_name
        if cached_file.exists():
            return cached_file.read_bytes()

    # 2. Download from Azure Blob Storage
    blob_client_service = client or get_blob_service_client()
    target_container = container_name or DEFAULT_CONTAINER_NAME
    container_client = blob_client_service.get_container_client(target_container)
    blob_client = container_client.get_blob_client(blob_name)

    download_stream = blob_client.download_blob()
    downloaded_bytes = download_stream.readall()

    # 3. Write to local cache if enabled
    if use_cache:
        cached_file = target_cache_dir / blob_name
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_bytes(downloaded_bytes)

    return downloaded_bytes


def download_xml(
    blob_url_or_name: str,
    container_name: Optional[str] = None,
    client: Optional[BlobServiceClient] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> str:
    """Download and decompress a paper XML from Azure Blob Storage or local cache.

    Args:
        blob_url_or_name: The full blob URL or the relative blob name/path.
        container_name: Azure container name (default: from env or 'paper-xmls').
        client: Optional existing BlobServiceClient.
        use_cache: If True, reads from / writes to local disk cache.
        cache_dir: Local base directory for cache (default: data/).

    Returns:
        Decompressed XML string.
    """
    blob_name = extract_blob_name(blob_url_or_name, container_name=container_name)
    raw_data = download_file(
        blob_url_or_name=blob_url_or_name,
        container_name=container_name,
        client=client,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    if blob_name.endswith(".gz"):
        return gzip.decompress(raw_data).decode("utf-8")
    return raw_data.decode("utf-8")


# This is not going to be used (seems like it) since we don't have a raw_content column anymore
def backfill_paper_contents_to_azure(
    source_db: Optional[Union[str, Path]] = None,
    target_db: Optional[Union[str, Path]] = None,
    provider: str = "ncbi",
    format: str = "xml",
    container_name: Optional[str] = None,
    client: Optional[BlobServiceClient] = None,
    limit: Optional[int] = None,
) -> int:
    """Backfill existing uncompressed XMLs from a database into Azure Blob Storage.

    Reads from source_db (default: data/papers_pre_azure.db) where raw_content was stored,
    uploads to Azure under {format}/{provider}/, and updates blob_url in target_db (default: data/papers.db).
    """
    import sqlite3

    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    src = Path(source_db) if source_db else data_dir / "papers_pre_azure.db"
    tgt = Path(target_db) if target_db else data_dir / "papers.db"

    if not src.exists():
        raise FileNotFoundError(f"Source database '{src}' does not exist.")
    if not tgt.exists():
        raise FileNotFoundError(f"Target database '{tgt}' does not exist.")

    src_conn = sqlite3.connect(src)
    src_cur = src_conn.cursor()

    tgt_conn = sqlite3.connect(tgt)
    tgt_cur = tgt_conn.cursor()

    query = """
        SELECT pc.id, pc.paper_id, pc.raw_content, pm.pmcid
        FROM paper_content pc
        LEFT JOIN paper_metadata pm ON pc.paper_id = pm.id
        WHERE pc.raw_content IS NOT NULL AND trim(pc.raw_content) != ''
    """
    if limit:
        query += f" LIMIT {limit}"

    src_cur.execute(query)
    rows = src_cur.fetchall()
    total = len(rows)
    print(f"Found {total} records to backfill to Azure...")

    blob_service = client or get_blob_service_client()
    count = 0

    for content_id, paper_id, raw_content, pmcid in rows:
        identifier = pmcid if (pmcid and pmcid.strip()) else f"paper_{paper_id}"
        blob_url = upload_xml(
            xml_content=raw_content,
            pmcid=identifier,
            provider=provider,
            format=format,
            container_name=container_name,
            client=blob_service,
        )
        tgt_cur.execute(
            "UPDATE paper_content SET blob_url = ? WHERE id = ?",
            (blob_url, content_id),
        )
        count += 1
        if count % 25 == 0 or count == total:
            tgt_conn.commit()
            print(f"Backfilled {count}/{total} records...")

    tgt_conn.commit()
    src_conn.close()
    tgt_conn.close()
    print(f"Backfill complete! Updated {count} records in '{tgt}'.")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azure Blob Storage management for peptide_extractor")
    parser.add_argument("--backfill", action="store_true", help="Backfill raw_content to Azure Blob Storage")
    parser.add_argument("--source-db", type=str, default=None, help="Source database with raw_content")
    parser.add_argument("--target-db", type=str, default=None, help="Target database to update blob_url")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to backfill")
    args = parser.parse_args()

    if args.backfill:
        backfill_paper_contents_to_azure(
            source_db=args.source_db,
            target_db=args.target_db,
            limit=args.limit,
        )
    else:
        parser.print_help()

