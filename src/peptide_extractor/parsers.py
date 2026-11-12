"""This file includes parser methods that turn raw full text into formats that can be cited by llms.

Supports JATS XML, TEI XML (from GROBID or converters), and PDF documents.
"""

from __future__ import annotations

import html
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO

import requests
import spacy
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, id, content, parent_id=None):
        self.id = id
        self.children = []
        self.parent_id = parent_id
        self.content = content


class RootNode(Node):
    pass


class SectionNode(Node):
    pass


class ParagraphNode(Node):
    pass


class SentenceNode(
    Node
):  # a bit trickier than others since there is no specific xml tag
    pass


class TableNode(Node):
    def __init__(self, id, content, parent_id, caption, columns, rows):
        super().__init__(id, content, parent_id)
        self.caption = caption
        self.columns = columns
        self.rows = rows


# specials (these will have additional fields and logic)
# class ImageNode(Node):
# pass


def get_body_tag_from_xml_root(xml_string: str) -> Tag | None:
    """Gets the body tag (entry point) from xml string (JATS or TEI)."""
    soup = BeautifulSoup(xml_string, "xml")
    body = soup.find("body")
    if body is not None:
        return body
    # Fallback to <text> if body tag is missing
    return soup.find("text") or soup


def construct_node_id(parent_id: str | None, current_id: str) -> str:
    """Constructs a node ID from a parent ID and current ID."""
    if parent_id is None or parent_id == "":
        return current_id
    return f"{parent_id}/{current_id}"


def is_table_tag(tag: Tag) -> bool:
    """Determines whether an XML Tag represents a table in JATS or TEI XML."""
    if tag.name in ("table-wrap", "table"):
        return True
    return bool(
        tag.name == "figure"
        and (tag.get("type") == "table" or tag.find("table") is not None)
    )


def parse_table_node(id: str, parent_id: str | None, table_tag: Tag) -> TableNode:
    """Parses a table tag (JATS <table-wrap> or TEI <figure type="table"> / <table>) into a TableNode."""
    caption_tag = table_tag.find(["caption", "head", "figDesc"])
    caption = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""

    tr_tags = table_tag.find_all("tr")
    row_tags = tr_tags if tr_tags else table_tag.find_all("row")

    columns = []
    rows = []

    if row_tags:
        header_tr = row_tags[0]
        header_cells = header_tr.find_all(["th", "td", "cell"])
        columns = [cell.get_text(separator=" ", strip=True) for cell in header_cells]

        for tr in row_tags[1:]:
            row_cells = tr.find_all(["td", "th", "cell"])
            rows.append(
                [cell.get_text(separator=" ", strip=True) for cell in row_cells]
            )

    content = caption

    return TableNode(
        id=id,
        content=content,
        parent_id=parent_id,
        caption=caption,
        columns=columns,
        rows=rows,
    )


def parse_xml(xml_string: str) -> RootNode:
    """Parses an XML string (JATS XML or TEI XML) into a unified tree of Node objects."""
    root_tag = get_body_tag_from_xml_root(xml_string)

    def _parse(
        root_tag: Tag,
        parent_node: Node | None = None,
        parent_id: str | None = None,
        level_index: int = 0,
    ):
        if parent_node is None:
            node = RootNode("", "", None)
        else:
            if root_tag.name in ("sec", "div"):
                id = construct_node_id(parent_id, f"sec-{level_index}")
                content_tag = root_tag.find(["title", "head"], recursive=False)
                if not content_tag:
                    content_tag = root_tag.find(["title", "head"])
                node = SectionNode(
                    id,
                    content_tag.get_text(separator=" ", strip=True)
                    if content_tag
                    else "",
                    parent_id,
                )
            elif root_tag.name == "p":
                id = construct_node_id(parent_id, f"p-{level_index}")
                nested_tables = [
                    child for child in root_tag.find_all(True) if is_table_tag(child)
                ]
                if nested_tables:
                    # Clone or extract table text out of p so .text doesn't swallow it
                    p_copy = BeautifulSoup(str(root_tag), "xml").find(root_tag.name)
                    for tw in p_copy.find_all(True):
                        if is_table_tag(tw):
                            tw.decompose()
                    content = p_copy.get_text(separator=" ", strip=True)
                else:
                    content = root_tag.get_text(separator=" ", strip=True)
                node = ParagraphNode(id, content, parent_id)
            elif is_table_tag(root_tag):
                id = construct_node_id(parent_id, f"table-{level_index}")
                node = parse_table_node(id, parent_id, root_tag)
            else:
                return
            parent_node.children.append(node)

        children = root_tag.find_all(recursive=False)

        tag_dict = {
            "sec": 0,
            "p": 0,
            "table": 0,
        }

        for child in children:
            if child.name in ("title", "head"):
                # Header title consumed as SectionNode content
                continue

            if child.name in ("sec", "div"):
                idx = tag_dict["sec"]
                tag_dict["sec"] += 1
                _parse(child, node, node.id, idx)
            elif child.name == "p":
                idx = tag_dict["p"]
                tag_dict["p"] += 1
                _parse(child, node, node.id, idx)
            elif is_table_tag(child):
                idx = tag_dict["table"]
                tag_dict["table"] += 1
                _parse(child, node, node.id, idx)
            else:
                # Other non-text container tags (e.g. lists, notes, or non-table figures)
                pass

        return node

    return _parse(root_tag)


def add_sentences(root: RootNode) -> RootNode:
    """Fills the node tree with sentences. Uses spacy model en_core_sci_sm for sentence segmentation."""
    nlp = spacy.load(
        "en_core_sci_sm",
        config={
            "vars": {"include_static_vectors": False},
            "components": {
                "tok2vec": {"model": {"embed": {"include_static_vectors": False}}}
            },
        },
    )

    def _add_sentences(root, nlp):
        for node in root.children:
            if isinstance(node, ParagraphNode):
                doc = nlp(node.content)
                sentences = list(doc.sents)
                for i, sentence in enumerate(sentences):
                    child_id = construct_node_id(node.id, f"s-{i}")
                    sentence_node = SentenceNode(
                        child_id, sentence.text.strip(), node.id
                    )
                    node.children.append(sentence_node)
            else:
                _add_sentences(node, nlp)

        return root

    return _add_sentences(root, nlp)


def calculate_level_from_name(name: str) -> int:
    """Calculates the level of a node based on its name (ID). Utility function"""
    tags = name.split("/")
    level = len(tags)
    return level


def to_llm_string(paper_tree: RootNode) -> str:
    """Returns a given node tree into an llm readable format."""
    tab_level = calculate_level_from_name(paper_tree.id) - 1
    s = ""
    if not isinstance(paper_tree, RootNode):
        s = "\t" * tab_level + f"[{paper_tree.id}]"

    if isinstance(paper_tree, ParagraphNode):
        s += "\n"
    elif isinstance(paper_tree, (SectionNode, SentenceNode)):
        s += f" {paper_tree.content}\n"
    elif isinstance(paper_tree, TableNode):
        s += f" {paper_tree.content}\n"

        indent = "\t" * tab_level

        header_str = " | ".join(paper_tree.columns)
        s += f"{indent}\tColumns: {header_str}\n"

        # rows with location annotation
        for idx, row in enumerate(paper_tree.rows):
            row_id = f"[{paper_tree.id}/row-{idx}]"
            row_str = " | ".join(str(cell) for cell in row)
            s += f"{indent}\t{row_id} {row_str}\n"

    for child in paper_tree.children:
        s += to_llm_string(child)

    return s


def to_debug_string(paper_tree: RootNode) -> str:
    """Returns a given node tree into a debug string."""

    def _render(node: Node, depth: int) -> list[str]:
        indent = "\t" * depth

        # Leaf node: collapsed on one line
        if not node.children:
            return [f"{indent}({type(node).__name__}, {node.id})"]

        # Parent node: open tag, render children, close tag
        lines = [f"{indent}({type(node).__name__}, {node.id}"]
        for child in node.children:
            lines.extend(_render(child, depth + 1))
        lines.append(f"{indent})")

        return lines

    return "\n".join(_render(paper_tree, 0))


def get_node_by_id(paper_tree: Node, id: str) -> Node | None:
    """Returns a node by its ID. Utility function for validation."""
    id = id.replace("[", "").replace("]", "")

    if paper_tree.id == id:
        return paper_tree
    for child in paper_tree.children:
        node = get_node_by_id(child, id)
        if node:
            return node
    return None


class GrobidClient:
    """Client for the GROBID REST API to convert scholarly PDFs to TEI XML."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 60,
    ):
        self.base_url = (
            base_url or os.getenv("GROBID_URL", "http://localhost:8070")
        ).rstrip("/")
        self.timeout = timeout

    def is_alive(self) -> bool:
        """Check if the GROBID service is running and responsive."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/isalive",
                timeout=min(5, self.timeout),
            )
            return resp.status_code == 200 and resp.text.strip().lower() in ("true", "")
        except requests.RequestException:
            return False

    def process_fulltext_document(
        self,
        pdf_input: bytes | str | Path | BinaryIO,
        consolidate_header: bool = False,
        consolidate_citations: bool = False,
        include_raw_citations: bool = False,
        segment_sentences: bool = False,
    ) -> str:
        """Convert a PDF document into TEI XML using GROBID's processFulltextDocument endpoint.

        Args:
            pdf_input: PDF bytes, a file path, or file-like object.
            consolidate_header: Whether to consolidate header metadata against CrossRef.
            consolidate_citations: Whether to consolidate citation metadata.
            include_raw_citations: Whether to include raw citations.
            segment_sentences: Whether to request GROBID sentence segmentation.

        Returns:
            The TEI XML document string.
        """
        if isinstance(pdf_input, (str, Path)):
            pdf_path = Path(pdf_input)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            filename = pdf_path.name
        elif isinstance(pdf_input, bytes):
            pdf_bytes = pdf_input
            filename = "document.pdf"
        elif hasattr(pdf_input, "read"):
            pdf_bytes = pdf_input.read()
            filename = getattr(pdf_input, "name", "document.pdf")
        else:
            raise TypeError(f"Unsupported pdf_input type: {type(pdf_input)}")

        files = {"input": (filename, pdf_bytes, "application/pdf")}
        data = {
            "consolidateHeader": "1" if consolidate_header else "0",
            "consolidateCitations": "1" if consolidate_citations else "0",
            "includeRawCitations": "1" if include_raw_citations else "0",
            "segmentSentences": "1" if segment_sentences else "0",
        }

        url = f"{self.base_url}/api/processFulltextDocument"
        try:
            response = requests.post(url, files=files, data=data, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Could not connect to GROBID service at {self.base_url}. "
                f"Ensure GROBID is running (e.g., docker run -t --rm -p 8070:8070 grobid/grobid:0.8.1) "
                f"or set the GROBID_URL environment variable."
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"GROBID processing failed with HTTP {response.status_code}: {response.text[:500]}"
            )

        return response.text


def _extract_text_from_pdf(pdf_input: bytes | str | Path | BinaryIO) -> str:
    """Extract plain text from a PDF using available tools (pdftotext or pypdf)."""
    # 1. Try pdftotext command line utility (from poppler-utils)
    if shutil.which("pdftotext"):
        try:
            if isinstance(pdf_input, (str, Path)):
                res = subprocess.run(
                    ["pdftotext", str(pdf_input), "-"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return res.stdout
            elif isinstance(pdf_input, bytes):
                res = subprocess.run(
                    ["pdftotext", "-", "-"],
                    input=pdf_input,
                    capture_output=True,
                    check=True,
                )
                return res.stdout.decode("utf-8", errors="replace")
            elif hasattr(pdf_input, "read"):
                raw_bytes = pdf_input.read()
                res = subprocess.run(
                    ["pdftotext", "-", "-"],
                    input=raw_bytes,
                    capture_output=True,
                    check=True,
                )
                return res.stdout.decode("utf-8", errors="replace")
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning(f"pdftotext extraction failed: {e}")

    # 2. Try pypdf if installed
    try:
        import io

        import pypdf

        if isinstance(pdf_input, (str, Path)):
            reader = pypdf.PdfReader(str(pdf_input))
        elif isinstance(pdf_input, bytes):
            reader = pypdf.PdfReader(io.BytesIO(pdf_input))
        else:
            reader = pypdf.PdfReader(pdf_input)
        text_pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(text_pages)
    except ImportError:
        pass

    raise RuntimeError(
        "Could not extract text from PDF. Neither GROBID, 'pdftotext' (poppler-utils), "
        "nor 'pypdf' are available."
    )


def pdf_to_tei_lightweight(
    pdf_input: bytes | str | Path | BinaryIO,
    title: str = "Document",
) -> str:
    """Lightweight fallback to convert a PDF into valid TEI XML without a running GROBID server."""
    text = _extract_text_from_pdf(pdf_input)
    raw_blocks = [
        b.strip() for b in text.replace("\x0c", "\n\n").split("\n\n") if b.strip()
    ]

    sections = []
    current_section = {"title": "Main", "paragraphs": []}

    known_headers = {
        "abstract",
        "introduction",
        "results",
        "discussion",
        "methods",
        "materials",
        "materials and methods",
        "experimental section",
        "experimental",
        "conclusion",
        "conclusions",
        "references",
        "background",
        "results and discussion",
    }

    for block in raw_blocks:
        lines = block.splitlines()
        first_line = lines[0].strip()
        is_heading = False

        cleaned_heading = first_line.lower().strip("0123456789. \t:-")
        if (
            cleaned_heading in known_headers
            or len(lines) == 1
            and len(first_line) <= 80
            and not first_line.endswith(".")
        ):
            is_heading = True

        if is_heading:
            if current_section["paragraphs"]:
                sections.append(current_section)
            current_section = {"title": first_line, "paragraphs": []}
            if len(lines) > 1:
                p_text = " ".join(l.strip() for l in lines[1:] if l.strip())
                if p_text:
                    current_section["paragraphs"].append(p_text)
        else:
            p_text = " ".join(l.strip() for l in lines if l.strip())
            if p_text:
                current_section["paragraphs"].append(p_text)

    if current_section["paragraphs"] or not sections:
        sections.append(current_section)

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">',
        "  <teiHeader>",
        "    <fileDesc>",
        f"      <titleStmt><title>{html.escape(title)}</title></titleStmt>",
        "    </fileDesc>",
        "  </teiHeader>",
        "  <text>",
        "    <body>",
    ]

    for s_idx, sec in enumerate(sections):
        xml_lines.append(f'      <div n="{s_idx}">')
        if sec["title"]:
            xml_lines.append(f"        <head>{html.escape(sec['title'])}</head>")
        for p in sec["paragraphs"]:
            xml_lines.append(f"        <p>{html.escape(p)}</p>")
        xml_lines.append("      </div>")

    xml_lines.extend(
        [
            "    </body>",
            "  </text>",
            "</TEI>",
        ]
    )

    return "\n".join(xml_lines)


def pdf_to_tei_xml(
    pdf_input: bytes | str | Path | BinaryIO,
    grobid_url: str | None = None,
    backend: str = "auto",
    timeout: int = 60,
    **grobid_kwargs,
) -> str:
    """Converts a PDF document to TEI XML via GROBID or lightweight fallback."""
    if backend == "auto":
        client = GrobidClient(base_url=grobid_url, timeout=timeout)
        if client.is_alive():
            logger.info("Using GROBID backend for PDF -> TEI XML conversion.")
            return client.process_fulltext_document(pdf_input, **grobid_kwargs)
        logger.info(
            "GROBID service not reachable; falling back to lightweight PDF -> TEI XML."
        )
        return pdf_to_tei_lightweight(pdf_input)
    elif backend == "grobid":
        client = GrobidClient(base_url=grobid_url, timeout=timeout)
        return client.process_fulltext_document(pdf_input, **grobid_kwargs)
    elif backend in ("lightweight", "pdftotext"):
        return pdf_to_tei_lightweight(pdf_input)
    else:
        raise ValueError(
            f"Unknown PDF backend: '{backend}'. Supported: 'auto', 'grobid', 'lightweight'."
        )


def is_tei_xml(xml_content: str) -> bool:
    """Returns True if the XML content represents TEI XML."""
    if not isinstance(xml_content, str):
        return False
    return (
        "<TEI" in xml_content
        or "<teiHeader" in xml_content
        or "http://www.tei-c.org/ns/1.0" in xml_content
    )


def is_pdf(content: str | bytes | Path) -> bool:
    """Returns True if content is a PDF file or raw PDF bytes."""
    if isinstance(content, bytes):
        return content.startswith(b"%PDF")
    if isinstance(content, Path):
        return content.suffix.lower() == ".pdf"
    if isinstance(content, str):
        try:
            p = Path(content)
            if p.suffix.lower() == ".pdf" and p.exists():
                return True
        except (OSError, ValueError):
            pass
    return False


def parse_document(
    content: str | bytes | Path,
    format: str = "auto",
    add_sentence_nodes: bool = True,
    grobid_url: str | None = None,
    backend: str = "auto",
) -> RootNode:
    """Unified single interface to parse documents (JATS XML, TEI XML, or PDF) into a Node tree."""
    resolved_format = format.lower()
    raw_text: str | None = None

    if resolved_format == "auto":
        if is_pdf(content):
            resolved_format = "pdf"
        else:
            if isinstance(content, bytes):
                raw_text = content.decode("utf-8", errors="replace")
            elif isinstance(content, Path) or (
                isinstance(content, str)
                and Path(content).exists()
                and Path(content).is_file()
            ):
                with open(content, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()
            else:
                raw_text = str(content)

            if is_tei_xml(raw_text):
                resolved_format = "tei"
            else:
                resolved_format = "xml"

    if resolved_format == "pdf":
        tei_xml = pdf_to_tei_xml(
            content,
            grobid_url=grobid_url,
            backend=backend,
        )
        root_node = parse_xml(tei_xml)
    else:
        if raw_text is None:
            if isinstance(content, bytes):
                raw_text = content.decode("utf-8", errors="replace")
            elif isinstance(content, Path) or (
                isinstance(content, str)
                and Path(content).exists()
                and Path(content).is_file()
            ):
                with open(content, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()
            else:
                raw_text = str(content)
        root_node = parse_xml(raw_text)

    if add_sentence_nodes:
        root_node = add_sentences(root_node)

    return root_node


def to_extraction_ready_string(
    content: str | bytes | Path,
    format: str = "auto",
    grobid_url: str | None = None,
    backend: str = "auto",
) -> str:
    """Single unified interface to convert JATS XML, TEI XML, or PDF into an LLM extraction-ready string."""
    root_node = parse_document(
        content=content,
        format=format,
        add_sentence_nodes=True,
        grobid_url=grobid_url,
        backend=backend,
    )
    return to_llm_string(root_node)


def extraction_ready_string_from_xml(xml_text: str) -> str:
    """Returns an extraction-ready string from an XML string (JATS or TEI)."""
    return to_extraction_ready_string(xml_text, format="xml")


def extraction_ready_string_from_pdf(
    pdf_content: bytes | str | Path,
    grobid_url: str | None = None,
    backend: str = "auto",
) -> str:
    """Returns an extraction-ready string from a PDF document (converted via TEI XML)."""
    return to_extraction_ready_string(
        pdf_content, format="pdf", grobid_url=grobid_url, backend=backend
    )
