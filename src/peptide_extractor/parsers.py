"""This file includes parser methods that turn raw full text into formats that can be cited by llms.

Supports JATS XML, TEI XML (from GROBID or converters), and PDF documents.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, BinaryIO

import requests
import spacy
from bs4 import BeautifulSoup, Tag
import time

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
    def __init__(
        self,
        id,
        content,
        parent_id,
        caption,
        columns,
        rows,
        footnotes: str = "",
        grid: list[list[dict[str, Any]]] | None = None,
        coords: str | None = None,
        image_bytes: bytes | None = None,
    ):
        super().__init__(id, content, parent_id)
        self.caption = caption
        self.columns = columns
        self.rows = rows
        self.footnotes = footnotes
        self.grid = grid
        self.coords = coords
        self.image_bytes = image_bytes


class ImageNode(Node):
    def __init__(
        self,
        id: str,
        content: str,
        parent_id: str | None,
        caption: str = "",
        coords: str | None = None,
        image_bytes: bytes | None = None,
    ):
        super().__init__(id, content, parent_id)
        self.caption = caption
        self.coords = coords
        self.image_bytes = image_bytes


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


def is_image_figure_tag(tag: Tag) -> bool:
    """Determines whether an XML Tag represents a non-table figure/image in JATS or TEI XML."""
    if tag.name == "fig":
        return True
    return bool(tag.name == "figure" and not is_table_tag(tag))


def _collect_coords(tag: Tag) -> str | None:
    """Collect deduplicated GROBID coords from a tag and its descendants."""
    all_coords: list[str] = []
    if tag.get("coords"):
        all_coords.append(str(tag.get("coords")))
    for child in tag.find_all(attrs={"coords": True}):
        if child.get("coords"):
            all_coords.append(str(child.get("coords")))
    return ";".join(dict.fromkeys(all_coords)) if all_coords else None


def parse_image_node(id: str, parent_id: str | None, fig_tag: Tag) -> ImageNode:
    """Parses a non-table figure tag (JATS <fig> or TEI <figure>) into an ImageNode."""
    parts: list[str] = []
    for tag_name in ("caption", "head", "figDesc"):
        t = fig_tag.find(tag_name)
        if t:
            txt = t.get_text(separator=" ", strip=True)
            if txt and txt not in parts:
                parts.append(txt)
    caption = " — ".join(parts) if parts else ""
    coords = _collect_coords(fig_tag)
    return ImageNode(
        id=id,
        content=caption,
        parent_id=parent_id,
        caption=caption,
        coords=coords,
    )


def parse_table_node(id: str, parent_id: str | None, table_tag: Tag) -> TableNode:
    """Parses a table tag (JATS <table-wrap> or TEI <figure type="table"> / <table>) into a TableNode."""
    caption_tag = table_tag.find(["caption", "head", "figDesc"])
    caption = caption_tag.get_text(separator=" ", strip=True) if caption_tag else ""

    foot_tag = table_tag.find(["table-wrap-foot", "note"])
    footnotes = foot_tag.get_text(separator=" ", strip=True) if foot_tag else ""

    tr_tags = table_tag.find_all("tr")
    row_tags = tr_tags if tr_tags else table_tag.find_all("row")

    columns = []
    rows = []
    grid: list[list[dict[str, Any]]] = []

    if row_tags:
        header_tr = row_tags[0]
        header_cells = header_tr.find_all(["th", "td", "cell"])
        columns = [cell.get_text(separator=" ", strip=True) for cell in header_cells]

        for tr in row_tags[1:]:
            row_cells = tr.find_all(["td", "th", "cell"])
            rows.append(
                [cell.get_text(separator=" ", strip=True) for cell in row_cells]
            )

        for r_idx, tr in enumerate(row_tags):
            in_thead = tr.find_parent("thead") is not None
            is_tei_label = tr.get("role") == "label"
            cells = tr.find_all(["th", "td", "cell"])
            grid_row: list[dict[str, Any]] = []
            for cell in cells:
                colspan_raw = cell.get("colspan") or cell.get("cols") or 1
                rowspan_raw = cell.get("rowspan") or cell.get("rows") or 1
                try:
                    colspan = max(1, int(colspan_raw))
                except (TypeError, ValueError):
                    colspan = 1
                try:
                    rowspan = max(1, int(rowspan_raw))
                except (TypeError, ValueError):
                    rowspan = 1
                is_header = (
                    in_thead
                    or is_tei_label
                    or cell.name == "th"
                    or (r_idx == 0 and not table_tag.find("thead"))
                )
                grid_row.append(
                    {
                        "text": cell.get_text(separator=" ", strip=True),
                        "colspan": colspan,
                        "rowspan": rowspan,
                        "is_header": is_header,
                    }
                )
            if grid_row:
                grid.append(grid_row)

    content = caption

    coords = _collect_coords(table_tag)

    return TableNode(
        id=id,
        content=content,
        parent_id=parent_id,
        caption=caption,
        columns=columns,
        rows=rows,
        footnotes=footnotes,
        grid=grid or None,
        coords=coords,
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
            elif is_image_figure_tag(root_tag):
                id = construct_node_id(parent_id, f"fig-{level_index}")
                node = parse_image_node(id, parent_id, root_tag)
            else:
                return
            parent_node.children.append(node)

        children = root_tag.find_all(recursive=False)

        tag_dict = {
            "sec": 0,
            "p": 0,
            "table": 0,
            "fig": 0,
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
            elif is_image_figure_tag(child):
                idx = tag_dict["fig"]
                tag_dict["fig"] += 1
                _parse(child, node, node.id, idx)
            else:
                # Other non-text container tags (e.g. lists, notes)
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
    elif isinstance(paper_tree, (SectionNode, SentenceNode, ImageNode)):
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


def _node_image_block(node: TableNode | ImageNode) -> dict[str, Any]:
    """Return a base64 PNG image_url block from cropped PDF bytes."""
    if node.image_bytes is None:
        raise ValueError(f"{type(node).__name__} {node.id} has no cropped image_bytes.")
    encoded = base64.b64encode(node.image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


def to_llm_blocks(
    paper_tree: RootNode,
    layout: str = "interleaved",
    tables_as_images: bool = True,
    images_as_images: bool = True,
) -> list[dict[str, Any]]:
    """Convert a parsed node tree into multimodal LangChain content blocks.

    Args:
        paper_tree: Parsed document RootNode.
        layout: 'interleaved' (insert visual PNGs immediately at their location in the text)
            or 'appendix' (keep location anchor in text and attach visual PNGs at the end).
        tables_as_images: When True, attach cropped PDF table PNGs when available.
        images_as_images: When True, attach cropped PDF figure PNGs when available.
    """
    if layout not in ("interleaved", "appendix"):
        raise ValueError(f"Unsupported layout '{layout}'. Expected 'interleaved' or 'appendix'.")

    if not tables_as_images and not images_as_images:
        return [{"type": "text", "text": to_llm_string(paper_tree)}]

    blocks: list[dict[str, Any]] = []
    text_buf: list[str] = []
    appendix_visuals: list[TableNode | ImageNode] = []

    def _flush_text() -> None:
        if text_buf:
            combined = "".join(text_buf)
            if combined.strip():
                blocks.append({"type": "text", "text": combined})
            text_buf.clear()

    def _walk(node: Node) -> None:
        tab_level = calculate_level_from_name(node.id) - 1
        indent = "\t" * max(0, tab_level)

        if isinstance(node, ParagraphNode):
            text_buf.append(f"{indent}[{node.id}]\n")
        elif isinstance(node, (SectionNode, SentenceNode)):
            text_buf.append(f"{indent}[{node.id}] {node.content}\n")
        elif isinstance(node, TableNode):
            caption_line = f"{indent}[{node.id}] {node.content}".rstrip()
            text_buf.append(f"{caption_line}\n")
            if node.footnotes:
                text_buf.append(f"{indent}\tFootnotes: {node.footnotes}\n")

            if tables_as_images and node.image_bytes is not None:
                if layout == "interleaved":
                    text_buf.append(f"{indent}\t>>> [TABLE IMAGE ATTACHED: {node.id}] <<<\n")
                    _flush_text()
                    blocks.append(_node_image_block(node))
                else:
                    text_buf.append(
                        f"{indent}\t>>> [SEE APPENDIX TABLE IMAGE: {node.id}] <<<\n"
                    )
                    appendix_visuals.append(node)
            else:
                header_str = " | ".join(node.columns)
                text_buf.append(f"{indent}\tColumns: {header_str}\n")
                for idx, row in enumerate(node.rows):
                    row_id = f"[{node.id}/row-{idx}]"
                    row_str = " | ".join(str(cell) for cell in row)
                    text_buf.append(f"{indent}\t{row_id} {row_str}\n")
            return
        elif isinstance(node, ImageNode):
            caption_line = f"{indent}[{node.id}] {node.content}".rstrip()
            text_buf.append(f"{caption_line}\n")
            if images_as_images and node.image_bytes is not None:
                if layout == "interleaved":
                    text_buf.append(f"{indent}\t>>> [FIGURE IMAGE ATTACHED: {node.id}] <<<\n")
                    _flush_text()
                    blocks.append(_node_image_block(node))
                else:
                    text_buf.append(
                        f"{indent}\t>>> [SEE APPENDIX FIGURE IMAGE: {node.id}] <<<\n"
                    )
                    appendix_visuals.append(node)
            return

        for child in node.children:
            _walk(child)

    _walk(paper_tree)
    _flush_text()

    if layout == "appendix" and appendix_visuals:
        for visual_node in appendix_visuals:
            kind = "TABLE" if isinstance(visual_node, TableNode) else "FIGURE"
            header = f"\n=== APPENDIX {kind} IMAGE: [{visual_node.id}] ===\n"
            if visual_node.caption:
                header += f"Caption: {visual_node.caption}\n"
            if isinstance(visual_node, TableNode) and visual_node.footnotes:
                header += f"Footnotes: {visual_node.footnotes}\n"
            blocks.append({"type": "text", "text": header})
            blocks.append(_node_image_block(visual_node))

    return blocks


def to_debug_string(paper_tree: RootNode) -> str:
    """Returns a given node tree into a debug string."""
    """
    (RootNode
        (SectionNode
            (ParagraphNode)
        )
        (SectionNode
            (ParagraphNode
                (SentenceNode)
                (SentenceNode)
                (SentenceNode)
            )
            (ParagraphNode)
            (SectionNode
                (ParagraphNode
                    (SentenceNode)
                    (SentenceNode)
                    (TableNode)
                )
            )
        )
    )
    """

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
        tei_coordinates: list[str] | None = None,
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
        data: dict[str, Any] = {
            "consolidateHeader": "1" if consolidate_header else "0",
            "consolidateCitations": "1" if consolidate_citations else "0",
            "includeRawCitations": "1" if include_raw_citations else "0",
            "segmentSentences": "1" if segment_sentences else "0",
        }
        if tei_coordinates:
            data["teiCoordinates"] = tei_coordinates

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

def pdf_to_tei_xml(
    pdf_input: bytes | str | Path | BinaryIO,
    grobid_url: str | None = None,
    backend: str = "grobid",
    timeout: int = 60,
    retry_timeout: int = 3,
    max_retry: int = 3,
    **grobid_kwargs,
) -> str:
    """Converts a PDF document to TEI XML via GROBID."""
    if backend not in ("grobid"):
        raise ValueError(
            f"Unknown PDF backend: '{backend}'. Supported: 'grobid'."
        )

    client = GrobidClient(base_url=grobid_url, timeout=timeout)
    attempts = max(1, max_retry)

    for attempt in range(1, attempts + 1):
        try:
            if client.is_alive():
                logger.info("Using GROBID backend for PDF -> TEI XML conversion.")
                return client.process_fulltext_document(pdf_input, **grobid_kwargs)
        except ConnectionError:
            pass

        if attempt < attempts:
            logger.warning(
                f"GROBID service not reachable. Waiting {retry_timeout}s. Retry {attempt}/{attempts}"
            )
            time.sleep(retry_timeout)

    raise ConnectionError("GROBID service not reachable")


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


def _read_text_content(content: str | bytes | Path) -> str:
    """Reads text from bytes, a file path, or returns a raw string as-is."""
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, Path):
        with open(content, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    if isinstance(content, str):
        try:
            p = Path(content)
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
        except (OSError, ValueError):
            pass
        return content
    return str(content)


def parse_document(
    content: str | bytes | Path,
    format: str = "auto",
    add_sentence_nodes: bool = True,
    grobid_url: str | None = None,
    backend: str = "auto",
    crop_pdf_tables: bool = False,
    crop_pdf_images: bool = False,
) -> RootNode:
    """Unified single interface to parse documents (JATS XML, TEI XML, or PDF) into a Node tree."""
    resolved_format = format.lower()
    raw_text: str | None = None

    if resolved_format == "auto":
        if is_pdf(content):
            resolved_format = "pdf"
        else:
            raw_text = _read_text_content(content)

            if is_tei_xml(raw_text):
                resolved_format = "tei"
            else:
                resolved_format = "xml"

    if resolved_format == "pdf":
        grobid_kwargs: dict[str, Any] = {}
        if crop_pdf_tables or crop_pdf_images:
            grobid_kwargs["tei_coordinates"] = ["figure"]
        tei_xml = pdf_to_tei_xml(
            content,
            grobid_url=grobid_url,
            backend=backend,
            **grobid_kwargs,
        )
        root_node = parse_xml(tei_xml)
        if crop_pdf_tables:
            from peptide_extractor.table_renderer import crop_tables_from_pdf

            crop_tables_from_pdf(root_node, content)
        if crop_pdf_images:
            from peptide_extractor.table_renderer import crop_figures_from_pdf

            crop_figures_from_pdf(root_node, content)
    else:
        if raw_text is None:
            raw_text = _read_text_content(content)
        root_node = parse_xml(raw_text)

    if add_sentence_nodes:
        root_node = add_sentences(root_node)

    return root_node


def to_extraction_ready_string(
    content: str | bytes | Path,
    format: str = "auto",
    grobid_url: str | None = None,
    backend: str = "grobid",
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
    backend: str = "grobid",
) -> str:
    """Returns an extraction-ready string from a PDF document (converted via TEI XML)."""
    return to_extraction_ready_string(
        pdf_content, format="pdf", grobid_url=grobid_url, backend=backend
    )


def to_extraction_ready_blocks(
    content: str | bytes | Path,
    format: str = "auto",
    layout: str = "interleaved",
    tables_as_images: bool = True,
    images_as_images: bool = True,
    grobid_url: str | None = None,
    backend: str = "grobid",
) -> list[dict[str, Any]]:
    """Convert JATS XML, TEI XML, or PDF into multimodal LLM extraction-ready content blocks."""
    root_node = parse_document(
        content=content,
        format=format,
        add_sentence_nodes=True,
        grobid_url=grobid_url,
        backend=backend,
        crop_pdf_tables=tables_as_images,
        crop_pdf_images=images_as_images,
    )
    return to_llm_blocks(
        root_node,
        layout=layout,
        tables_as_images=tables_as_images,
        images_as_images=images_as_images,
    )


def extraction_ready_blocks_from_xml(
    xml_text: str,
    layout: str = "interleaved",
    tables_as_images: bool = True,
    images_as_images: bool = True,
) -> list[dict[str, Any]]:
    """Returns multimodal extraction-ready blocks from an XML string (JATS or TEI)."""
    return to_extraction_ready_blocks(
        xml_text,
        format="xml",
        layout=layout,
        tables_as_images=tables_as_images,
        images_as_images=images_as_images,
    )


def extraction_ready_blocks_from_pdf(
    pdf_content: bytes | str | Path,
    layout: str = "interleaved",
    tables_as_images: bool = True,
    images_as_images: bool = True,
    grobid_url: str | None = None,
    backend: str = "grobid",
) -> list[dict[str, Any]]:
    """Returns multimodal extraction-ready blocks from a PDF document (converted via TEI XML)."""
    return to_extraction_ready_blocks(
        pdf_content,
        format="pdf",
        layout=layout,
        tables_as_images=tables_as_images,
        images_as_images=images_as_images,
        grobid_url=grobid_url,
        backend=backend,
    )
