"""Crop original table images directly from PDF pages using GROBID coordinates."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from peptide_extractor.parsers import ImageNode, Node, RootNode, TableNode

logger = logging.getLogger(__name__)


def parse_grobid_coords(coords_str: str) -> list[tuple[int, float, float, float, float]]:
    """Parse a GROBID coords string ('page,x,y,w,h;...') into a list of bounding boxes."""
    boxes: list[tuple[int, float, float, float, float]] = []
    if not coords_str:
        return boxes
    for part in coords_str.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = part.split(",")
        if len(fields) != 5:
            continue
        try:
            page_num = int(float(fields[0]))
            x, y, w, h = (
                float(fields[1]),
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
            )
            if page_num >= 1 and w > 0 and h > 0:
                boxes.append((page_num, x, y, w, h))
        except ValueError:
            continue
    return boxes


def _iter_nodes_of_type(node: Node, cls: type) -> list[Any]:
    found: list[Any] = []
    if isinstance(node, cls):
        found.append(node)
    for child in node.children:
        found.extend(_iter_nodes_of_type(child, cls))
    return found


def _stitch_page_images(page_images: list[Image.Image]) -> bytes:
    if len(page_images) == 1:
        out_buf = io.BytesIO()
        page_images[0].save(out_buf, format="PNG")
        return out_buf.getvalue()
    max_w = max(im.width for im in page_images)
    total_h = sum(im.height for im in page_images)
    stitched = Image.new("RGB", (max_w, total_h), color="#FFFFFF")
    y_off = 0
    for im in page_images:
        stitched.paste(im, (0, y_off))
        y_off += im.height
    out_buf = io.BytesIO()
    stitched.save(out_buf, format="PNG")
    return out_buf.getvalue()


def _open_fitz_doc(pdf_input: bytes | str | Path):
    import fitz  # PyMuPDF

    if isinstance(pdf_input, bytes):
        return fitz.open(stream=pdf_input, filetype="pdf"), fitz
    if isinstance(pdf_input, (str, Path)):
        return fitz.open(str(pdf_input)), fitz
    raise TypeError(f"Unsupported pdf_input type for PDF cropping: {type(pdf_input)}")


def crop_tables_from_pdf(
    root_node: RootNode,
    pdf_input: bytes | str | Path,
    dpi: int = 200,
    padding: float = 8.0,
) -> int:
    """Crop original table regions directly from a PDF using GROBID coords.

    Populates `table_node.image_bytes` in-place for every TableNode that has
    valid `coords`. Returns the number of tables cropped.
    """
    from peptide_extractor.parsers import TableNode

    table_nodes: list[TableNode] = [
        t for t in _iter_nodes_of_type(root_node, TableNode) if getattr(t, "coords", None)
    ]
    if not table_nodes:
        return 0

    try:
        doc, fitz = _open_fitz_doc(pdf_input)
    except ImportError:
        logger.warning(
            "pymupdf (fitz) is not installed; cannot crop PDF table images. "
            "Install pymupdf (e.g. 'uv sync') to enable direct PDF table cropping."
        )
        return 0

    cropped_count = 0
    try:
        page_table_tops: dict[int, list[float]] = {}
        parsed_tables: list[
            tuple[TableNode, dict[int, list[tuple[float, float, float, float]]]]
        ] = []
        for table_node in table_nodes:
            boxes = parse_grobid_coords(table_node.coords or "")
            if not boxes:
                continue
            by_page: dict[int, list[tuple[float, float, float, float]]] = {}
            for page_num, x, y, w, h in boxes:
                if 1 <= page_num <= len(doc):
                    by_page.setdefault(page_num, []).append((x, y, x + w, y + h))
            if not by_page:
                continue
            parsed_tables.append((table_node, by_page))
            for page_num, rects in by_page.items():
                page_table_tops.setdefault(page_num, []).append(min(r[1] for r in rects))

        for tops in page_table_tops.values():
            tops.sort()

        for table_node, by_page in parsed_tables:
            page_images: list[Image.Image] = []
            for page_num in sorted(by_page):
                page = doc[page_num - 1]
                page_rect = page.rect
                rects = by_page[page_num]
                table_y0 = min(r[1] for r in rects)
                x0 = page_rect.x0
                x1 = page_rect.x1
                y0 = max(page_rect.y0, table_y0 - padding)

                next_tops = [
                    t for t in page_table_tops.get(page_num, []) if t > table_y0 + 40.0
                ]
                if next_tops:
                    y1 = min(page_rect.y1, next_tops[0] - 4.0)
                else:
                    y1 = page_rect.y1

                if x1 <= x0 or y1 <= y0:
                    continue

                clip = fitz.Rect(x0, y0, x1, y1)
                pix = page.get_pixmap(clip=clip, dpi=dpi)
                page_images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))

            if not page_images:
                continue

            table_node.image_bytes = _stitch_page_images(page_images)
            cropped_count += 1
    finally:
        doc.close()

    return cropped_count


def crop_figures_from_pdf(
    root_node: RootNode,
    pdf_input: bytes | str | Path,
    dpi: int = 200,
    padding: float = 8.0,
) -> int:
    """Crop original figure/image regions directly from a PDF using GROBID coords.

    Populates `image_node.image_bytes` in-place for every ImageNode that has
    valid `coords`. Returns the number of figures cropped.
    """
    from peptide_extractor.parsers import ImageNode

    image_nodes: list[ImageNode] = [
        img for img in _iter_nodes_of_type(root_node, ImageNode) if getattr(img, "coords", None)
    ]
    if not image_nodes:
        return 0

    try:
        doc, fitz = _open_fitz_doc(pdf_input)
    except ImportError:
        logger.warning(
            "pymupdf (fitz) is not installed; cannot crop PDF figure images. "
            "Install pymupdf (e.g. 'uv sync') to enable direct PDF figure cropping."
        )
        return 0

    cropped_count = 0
    try:
        for image_node in image_nodes:
            boxes = parse_grobid_coords(image_node.coords or "")
            if not boxes:
                continue

            by_page: dict[int, list[tuple[float, float, float, float]]] = {}
            for page_num, x, y, w, h in boxes:
                if 1 <= page_num <= len(doc):
                    by_page.setdefault(page_num, []).append((x, y, x + w, y + h))
            if not by_page:
                continue

            page_images: list[Image.Image] = []
            for page_num in sorted(by_page):
                page = doc[page_num - 1]
                page_rect = page.rect
                rects = by_page[page_num]
                x0 = max(page_rect.x0, min(r[0] for r in rects) - padding)
                y0 = max(page_rect.y0, min(r[1] for r in rects) - padding)
                x1 = min(page_rect.x1, max(r[2] for r in rects) + padding)
                y1 = min(page_rect.y1, max(r[3] for r in rects) + padding)
                if x1 <= x0 or y1 <= y0:
                    continue
                clip = fitz.Rect(x0, y0, x1, y1)
                pix = page.get_pixmap(clip=clip, dpi=dpi)
                page_images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))

            if not page_images:
                continue

            image_node.image_bytes = _stitch_page_images(page_images)
            cropped_count += 1
    finally:
        doc.close()

    return cropped_count
