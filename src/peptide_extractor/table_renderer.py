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


def _box_contains_significant_graphic(
    page: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    min_graphic_area_ratio: float = 0.15,
) -> bool:
    """Return True if [x0, y0, x1, y1] already contains substantial raster/vector graphics."""
    box_area = max(1.0, (x1 - x0) * (y1 - y0))
    covered_area = 0.0
    try:
        for img_info in page.get_images(full=True):
            for rect in page.get_image_rects(img_info[0]):
                ix0 = max(x0, rect.x0)
                iy0 = max(y0, rect.y0)
                ix1 = min(x1, rect.x1)
                iy1 = min(y1, rect.y1)
                if ix1 > ix0 and iy1 > iy0:
                    covered_area += (ix1 - ix0) * (iy1 - iy0)
                    if covered_area / box_area >= min_graphic_area_ratio:
                        return True
    except Exception:
        pass
    return False


def _expand_caption_only_figure_rect(
    page: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_rect: Any,
    padding: float,
    min_graphic_height: float = 90.0,
) -> tuple[float, float, float, float]:
    """Expand a caption-only GROBID figure box to include the graphic rendered above it.

    GROBID frequently emits coordinates only for the figure caption (<figDesc>) rather
    than the embedded plot, phase diagram, or chemical scheme rendered above it—even when
    a long multi-line caption is >100pt tall (e.g. Nature Communications Figure 1).
    Expand upward whenever the box is either short (< `min_graphic_height`) OR contains
    no significant raster graphic while graphics/drawings exist directly above it.
    """
    box_height = y1 - y0
    has_graphic_inside = _box_contains_significant_graphic(page, x0, y0, x1, y1)
    if box_height >= min_graphic_height and has_graphic_inside:
        return x0, y0, x1, y1

    # Look for embedded raster images or vector drawings positioned above the caption
    # (allow up to 85% of page height above y0 so full-page figures like Nature Fig 1 are captured).
    page_height = page_rect.y1 - page_rect.y0
    max_upward_span = page_height * 0.85
    candidate_rects: list[tuple[float, float, float, float]] = []

    try:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            for rect in page.get_image_rects(xref):
                if rect.is_empty or (rect.height < 8.0 and rect.width < 8.0):
                    continue
                if rect.y0 < y0 and rect.y1 <= y1 + 20.0 and (y0 - rect.y0) <= max_upward_span:
                    candidate_rects.append((rect.x0, rect.y0, rect.x1, rect.y1))
    except Exception:
        pass

    if not candidate_rects:
        try:
            for drawing in page.get_drawings():
                rect = drawing.get("rect")
                if rect is None or rect.is_empty:
                    continue
                if rect.height < 10.0 and rect.width < 10.0:
                    continue
                if rect.y0 < y0 and rect.y1 <= y1 + 10.0 and (y0 - rect.y0) <= max_upward_span:
                    candidate_rects.append((rect.x0, rect.y0, rect.x1, rect.y1))
        except Exception:
            pass

    if candidate_rects:
        new_x0 = max(page_rect.x0, min(x0, *(r[0] for r in candidate_rects)) - padding)
        new_y0 = max(page_rect.y0, min(y0, *(r[1] for r in candidate_rects)) - padding)
        new_x1 = min(page_rect.x1, max(x1, *(r[2] for r in candidate_rects)) + padding)
        new_y1 = min(page_rect.y1, max(y1, *(r[3] for r in candidate_rects)) + padding)
        return new_x0, new_y0, new_x1, new_y1

    if box_height < min_graphic_height:
        # Last-resort fallback for thin caption boxes when no raster/vector objects are exposed
        fallback_y0 = max(page_rect.y0, y0 - page_height * 0.35)
        return page_rect.x0, fallback_y0, page_rect.x1, y1

    return x0, y0, x1, y1


def _locate_caption_boxes_in_pdf(
    doc: Any, caption: str
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Fallback when GROBID omits coords on an ImageNode: search for the caption prefix in the PDF."""
    if not caption or not caption.strip():
        return {}
    # Use the leading ~40 characters (e.g. 'Figure 1 | Morphological characterization')
    cleaned = " ".join(caption.split())
    queries = [cleaned[:45], cleaned[:25]]
    for q in queries:
        if len(q) < 8:
            continue
        for idx in range(len(doc)):
            hits = doc[idx].search_for(q)
            if hits:
                return {idx + 1: [(r.x0, r.y0, r.x1, r.y1) for r in hits]}
    return {}


def crop_figures_from_pdf(
    root_node: RootNode,
    pdf_input: bytes | str | Path,
    dpi: int = 200,
    padding: float = 8.0,
) -> int:
    """Crop original figure/image regions directly from a PDF using GROBID coords (with caption fallback).

    Populates `image_node.image_bytes` in-place for every ImageNode that has
    valid `coords` or a caption locatable in the PDF. Returns the number of figures cropped.
    """
    from peptide_extractor.parsers import ImageNode

    image_nodes: list[ImageNode] = [
        img
        for img in _iter_nodes_of_type(root_node, ImageNode)
        if getattr(img, "coords", None) or getattr(img, "caption", None)
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
            by_page: dict[int, list[tuple[float, float, float, float]]] = {}
            for page_num, x, y, w, h in boxes:
                if 1 <= page_num <= len(doc):
                    by_page.setdefault(page_num, []).append((x, y, x + w, y + h))
            if not by_page:
                by_page = _locate_caption_boxes_in_pdf(doc, image_node.caption or "")
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
                x0, y0, x1, y1 = _expand_caption_only_figure_rect(
                    page, x0, y0, x1, y1, page_rect, padding
                )
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
