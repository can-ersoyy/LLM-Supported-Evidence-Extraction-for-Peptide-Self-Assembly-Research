from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from peptide_extractor.parsers import (
    GrobidClient,
    ImageNode,
    ParagraphNode,
    RootNode,
    SectionNode,
    SentenceNode,
    TableNode,
    add_sentences,
    construct_node_id,
    extraction_ready_string_from_pdf,
    extraction_ready_string_from_xml,
    get_node_by_id,
    is_tei_xml,
    parse_document,
    parse_xml,
    pdf_to_tei_xml,
    to_debug_string,
    to_extraction_ready_string,
    to_llm_string,
)

SAMPLE_JATS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <body>
    <sec id="sec1">
      <title>Introduction</title>
      <p>Peptide self-assembly is a spontaneous process. It enables the fabrication of diverse nanostructures.</p>
    </sec>
    <sec id="sec2">
      <title>Experimental Results</title>
      <p>We synthesized diphenylalanine (FF) peptides.</p>
      <table-wrap id="tbl1">
        <caption>Table 1: Experimental observations</caption>
        <table>
          <thead>
            <tr>
              <th>Peptide</th>
              <th>Solvent</th>
              <th>Morphology</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>FF</td>
              <td>Water</td>
              <td>nanotube</td>
            </tr>
            <tr>
              <td>KLVFF</td>
              <td>PBS</td>
              <td>nanofiber</td>
            </tr>
          </tbody>
        </table>
      </table-wrap>
      <sec id="sec2-1">
        <title>Morphological Analysis</title>
        <p>TEM imaging confirmed nanotube formation.</p>
      </sec>
    </sec>
  </body>
</article>
"""

SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Self-assembly of diphenylalanine peptides</title></titleStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div n="1">
        <head>Introduction</head>
        <p>Diphenylalanine peptides assemble into robust nanotubes. These structures exhibit high mechanical rigidity.</p>
      </div>
      <div n="2">
        <head>Results</head>
        <p>We observed tubular assemblies under physiological pH.</p>
        <figure type="table" xml:id="tab_0">
          <head>Table 1: Assembly characteristics</head>
          <table>
            <row role="label">
              <cell>Sequence</cell>
              <cell>Solvent</cell>
              <cell>Outcome</cell>
            </row>
            <row>
              <cell>FF</cell>
              <cell>H2O</cell>
              <cell>nanotube</cell>
            </row>
            <row>
              <cell>Boc-FF</cell>
              <cell>EtOH</cell>
              <cell>nanosphere</cell>
            </row>
          </table>
        </figure>
        <div n="2.1">
          <head>Secondary Structure Analysis</head>
          <p>FTIR spectra displayed characteristic beta-sheet peaks.</p>
        </div>
      </div>
      <div n="3">
        <!-- Figure that is an image, not a table - should not become a TableNode -->
        <figure xml:id="fig_0">
          <head>Figure 1: TEM micrograph</head>
          <figDesc>Scale bar 200 nm.</figDesc>
        </figure>
        <p>Conclusion of the study.</p>
      </div>
    </body>
  </text>
</TEI>
"""


def test_construct_node_id():
    assert construct_node_id(None, "sec-0") == "sec-0"
    assert construct_node_id("", "sec-0") == "sec-0"
    assert construct_node_id("sec-0", "p-1") == "sec-0/p-1"
    assert construct_node_id("sec-0/p-1", "s-2") == "sec-0/p-1/s-2"


def test_is_tei_xml():
    assert is_tei_xml(SAMPLE_TEI_XML) is True
    assert is_tei_xml(SAMPLE_JATS_XML) is False
    assert is_tei_xml("<xml></xml>") is False
    assert is_tei_xml(123) is False


def test_parse_jats_xml():
    root = parse_xml(SAMPLE_JATS_XML)
    assert isinstance(root, RootNode)
    assert len(root.children) == 2

    # First section: Introduction
    sec0 = root.children[0]
    assert isinstance(sec0, SectionNode)
    assert sec0.id == "sec-0"
    assert sec0.content == "Introduction"
    assert len(sec0.children) == 1
    p0 = sec0.children[0]
    assert isinstance(p0, ParagraphNode)
    assert p0.id == "sec-0/p-0"
    assert "Peptide self-assembly" in p0.content

    # Second section: Experimental Results
    sec1 = root.children[1]
    assert isinstance(sec1, SectionNode)
    assert sec1.id == "sec-1"
    assert sec1.content == "Experimental Results"
    assert len(sec1.children) == 3

    p1 = sec1.children[0]
    assert isinstance(p1, ParagraphNode)
    assert p1.id == "sec-1/p-0"

    tbl0 = sec1.children[1]
    assert isinstance(tbl0, TableNode)
    assert tbl0.id == "sec-1/table-0"
    assert "Table 1" in tbl0.caption
    assert tbl0.columns == ["Peptide", "Solvent", "Morphology"]
    assert len(tbl0.rows) == 2
    assert tbl0.rows[0] == ["FF", "Water", "nanotube"]

    # Nested section
    subsec0 = sec1.children[2]
    assert isinstance(subsec0, SectionNode)
    assert subsec0.id == "sec-1/sec-0"
    assert subsec0.content == "Morphological Analysis"


def test_parse_tei_xml():
    root = parse_xml(SAMPLE_TEI_XML)
    assert isinstance(root, RootNode)
    assert len(root.children) == 3

    # Section 0
    sec0 = root.children[0]
    assert isinstance(sec0, SectionNode)
    assert sec0.id == "sec-0"
    assert sec0.content == "Introduction"
    assert len(sec0.children) == 1
    assert isinstance(sec0.children[0], ParagraphNode)
    assert sec0.children[0].id == "sec-0/p-0"

    # Section 1
    sec1 = root.children[1]
    assert isinstance(sec1, SectionNode)
    assert sec1.id == "sec-1"
    assert sec1.content == "Results"
    assert len(sec1.children) == 3

    p0 = sec1.children[0]
    assert p0.id == "sec-1/p-0"

    tbl0 = sec1.children[1]
    assert isinstance(tbl0, TableNode)
    assert tbl0.id == "sec-1/table-0"
    assert "Table 1: Assembly characteristics" in tbl0.caption
    assert tbl0.columns == ["Sequence", "Solvent", "Outcome"]
    assert len(tbl0.rows) == 2
    assert tbl0.rows[0] == ["FF", "H2O", "nanotube"]

    # Nested section in TEI
    subsec0 = sec1.children[2]
    assert isinstance(subsec0, SectionNode)
    assert subsec0.id == "sec-1/sec-0"
    assert subsec0.content == "Secondary Structure Analysis"

    # Section 2: Non-table figure should become an ImageNode (not TableNode)
    sec2 = root.children[2]
    assert len(sec2.children) == 2
    assert isinstance(sec2.children[0], ImageNode)
    assert sec2.children[0].id == "sec-2/fig-0"
    assert "Figure 1: TEM micrograph" in sec2.children[0].caption
    assert isinstance(sec2.children[1], ParagraphNode)
    assert sec2.children[1].content == "Conclusion of the study."


def test_add_sentences_and_get_node_by_id():
    root = parse_xml(SAMPLE_TEI_XML)
    root = add_sentences(root)

    sec0_p0 = root.children[0].children[0]
    assert len(sec0_p0.children) == 2
    s0 = sec0_p0.children[0]
    assert isinstance(s0, SentenceNode)
    assert s0.id == "sec-0/p-0/s-0"
    assert "Diphenylalanine peptides assemble" in s0.content

    s1 = sec0_p0.children[1]
    assert s1.id == "sec-0/p-0/s-1"
    assert "mechanical rigidity" in s1.content

    # get_node_by_id checks
    found_s0 = get_node_by_id(root, "[sec-0/p-0/s-0]")
    assert found_s0 is s0
    found_tbl = get_node_by_id(root, "sec-1/table-0")
    assert isinstance(found_tbl, TableNode)
    assert get_node_by_id(root, "non-existent") is None


def test_to_llm_string_and_debug_string():
    root = parse_xml(SAMPLE_TEI_XML)
    root = add_sentences(root)
    llm_str = to_llm_string(root)

    assert "[sec-0] Introduction" in llm_str
    assert "[sec-0/p-0]" in llm_str
    assert (
        "[sec-0/p-0/s-0] Diphenylalanine peptides assemble into robust nanotubes."
        in llm_str
    )
    assert "[sec-1/table-0] Table 1: Assembly characteristics" in llm_str
    assert "Columns: Sequence | Solvent | Outcome" in llm_str
    assert "[sec-1/table-0/row-0] FF | H2O | nanotube" in llm_str
    assert "[sec-2/fig-0] Figure 1: TEM micrograph" in llm_str

    debug_str = to_debug_string(root)
    assert "(RootNode," in debug_str
    assert "(SectionNode, sec-0" in debug_str
    assert "(SentenceNode, sec-0/p-0/s-0)" in debug_str


def test_unified_parse_document_and_to_extraction_ready_string():
    # 1. JATS XML string
    jats_llm = to_extraction_ready_string(SAMPLE_JATS_XML)
    assert "[sec-0] Introduction" in jats_llm
    assert "[sec-1/table-0]" in jats_llm

    # 2. TEI XML string
    tei_llm = to_extraction_ready_string(SAMPLE_TEI_XML)
    assert "[sec-0] Introduction" in tei_llm
    assert "[sec-1/table-0]" in tei_llm

    # 3. Backwards-compatible extraction_ready_string_from_xml
    legacy_out = extraction_ready_string_from_xml(SAMPLE_JATS_XML)
    assert "[sec-0] Introduction" in legacy_out

    # 4. Long XML string (> PATH_MAX) should not trigger OSError in Path.exists()
    long_p = "<p>" + ("Peptide self-assembly occurs in water. " * 200) + "</p>"
    long_jats = f"<article><body><sec><title>Results</title>{long_p}</sec></body></article>"
    assert len(long_jats) > 4096
    long_llm = to_extraction_ready_string(long_jats, format="xml")
    assert "[sec-0] Results" in long_llm
    long_llm_auto = to_extraction_ready_string(long_jats, format="auto")
    assert "[sec-0] Results" in long_llm_auto


def test_grobid_client_is_alive():
    client = GrobidClient("http://localhost:8070")

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "true"
        assert client.is_alive() is True

    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError()
        assert client.is_alive() is False


def test_grobid_client_process_fulltext_document():
    client = GrobidClient("http://localhost:8070")
    dummy_pdf = b"%PDF-1.4 dummy content"

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = SAMPLE_TEI_XML

        result = client.process_fulltext_document(dummy_pdf)
        assert result == SAMPLE_TEI_XML
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:8070/api/processFulltextDocument"
        assert "input" in kwargs["files"]


def test_grobid_client_connection_error():
    client = GrobidClient("http://localhost:8070")
    dummy_pdf = b"%PDF-1.4 dummy"

    with (
        patch("requests.post", side_effect=requests.exceptions.ConnectionError()),
        pytest.raises(ConnectionError, match="Could not connect to GROBID service"),
    ):
        client.process_fulltext_document(dummy_pdf)


def test_pdf_to_tei_xml_retries_when_grobid_unavailable():
    dummy_pdf = b"%PDF-1.4 dummy"

    with (
        patch.object(GrobidClient, "is_alive", side_effect=[False, False, True]) as mock_alive,
        patch.object(GrobidClient, "process_fulltext_document", return_value=SAMPLE_TEI_XML) as mock_proc,
        patch("peptide_extractor.parsers.time.sleep") as mock_sleep,
    ):
        result = pdf_to_tei_xml(dummy_pdf, retry_timeout=2, max_retry=3)
        assert result == SAMPLE_TEI_XML
        assert mock_alive.call_count == 3
        mock_proc.assert_called_once_with(dummy_pdf)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(2)


def test_pdf_to_tei_xml_raises_after_max_retries():
    dummy_pdf = b"%PDF-1.4 dummy"

    with (
        patch.object(GrobidClient, "is_alive", return_value=False) as mock_alive,
        patch("peptide_extractor.parsers.time.sleep") as mock_sleep,
        pytest.raises(ConnectionError, match="GROBID service not reachable"),
    ):
        pdf_to_tei_xml(dummy_pdf, retry_timeout=1, max_retry=3)

    assert mock_alive.call_count == 3
    assert mock_sleep.call_count == 2


def test_pdf_to_tei_xml_does_not_swallow_processing_errors():
    dummy_pdf = b"%PDF-1.4 dummy"

    with (
        patch.object(GrobidClient, "is_alive", return_value=True),
        patch.object(
            GrobidClient,
            "process_fulltext_document",
            side_effect=RuntimeError("GROBID processing failed with HTTP 500"),
        ),
        patch("peptide_extractor.parsers.time.sleep") as mock_sleep,
        pytest.raises(RuntimeError, match="HTTP 500"),
    ):
        pdf_to_tei_xml(dummy_pdf, retry_timeout=1, max_retry=3)

    mock_sleep.assert_not_called()


def test_parse_table_with_spans_and_footnotes():
    xml_with_spans = """<?xml version="1.0" encoding="UTF-8"?>
    <article><body><sec><title>Results</title>
      <table-wrap id="tbl_span">
        <caption>Table 2: Matrix screening</caption>
        <table>
          <thead>
            <tr>
              <th rowspan="2">Peptide</th>
              <th colspan="2">Conditions</th>
            </tr>
            <tr>
              <th>pH</th>
              <th>Morphology</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>KLVFF</td>
              <td>7.4</td>
              <td>nanofiber</td>
            </tr>
          </tbody>
        </table>
        <table-wrap-foot><p>Incubated at 25 °C for 24 h.</p></table-wrap-foot>
      </table-wrap>
    </sec></body></article>
    """
    root = parse_xml(xml_with_spans)
    tbl = root.children[0].children[0]
    assert isinstance(tbl, TableNode)
    assert "Incubated at 25 °C" in tbl.footnotes
    assert tbl.grid is not None
    assert tbl.grid[0][0]["rowspan"] == 2
    assert tbl.grid[0][1]["colspan"] == 2


def test_to_llm_blocks_interleaved_and_appendix():
    import base64
    from io import BytesIO
    from PIL import Image
    from peptide_extractor.parsers import (
        extraction_ready_blocks_from_xml,
        to_llm_blocks,
    )

    root = parse_xml(SAMPLE_JATS_XML)
    root = add_sentences(root)

    # Without PDF-cropped image_bytes, XML tables fall back to text
    xml_fallback = to_llm_blocks(root, layout="interleaved", tables_as_images=True)
    assert len(xml_fallback) == 1
    assert xml_fallback[0]["type"] == "text"
    assert "[sec-1/table-0/row-0] FF | Water | nanotube" in xml_fallback[0]["text"]

    # Attach cropped PDF PNG bytes to TableNode
    buf = BytesIO()
    Image.new("RGB", (100, 50), color="#FFFFFF").save(buf, format="PNG")
    tbl = get_node_by_id(root, "sec-1/table-0")
    assert isinstance(tbl, TableNode)
    tbl.image_bytes = buf.getvalue()

    # Interleaved layout: text -> table image -> remaining text
    interleaved = to_llm_blocks(root, layout="interleaved", tables_as_images=True)
    assert len(interleaved) == 3
    assert interleaved[0]["type"] == "text"
    assert "[sec-1/table-0] Table 1: Experimental observations" in interleaved[0]["text"]
    assert ">>> [TABLE IMAGE ATTACHED: sec-1/table-0] <<<" in interleaved[0]["text"]
    assert "FF | Water | nanotube" not in interleaved[0]["text"]

    assert interleaved[1]["type"] == "image_url"
    url = interleaved[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded_png = base64.b64decode(url.split(",", 1)[1])
    assert decoded_png.startswith(b"\x89PNG\r\n\x1a\n")

    assert interleaved[2]["type"] == "text"
    assert "[sec-1/sec-0] Morphological Analysis" in interleaved[2]["text"]

    # Appendix layout: full text first, then labeled appendix table image
    appendix = to_llm_blocks(root, layout="appendix", tables_as_images=True)
    assert len(appendix) == 3
    assert appendix[0]["type"] == "text"
    assert ">>> [SEE APPENDIX TABLE IMAGE: sec-1/table-0] <<<" in appendix[0]["text"]
    assert "[sec-1/sec-0] Morphological Analysis" in appendix[0]["text"]
    assert appendix[1]["type"] == "text"
    assert "=== APPENDIX TABLE IMAGE: [sec-1/table-0] ===" in appendix[1]["text"]
    assert appendix[2]["type"] == "image_url"

    # Convenience wrapper + tables_as_images=False fallback
    via_wrapper = extraction_ready_blocks_from_xml(
        SAMPLE_JATS_XML, tables_as_images=False, images_as_images=False
    )
    assert len(via_wrapper) == 1
    assert via_wrapper[0]["type"] == "text"
    assert "[sec-1/table-0/row-0] FF | Water | nanotube" in via_wrapper[0]["text"]


def test_pdf_table_cropping_uses_original_bytes_without_rerendering():
    import base64
    import sys
    import types
    from io import BytesIO
    from PIL import Image
    from peptide_extractor.parsers import to_llm_blocks
    from peptide_extractor.table_renderer import crop_tables_from_pdf, parse_grobid_coords

    tei_with_coords = """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body><div n="1"><head>Results</head>
        <figure type="table" coords="2,40.0,100.0,300.0,150.0">
          <head>Table 1: Original PDF Table</head>
          <table><row><cell>A</cell><cell>B</cell></row></table>
        </figure>
      </div></body></text>
    </TEI>
    """
    assert parse_grobid_coords("2,40.0,100.0,300.0,150.0") == [(2, 40.0, 100.0, 300.0, 150.0)]

    root = parse_xml(tei_with_coords)
    tbl = root.children[0].children[0]
    assert isinstance(tbl, TableNode)
    assert tbl.coords == "2,40.0,100.0,300.0,150.0"

    buf = BytesIO()
    Image.new("RGB", (120, 60), color="#123456").save(buf, format="PNG")
    fake_png = buf.getvalue()

    class FakePixmap:
        def tobytes(self, fmt: str) -> bytes:
            return fake_png

    class FakeRect:
        def __init__(self, x0, y0, x1, y1):
            self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    class FakePage:
        rect = FakeRect(0.0, 0.0, 600.0, 800.0)

        def get_pixmap(self, clip=None, dpi=200):
            return FakePixmap()

    class FakeDoc:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return FakePage()

        def close(self):
            pass

    fake_fitz = types.SimpleNamespace(
        open=lambda **kwargs: FakeDoc(),
        Rect=FakeRect,
    )

    with patch.dict(sys.modules, {"fitz": fake_fitz}):
        cropped = crop_tables_from_pdf(root, b"%PDF-1.4 fake")

    assert cropped == 1
    assert tbl.image_bytes is not None

    blocks = to_llm_blocks(root, layout="interleaved", tables_as_images=True)
    img_block = next(b for b in blocks if b["type"] == "image_url")
    decoded = base64.b64decode(img_block["image_url"]["url"].split(",", 1)[1])
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_pdf_figure_cropping_and_image_node_blocks():
    import base64
    import sys
    import types
    from io import BytesIO
    from PIL import Image
    from peptide_extractor.parsers import to_llm_blocks
    from peptide_extractor.table_renderer import crop_figures_from_pdf

    tei_with_fig = """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body><div n="1"><head>Results</head>
        <figure xml:id="fig_0" coords="2,50.0,120.0,250.0,180.0">
          <head>Figure 1: TEM</head>
          <figDesc>Nanofibers observed at 10 mg/ml.</figDesc>
        </figure>
      </div></body></text>
    </TEI>
    """
    root = parse_xml(tei_with_fig)
    fig = root.children[0].children[0]
    assert isinstance(fig, ImageNode)
    assert fig.id == "sec-0/fig-0"
    assert fig.coords == "2,50.0,120.0,250.0,180.0"

    buf = BytesIO()
    Image.new("RGB", (100, 80), color="#654321").save(buf, format="PNG")
    fake_png = buf.getvalue()

    class FakePixmap:
        def tobytes(self, fmt: str) -> bytes:
            return fake_png

    class FakeRect:
        def __init__(self, x0, y0, x1, y1):
            self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    class FakePage:
        rect = FakeRect(0.0, 0.0, 600.0, 800.0)

        def get_pixmap(self, clip=None, dpi=200):
            return FakePixmap()

    class FakeDoc:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return FakePage()

        def close(self):
            pass

    fake_fitz = types.SimpleNamespace(
        open=lambda **kwargs: FakeDoc(),
        Rect=FakeRect,
    )

    with patch.dict(sys.modules, {"fitz": fake_fitz}):
        cropped = crop_figures_from_pdf(root, b"%PDF-1.4 fake")

    assert cropped == 1
    assert fig.image_bytes is not None

    interleaved = to_llm_blocks(root, layout="interleaved", images_as_images=True)
    assert len(interleaved) == 2
    assert ">>> [FIGURE IMAGE ATTACHED: sec-0/fig-0] <<<" in interleaved[0]["text"]
    assert interleaved[1]["type"] == "image_url"
    decoded = base64.b64decode(interleaved[1]["image_url"]["url"].split(",", 1)[1])
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")

    appendix = to_llm_blocks(root, layout="appendix", images_as_images=True)
    assert len(appendix) == 3
    assert ">>> [SEE APPENDIX FIGURE IMAGE: sec-0/fig-0] <<<" in appendix[0]["text"]
    assert "=== APPENDIX FIGURE IMAGE: [sec-0/fig-0] ===" in appendix[1]["text"]
    assert appendix[2]["type"] == "image_url"



