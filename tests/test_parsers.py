from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from peptide_extractor.parsers import (
    GrobidClient,
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
    pdf_to_tei_lightweight,
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

    # Section 2: Non-table figure should not become a TableNode
    sec2 = root.children[2]
    assert len(sec2.children) == 1  # Only paragraph, non-table figure skipped
    assert isinstance(sec2.children[0], ParagraphNode)
    assert sec2.children[0].content == "Conclusion of the study."


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


def test_pdf_to_tei_lightweight_with_real_pdf():
    pdf_path = Path("docs/system_design/idp-initial-system_v1.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF fixture not found")

    tei_xml = pdf_to_tei_lightweight(pdf_path, title="System Design")
    assert is_tei_xml(tei_xml)
    assert "<TEI" in tei_xml
    assert "<body>" in tei_xml

    # Parse resulting TEI XML into object tree
    root = parse_xml(tei_xml)
    assert isinstance(root, RootNode)
    assert len(root.children) > 0
    first_sec = root.children[0]
    assert isinstance(first_sec, SectionNode)
    assert first_sec.id == "sec-0"


def test_end_to_end_pdf_to_object_tree():
    pdf_path = Path("docs/system_design/idp-initial-system_v1.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF fixture not found")

    # With backend='lightweight'
    root = parse_document(pdf_path, format="pdf", backend="lightweight")
    assert isinstance(root, RootNode)
    assert len(root.children) > 0

    llm_str = extraction_ready_string_from_pdf(pdf_path, backend="lightweight")
    assert "[sec-0]" in llm_str
    assert "[sec-0/p-0/s-0]" in llm_str
