"""Unit and integration tests for GoldenDisagreementAudit and golden_auditor.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from sqlmodel import Session, SQLModel, create_engine, select

from peptide_extractor.db import (
    AuditType,
    DisagreementVerdict,
    ExtractionMode,
    GoldenDisagreementAudit,
    PaperContent,
    PaperMetadata,
    PeptideEntry,
)
from peptide_extractor.golden_auditor import (
    FieldAdjudicationVerdict,
    UnmatchedEntryAdjudicationVerdict,
    audit_run_against_golden,
    collect_disagreements,
    format_audit_report,
    resolve_evidence_for_field,
)
from peptide_extractor.parsers import parse_document


def _make_entry(**kwargs) -> PeptideEntry:
    defaults = {
        "sequence": "FF",
        "length": 2,
        "pmid": "12345678",
        "type": "hydrogel",
        "n_terminal": "Fmoc",
        "c_terminal": "Free",
        "temperature": 25.0,
        "ph": 7.0,
        "solution": "DMSO/H2O",
        "volume_ratio": "5/95",
        "extraction_mode": ExtractionMode.MANUAL,
        "run_identifier": 0,
    }
    defaults.update(kwargs)
    return PeptideEntry(**defaults)


def test_collect_disagreements_finds_field_mismatches_and_unmatched_llm_entries():
    manual = [
        _make_entry(id=1, sequence="FF", ph=7.4, solution="PBS"),
    ]
    llm = [
        # Matched to FF, but pH disagrees (7.0 vs 7.4 -> outside 0.25 tol)
        _make_entry(
            id=10,
            sequence="FF",
            ph=7.0,
            solution="PBS",
            extraction_mode=ExtractionMode.LLM,
            run_identifier=99,
        ),
        # Unmatched LLM-only sequence
        _make_entry(
            id=11,
            sequence="KLVFF",
            length=5,
            type="fiber",
            solution="H2O",
            extraction_mode=ExtractionMode.LLM,
            run_identifier=99,
        ),
    ]

    candidates = collect_disagreements(manual, llm, include_unmatched_llm=True)
    field_cands = [c for c in candidates if c.audit_type == AuditType.FIELD_MISMATCH]
    unmatched_cands = [c for c in candidates if c.audit_type == AuditType.LLM_ONLY_ENTRY]

    assert len(field_cands) == 1
    assert field_cands[0].field_name == "ph"
    assert field_cands[0].manual_value == "7.4"
    assert field_cands[0].llm_value == "7.0"

    assert len(unmatched_cands) == 1
    assert unmatched_cands[0].sequence == "KLVFF"
    assert unmatched_cands[0].manual_entry is None


def test_resolve_evidence_for_field_retrieves_exact_sentence_quote():
    xml = (
        "<article><body><sec><title>Methods</title>"
        "<p>Fmoc-FF was dissolved at pH 7.0 and incubated at 25 C.</p>"
        "</sec></body></article>"
    )
    tree = parse_document(xml, format="xml")
    ev_json = json.dumps({"ph": "[sec-0/p-0/s-0]", "sequence": "[sec-0/p-0/s-0]"})

    tag, quote, img = resolve_evidence_for_field(tree, ev_json, "ph")
    assert tag == "[sec-0/p-0/s-0]"
    assert "pH 7.0" in quote
    assert img is None


def test_audit_run_against_golden_persists_db_rows_and_exports_reports(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    sample_xml = (
        "<article><body><sec><title>Results</title>"
        "<p>Fmoc-FF formed a hydrogel in PBS at pH 7.0 and 25 C.</p>"
        "<p>Control peptide KLVFF also formed nanofibers in water.</p>"
        "</sec></body></article>"
    )

    def _mock_controller_factory(*args, **kwargs):
        schema_cls = kwargs.get("structured_output")
        ctrl = MagicMock()
        if schema_cls is FieldAdjudicationVerdict:
            ctrl.extract_peptides.return_value = FieldAdjudicationVerdict(
                verdict="llm_correct_golden_wrong",
                confidence=0.95,
                reason="Source sentence explicitly states pH 7.0, not 7.4.",
            )
        else:
            ctrl.extract_peptides.return_value = UnmatchedEntryAdjudicationVerdict(
                verdict="valid_uncurated_in_golden",
                confidence=0.92,
                reason="Source text explicitly reports KLVFF nanofibers in water.",
            )
        return ctrl

    md_path = tmp_path / "audit_report.md"
    csv_path = tmp_path / "audit_report.csv"

    with Session(engine) as session:
        paper = PaperMetadata(
            title="Fmoc-FF and KLVFF",
            abstract="Peptide self-assembly.",
            pmid="88888888",
            doi="10.1000/test.audit",
            is_open_access=True,
        )
        session.add(paper)
        session.commit()
        session.refresh(paper)

        content = PaperContent(
            paper_id=paper.id,
            format="xml",
            source="pmc",
            blob_url="https://example.invalid/paper_88888888.xml",
        )
        session.add(content)

        m_entry = _make_entry(
            paper_id=paper.id,
            pmid="88888888",
            sequence="FF",
            ph=7.4,
            solution="PBS",
            extraction_mode=ExtractionMode.MANUAL,
            run_identifier=0,
        )
        l_entry_match = _make_entry(
            paper_id=paper.id,
            pmid="88888888",
            sequence="FF",
            ph=7.0,
            solution="PBS",
            evidence_span=json.dumps({"ph": "[sec-0/p-0/s-0]", "sequence": "[sec-0/p-0/s-0]"}),
            extraction_mode=ExtractionMode.LLM,
            run_identifier=42,
        )
        l_entry_extra = _make_entry(
            paper_id=paper.id,
            pmid="88888888",
            sequence="KLVFF",
            length=5,
            type="fiber",
            solution="water",
            evidence_span=json.dumps({"sequence": "[sec-0/p-1/s-0]", "type": "[sec-0/p-1/s-0]"}),
            extraction_mode=ExtractionMode.LLM,
            run_identifier=42,
        )
        session.add_all([m_entry, l_entry_match, l_entry_extra])
        session.commit()

        with (
            patch(
                "peptide_extractor.golden_auditor.LLMController",
                side_effect=_mock_controller_factory,
            ),
            patch.object(PaperContent, "get_content", return_value=sample_xml),
        ):
            audits, md_report = audit_run_against_golden(
                llm_run_identifier=42,
                manual_run_identifier=0,
                pmids=["88888888"],
                save=True,
                audit_run_identifier=1001,
                export_markdown_path=md_path,
                export_csv_path=csv_path,
                session=session,
            )

        assert len(audits) == 2
        saved = session.exec(
            select(GoldenDisagreementAudit).where(
                GoldenDisagreementAudit.audit_run_identifier == 1001
            )
        ).all()
        assert len(saved) == 2

        by_type = {row.audit_type: row for row in saved}
        assert by_type[AuditType.FIELD_MISMATCH].verdict == DisagreementVerdict.LLM_CORRECT_GOLDEN_WRONG
        assert by_type[AuditType.FIELD_MISMATCH].manual_entry_id == m_entry.id
        assert by_type[AuditType.FIELD_MISMATCH].llm_entry_id == l_entry_match.id
        assert "pH 7.0" in (by_type[AuditType.FIELD_MISMATCH].source_quote or "")

        assert by_type[AuditType.LLM_ONLY_ENTRY].verdict == DisagreementVerdict.VALID_UNCURATED_IN_GOLDEN
        assert by_type[AuditType.LLM_ONLY_ENTRY].manual_entry_id is None
        assert by_type[AuditType.LLM_ONLY_ENTRY].llm_entry_id == l_entry_extra.id

        assert md_path.exists()
        assert csv_path.exists()
        assert "llm_correct_golden_wrong" in md_report
        assert "valid_uncurated_in_golden" in format_audit_report(saved)


def test_resolve_evidence_handles_table_row_tags_and_string_footnotes():
    from peptide_extractor.parsers import RootNode, SectionNode, TableNode

    root = RootNode("root", "")
    sec = SectionNode("sec-0", "Results", "root")
    tbl = TableNode(
        id="sec-0/table-0",
        content="Table 1. Assembly outcomes",
        parent_id="sec-0",
        caption="Table 1. Assembly outcomes",
        columns=["Peptide", "Solvent", "Result"],
        rows=[
            ["FF", "PBS", "hydrogel"],
            ["KLVFF", "H2O", "fiber"],
        ],
        footnotes="All experiments run at 25 C.",
    )
    sec.children.append(tbl)
    root.children.append(sec)

    ev_json = json.dumps({"type": "[sec-0/table-0/row-1]"})
    tag, quote, img = resolve_evidence_for_field(root, ev_json, "type")
    assert tag == "[sec-0/table-0/row-1]"
    assert ">>> [row-1] KLVFF | H2O | fiber" in quote
    assert "Footnotes: All experiments run at 25 C." in quote
    assert img is None


def test_resolve_evidence_handles_multitag_whitespace_and_parent_fallback():
    xml = (
        "<article><body><sec><title>Methods</title>"
        "<p>Fmoc-FF was dissolved at pH 7.0 and incubated at 25 C.</p>"
        "</sec></body></article>"
    )
    tree = parse_document(xml, format="xml")

    # Whitespace + multiple comma-separated tags
    ev_multi = json.dumps({"ph": " [sec-0/p-0/s-0], [sec-0/p-0/s-1] "})
    tag, quote, _ = resolve_evidence_for_field(tree, ev_multi, "ph")
    assert tag == "[sec-0/p-0/s-0]"
    assert "pH 7.0" in quote

    # Out-of-range sentence index falls back to parent ParagraphNode
    ev_oob = json.dumps({"ph": "[sec-0/p-0/s-99]"})
    tag_fb, quote_fb, _ = resolve_evidence_for_field(tree, ev_oob, "ph")
    assert tag_fb == "[sec-0/p-0/s-99]"
    assert "pH 7.0" in quote_fb


def test_audit_run_falls_back_across_multiple_paper_content_formats_and_reports_parse_errors():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    xml_without_match = (
        "<article><body><sec><title>Intro</title>"
        "<p>Short intro paragraph only.</p>"
        "</sec></body></article>"
    )
    pdf_tei_with_match = (
        "<article><body>"
        "<sec><title>Intro</title><p>Intro.</p></sec>"
        "<sec><title>Results</title>"
        "<p>Peptide FF formed a hydrogel at pH 7.0.</p>"
        "</sec></body></article>"
    )
    pdf_tree = parse_document(pdf_tei_with_match, format="xml")

    def _mock_ctrl(*args, **kwargs):
        ctrl = MagicMock()
        ctrl.extract_peptides.return_value = FieldAdjudicationVerdict(
            verdict="llm_correct_golden_wrong",
            confidence=0.9,
            reason="PDF tree states pH 7.0.",
        )
        return ctrl

    with Session(engine) as session:
        paper = PaperMetadata(
            title="Multi-format test",
            abstract="Abstract.",
            pmid="77777777",
            doi="10.1000/test.multiformat",
            is_open_access=True,
        )
        session.add(paper)
        session.commit()
        session.refresh(paper)

        # Paper has XML first, PDF second; LLM tag `[sec-1/p-0/s-0]` only exists in PDF tree
        pc_xml = PaperContent(
            paper_id=paper.id,
            format="xml",
            source="pmc",
            blob_url="https://example.invalid/77777777.xml",
        )
        pc_pdf = PaperContent(
            paper_id=paper.id,
            format="pdf",
            source="pmc",
            blob_url="https://example.invalid/77777777.pdf",
        )
        session.add_all([pc_xml, pc_pdf])

        m_entry = _make_entry(
            paper_id=paper.id,
            pmid="77777777",
            sequence="FF",
            ph=7.4,
            solution="PBS",
            extraction_mode=ExtractionMode.MANUAL,
            run_identifier=0,
        )
        l_entry = _make_entry(
            paper_id=paper.id,
            pmid="77777777",
            sequence="FF",
            ph=7.0,
            solution="PBS",
            evidence_span=json.dumps({"ph": "[sec-1/p-0/s-0]"}),
            extraction_mode=ExtractionMode.LLM,
            run_identifier=55,
        )
        session.add_all([m_entry, l_entry])
        session.commit()

        def _fake_get_content(self, use_cache=True):
            return xml_without_match if self.format == "xml" else b"%PDF-1.4 fake"

        real_parse_document = parse_document

        def _fake_parse_document(raw, format="auto", **kwargs):
            if format == "pdf":
                return pdf_tree
            return real_parse_document(raw, format=format, **kwargs)

        with (
            patch("peptide_extractor.golden_auditor.LLMController", side_effect=_mock_ctrl),
            patch.object(PaperContent, "get_content", _fake_get_content),
            patch("peptide_extractor.golden_auditor.parse_document", side_effect=_fake_parse_document),
        ):
            audits, _ = audit_run_against_golden(
                llm_run_identifier=55,
                manual_run_identifier=0,
                pmids=["77777777"],
                save=False,
                session=session,
            )

        assert len(audits) == 1
        assert audits[0].verdict == DisagreementVerdict.LLM_CORRECT_GOLDEN_WRONG
        assert "pH 7.0" in (audits[0].source_quote or "")

        # Now verify clear diagnostic when content_format='pdf' and GROBID fails
        with (
            patch("peptide_extractor.golden_auditor.LLMController", side_effect=_mock_ctrl),
            patch.object(PaperContent, "get_content", _fake_get_content),
            patch(
                "peptide_extractor.golden_auditor.parse_document",
                side_effect=RuntimeError("GROBID service unreachable at http://localhost:8070"),
            ),
        ):
            audits_err, _ = audit_run_against_golden(
                llm_run_identifier=55,
                manual_run_identifier=0,
                pmids=["77777777"],
                content_format="pdf",
                save=False,
                session=session,
            )

        assert len(audits_err) == 1
        assert audits_err[0].verdict == DisagreementVerdict.AMBIGUOUS_NEEDS_HUMAN
        assert "GROBID service unreachable" in (audits_err[0].reason or "")

