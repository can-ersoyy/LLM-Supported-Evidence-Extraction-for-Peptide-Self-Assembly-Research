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
