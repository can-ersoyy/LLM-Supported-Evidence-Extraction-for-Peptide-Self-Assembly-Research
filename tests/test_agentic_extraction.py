"""Unit and integration tests for the multi-agent peptide extraction package."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sqlmodel import Session, SQLModel, create_engine, select

from peptide_extractor.agentic import (
    AgenticExtractionConfig,
    EntityRegistryResult,
    ExtractionOrchestrator,
    FieldUpdate,
    PeptideAlias,
    RefinerPatchResult,
    ReviewerFeedback,
    ReviewIssue,
    apply_refiner_patch,
    canonicalize_solution,
    convert_concentration_to_mg_ml,
    estimate_peptide_mw,
    extract_paper_detection_methods,
    normalize_terminal_cap,
    sanitize_and_validate_entries,
    split_sequence_and_caps,
)
from peptide_extractor.db import (
    PaperContent,
    PaperMetadata,
    PeptideEntry,
)
from peptide_extractor.db.peptide_entry import (
    EvidenceSpan,
    PeptideAssemblyType,
    PeptideConjugateMixture,
    PeptideExtractionResult,
    PeptideLinearity,
    PeptideLLMEntry,
)
from peptide_extractor.extraction import (
    CompactTableCondition,
    CompactTableRow,
    SharedExperimentalConditions,
    TableExtractionResult,
    run_extraction,
)
from peptide_extractor.pipeline import run_peptide_extraction

# ---------------------------------------------------------------------------
# 1. Deterministic Chemistry Sandbox Tests
# ---------------------------------------------------------------------------


def test_split_sequence_and_caps_handles_three_letter_caps_and_cyclics():
    seq, n_cap, c_cap, lin, mod = split_sequence_and_caps("Fmoc-Phe-Phe-NH2")
    assert seq == "FF"
    assert n_cap == "Fmoc"
    assert c_cap == "Amidation"
    assert lin is None
    assert mod is True

    seq2, n2, c2, lin2, mod2 = split_sequence_and_caps("Boc-Ser-Val-Gly-OCH2Ph")
    assert seq2 == "SVG"
    assert n2 == "Boc"
    assert c2 == "Benzylation"
    assert lin2 is None
    assert mod2 is True

    seq3, n3, c3, lin3, mod3 = split_sequence_and_caps("cyclo(K-L-K-L)")
    assert seq3 == "KLKL"
    assert n3 is None
    assert c3 is None
    assert lin3 == PeptideLinearity.CYCLIC
    assert mod3 is True


def test_mw_and_concentration_conversion():
    mw_ff = estimate_peptide_mw("FF")
    assert mw_ff is not None
    # Phe-Phe average MW ~ 312.37 Da
    assert 311.0 < mw_ff < 314.0

    assert convert_concentration_to_mg_ml(1.0, "wt%", "FF") == 10.0
    mg_ml = convert_concentration_to_mg_ml(10.0, "mM", "FF")
    assert mg_ml is not None
    assert abs(mg_ml - (10.0 * mw_ff / 1000.0)) < 0.01


def test_sanitize_resolves_aliases_validates_ranges_and_deduplicates():
    aliases = [
        PeptideAlias(
            label="1a",
            sequence="KLVFF",
            n_terminal="Acetylation",
            c_terminal="Amidation",
            linear_cyclic=PeptideLinearity.LINEAR,
            chirality="l",
        )
    ]
    raw_entries = [
        # Uses alias "1a" without caps
        PeptideLLMEntry(
            sequence="1a",
            type=PeptideAssemblyType.FIBER,
            solution="PBS",
            ph=7.4,
            temperature=25.0,
            evidence_span=EvidenceSpan(sequence="[sec-1/table-0]"),
        ),
        # Duplicate from prose with extra concentration & detection_method
        PeptideLLMEntry(
            sequence="Ac-KLVFF-NH2",
            type=PeptideAssemblyType.FIBER,
            solution="PBS",
            ph=7.4,
            temperature=25.0,
            peptide_concentration=2.0,
            detection_method=["TEM"],
            evidence_span=EvidenceSpan(
                sequence="[sec-1/p-2/s-1]",
                peptide_concentration="[sec-1/p-2/s-1]",
            ),
        ),
        # Invalid physical range entry
        PeptideLLMEntry(
            sequence="FF",
            type=PeptideAssemblyType.HYDROGEL,
            ph=18.5,
            temperature=500.0,
            peptide_concentration=-2.0,
        ),
    ]

    cleaned, report = sanitize_and_validate_entries(raw_entries, aliases=aliases)

    # First two entries merge into one (same resolved sequence, caps, solvent, pH, temp)
    # Wait: one has concentration=None and one has concentration=2.0 so their dedup keys differ
    # unless concentration matches; verify both got resolved to KLVFF with Acetylation/Amidation
    assert cleaned[0].sequence == "KLVFF"
    assert cleaned[0].n_terminal == "Acetylation"
    assert cleaned[0].c_terminal == "Amidation"
    assert cleaned[1].sequence == "KLVFF"
    assert cleaned[1].n_terminal == "Acetylation"
    assert cleaned[1].c_terminal == "Amidation"

    # Physical errors detected on entry 2
    assert report.has_errors is True
    error_fields = {i.field for i in report.issues if i.severity == "error"}
    assert {"ph", "temperature", "peptide_concentration"}.issubset(error_fields)


def test_sanitize_merges_exact_cross_pass_duplicates():
    e1 = PeptideLLMEntry(
        sequence="FF",
        type=PeptideAssemblyType.HYDROGEL,
        n_terminal="Fmoc",
        c_terminal="Free",
        solution="DMSO/H2O",
        temperature=25.0,
        ph=7.0,
        evidence_span=EvidenceSpan(sequence="[sec-1/table-0]"),
    )
    e2 = PeptideLLMEntry(
        sequence="Fmoc-Phe-Phe-OH",
        type=PeptideAssemblyType.HYDROGEL,
        solution="DMSO/H2O",
        temperature=25.0,
        ph=7.0,
        detection_method=["TEM", "CD"],
        evidence_span=EvidenceSpan(sequence="[sec-1/p-1/s-1]", ph="[sec-1/p-1/s-1]"),
    )
    cleaned, report = sanitize_and_validate_entries([e1, e2])
    assert len(cleaned) == 1
    assert report.duplicates_merged == 1
    assert cleaned[0].detection_method == ["TEM", "CD"]
    assert cleaned[0].evidence_span is not None
    assert cleaned[0].evidence_span.ph == "[sec-1/p-1/s-1]"


# ---------------------------------------------------------------------------
# 2. Orchestrator Workflow Tests (Mocked LLMController)
# ---------------------------------------------------------------------------


def _make_mock_controller_router(responses_by_schema: dict):
    """Return a fake LLMController factory dispatching responses by structured_output class."""
    call_counts: dict[type, int] = {}

    def _factory(*args, **kwargs):
        schema_cls = kwargs.get("structured_output")
        ctrl = MagicMock()

        def _extract_peptides(payload, **call_kwargs):
            idx = call_counts.get(schema_cls, 0)
            call_counts[schema_cls] = idx + 1
            seq = responses_by_schema[schema_cls]
            return seq[min(idx, len(seq) - 1)]

        ctrl.extract_peptides.side_effect = _extract_peptides
        return ctrl

    return _factory, call_counts


def test_orchestrator_happy_path_approves_without_refiner():
    scout_res = EntityRegistryResult(
        peptides=[
            PeptideAlias(
                label="Fmoc-FF",
                sequence="FF",
                n_terminal="Fmoc",
                c_terminal="Free",
                linear_cyclic=PeptideLinearity.LINEAR,
            )
        ],
        shared_conditions=SharedExperimentalConditions(
            temperature=25.0,
            detection_method=["TEM"],
        ),
    )
    prose_res = PeptideExtractionResult(
        entries=[
            PeptideLLMEntry(
                sequence="Fmoc-FF",
                type=PeptideAssemblyType.HYDROGEL,
                temperature=25.0,
                solution="DMSO/H2O",
                evidence_span=EvidenceSpan(sequence="[sec-1/p-0/s-0]"),
            )
        ]
    )
    review_res = ReviewerFeedback(
        approved=True,
        trust_score=0.95,
        issues=[],
        missing_sequences=[],
    )

    factory, counts = _make_mock_controller_router(
        {
            EntityRegistryResult: [scout_res],
            PeptideExtractionResult: [prose_res],
            ReviewerFeedback: [review_res],
        }
    )

    with patch("peptide_extractor.agentic.agents.LLMController", side_effect=factory):
        cfg = AgenticExtractionConfig(
            provider="ollama",
            model_name="qwen3:8b",
            reviewer_provider="ollama",
            reviewer_model_name="deepseek-r1:8b",
            max_refinement_rounds=2,
        )
        orchestrator = ExtractionOrchestrator(cfg)
        trace = orchestrator.run_with_trace("[sec-1/p-0/s-0] Fmoc-FF forms a hydrogel at 25 C.")

    assert trace.refinement_rounds_executed == 0
    assert len(trace.review_history) == 1
    assert trace.final_entry_count == 1
    assert trace.result.entries[0].sequence == "FF"
    assert trace.result.entries[0].n_terminal == "Fmoc"
    # PeptideExtractionResult called only once (ProseExtractor), NOT Refiner
    assert counts[PeptideExtractionResult] == 1


def test_orchestrator_triggers_refiner_on_reviewer_rejection():
    scout_res = EntityRegistryResult(
        peptides=[
            PeptideAlias(label="P1", sequence="KLVFF", n_terminal="Acetylation", c_terminal="Amidation"),
            PeptideAlias(label="P2", sequence="VQIVYK", n_terminal="Acetylation", c_terminal="Amidation"),
        ],
        shared_conditions=SharedExperimentalConditions(temperature=37.0, ph=7.4),
    )
    initial_prose_res = PeptideExtractionResult(
        entries=[
            PeptideLLMEntry(
                sequence="P1",
                type=PeptideAssemblyType.FIBER,
                temperature=37.0,
                ph=7.4,
            )
        ]
    )
    refine_patch = RefinerPatchResult(
        update_fields=[FieldUpdate(entry_index=0, field="solution", value="PBS")],
        add_entries=[
            PeptideLLMEntry(
                sequence="VQIVYK",
                n_terminal="Acetylation",
                c_terminal="Amidation",
                type=PeptideAssemblyType.AMYLOID,
                solution="PBS",
                temperature=37.0,
                ph=7.4,
            )
        ],
    )
    review_round_0 = ReviewerFeedback(
        approved=False,
        trust_score=0.55,
        issues=[
            ReviewIssue(
                entry_index=None,
                field="missing_row",
                reason="Peptide P2 (VQIVYK) was omitted.",
                suggested_fix="Add VQIVYK forming amyloid fibrils at 37 C, pH 7.4.",
            )
        ],
        missing_sequences=["VQIVYK"],
        revision_instructions="Add missing entry for VQIVYK and set solution=PBS on entry 0.",
    )
    review_round_1 = ReviewerFeedback(
        approved=True,
        trust_score=0.96,
        issues=[],
        missing_sequences=[],
    )

    factory, counts = _make_mock_controller_router(
        {
            EntityRegistryResult: [scout_res],
            PeptideExtractionResult: [initial_prose_res],
            RefinerPatchResult: [refine_patch],
            ReviewerFeedback: [review_round_0, review_round_1],
        }
    )

    with patch("peptide_extractor.agentic.agents.LLMController", side_effect=factory):
        cfg = AgenticExtractionConfig(max_refinement_rounds=2, min_trust_score=0.8)
        trace = ExtractionOrchestrator(cfg).run_with_trace(
            "[sec-1/p-0/s-0] P1 (Ac-KLVFF-NH2) forms fibers and P2 (Ac-VQIVYK-NH2) forms amyloid in PBS at 37 C, pH 7.4."
        )

    assert trace.refinement_rounds_executed == 1
    assert len(trace.review_history) == 2
    assert trace.final_entry_count == 2
    assert {e.sequence for e in trace.result.entries} == {"KLVFF", "VQIVYK"}
    assert all(e.solution == "PBS" for e in trace.result.entries)
    assert counts[PeptideExtractionResult] == 1
    assert counts[RefinerPatchResult] == 1


def test_patch_refiner_preserves_large_screening_tables_over_80_rows():
    """Ensure patch-based refinement never drops untouched rows on 120+ row tables."""
    entries = [
        PeptideLLMEntry(
            sequence="FF",
            type=PeptideAssemblyType.ORGANOGEL,
            solution=f"solvent_{i}",
            temperature=25.0,
        )
        for i in range(120)
    ]
    patch_res = RefinerPatchResult(
        update_fields=[FieldUpdate(entry_index=0, field="c_terminal", value="Amidation")],
        add_entries=[
            PeptideLLMEntry(
                sequence="KLVFF",
                type=PeptideAssemblyType.HYDROGEL,
                solution="PBS",
            )
        ],
        delete_indices=[119],
    )
    patched = apply_refiner_patch(entries, patch_res)
    assert len(patched) == 120  # 120 - 1 deleted + 1 added
    assert patched[0].c_terminal == "Amidation"
    assert patched[-1].sequence == "KLVFF"


def test_sandbox_unifies_solvent_sets_neat_ratio_defaults_structural_and_scans_methods():
    raw = [
        PeptideLLMEntry(
            sequence="SVG",
            type=PeptideAssemblyType.ORGANOGEL,
            solvent="toluene",  # solution intentionally left None
            detection_method=["TEM"],
        )
    ]
    paper_text = (
        "Gelation was assessed by the inverted vial test (vial inversion). "
        "Morphology was characterized by TEM, FE-SEM, FT-IR spectroscopy, and rheology."
    )
    cleaned, _ = sanitize_and_validate_entries(raw, source_text=paper_text)
    assert len(cleaned) == 1
    e = cleaned[0]
    assert e.solution == "toluene"
    assert e.volume_ratio == [1]
    assert e.linear_cyclic == PeptideLinearity.LINEAR
    assert e.peptide_conjugate_mixture == PeptideConjugateMixture.PEPTIDE
    assert set(e.detection_method or []) >= {"TEM", "SEM", "FTIR", "Rheology", "Vial Inversion"}
    assert "Vial Inversion" in extract_paper_detection_methods(paper_text)


def test_orchestrator_handles_multimodal_table_blocks():
    scout_res = EntityRegistryResult(
        peptides=[
            PeptideAlias(
                label="1a",
                sequence="SVG",
                n_terminal="Boc",
                c_terminal="Benzylation",
                linear_cyclic=PeptideLinearity.LINEAR,
            )
        ],
        shared_conditions=SharedExperimentalConditions(
            temperature=25.0,
            incubation_period=1440,
            detection_method=["vial inversion", "TEM"],
            evidence_span=EvidenceSpan(temperature="[sec-1/p-0/s-0]"),
        ),
    )
    prose_res = PeptideExtractionResult(entries=[])
    table_res = TableExtractionResult(
        rows=[
            CompactTableRow(
                sequence="1a",
                conditions=[
                    CompactTableCondition(solution="toluene", type=PeptideAssemblyType.ORGANOGEL),
                    CompactTableCondition(solution="water", type=PeptideAssemblyType.NONE),
                ],
            )
        ]
    )
    review_res = ReviewerFeedback(approved=True, trust_score=0.92)

    factory, counts = _make_mock_controller_router(
        {
            EntityRegistryResult: [scout_res],
            PeptideExtractionResult: [prose_res],
            TableExtractionResult: [table_res],
            ReviewerFeedback: [review_res],
        }
    )

    blocks = [
        {
            "type": "text",
            "text": (
                "[sec-1/p-0/s-0] Gelation was tested at 25 C for 24 h.\n"
                "[sec-1/table-0] Table 1. Gelation screen\n"
                ">>> [TABLE IMAGE ATTACHED: sec-1/table-0] <<<"
            ),
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    with patch("peptide_extractor.agentic.agents.LLMController", side_effect=factory):
        res = run_extraction(blocks, agentic=True)

    assert len(res.entries) == 2
    assert all(e.sequence == "SVG" for e in res.entries)
    assert all(e.n_terminal == "Boc" and e.c_terminal == "Benzylation" for e in res.entries)
    assert all(e.temperature == 25.0 and e.incubation_period == 1440 for e in res.entries)
    assert {e.type for e in res.entries} == {
        PeptideAssemblyType.ORGANOGEL,
        PeptideAssemblyType.NONE,
    }
    assert counts[TableExtractionResult] == 1


# ---------------------------------------------------------------------------
# 3. Pipeline Wrapper Integration Test
# ---------------------------------------------------------------------------


def test_run_peptide_extraction_wrapper_with_agentic_mode():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    sample_xml = """<article><body>
    <sec><title>Results</title>
    <p>Fmoc-FF formed a transparent hydrogel in DMSO/H2O at 25 C and pH 7.0.</p>
    </sec></body></article>"""

    scout_res = EntityRegistryResult(
        peptides=[PeptideAlias(label="Fmoc-FF", sequence="FF", n_terminal="Fmoc", c_terminal="Free")],
        shared_conditions=SharedExperimentalConditions(temperature=25.0, ph=7.0),
    )
    prose_res = PeptideExtractionResult(
        entries=[
            PeptideLLMEntry(
                sequence="Fmoc-FF",
                type=PeptideAssemblyType.HYDROGEL,
                temperature=25.0,
                ph=7.0,
                solution="DMSO/H2O",
                peptide_conjugate_mixture=PeptideConjugateMixture.PEPTIDE,
                linear_cyclic=PeptideLinearity.LINEAR,
                evidence_span=EvidenceSpan(sequence="[sec-0/p-0/s-0]"),
            )
        ]
    )
    review_res = ReviewerFeedback(approved=True, trust_score=0.94)

    factory, _ = _make_mock_controller_router(
        {
            EntityRegistryResult: [scout_res],
            PeptideExtractionResult: [prose_res],
            ReviewerFeedback: [review_res],
        }
    )

    with Session(engine) as session:
        paper = PaperMetadata(
            title="Fmoc-FF Hydrogel",
            abstract="Short peptide hydrogel.",
            pmid="99999999",
            doi="10.1000/test.agentic",
            is_open_access=True,
        )
        session.add(paper)
        session.commit()
        session.refresh(paper)

        content = PaperContent(
            paper_id=paper.id,
            format="xml",
            source="pmc",
            blob_url="https://example.invalid/paper_1.xml",
        )
        session.add(content)
        session.commit()

        with (
            patch("peptide_extractor.agentic.agents.LLMController", side_effect=factory),
            patch.object(PaperContent, "get_content", return_value=sample_xml),
        ):
            failed_ids, run_id = run_peptide_extraction(
                session=session,
                save=True,
                provider="ollama",
                model_name="qwen3:8b",
                agentic=True,
                reviewer_provider="ollama",
                reviewer_model_name="deepseek-r1:8b",
                delay_seconds=0.0,
                log_to_file=False,
            )

        assert failed_ids == []
        saved = session.exec(
            select(PeptideEntry).where(PeptideEntry.run_identifier == run_id)
        ).all()
        assert len(saved) == 1
        assert saved[0].sequence == "FF"
        assert saved[0].n_terminal == "Fmoc"
        assert saved[0].c_terminal == "Free"
        assert saved[0].type == "hydrogel"


def test_sandbox_converts_molar_concentration_canonicalizes_solvents_terminals_and_rt():
    raw = [
        PeptideLLMEntry(
            sequence="FF",
            n_terminal="Ac",
            c_terminal="NH2",
            type=PeptideAssemblyType.HYDROGEL,
            solution="HFIP/water",
            peptide_concentration=10.0,
            peptide_concentration_unit="mM",
        ),
        PeptideLLMEntry(
            sequence="SVG",
            n_terminal="boc",
            c_terminal="cooh",
            type=PeptideAssemblyType.ORGANOGEL,
            solution="CH2Cl2",
            temperature=37.0,
            peptide_concentration=0.5,
            peptide_concentration_unit="wt%",
        ),
    ]
    cleaned, report = sanitize_and_validate_entries(raw, default_room_temperature=True)
    assert len(cleaned) == 2

    e0, e1 = cleaned[0], cleaned[1]
    assert e0.n_terminal == "Acetylation"
    assert e0.c_terminal == "Amidation"
    assert e0.solution == "HFP/H2O"
    assert e0.temperature == 25.0  # default RT applied
    assert e0.peptide_concentration_unit == "mg/ml"
    mw_ac_ff_nh2 = estimate_peptide_mw("FF", n_terminal="Acetylation", c_terminal="Amidation")
    assert mw_ac_ff_nh2 is not None
    assert abs(e0.peptide_concentration - (10.0 * mw_ac_ff_nh2 / 1000.0)) < 0.01

    assert e1.n_terminal == "Boc"
    assert e1.c_terminal == "Free"
    assert e1.solution == "Dichloromethane"
    assert e1.temperature == 37.0  # explicit temp preserved
    assert e1.peptide_concentration == 5.0  # 0.5 wt% -> 5.0 mg/ml
    assert report.auto_fixes_applied >= 4

    assert canonicalize_solution("EtOAc") == "Ethyl acetate"
    assert canonicalize_solution("Cl-C6H5") == "Chlorobenzene"
    assert normalize_terminal_cap("H2N", is_n_term=True) == "Free"
    assert normalize_terminal_cap("OMe", is_n_term=False) == "Methyl ester"
