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


# ---------------------------------------------------------------------------
# 5. PDF Figure Cropping & FigureExtractorAgent Tests
# ---------------------------------------------------------------------------


def test_split_blocks_for_visual_extraction_separates_tables_and_figures():
    from peptide_extractor.extraction import _split_blocks_for_visual_extraction

    blocks = [
        {
            "type": "text",
            "text": (
                "[sec-1] Results\n"
                "[sec-1/table-0] Table 1. Gelation screening.\n"
                "\t>>> [TABLE IMAGE ATTACHED: sec-1/table-0] <<<\n"
            ),
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,TABLE0"}},
        {
            "type": "text",
            "text": (
                "[sec-1/fig-0] Figure 1. Phase diagram of Boc-FF in ethanol/water.\n"
                "\t>>> [FIGURE IMAGE ATTACHED: sec-1/fig-0] <<<\n"
            ),
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,FIG0"}},
        {
            "type": "text",
            "text": "=== APPENDIX FIGURE IMAGE: [sec-2/fig-1] ===\nCaption: Scheme 1. Structures.\n",
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,FIG1"}},
    ]

    prose, table_items, figure_items = _split_blocks_for_visual_extraction(blocks)
    assert "Results" in prose
    assert len(table_items) == 1
    assert table_items[0][0] == "sec-1/table-0"
    assert "Table 1" in table_items[0][1]

    assert len(figure_items) == 2
    assert figure_items[0][0] == "sec-1/fig-0"
    assert "Phase diagram" in figure_items[0][1]
    assert figure_items[1][0] == "sec-2/fig-1"
    assert "Scheme 1" in figure_items[1][1]


def test_crop_figures_from_pdf_expands_caption_only_grobid_box():
    import fitz

    from peptide_extractor.parsers import ImageNode, RootNode
    from peptide_extractor.table_renderer import crop_figures_from_pdf

    # Build a synthetic 1-page PDF with a vector plot box above a thin caption
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    # Vector plot graphic from y=100..260
    page.draw_rect(fitz.Rect(80, 100, 520, 260), color=(0, 0, 1), fill=(0.9, 0.9, 1))
    # Thin caption below at y=280..300 (height=20 < 90)
    page.insert_text((80, 295), "Figure 1. Phase diagram of Boc-FF.", fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()

    root = RootNode("", "", None)
    fig = ImageNode(
        id="fig-0",
        content="Figure 1. Phase diagram of Boc-FF.",
        parent_id="",
        caption="Figure 1. Phase diagram of Boc-FF.",
        coords="1,80,280,440,20",  # caption-only GROBID box!
    )
    root.children.append(fig)

    count = crop_figures_from_pdf(root, pdf_bytes, dpi=72)
    assert count == 1
    assert fig.image_bytes is not None

    from io import BytesIO
    from PIL import Image

    cropped = Image.open(BytesIO(fig.image_bytes))
    # Expanded box must cover the vector graphic above (y=100..300 -> height >= 190 px at 72 dpi)
    assert cropped.height >= 190


def test_orchestrator_runs_figure_extractor_and_attaches_schemes_to_scout():
    blocks = [
        {
            "type": "text",
            "text": (
                "[sec-0/fig-0] Scheme 1. Chemical structures of peptide amphiphiles 1-2.\n"
                "\t>>> [FIGURE IMAGE ATTACHED: sec-0/fig-0] <<<\n"
            ),
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,SCHEME"}},
        {
            "type": "text",
            "text": (
                "[sec-1/table-0] Table 1. Summary.\n"
                "\t>>> [TABLE IMAGE ATTACHED: sec-1/table-0] <<<\n"
            ),
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,TABLE"}},
        {
            "type": "text",
            "text": (
                "[sec-1/fig-1] Figure 1. Phase diagram of Boc-FF in ethanol/water.\n"
                "\t>>> [FIGURE IMAGE ATTACHED: sec-1/fig-1] <<<\n"
            ),
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,PHASE"}},
    ]

    registry = EntityRegistryResult(
        peptides=[
            PeptideAlias(
                label="Boc-FF",
                sequence="FF",
                n_terminal="Boc",
                c_terminal="Free",
            )
        ],
        shared_conditions=SharedExperimentalConditions(temperature=25.0),
    )

    scout_mock = MagicMock()
    scout_mock.run.return_value = registry

    prose_mock = MagicMock()
    prose_mock.run.return_value = PeptideExtractionResult(entries=[])

    table_mock = MagicMock()
    table_mock.run.return_value = TableExtractionResult(
        rows=[
            CompactTableRow(
                sequence="FF",
                n_terminal="Boc",
                c_terminal="Free",
                conditions=[
                    CompactTableCondition(
                        type=PeptideAssemblyType.TUBE,
                        solution="Ethanol/H2O",
                        volume_ratio=[10, 90],
                        peptide_concentration=2.0,
                    )
                ],
            )
        ]
    )

    figure_mock = MagicMock()
    # Scheme figure returns empty rows; phase-diagram figure returns 2 points (1 sphere + 1 none)
    figure_mock.run.side_effect = [
        TableExtractionResult(rows=[]),
        TableExtractionResult(
            rows=[
                CompactTableRow(
                    sequence="FF",
                    n_terminal="Boc",
                    c_terminal="Free",
                    conditions=[
                        CompactTableCondition(
                            type=PeptideAssemblyType.NANOSPHERE,
                            solution="Ethanol/H2O",
                            volume_ratio=[50, 50],
                            peptide_concentration=4.0,
                        ),
                        CompactTableCondition(
                            type=PeptideAssemblyType.NONE,
                            solution="Ethanol/H2O",
                            volume_ratio=[80, 20],
                            peptide_concentration=0.5,
                        ),
                    ],
                )
            ]
        ),
    ]

    cfg = AgenticExtractionConfig(enable_reviewer=False)
    with (
        patch("peptide_extractor.agentic.orchestrator.EntityScoutAgent", return_value=scout_mock),
        patch("peptide_extractor.agentic.orchestrator.ProseExtractorAgent", return_value=prose_mock),
        patch("peptide_extractor.agentic.orchestrator.TableExtractorAgent", return_value=table_mock),
        patch("peptide_extractor.agentic.orchestrator.FigureExtractorAgent", return_value=figure_mock),
    ):
        trace = ExtractionOrchestrator(cfg).run_with_trace(blocks, filename="paper_44.pdf")

    # Scheme 1 was attached to EntityScoutAgent
    scout_kwargs = scout_mock.run.call_args.kwargs
    assert scout_kwargs["scheme_images"] is not None
    assert len(scout_kwargs["scheme_images"]) == 1
    assert scout_kwargs["scheme_images"][0][0] == "sec-0/fig-0"

    # TableExtractorAgent ran ONLY on the table, FigureExtractorAgent ran on BOTH figures
    assert table_mock.run.call_count == 1
    assert figure_mock.run.call_count == 2
    assert trace.figures_processed == 2

    # Final entries include 1 table row + 2 figure phase-diagram rows
    assert trace.final_entry_count == 3
    fig_entries = [
        e for e in trace.result.entries if e.evidence_span and e.evidence_span.sequence == "[sec-1/fig-1]"
    ]
    assert len(fig_entries) == 2
    assert {e.type for e in fig_entries} == {
        PeptideAssemblyType.NANOSPHERE,
        PeptideAssemblyType.NONE,
    }


def test_merge_table_does_not_bleed_shared_solvent_when_cell_overrides_solution():
    from peptide_extractor.extraction import merge_table_result_with_shared

    shared = SharedExperimentalConditions(solvent="toluene", temperature=25.0)
    table_res = TableExtractionResult(
        table_defaults=SharedExperimentalConditions(peptide_concentration=100.0),
        rows=[
            CompactTableRow(
                sequence="AAA",
                conditions=[
                    CompactTableCondition(
                        type=PeptideAssemblyType.NONE,
                        solution="Chloroform",
                    )
                ],
            )
        ],
    )
    merged = merge_table_result_with_shared(table_res, shared, "table-0")
    assert len(merged) == 1
    assert merged[0].solution == "Chloroform"
    assert merged[0].solvent is None
    assert merged[0].peptide_concentration == 100.0


def test_dehydrophenylalanine_and_t_butyloxycarbonyl_normalization():
    seq, n_cap, c_cap, _, _ = split_sequence_and_caps("H-Glu-ΔPhe-OH")
    assert seq == "EF"
    assert n_cap == "Free"
    assert c_cap == "Free"
    assert normalize_terminal_cap("t-butyloxycarbonyl", is_n_term=True) == "Boc"


def test_crop_figures_from_pdf_caption_fallback_when_coords_missing():
    import fitz
    from io import BytesIO
    from PIL import Image

    from peptide_extractor.parsers import ImageNode, RootNode
    from peptide_extractor.table_renderer import crop_figures_from_pdf

    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.draw_rect(fitz.Rect(60, 80, 540, 420), color=(0, 0, 1), fill=(0.9, 0.9, 1))
    page.insert_text((60, 450), "Figure 1 | Phase diagram of Boc-FF in ethanol/water.", fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()

    root = RootNode("", "", None)
    fig = ImageNode(
        id="fig-0",
        content="Figure 1 | Phase diagram of Boc-FF in ethanol/water.",
        parent_id="",
        caption="Figure 1 | Phase diagram of Boc-FF in ethanol/water.",
        coords=None,  # GROBID omitted coords completely
    )
    root.children.append(fig)

    count = crop_figures_from_pdf(root, pdf_bytes, dpi=72)
    assert count == 1
    assert fig.image_bytes is not None
    cropped = Image.open(BytesIO(fig.image_bytes))
    assert cropped.height >= 350


def test_peptideminer_protocol_opt_in_prompt_and_ph_estimation():
    from peptide_extractor.agentic.agents import (
        PEPTIDEMINER_PROTOCOL_PROMPT_ADDENDUM,
        PROSE_EXTRACTOR_SYSTEM_PROMPT,
        _with_peptideminer_addendum,
    )
    from peptide_extractor.agentic.chemistry_sandbox import (
        estimate_ph_from_solution,
        sanitize_and_validate_entries,
    )

    # 1. Default config leaves prompts untouched
    default_cfg = AgenticExtractionConfig()
    assert default_cfg.peptideminer_protocol is False
    assert (
        _with_peptideminer_addendum(PROSE_EXTRACTOR_SYSTEM_PROMPT, default_cfg)
        == PROSE_EXTRACTOR_SYSTEM_PROMPT
    )

    # 2. Opt-in config appends PeptideMiner protocol addendum
    pm_cfg = AgenticExtractionConfig(peptideminer_protocol=True)
    pm_prompt = _with_peptideminer_addendum(PROSE_EXTRACTOR_SYSTEM_PROMPT, pm_cfg)
    assert pm_prompt.endswith(PEPTIDEMINER_PROTOCOL_PROMPT_ADDENDUM)
    assert "UNIFORM RANGE SAMPLING" in pm_prompt
    assert "1.1x, 1.2x" in pm_prompt

    # 3. Solvent-based pH estimation (Principle 4)
    assert estimate_ph_from_solution("DMSO/PBS") == 7.4
    assert estimate_ph_from_solution("Methanol/H2O") == 8.3
    assert estimate_ph_from_solution("H2O") == 7.0
    assert estimate_ph_from_solution("acetic acid") == 4.0
    assert estimate_ph_from_solution("NaOH/H2O") == 11.0
    assert estimate_ph_from_solution(None) is None

    # 4. Sandbox respects opt-in estimate_solvent_ph without affecting default behavior
    raw = [
        PeptideLLMEntry(
            sequence="FF",
            type=PeptideAssemblyType.HYDROGEL,
            solution="Methanol/H2O",
            ph=None,
            temperature=None,
        )
    ]
    cleaned_default, _ = sanitize_and_validate_entries(
        raw, default_room_temperature=False, estimate_solvent_ph=False
    )
    assert cleaned_default[0].ph is None
    assert cleaned_default[0].temperature is None

    cleaned_pm, _ = sanitize_and_validate_entries(
        raw, default_room_temperature=True, estimate_solvent_ph=True
    )
    assert cleaned_pm[0].ph == 8.3
    assert cleaned_pm[0].temperature == 25.0


def test_hybrid_protocol_prompt_and_deterministic_hygiene():
    from peptide_extractor.agentic.agents import (
        HYBRID_PROTOCOL_PROMPT_ADDENDUM,
        PROSE_EXTRACTOR_SYSTEM_PROMPT,
        _with_peptideminer_addendum,
    )
    from peptide_extractor.agentic.chemistry_sandbox import sanitize_and_validate_entries

    cfg = AgenticExtractionConfig(hybrid_protocol=True)
    prompt = _with_peptideminer_addendum(PROSE_EXTRACTOR_SYSTEM_PROMPT, cfg)
    assert prompt.endswith(HYBRID_PROTOCOL_PROMPT_ADDENDUM)
    assert "Do NOT synthetically sample ranges" in prompt
    assert "leave `ph: null`" in prompt

    raw = [
        PeptideLLMEntry(
            sequence="FF",
            type=PeptideAssemblyType.HYDROGEL,
            n_terminal=None,
            c_terminal=None,
            peptide_conjugate_mixture=PeptideConjugateMixture.CONJUGATE,
            conjugate_partner="Fmoc",
            solution="H2O",
            ph=None,
            temperature=None,
        )
    ]
    cleaned, _ = sanitize_and_validate_entries(
        raw,
        default_room_temperature=True,
        estimate_solvent_ph=False,
        strict_terminal_and_conjugate_hygiene=True,
    )
    assert cleaned[0].n_terminal == "Free"
    assert cleaned[0].c_terminal == "Free"
    assert cleaned[0].conjugate_partner is None
    assert cleaned[0].peptide_conjugate_mixture == PeptideConjugateMixture.PEPTIDE
    assert cleaned[0].ph is None
    assert cleaned[0].temperature == 25.0
