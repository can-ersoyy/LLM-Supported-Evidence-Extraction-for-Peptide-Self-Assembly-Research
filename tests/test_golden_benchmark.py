import pytest

from peptide_extractor.db import ExtractionMode, PeptideEntry
from peptide_extractor.golden_benchmark import (
    PEPTIDEMINER_NUMERICAL_BOUNDS,
    PEPTIDEMINER_TEST_PMIDS,
    compare_runs,
    compare_runs_peptideminer,
    extract_detection_tokens,
    extract_solvent_tokens,
    format_peptideminer_comparison,
    load_peptideminer_paper_baselines,
    normalize_c_terminal,
    normalize_conjugate_partner,
    normalize_n_terminal,
    normalize_phase,
    normalize_volume_ratio,
    parse_volume_ratio_float,
)


def _entry(**kwargs) -> PeptideEntry:
    defaults = dict(
        pmid="12345",
        sequence="FF",
        length=2,
        extraction_mode=ExtractionMode.MANUAL,
    )
    defaults.update(kwargs)
    return PeptideEntry(**defaults)


def test_terminal_normalization():
    assert normalize_n_terminal("Free") == "free"
    assert normalize_n_terminal("NH2") == "free"
    assert normalize_n_terminal("H") == "free"
    assert normalize_n_terminal("Ac") == "acetylation"

    assert normalize_c_terminal("Free") == "free"
    assert normalize_c_terminal("COOH") == "free"
    assert normalize_c_terminal("NH2") == "amidation"
    assert normalize_c_terminal("OMe") == "methyl ester"


def test_solvent_and_volume_ratio_normalization():
    assert extract_solvent_tokens("HFP/H2O", "Aqueous dispersion") == {"hfp", "h2o"}
    assert extract_solvent_tokens("water", "1,1,1,3,3,3-hexafluoro-2-propanol") == {"hfp", "h2o"}
    assert normalize_volume_ratio("100") == "1"
    assert normalize_volume_ratio(" 5 / 95 ") == "5/95"


def test_detection_tokens():
    m_tokens = extract_detection_tokens("NMR spectroscopy")
    l_tokens = extract_detection_tokens("CPMAS NMR,Confocal fluorescence microscopy,TEM")
    assert m_tokens == {"nmr"}
    assert m_tokens.issubset(l_tokens)


def test_compare_runs_condition_alignment():
    manual = [
        _entry(
            type="hydrogel",
            n_terminal="Free",
            c_terminal="Free",
            solution="HFP/H2O",
            volume_ratio="5/95",
            ph=7.0,
            detection_method="Transmission Electron Microscopy (TEM), FTIR",
        ),
        _entry(
            type="none",
            n_terminal="Free",
            c_terminal="Free",
            solution="Chloroform",
            volume_ratio="1",
            ph=7.0,
        ),
    ]
    llm = [
        _entry(
            extraction_mode=ExtractionMode.LLM,
            type="hydrogel",
            n_terminal="NH2",
            c_terminal="COOH",
            solution="water",
            solvent="hexafluoro-2-propanol",
            volume_ratio="95/5",
            ph=7.1,
            detection_method="TEM,FT-IR,CD",
        )
    ]

    report = compare_runs(manual, llm)
    assert report.covered_pmids == 1
    assert report.sequence_recall == 1.0
    assert report.sequence_precision == 1.0
    assert report.sequence_f1 == 1.0
    assert report.matched_condition_pairs == 1
    assert report.matched_assembling_pairs == 1
    assert report.matched_none_pairs == 0
    assert report.assembling_condition_recall == 1.0
    assert report.none_condition_recall == 0.0
    assert report.condition_precision == 1.0
    assert report.unmatched_manual_rows == 1
    assert report.field_stats["n_terminal"].normalized_match == 1
    assert report.field_stats["c_terminal"].normalized_match == 1
    assert report.field_stats["solution"].normalized_match == 1
    assert report.field_stats["volume_ratio"].normalized_match == 1
    assert report.field_stats["detection_method"].normalized_match == 1
    assert "PMIDs covered: 1/1" in report.format_report()


def test_none_and_assembling_entries_cannot_match():
    manual = [
        _entry(
            type="none",
            solution="H2O",
            ph=7.0,
            temperature=25.0,
        ),
    ]
    llm = [
        _entry(
            extraction_mode=ExtractionMode.LLM,
            type="fiber",
            solution="H2O",
            ph=7.0,
            temperature=25.0,
        ),
    ]

    report = compare_runs(manual, llm)
    assert report.matched_condition_pairs == 0
    assert report.matched_none_pairs == 0
    assert report.matched_assembling_pairs == 0
    assert report.none_condition_recall == 0.0
    assert report.condition_precision == 0.0
    assert report.unmatched_manual_rows == 1
    assert report.unmatched_llm_rows == 1


def test_similarity_threshold_blocks_unrelated_entries():
    manual = [
        _entry(
            type="hydrogel",
            solution="H2O",
            ph=7.0,
        ),
    ]
    llm = [
        _entry(
            extraction_mode=ExtractionMode.LLM,
            type="fiber",
            solution="DMSO",
            ph=3.0,
        ),
    ]

    report = compare_runs(manual, llm)
    assert report.matched_condition_pairs == 0
    assert report.unmatched_manual_rows == 1
    assert report.unmatched_llm_rows == 1

    report_no_threshold = compare_runs(manual, llm, min_similarity=0.0)
    assert report_no_threshold.matched_condition_pairs == 1


def test_peptideminer_test_pmids_and_volume_ratio_float():
    assert len(PEPTIDEMINER_TEST_PMIDS) == 23
    assert len(set(PEPTIDEMINER_TEST_PMIDS)) == 23
    assert parse_volume_ratio_float("1") == pytest.approx(1.0)
    assert parse_volume_ratio_float("100") == pytest.approx(1.0)
    assert parse_volume_ratio_float("5/95") == pytest.approx(0.05)
    assert parse_volume_ratio_float("50:50") == pytest.approx(0.5)
    assert parse_volume_ratio_float("0.2") == pytest.approx(0.2)
    assert parse_volume_ratio_float(None) is None
    assert parse_volume_ratio_float("invalid") is None


def test_compare_runs_peptideminer_protocol():
    manual = [
        _entry(
            pmid="18070345",
            sequence="FF",
            type="hydrogel",
            n_terminal="Fmoc",
            c_terminal="Free",
            peptide_conjugate_mixture="peptide",
            conjugate_partner=None,
            solution="DMSO/PBS",
            volume_ratio="5/95",
            ph=7.0,
            temperature=37.0,
            peptide_concentration=5.0,
        ),
        _entry(
            pmid="18070345",
            sequence="FF",
            type="hydrogel",
            n_terminal="Fmoc",
            c_terminal="Free",
            peptide_conjugate_mixture="peptide",
            conjugate_partner=None,
            solution="DMSO/PBS",
            volume_ratio="10/90",
            ph=7.0,
            temperature=25.0,
            peptide_concentration=10.0,
        ),
        _entry(
            pmid="18070345",
            sequence="FFK",
            type="tube",
            n_terminal="Acetylation",
            c_terminal="Amidation",
            peptide_conjugate_mixture="peptide",
            conjugate_partner=None,
            solution="HFP/H2O",
            volume_ratio="5/95",
            ph=5.5,
            temperature=25.0,
            peptide_concentration=1.0,
        ),
    ]

    llm = [
        # Direct match on (FF, hydrogel); tie-broken to 2nd manual row (temp=25, conc=10, vr=10/90)
        _entry(
            pmid="18070345",
            extraction_mode=ExtractionMode.LLM,
            sequence="FF",
            type="hydrogel",
            n_terminal="Fmoc",
            c_terminal="COOH",
            peptide_conjugate_mixture="peptide",
            conjugate_partner=None,
            solution="DMSO/PBS",
            volume_ratio="10/90",
            ph=7.5,
            temperature=25.0,
            peptide_concentration=12.0,
        ),
        # No direct (sequence, phase) match: phase is 'fiber' instead of 'tube',
        # should fall back to FFK row; missing temperature should be excluded from MAE.
        _entry(
            pmid="18070345",
            extraction_mode=ExtractionMode.LLM,
            sequence="FFK",
            type="fiber",
            n_terminal="Ac",
            c_terminal="NH2",
            peptide_conjugate_mixture="peptide",
            conjugate_partner=None,
            solution="H2O",  # missing HFP -> False under full solution alignment
            volume_ratio="1",
            ph=5.5,
            temperature=None,
            peptide_concentration=2.0,
        ),
    ]

    report = compare_runs_peptideminer(manual, llm)
    assert report.covered_pmids == 1
    assert report.manual_rows_on_covered_pmids == 3
    assert report.llm_rows_evaluated == 2

    # Categorical: sequence 2/2, phase 1/2, n_terminal norm 2/2, c_terminal norm 2/2,
    # category 2/2, conjugate_partner 2/2 (both null), solution 1/2
    assert report.categorical_stats["sequence"].normalized_accuracy == pytest.approx(1.0)
    assert report.categorical_stats["phase"].normalized_accuracy == pytest.approx(0.5)
    assert report.categorical_stats["n_terminal"].normalized_accuracy == pytest.approx(1.0)
    assert report.categorical_stats["c_terminal"].normalized_accuracy == pytest.approx(1.0)
    assert report.categorical_stats["category"].normalized_accuracy == pytest.approx(1.0)
    assert report.categorical_stats["conjugate_partner"].normalized_accuracy == pytest.approx(1.0)
    assert report.categorical_stats["conjugate_partner"].both_nonnull == 0
    assert report.categorical_stats["solution"].normalized_accuracy == pytest.approx(0.5)

    # Numerical MAE:
    # ph: |7.5-7.0|=0.5 and |5.5-5.5|=0.0 -> raw MAE = 0.25
    assert report.numerical_stats["ph"].evaluated == 2
    assert report.numerical_stats["ph"].raw_mae == pytest.approx(0.25)
    ph_lo, ph_hi = PEPTIDEMINER_NUMERICAL_BOUNDS["ph"]
    assert report.numerical_stats["ph"].normalized_mae == pytest.approx(
        0.25 / (ph_hi - ph_lo)
    )
    # temperature: only 1st pred has temp (25 vs 25 -> 0.0); 2nd pred is None -> excluded
    assert report.numerical_stats["temperature"].evaluated == 1
    assert report.numerical_stats["temperature"].manual_nonnull == 2
    assert report.numerical_stats["temperature"].raw_mae == pytest.approx(0.0)
    # concentration: |12-10|=2.0 and |2-1|=1.0 -> raw MAE = 1.5
    assert report.numerical_stats["concentration"].raw_mae == pytest.approx(1.5)
    # solvent_ratio: |0.1-0.1|=0.0 and |1.0-0.05|=0.95 -> raw MAE = 0.475
    assert report.numerical_stats["solvent_ratio"].raw_mae == pytest.approx(0.475)

    formatted = report.format_report()
    assert "Avg Categorical Accuracy" in formatted
    assert "NormMAE" in formatted
    assert "solvent_ratio" in formatted
    assert len(report.numerical_stats["ph"].normalized_errors) == 2
    assert len(report.numerical_stats["temperature"].raw_errors) == 1


def test_load_peptideminer_paper_baselines():
    baselines = load_peptideminer_paper_baselines()
    assert "PeptideMiner Pretrained (GPT-3.5)" in baselines
    assert "PeptideMiner Fine-tuned (GPT-3.5)" in baselines
    assert len(baselines["PeptideMiner Pretrained (GPT-3.5)"]) == 60
    assert len(baselines["PeptideMiner Fine-tuned (GPT-3.5)"]) == 175
    for entries in baselines.values():
        assert {e.pmid for e in entries} == set(PEPTIDEMINER_TEST_PMIDS)


def test_format_peptideminer_comparison():
    manual = [
        _entry(
            pmid="18070345",
            sequence="FF",
            type="hydrogel",
            n_terminal="Fmoc",
            c_terminal="Free",
            peptide_conjugate_mixture="peptide",
            solution="DMSO/PBS",
            volume_ratio="5/95",
            ph=7.0,
            temperature=37.0,
            peptide_concentration=5.0,
        )
    ]
    llm_a = [
        _entry(
            pmid="18070345",
            extraction_mode=ExtractionMode.LLM,
            sequence="FF",
            type="hydrogel",
            n_terminal="Fmoc",
            c_terminal="Free",
            peptide_conjugate_mixture="peptide",
            solution="DMSO/PBS",
            volume_ratio="5/95",
            ph=7.0,
            temperature=37.0,
            peptide_concentration=5.0,
        )
    ]
    llm_b = [
        _entry(
            pmid="18070345",
            extraction_mode=ExtractionMode.LLM,
            sequence="FF",
            type="fiber",
            n_terminal="Boc",
            c_terminal="Free",
            peptide_conjugate_mixture="peptide",
            solution="H2O",
            volume_ratio="1",
            ph=6.0,
            temperature=25.0,
            peptide_concentration=10.0,
        )
    ]
    table = format_peptideminer_comparison(
        {
            "Model A": compare_runs_peptideminer(manual, llm_a),
            "Model B": compare_runs_peptideminer(manual, llm_b),
        }
    )
    assert "Model A" in table
    assert "Model B" in table
    assert "Avg Categorical Acc" in table
    assert "ph NormMAE (Raw)" in table


def test_extended_peptideminer_normalizers():
    # N-terminal: PeptideMiner 'Napthalene' typo + derivatives + Ferrocene
    assert normalize_n_terminal("Napthalene") == "naphthalene"
    assert normalize_n_terminal("Naphthalene derivative") == "naphthalene"
    assert normalize_n_terminal("2-naphthaleneacetic acid") == "naphthalene"
    assert normalize_n_terminal("6-bromo-2-naphthylmethoxycarbonyl") == "naphthalene"
    assert normalize_n_terminal("Nap") == "naphthalene"
    assert normalize_n_terminal("Fc") == "ferrocene"

    # C-terminal: PeptideMiner 'Methoxy' == 'OMe' == 'Methyl ester'
    assert normalize_c_terminal("Methoxy") == "methyl ester"
    assert normalize_c_terminal("OCH2Ph") == "benzylation"

    # Conjugate partner: 'Napthalene' typo, naphthols, and NDI
    assert normalize_conjugate_partner("Napthalene") == "naphthalene"
    assert normalize_conjugate_partner("6-cyano-2-naphthol") == "naphthalene"
    assert (
        normalize_conjugate_partner("NDI : 1,4,5,8-naphthalenetetracarboxylic acid diimide")
        == "ndi"
    )
    assert normalize_conjugate_partner("Octyl-NDI") == "ndi"

    # Phase synonyms
    assert normalize_phase("nanofibers") == "fiber"
    assert normalize_phase("liquid solution") == "none"
    assert normalize_phase("hydrogels") == "hydrogel"
    assert normalize_phase("nanovesicles") == "vesicle"

    # Solvents: PB, TFE, isopropanol, EtOAc
    assert extract_solvent_tokens("PB") == {"pb"}
    assert extract_solvent_tokens("sodium phosphate buffer") == {"pb"}
    assert extract_solvent_tokens("TFE/H2O") == {"tfe", "h2o"}
    assert extract_solvent_tokens("2-propanol") == {"isopropanol"}
    assert extract_solvent_tokens("EtOAc") == {"ethyl acetate"}

