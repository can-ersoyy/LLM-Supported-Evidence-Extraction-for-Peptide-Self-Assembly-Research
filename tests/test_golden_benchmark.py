from peptide_extractor.db import ExtractionMode, PeptideEntry
from peptide_extractor.golden_benchmark import (
    compare_runs,
    extract_detection_tokens,
    extract_solvent_tokens,
    normalize_c_terminal,
    normalize_n_terminal,
    normalize_volume_ratio,
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

