"""Benchmarking and comparison utilities for LLM extractions vs the golden dataset."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Optional, Sequence

from sqlmodel import Session, select

from peptide_extractor.db import ExtractionMode, PeptideEntry, get_session_context

DEV_BENCHMARK_PMIDS: tuple[str, ...] = (
    "20460796",
    "28589640",
    "29303206",
    "18052148",
    "25391268",
    "28767108",
    "25068387",
    "25229206",
    "25694059",
    "18358051",
)

# Exact 23 held-out test papers from lamm-mit/PeptideMiner/data/testset.jsonl
# (Yang et al., arXiv:2411.05421v1, 316 ground-truth rows total).
PEPTIDEMINER_TEST_PMIDS: tuple[str, ...] = (
    "30452865",
    "21995651",
    "22005767",
    "18035858",
    "22159641",
    "25391268",
    "20356006",
    "20695592",
    "21833387",
    "28518205",
    "24837840",
    "24895323",
    "16430299",
    "22129632",
    "25068387",
    "25694059",
    "23871602",
    "24700146",
    "19705843",
    "22955637",
    "30104564",
    "28994295",
    "30267640",
)

# Min/max bounds across all 1,017 PeptideMiner manual entries for [0, 1] min-max
# normalized MAE (matching Yang et al. Fig. 2b & Fig. 4d).
PEPTIDEMINER_NUMERICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "ph": (3.0, 12.0),
    "temperature": (-80.0, 200.0),
    "peptide_concentration": (0.005, 144.0),
    "volume_ratio": (0.01, 1.0),
}

_N_TERM_FREE = {"free", "nh2", "h2n", "h", "-h", "unmodified", "none"}
_C_TERM_FREE = {"free", "cooh", "oh", "-oh", "unmodified", "none"}

_N_TERM_ALIASES = {
    "ac": "acetylation",
    "acetyl": "acetylation",
    "acetylation": "acetylation",
    "fmoc": "fmoc",
    "9-fluorenylmethoxycarbonyl": "fmoc",
    "boc": "boc",
    "t-boc": "boc",
    "tert-butyloxycarbonyl": "boc",
    "t-butyloxycarbonyl": "boc",
    "cbz": "cbz",
    "z": "cbz",
    "nap": "naphthalene",
    "2-nap": "naphthalene",
    "napthalene": "naphthalene",
    "naphthalene": "naphthalene",
    "ferrocene": "ferrocene",
    "fc": "ferrocene",
    "chlorination": "chlorination",
}

_N_TERM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"nap(?:hth|th)(?:alene|yl|ol)|(?:^|[\s-])nap(?:$|[\s-])"), "naphthalene"),
    (re.compile(r"fluorenylmethoxycarbonyl|\bfmoc\b"), "fmoc"),
    (re.compile(r"butyloxycarbonyl|\bt?-?boc\b"), "boc"),
    (re.compile(r"\bferrocen|\bfc\b"), "ferrocene"),
    (re.compile(r"\bacet(?:yl|ylation)\b"), "acetylation"),
]

_C_TERM_ALIASES = {
    "nh2": "amidation",
    "conh2": "amidation",
    "amide": "amidation",
    "amidation": "amidation",
    "ome": "methyl ester",
    "och3": "methyl ester",
    "methoxy": "methyl ester",
    "methyl ester": "methyl ester",
    "oet": "ethyl ester",
    "ethyl ester": "ethyl ester",
    "obn": "benzylation",
    "och2ph": "benzylation",
    "obzl": "benzylation",
    "benzyl ester": "benzylation",
    "benzylation": "benzylation",
    "n-hexadecylamine": "n-hexadecylamine",
    "hexadecylamine": "n-hexadecylamine",
}

_CONJUGATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bndi\b|naphthalenetetracarboxylic|naphthalene\s+diimide"), "ndi"),
    (re.compile(r"nap(?:hth|th)(?:alene|yl|ol)|(?:^|[\s-])nap(?:$|[\s-])"), "naphthalene"),
    (re.compile(r"\bferrocen|\bfc\b"), "ferrocene"),
    (re.compile(r"poly\s*\(?\s*ethylene\s+glycol\s*\)?|\bpeg\b"), "peg"),
]

_PHASE_ALIASES = {
    "none": "none",
    "no assembly": "none",
    "no-assembly": "none",
    "liquid solution": "none",
    "solution": "none",
    "soluble": "none",
    "clear solution": "none",
    "hydrogel": "hydrogel",
    "hydrogels": "hydrogel",
    "gel": "hydrogel",
    "gels": "hydrogel",
    "organogel": "organogel",
    "organogels": "organogel",
    "fiber": "fiber",
    "fibers": "fiber",
    "fibril": "fiber",
    "fibrils": "fiber",
    "nanofiber": "fiber",
    "nanofibers": "fiber",
    "nanofibril": "fiber",
    "nanofibrils": "fiber",
    "filament": "fiber",
    "filaments": "fiber",
    "tube": "tube",
    "tubes": "tube",
    "nanotube": "tube",
    "nanotubes": "tube",
    "microtube": "tube",
    "microtubes": "tube",
    "nanosphere": "nanosphere",
    "nanospheres": "nanosphere",
    "sphere": "nanosphere",
    "spheres": "nanosphere",
    "microsphere": "nanosphere",
    "microspheres": "nanosphere",
    "ribbon": "ribbon",
    "ribbons": "ribbon",
    "nanoribbon": "ribbon",
    "nanoribbons": "ribbon",
    "tape": "ribbon",
    "tapes": "ribbon",
    "nanotape": "ribbon",
    "nanotapes": "ribbon",
    "belt": "ribbon",
    "belts": "ribbon",
    "vesicle": "vesicle",
    "vesicles": "vesicle",
    "nanovesicle": "vesicle",
    "nanovesicles": "vesicle",
    "particle": "particle",
    "particles": "particle",
    "nanoparticle": "particle",
    "nanoparticles": "particle",
    "microparticle": "particle",
    "microparticles": "particle",
    "amyloid": "amyloid",
    "amyloids": "amyloid",
}

_SOLVENT_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:1,1,1,3,3,3-)?hexafluoro-?(?:2-)?(?:propanol|isopropanol)\b|\bhfip\b|\bhfp\b"
        ),
        "hfp",
    ),
    (re.compile(r"\b(?:2,2,2-)?trifluoroethanol\b|\btfe\b"), "tfe"),
    (re.compile(r"\bdimethyl\s*sulfoxide\b|\bdimethylsulfoxide\b|\bdmso\b"), "dmso"),
    (re.compile(r"\bdimethyl\s*formamide\b|\bdimethylformamide\b|\bdmf\b"), "dmf"),
    (re.compile(r"\bphosphate[- ]buffered saline\b|\bpbs(?:\s*1x)?\b"), "pbs"),
    (
        re.compile(
            r"\b(?:sodium\s+|potassium\s+)?phosphate\s+(?:buffer|solution)\b|\bpb\b"
        ),
        "pb",
    ),
    (
        re.compile(
            r"\b(?:deionized\s+|distilled\s+|milli-?q\s+|ultra-?pure\s+)?water\b|\bh2o\b|\baqueous(?:\s+dispersion|\s+buffer)?\b"
        ),
        "h2o",
    ),
    (re.compile(r"\bdichloromethane\b|\bdcm\b"), "dcm"),
    (re.compile(r"\btetrahydrofuran\b|\bthf\b"), "thf"),
    (re.compile(r"\bacetonitrile\b|\bmecn\b|\bch3cn\b"), "acetonitrile"),
    (re.compile(r"\bmethanol\b|\bmeoh\b"), "methanol"),
    (re.compile(r"\bethanol\b|\betoh\b"), "ethanol"),
    (re.compile(r"(?<![a-z0-9-])(?:2-propanol|isopropanol|ipa)\b"), "isopropanol"),
    (re.compile(r"\bchloroform\b|\bchcl3\b"), "chloroform"),
    (re.compile(r"\bacetone\b"), "acetone"),
    (re.compile(r"\btoluene\b"), "toluene"),
    (re.compile(r"\bhexane\b"), "hexane"),
    (re.compile(r"\bethyl\s*acetate\b|\betoac\b"), "ethyl acetate"),
    (re.compile(r"\bchlorobenzene\b"), "chlorobenzene"),
    (re.compile(r"\bacetic\s*acid\b"), "acetic acid"),
    (re.compile(r"\btris(?:\s+buffer)?\b"), "tris"),
    (re.compile(r"\bhepes\b"), "hepes"),
]

_DETECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"cryo[- ]?(?:em|electron\s+microscopy)"), "cryo-em"),
    (re.compile(r"transmission\s+electron\s+microscop|\btem\b"), "tem"),
    (re.compile(r"scanning\s+electron\s+microscop|\bsem\b"), "sem"),
    (re.compile(r"atomic\s+force\s+microscop|\bafm\b"), "afm"),
    (re.compile(r"confocal"), "confocal"),
    (
        re.compile(r"optical\s+microscop|light\s+microscop|polarizing\s+microscop"),
        "optical microscopy",
    ),
    (re.compile(r"circular\s+dichroism|\bcd\b"), "cd"),
    (re.compile(r"fourier\s+transform\s+infrared|\bft-?ir\b|\batr-?ir\b"), "ftir"),
    (re.compile(r"thioflavin|\btht\b"), "tht"),
    (re.compile(r"\bnmr\b|nuclear\s+magnetic\s+resonance"), "nmr"),
    (re.compile(r"x-?ray\s+diffraction|\bxrd\b|\bsaxs\b|\bwaxs\b"), "xrd"),
    (re.compile(r"dynamic\s+light\s+scattering|\bdls\b"), "dls"),
    (re.compile(r"uv[-/–]?vis"), "uv-vis"),
    (re.compile(r"fluorescence"), "fluorescence"),
    (re.compile(r"rheolog"), "rheology"),
    (re.compile(r"congo\s+red"), "congo red"),
    (re.compile(r"mass\s+spectrom|\bmaldi\b"), "ms"),
    (re.compile(r"raman"), "raman"),
    (re.compile(r"turbidit"), "turbidity"),
    (re.compile(r"vial\s+inversion|tube\s+inversion"), "vial inversion"),
]


def normalize_n_terminal(value: Optional[str]) -> Optional[str]:
    """Normalize N-terminal modification names into canonical lowercase tokens."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned in _N_TERM_FREE:
        return "free"
    if cleaned in _N_TERM_ALIASES:
        return _N_TERM_ALIASES[cleaned]
    for pattern, canonical in _N_TERM_PATTERNS:
        if pattern.search(cleaned):
            return canonical
    return cleaned


def normalize_c_terminal(value: Optional[str]) -> Optional[str]:
    """Normalize C-terminal modification names into canonical lowercase tokens."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned in _C_TERM_FREE:
        return "free"
    return _C_TERM_ALIASES.get(cleaned, cleaned)


def normalize_conjugate_partner(value: Optional[str]) -> Optional[str]:
    """Normalize conjugate partner names (including PeptideMiner 'Napthalene' typo & NDI)."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned or cleaned in {"none", "nan", "np", "n/a", "-"}:
        return None
    for pattern, canonical in _CONJUGATE_PATTERNS:
        if pattern.search(cleaned):
            return canonical
    return cleaned


def normalize_phase(value: Optional[str]) -> Optional[str]:
    """Normalize self-assembly phase/morphology labels into canonical PeptideMiner classes."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    return _PHASE_ALIASES.get(cleaned, cleaned)


def normalize_volume_ratio(value: Optional[str]) -> Optional[str]:
    """Normalize volume ratio strings (e.g. '100' -> '1', whitespace stripped)."""
    if value is None:
        return None
    cleaned = re.sub(r"\s+", "", value.strip())
    if not cleaned:
        return None
    if cleaned in {"100", "100/0", "100:0", "1/0"}:
        return "1"
    return cleaned


def extract_solvent_tokens(
    solution: Optional[str], solvent: Optional[str] = None
) -> set[str]:
    """Extract canonical solvent component tokens from solution and solvent strings."""
    combined = " ".join(part for part in (solution, solvent) if part).lower()
    if not combined.strip():
        return set()
    tokens: set[str] = set()
    for pattern, canonical in _SOLVENT_ALIASES:
        if pattern.search(combined):
            tokens.add(canonical)
    if not tokens:
        for chunk in re.split(r"[,/+;]|\band\b", combined):
            c = chunk.strip()
            if c and c not in {"dispersion", "medium", "solution", "buffer"}:
                tokens.add(c)
    return tokens


def extract_detection_tokens(detection_method: Optional[str]) -> set[str]:
    """Extract canonical detection-method tokens from a free-text or comma-separated string."""
    if not detection_method or not detection_method.strip():
        return set()
    text = detection_method.lower()
    tokens: set[str] = set()
    for pattern, canonical in _DETECTION_PATTERNS:
        if pattern.search(text):
            tokens.add(canonical)
    if not tokens:
        for part in re.split(r"[,;]", text):
            p = part.strip()
            if p:
                tokens.add(p)
    return tokens


def condition_similarity(manual: PeptideEntry, llm: PeptideEntry) -> float:
    """Compute a similarity score in [0, 1] between two entries sharing (pmid, sequence)."""
    score = 0.0
    weight = 0.0

    def _add(match: bool, w: float) -> None:
        nonlocal score, weight
        weight += w
        if match:
            score += w

    if manual.type is not None and llm.type is not None:
        _add(manual.type.strip().lower() == llm.type.strip().lower(), 2.0)

    m_sol_only = extract_solvent_tokens(manual.solution)
    l_sol_only = extract_solvent_tokens(llm.solution)
    m_sol_all = extract_solvent_tokens(manual.solution, manual.solvent)
    l_sol_all = extract_solvent_tokens(llm.solution, llm.solvent)
    jaccards: list[float] = []
    if m_sol_only and l_sol_only:
        jaccards.append(len(m_sol_only & l_sol_only) / len(m_sol_only | l_sol_only))
    if m_sol_all and l_sol_all:
        jaccards.append(len(m_sol_all & l_sol_all) / len(m_sol_all | l_sol_all))
    if jaccards:
        weight += 2.0
        score += 2.0 * max(jaccards)

    m_vr = normalize_volume_ratio(manual.volume_ratio)
    l_vr = normalize_volume_ratio(llm.volume_ratio)
    if m_vr and l_vr:
        _add(
            m_vr == l_vr
            or ("/" in m_vr and "/".join(reversed(m_vr.split("/"))) == l_vr),
            1.5,
        )

    if manual.ph is not None and llm.ph is not None:
        _add(abs(manual.ph - llm.ph) <= 0.25, 1.5)

    if manual.temperature is not None and llm.temperature is not None:
        _add(abs(manual.temperature - llm.temperature) <= 2.0, 1.0)

    if (
        manual.peptide_concentration is not None
        and llm.peptide_concentration is not None
    ):
        tol = max(0.05, 0.05 * abs(manual.peptide_concentration))
        _add(abs(manual.peptide_concentration - llm.peptide_concentration) <= tol, 1.5)

    return score / weight if weight > 0 else 0.0


@dataclass
class FieldMatchStats:
    """Exact vs normalized agreement counts for a single field on matched pairs."""

    manual_nonnull: int = 0
    llm_nonnull: int = 0
    both_nonnull: int = 0
    exact_match: int = 0
    normalized_match: int = 0


@dataclass
class GoldenBenchmarkReport:
    """Summary of comparing an LLM extraction run against golden-dataset manual entries."""

    manual_pmids: int
    llm_pmids: int
    covered_pmids: int
    manual_rows_total: int
    manual_rows_on_covered_pmids: int
    manual_assembling_rows_on_covered_pmids: int
    manual_none_rows_on_covered_pmids: int
    llm_rows_total: int
    llm_none_rows: int
    manual_unique_sequences: int
    llm_unique_sequences: int
    sequence_overlap: int
    sequence_recall: float
    sequence_precision: float
    sequence_f1: float
    matched_condition_pairs: int
    matched_assembling_pairs: int
    matched_none_pairs: int
    assembling_condition_recall: float
    none_condition_recall: float
    condition_precision: float
    unmatched_manual_rows: int
    unmatched_llm_rows: int
    field_stats: dict[str, FieldMatchStats] = field(default_factory=dict)

    def format_report(self) -> str:
        """Return a human-readable scorecard of sequence, condition, and field metrics."""
        lines = [
            f"PMIDs covered: {self.covered_pmids}/{self.manual_pmids}",
            (
                f"Sequences: Manual={self.manual_unique_sequences}, LLM={self.llm_unique_sequences}, "
                f"Overlap={self.sequence_overlap} | R={self.sequence_recall:.1%}, "
                f"P={self.sequence_precision:.1%}, F1={self.sequence_f1:.1%}"
            ),
            (
                f"Condition rows: Manual={self.manual_rows_on_covered_pmids} "
                f"(assembling={self.manual_assembling_rows_on_covered_pmids}, none={self.manual_none_rows_on_covered_pmids}) | "
                f"LLM={self.llm_rows_total} (none={self.llm_none_rows})"
            ),
            (
                f"Matched pairs: {self.matched_condition_pairs} | "
                f"Assembling R={self.assembling_condition_recall:.1%} ({self.matched_assembling_pairs}/{self.manual_assembling_rows_on_covered_pmids}), "
                f"None R={self.none_condition_recall:.1%} ({self.matched_none_pairs}/{self.manual_none_rows_on_covered_pmids}), "
                f"Condition P={self.condition_precision:.1%}"
            ),
            "",
            f"{'Field':<26} {'Cov':>7} {'Exact':>7} {'Norm':>7}",
        ]
        for fname, st in self.field_stats.items():
            cov = f"{st.both_nonnull}/{st.manual_nonnull}"
            ex = (
                f"{(st.exact_match / st.both_nonnull * 100):.1f}%"
                if st.both_nonnull
                else "-"
            )
            nm = (
                f"{(st.normalized_match / st.both_nonnull * 100):.1f}%"
                if st.both_nonnull
                else "-"
            )
            lines.append(f"{fname:<26} {cov:>7} {ex:>7} {nm:>7}")
        return "\n".join(lines)


def _is_none_type(entry: PeptideEntry) -> bool:
    """Return True if an entry represents a non-assembling ('none') condition."""
    return (entry.type or "").strip().lower() == "none"


def _match_entries_for_sequence(
    manual_list: Sequence[PeptideEntry],
    llm_list: Sequence[PeptideEntry],
    min_similarity: float = 0.1,
) -> tuple[
    list[tuple[PeptideEntry, PeptideEntry]], list[PeptideEntry], list[PeptideEntry]
]:
    """Greedily pair manual and LLM entries for the same (pmid, sequence) by condition similarity.

    Entries with ``type == 'none'`` (non-assembling) are never paired with assembling entries,
    and candidate pairs with similarity below ``min_similarity`` are not matched.
    """
    candidates: list[tuple[float, int, int]] = []
    for mi, m in enumerate(manual_list):
        m_none = _is_none_type(m)
        for li, l in enumerate(llm_list):
            if m_none != _is_none_type(l):
                continue
            sim = condition_similarity(m, l)
            if sim < min_similarity:
                continue
            candidates.append((sim, mi, li))
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_m: set[int] = set()
    used_l: set[int] = set()
    pairs: list[tuple[PeptideEntry, PeptideEntry]] = []
    for _, mi, li in candidates:
        if mi not in used_m and li not in used_l:
            used_m.add(mi)
            used_l.add(li)
            pairs.append((manual_list[mi], llm_list[li]))

    unmatched_m = [m for i, m in enumerate(manual_list) if i not in used_m]
    unmatched_l = [l for i, l in enumerate(llm_list) if i not in used_l]
    return pairs, unmatched_m, unmatched_l


def _values_match_normalized(
    field_name: str, m: PeptideEntry, l: PeptideEntry
) -> tuple[bool, bool]:
    """Return (exact_match, normalized_match) for a field on a paired (manual, llm) entry."""
    mv = getattr(m, field_name)
    lv = getattr(l, field_name)
    exact = mv == lv

    if field_name == "type":
        return exact, normalize_phase(mv) == normalize_phase(lv)
    if field_name == "n_terminal":
        return exact, normalize_n_terminal(mv) == normalize_n_terminal(lv)
    if field_name == "c_terminal":
        return exact, normalize_c_terminal(mv) == normalize_c_terminal(lv)
    if field_name == "conjugate_partner":
        return exact, normalize_conjugate_partner(mv) == normalize_conjugate_partner(lv)
    if field_name == "volume_ratio":
        mvr, lvr = normalize_volume_ratio(mv), normalize_volume_ratio(lv)
        if mvr is None or lvr is None:
            return exact, False
        flipped = "/" in mvr and "/".join(reversed(mvr.split("/"))) == lvr
        return exact, (mvr == lvr or flipped)
    if field_name == "solution":
        if exact:
            return True, True
        ms_sol = extract_solvent_tokens(m.solution)
        ls_sol = extract_solvent_tokens(l.solution)
        if ms_sol and ls_sol and ms_sol == ls_sol:
            return exact, True
        ms_all = extract_solvent_tokens(m.solution, m.solvent)
        ls_all = extract_solvent_tokens(l.solution, l.solvent)
        return exact, bool(ms_all and ls_all and ms_all == ls_all)
    if field_name == "detection_method":
        md = extract_detection_tokens(mv)
        ld = extract_detection_tokens(lv)
        return exact, bool(md and ld and md.issubset(ld))
    if field_name in {
        "temperature",
        "ph",
        "peptide_concentration",
        "incubation_period",
    }:
        if mv is None or lv is None:
            return exact, False
        tol = (
            0.25
            if field_name == "ph"
            else (
                2.0 if field_name == "temperature" else max(0.05, 0.05 * abs(float(mv)))
            )
        )
        return exact, abs(float(mv) - float(lv)) <= tol
    if isinstance(mv, str) and isinstance(lv, str):
        return exact, mv.strip().lower() == lv.strip().lower()
    return exact, exact


def compare_runs(
    manual_entries: Sequence[PeptideEntry],
    llm_entries: Sequence[PeptideEntry],
    min_similarity: float = 0.1,
) -> GoldenBenchmarkReport:
    """Compare LLM-extracted entries against golden-dataset manual entries."""
    manual_pmids = {e.pmid for e in manual_entries}
    llm_pmids = {e.pmid for e in llm_entries}
    covered_pmids = manual_pmids & llm_pmids

    manual_covered = [e for e in manual_entries if e.pmid in covered_pmids]
    manual_none_covered = sum(
        1 for e in manual_covered if (e.type or "").lower() == "none"
    )
    manual_assembling_covered = len(manual_covered) - manual_none_covered
    llm_none_rows = sum(1 for e in llm_entries if (e.type or "").lower() == "none")

    m_seqs = {(e.pmid, e.sequence.upper()) for e in manual_covered}
    l_seqs = {
        (e.pmid, e.sequence.upper()) for e in llm_entries if e.pmid in covered_pmids
    }
    seq_overlap = len(m_seqs & l_seqs)

    m_by_ps: dict[tuple[str, str], list[PeptideEntry]] = defaultdict(list)
    for e in manual_covered:
        m_by_ps[(e.pmid, e.sequence.upper())].append(e)

    l_by_ps: dict[tuple[str, str], list[PeptideEntry]] = defaultdict(list)
    for e in llm_entries:
        if e.pmid in covered_pmids:
            l_by_ps[(e.pmid, e.sequence.upper())].append(e)

    matched_pairs: list[tuple[PeptideEntry, PeptideEntry]] = []
    unmatched_m_count = 0
    unmatched_l_count = 0

    for key in set(m_by_ps.keys()) | set(l_by_ps.keys()):
        pairs, um, ul = _match_entries_for_sequence(
            m_by_ps.get(key, []),
            l_by_ps.get(key, []),
            min_similarity=min_similarity,
        )
        matched_pairs.extend(pairs)
        unmatched_m_count += len(um)
        unmatched_l_count += len(ul)

    eval_fields = [
        "type",
        "n_terminal",
        "c_terminal",
        "temperature",
        "ph",
        "peptide_concentration",
        "solution",
        "volume_ratio",
        "incubation_period",
        "peptide_conjugate_mixture",
        "linear_cyclic",
        "detection_method",
    ]
    field_stats: dict[str, FieldMatchStats] = {
        f: FieldMatchStats() for f in eval_fields
    }

    for m, l in matched_pairs:
        for f in eval_fields:
            st = field_stats[f]
            mv = getattr(m, f)
            lv = getattr(l, f)
            m_ok = mv is not None and str(mv).strip() != ""
            l_ok = lv is not None and str(lv).strip() != ""
            if m_ok:
                st.manual_nonnull += 1
            if l_ok:
                st.llm_nonnull += 1
            if m_ok and l_ok:
                st.both_nonnull += 1
                ex, norm = _values_match_normalized(f, m, l)
                if ex:
                    st.exact_match += 1
                if norm:
                    st.normalized_match += 1

    seq_rec = seq_overlap / len(m_seqs) if m_seqs else 0.0
    seq_prec = seq_overlap / len(l_seqs) if l_seqs else 0.0
    seq_f1 = (
        (2 * seq_rec * seq_prec / (seq_rec + seq_prec)) if (seq_rec + seq_prec) else 0.0
    )

    matched_none = sum(1 for m, _ in matched_pairs if (m.type or "").lower() == "none")
    matched_assembling = len(matched_pairs) - matched_none

    return GoldenBenchmarkReport(
        manual_pmids=len(manual_pmids),
        llm_pmids=len(llm_pmids),
        covered_pmids=len(covered_pmids),
        manual_rows_total=len(manual_entries),
        manual_rows_on_covered_pmids=len(manual_covered),
        manual_assembling_rows_on_covered_pmids=manual_assembling_covered,
        manual_none_rows_on_covered_pmids=manual_none_covered,
        llm_rows_total=len(llm_entries),
        llm_none_rows=llm_none_rows,
        manual_unique_sequences=len(m_seqs),
        llm_unique_sequences=len(l_seqs),
        sequence_overlap=seq_overlap,
        sequence_recall=seq_rec,
        sequence_precision=seq_prec,
        sequence_f1=seq_f1,
        matched_condition_pairs=len(matched_pairs),
        matched_assembling_pairs=matched_assembling,
        matched_none_pairs=matched_none,
        assembling_condition_recall=(
            matched_assembling / manual_assembling_covered
            if manual_assembling_covered
            else 0.0
        ),
        none_condition_recall=(
            matched_none / manual_none_covered if manual_none_covered else 0.0
        ),
        condition_precision=(
            len(matched_pairs) / len(llm_entries) if llm_entries else 0.0
        ),
        unmatched_manual_rows=unmatched_m_count,
        unmatched_llm_rows=unmatched_l_count,
        field_stats=field_stats,
    )


def benchmark_run_against_golden(
    llm_run_identifier: int,
    manual_run_identifier: int = 0,
    pmids: Optional[Sequence[str]] = None,
    session: Optional[Session] = None,
    min_similarity: float = 0.1,
) -> GoldenBenchmarkReport:
    """Load manual and LLM PeptideEntry rows from the DB and compute GoldenBenchmarkReport."""

    def _run(s: Session) -> GoldenBenchmarkReport:
        m_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == manual_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.MANUAL,
        )
        l_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == llm_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.LLM,
        )
        if pmids is not None:
            pmid_list = list(pmids)
            m_stmt = m_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]
            l_stmt = l_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]
        manual = s.exec(m_stmt).all()
        llm = s.exec(l_stmt).all()
        return compare_runs(manual, llm, min_similarity=min_similarity)

    if session is not None:
        return _run(session)
    with get_session_context() as s:
        return _run(s)


@dataclass
class PmidDiagnostic:
    """Per-PMID breakdown of sequence, assembling/none row recall, precision, and solution accuracy."""

    pmid: str
    manual_seqs: int
    llm_seqs: int
    seq_overlap: int
    missing_seqs: list[str]
    extra_seqs: list[str]
    manual_assembling: int
    llm_assembling: int
    matched_assembling: int
    assembling_recall: float
    manual_none: int
    llm_none: int
    matched_none: int
    none_recall: float
    condition_precision: float
    solution_both_nonnull: int
    solution_norm_match: int

    @property
    def missed_assembling(self) -> int:
        return self.manual_assembling - self.matched_assembling


def compare_runs_per_pmid(
    manual_entries: Sequence[PeptideEntry],
    llm_entries: Sequence[PeptideEntry],
) -> list[PmidDiagnostic]:
    """Compute per-PMID diagnostic metrics sorted by missed assembling rows (descending)."""
    m_by_pmid: dict[str, list[PeptideEntry]] = defaultdict(list)
    for e in manual_entries:
        m_by_pmid[e.pmid].append(e)

    l_by_pmid: dict[str, list[PeptideEntry]] = defaultdict(list)
    for e in llm_entries:
        l_by_pmid[e.pmid].append(e)

    all_pmids = sorted(set(m_by_pmid.keys()) | set(l_by_pmid.keys()))
    diagnostics: list[PmidDiagnostic] = []

    for pmid in all_pmids:
        m_list = m_by_pmid.get(pmid, [])
        l_list = l_by_pmid.get(pmid, [])

        m_seq_set = {e.sequence.upper() for e in m_list}
        l_seq_set = {e.sequence.upper() for e in l_list}
        missing = sorted(m_seq_set - l_seq_set)
        extra = sorted(l_seq_set - m_seq_set)

        m_none = sum(1 for e in m_list if (e.type or "").strip().lower() == "none")
        m_asm = len(m_list) - m_none
        l_none = sum(1 for e in l_list if (e.type or "").strip().lower() == "none")
        l_asm = len(l_list) - l_none

        if m_list and l_list:
            rep = compare_runs(m_list, l_list)
            matched_asm = rep.matched_assembling_pairs
            matched_none = rep.matched_none_pairs
            asm_rec = rep.assembling_condition_recall
            none_rec = rep.none_condition_recall
            cond_prec = rep.condition_precision
            sol_stats = rep.field_stats.get("solution", FieldMatchStats())
            sol_both = sol_stats.both_nonnull
            sol_norm = sol_stats.normalized_match
        else:
            matched_asm = 0
            matched_none = 0
            asm_rec = 0.0
            none_rec = 0.0
            cond_prec = 0.0
            sol_both = 0
            sol_norm = 0

        diagnostics.append(
            PmidDiagnostic(
                pmid=pmid,
                manual_seqs=len(m_seq_set),
                llm_seqs=len(l_seq_set),
                seq_overlap=len(m_seq_set & l_seq_set),
                missing_seqs=missing,
                extra_seqs=extra,
                manual_assembling=m_asm,
                llm_assembling=l_asm,
                matched_assembling=matched_asm,
                assembling_recall=asm_rec,
                manual_none=m_none,
                llm_none=l_none,
                matched_none=matched_none,
                none_recall=none_rec,
                condition_precision=cond_prec,
                solution_both_nonnull=sol_both,
                solution_norm_match=sol_norm,
            )
        )

    diagnostics.sort(key=lambda d: (d.missed_assembling, d.manual_assembling), reverse=True)
    return diagnostics


def format_per_pmid_diagnostics(diagnostics: Sequence[PmidDiagnostic]) -> str:
    """Format per-PMID diagnostics as an aligned summary table plus missing/extra sequence details."""
    header = (
        f"{'PMID':<10} {'Seq(M/L/O)':>11} {'Asm(M/L/Hit)':>13} {'AsmR':>6} "
        f"{'None(M/L/Hit)':>14} {'NoneR':>6} {'CondP':>6} {'SolNorm':>10}"
    )
    lines = [header, "-" * len(header)]
    for d in diagnostics:
        seq_col = f"{d.manual_seqs}/{d.llm_seqs}/{d.seq_overlap}"
        asm_col = f"{d.manual_assembling}/{d.llm_assembling}/{d.matched_assembling}"
        none_col = f"{d.manual_none}/{d.llm_none}/{d.matched_none}"
        sol_col = (
            f"{d.solution_norm_match}/{d.solution_both_nonnull}"
            if d.solution_both_nonnull
            else "-"
        )
        lines.append(
            f"{d.pmid:<10} {seq_col:>11} {asm_col:>13} {d.assembling_recall:>5.1%} "
            f"{none_col:>14} {d.none_recall:>5.1%} {d.condition_precision:>5.1%} {sol_col:>10}"
        )

    detail_lines = []
    for d in diagnostics:
        if d.missing_seqs or d.extra_seqs:
            parts = [f"PMID {d.pmid}:"]
            if d.missing_seqs:
                parts.append(f"missing_seqs={d.missing_seqs}")
            if d.extra_seqs:
                parts.append(f"extra_seqs={d.extra_seqs}")
            detail_lines.append("  " + " | ".join(parts))

    if detail_lines:
        lines.append("")
        lines.append("Sequence Mismatches:")
        lines.extend(detail_lines)

    return "\n".join(lines)


def diagnose_run_per_pmid(
    llm_run_identifier: int,
    manual_run_identifier: int = 0,
    pmids: Optional[Sequence[str]] = None,
    session: Optional[Session] = None,
) -> tuple[list[PmidDiagnostic], str]:
    """Load manual and LLM rows from DB and return (per-PMID diagnostics, formatted table)."""

    def _run(s: Session) -> tuple[list[PmidDiagnostic], str]:
        m_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == manual_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.MANUAL,
        )
        l_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == llm_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.LLM,
        )
        if pmids is not None:
            pmid_list = list(pmids)
            m_stmt = m_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]
            l_stmt = l_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]
        manual = s.exec(m_stmt).all()
        llm = s.exec(l_stmt).all()
        diags = compare_runs_per_pmid(manual, llm)
        return diags, format_per_pmid_diagnostics(diags)

    if session is not None:
        return _run(session)
    with get_session_context() as s:
        return _run(s)


# ---------------------------------------------------------------------------
# PeptideMiner Paper Evaluation Protocol (Yang et al., arXiv:2411.05421v1)
# ---------------------------------------------------------------------------

PEPTIDEMINER_CATEGORICAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("sequence", "sequence"),
    ("phase", "type"),
    ("n_terminal", "n_terminal"),
    ("c_terminal", "c_terminal"),
    ("category", "peptide_conjugate_mixture"),
    ("conjugate_partner", "conjugate_partner"),
    ("solution", "solution"),
)

PEPTIDEMINER_NUMERICAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("ph", "ph"),
    ("temperature", "temperature"),
    ("concentration", "peptide_concentration"),
    ("solvent_ratio", "volume_ratio"),
)


def parse_volume_ratio_float(value: Optional[str]) -> Optional[float]:
    """Parse a volume_ratio string into a numeric fraction in [0, 1].

    Examples:
        '1', '100', '100/0' -> 1.0
        '5/95'              -> 0.05
        '50:50'             -> 0.5
        '0.05'              -> 0.05
    """
    norm = normalize_volume_ratio(value)
    if norm is None:
        return None
    if norm == "1":
        return 1.0
    sep = "/" if "/" in norm else (":" if ":" in norm else None)
    if sep is not None:
        parts = norm.split(sep)
        if len(parts) == 2:
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                return None
            if a < 0 or b < 0 or (a + b) == 0:
                return None
            return a / (a + b)
        return None
    try:
        val = float(norm)
    except ValueError:
        return None
    if val < 0:
        return None
    if val > 1.0 and val <= 100.0:
        return val / 100.0
    return val


def _is_nonnull(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return bool(cleaned) and cleaned not in {"nan", "np", "n/a", "null"}
    return True


def _peptideminer_categorical_match(
    attr_name: str, manual: PeptideEntry, llm: PeptideEntry
) -> tuple[bool, bool, bool]:
    """Return (exact_match, normalized_match, both_nonnull) under PeptideMiner rules.

    Following Yang et al. (Methods, p. 10), both-empty / both-'nan' predictions
    count as True (aligned), while one-missing vs one-present counts as False.
    For 'solution', both solute and solvent components must match completely.
    """
    mv = getattr(manual, attr_name)
    lv = getattr(llm, attr_name)
    m_ok = _is_nonnull(mv)
    l_ok = _is_nonnull(lv)

    if not m_ok and not l_ok:
        return True, True, False
    if m_ok != l_ok:
        return False, False, False

    if attr_name == "sequence":
        m_str = str(mv).strip()
        l_str = str(lv).strip()
        return m_str == l_str, m_str.upper() == l_str.upper(), True
    if attr_name == "type":
        m_str = str(mv).strip()
        l_str = str(lv).strip()
        return m_str == l_str, normalize_phase(m_str) == normalize_phase(l_str), True
    if attr_name == "n_terminal":
        return mv == lv, normalize_n_terminal(mv) == normalize_n_terminal(lv), True
    if attr_name == "c_terminal":
        return mv == lv, normalize_c_terminal(mv) == normalize_c_terminal(lv), True
    if attr_name == "conjugate_partner":
        return (
            mv == lv,
            normalize_conjugate_partner(mv) == normalize_conjugate_partner(lv),
            True,
        )
    if attr_name == "solution":
        exact = str(mv).strip().lower() == str(lv).strip().lower()
        if exact:
            return True, True, True
        ms_sol = extract_solvent_tokens(manual.solution)
        ls_sol = extract_solvent_tokens(llm.solution)
        if ms_sol and ls_sol and ms_sol == ls_sol:
            return False, True, True
        ms_all = extract_solvent_tokens(manual.solution, manual.solvent)
        ls_all = extract_solvent_tokens(llm.solution, llm.solvent)
        return False, bool(ms_all and ls_all and ms_all == ls_all), True

    m_str = str(mv).strip()
    l_str = str(lv).strip()
    return m_str == l_str, m_str.lower() == l_str.lower(), True


def _get_numeric_value(entry: PeptideEntry, attr_name: str) -> Optional[float]:
    if attr_name == "volume_ratio":
        return parse_volume_ratio_float(entry.volume_ratio)
    val = getattr(entry, attr_name)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _peptideminer_alignment_score(
    manual: PeptideEntry,
    llm: PeptideEntry,
    bounds: Mapping[str, tuple[float, float]],
    include_seq_phase: bool = False,
) -> float:
    """Score closeness of a ground-truth row to an LLM prediction on target entities."""
    score = 0.0
    if include_seq_phase:
        if manual.sequence.strip().upper() == llm.sequence.strip().upper():
            score += 2.0
        if (manual.type or "").strip().lower() == (llm.type or "").strip().lower():
            score += 2.0

    for _, attr in PEPTIDEMINER_CATEGORICAL_FIELDS:
        if attr in {"sequence", "type"}:
            continue
        _, norm_match, _ = _peptideminer_categorical_match(attr, manual, llm)
        if norm_match:
            score += 1.0

    for _, attr in PEPTIDEMINER_NUMERICAL_FIELDS:
        mv = _get_numeric_value(manual, attr)
        lv = _get_numeric_value(llm, attr)
        if mv is not None and lv is not None:
            lo, hi = bounds.get(attr, (0.0, 1.0))
            span = (hi - lo) if hi > lo else 1.0
            norm_err = min(1.0, abs(mv - lv) / span)
            score += 1.0 - norm_err
    return score


def _find_best_peptideminer_match(
    pred: PeptideEntry,
    manual_candidates: Sequence[PeptideEntry],
    bounds: Mapping[str, tuple[float, float]],
) -> PeptideEntry:
    """Select the ground-truth entry for a prediction per Yang et al. Methods (p. 10).

    1. Filter manual entries for the same PMID that match both predicted sequence
       and self-assembly phase.
    2. If multiple direct matches exist, pick the one closest on remaining entities.
    3. If no direct match exists, pick the closest manual entry across all entities.
    """
    pred_seq = pred.sequence.strip().upper()
    pred_phase = (pred.type or "").strip().lower()

    direct = [
        m
        for m in manual_candidates
        if m.sequence.strip().upper() == pred_seq
        and (m.type or "").strip().lower() == pred_phase
    ]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        return max(
            direct,
            key=lambda m: _peptideminer_alignment_score(
                m, pred, bounds, include_seq_phase=False
            ),
        )
    return max(
        manual_candidates,
        key=lambda m: _peptideminer_alignment_score(
            m, pred, bounds, include_seq_phase=True
        ),
    )


@dataclass
class PeptideMinerCategoricalStats:
    """Prediction-driven categorical accuracy stats matching Yang et al. Fig. 4c."""

    total: int = 0
    exact_correct: int = 0
    normalized_correct: int = 0
    both_nonnull: int = 0
    both_nonnull_normalized_correct: int = 0

    @property
    def exact_accuracy(self) -> float:
        return self.exact_correct / self.total if self.total else 0.0

    @property
    def normalized_accuracy(self) -> float:
        return self.normalized_correct / self.total if self.total else 0.0

    @property
    def nonnull_accuracy(self) -> float:
        return (
            self.both_nonnull_normalized_correct / self.both_nonnull
            if self.both_nonnull
            else 0.0
        )


@dataclass
class PeptideMinerNumericalStats:
    """Numerical MAE stats matching Yang et al. Fig. 4d (excluding missing predictions)."""

    manual_nonnull: int = 0
    evaluated: int = 0
    raw_mae: Optional[float] = None
    normalized_mae: Optional[float] = None
    median_abs_error: Optional[float] = None
    max_abs_error: Optional[float] = None
    raw_errors: list[float] = field(default_factory=list)
    normalized_errors: list[float] = field(default_factory=list)


@dataclass
class PeptideMinerBenchmarkReport:
    """Benchmark report implementing the PeptideMiner paper evaluation protocol."""

    manual_pmids: int
    llm_pmids: int
    covered_pmids: int
    manual_rows_on_covered_pmids: int
    llm_rows_evaluated: int
    categorical_stats: dict[str, PeptideMinerCategoricalStats] = field(
        default_factory=dict
    )
    numerical_stats: dict[str, PeptideMinerNumericalStats] = field(default_factory=dict)

    @property
    def average_categorical_exact_accuracy(self) -> float:
        vals = [st.exact_accuracy for st in self.categorical_stats.values() if st.total]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def average_categorical_normalized_accuracy(self) -> float:
        vals = [
            st.normalized_accuracy for st in self.categorical_stats.values() if st.total
        ]
        return sum(vals) / len(vals) if vals else 0.0

    def format_report(self) -> str:
        """Format categorical accuracy (Fig. 4c) and numerical MAE (Fig. 4d) tables."""
        lines = [
            f"PMIDs covered: {self.covered_pmids}/{self.manual_pmids} "
            f"(LLM PMIDs: {self.llm_pmids})",
            f"Entries: Manual (covered PMIDs)={self.manual_rows_on_covered_pmids} | "
            f"LLM evaluated={self.llm_rows_evaluated}",
            f"Avg Categorical Accuracy: Exact={self.average_categorical_exact_accuracy:.1%}, "
            f"Normalized={self.average_categorical_normalized_accuracy:.1%}",
            "",
            f"{'Categorical Field':<22} {'Exact':>8} {'Norm':>8} {'NonNull':>9} {'N':>6}",
        ]
        for label, st in self.categorical_stats.items():
            ex = f"{st.exact_accuracy:.1%}" if st.total else "-"
            nm = f"{st.normalized_accuracy:.1%}" if st.total else "-"
            nn = (
                f"{st.nonnull_accuracy:.1%}"
                if st.both_nonnull
                else "-"
            )
            lines.append(f"{label:<22} {ex:>8} {nm:>8} {nn:>9} {st.total:>6}")

        lines.extend(
            [
                "",
                f"{'Numerical Field':<22} {'NormMAE':>8} {'RawMAE':>9} {'MedAE':>8} {'Eval/Man':>10}",
            ]
        )
        for label, st in self.numerical_stats.items():
            nmae = f"{st.normalized_mae:.4f}" if st.normalized_mae is not None else "-"
            rmae = f"{st.raw_mae:.4f}" if st.raw_mae is not None else "-"
            med = (
                f"{st.median_abs_error:.4f}"
                if st.median_abs_error is not None
                else "-"
            )
            cov = f"{st.evaluated}/{st.manual_nonnull}"
            lines.append(f"{label:<22} {nmae:>8} {rmae:>9} {med:>8} {cov:>10}")

        lines.append("")
        lines.append(
            "Note: Mirrors Yang et al. (arXiv:2411.05421v1) prediction-driven matching, "
            "null-aligned categorical accuracy, and [0,1]-normalized MAE (excluding missing predictions). "
            "Non-terminal modification and thermal process are omitted as they are not in PeptideEntry."
        )
        return "\n".join(lines)


def compare_runs_peptideminer(
    manual_entries: Sequence[PeptideEntry],
    llm_entries: Sequence[PeptideEntry],
    numerical_bounds: Optional[Mapping[str, tuple[float, float]]] = None,
) -> PeptideMinerBenchmarkReport:
    """Evaluate LLM extractions against manual entries using the PeptideMiner paper protocol."""
    bounds = (
        dict(numerical_bounds)
        if numerical_bounds is not None
        else dict(PEPTIDEMINER_NUMERICAL_BOUNDS)
    )

    m_by_pmid: dict[str, list[PeptideEntry]] = defaultdict(list)
    for e in manual_entries:
        m_by_pmid[e.pmid].append(e)

    l_by_pmid: dict[str, list[PeptideEntry]] = defaultdict(list)
    for e in llm_entries:
        l_by_pmid[e.pmid].append(e)

    manual_pmids = set(m_by_pmid.keys())
    llm_pmids = set(l_by_pmid.keys())
    covered_pmids = manual_pmids & llm_pmids

    manual_rows_covered = sum(len(m_by_pmid[p]) for p in covered_pmids)

    cat_stats: dict[str, PeptideMinerCategoricalStats] = {
        label: PeptideMinerCategoricalStats()
        for label, _ in PEPTIDEMINER_CATEGORICAL_FIELDS
    }
    raw_errors: dict[str, list[float]] = {
        label: [] for label, _ in PEPTIDEMINER_NUMERICAL_FIELDS
    }
    norm_errors: dict[str, list[float]] = {
        label: [] for label, _ in PEPTIDEMINER_NUMERICAL_FIELDS
    }
    num_manual_nonnull: dict[str, int] = {
        label: 0 for label, _ in PEPTIDEMINER_NUMERICAL_FIELDS
    }

    llm_rows_evaluated = 0
    for pmid in sorted(covered_pmids):
        m_list = m_by_pmid[pmid]
        for pred in l_by_pmid[pmid]:
            llm_rows_evaluated += 1
            gt = _find_best_peptideminer_match(pred, m_list, bounds)

            for label, attr in PEPTIDEMINER_CATEGORICAL_FIELDS:
                st = cat_stats[label]
                st.total += 1
                ex, nm, both_nn = _peptideminer_categorical_match(attr, gt, pred)
                if ex:
                    st.exact_correct += 1
                if nm:
                    st.normalized_correct += 1
                if both_nn:
                    st.both_nonnull += 1
                    if nm:
                        st.both_nonnull_normalized_correct += 1

            for label, attr in PEPTIDEMINER_NUMERICAL_FIELDS:
                mv = _get_numeric_value(gt, attr)
                lv = _get_numeric_value(pred, attr)
                if mv is not None:
                    num_manual_nonnull[label] += 1
                if mv is not None and lv is not None:
                    err = abs(lv - mv)
                    raw_errors[label].append(err)
                    lo, hi = bounds.get(attr, (0.0, 1.0))
                    span = (hi - lo) if hi > lo else 1.0
                    norm_errors[label].append(err / span)

    num_stats: dict[str, PeptideMinerNumericalStats] = {}
    for label, _ in PEPTIDEMINER_NUMERICAL_FIELDS:
        errs = raw_errors[label]
        nerrs = norm_errors[label]
        if errs:
            num_stats[label] = PeptideMinerNumericalStats(
                manual_nonnull=num_manual_nonnull[label],
                evaluated=len(errs),
                raw_mae=sum(errs) / len(errs),
                normalized_mae=sum(nerrs) / len(nerrs),
                median_abs_error=median(errs),
                max_abs_error=max(errs),
                raw_errors=list(errs),
                normalized_errors=list(nerrs),
            )
        else:
            num_stats[label] = PeptideMinerNumericalStats(
                manual_nonnull=num_manual_nonnull[label],
                evaluated=0,
            )

    return PeptideMinerBenchmarkReport(
        manual_pmids=len(manual_pmids),
        llm_pmids=len(llm_pmids),
        covered_pmids=len(covered_pmids),
        manual_rows_on_covered_pmids=manual_rows_covered,
        llm_rows_evaluated=llm_rows_evaluated,
        categorical_stats=cat_stats,
        numerical_stats=num_stats,
    )


def benchmark_run_peptideminer(
    llm_run_identifier: int,
    manual_run_identifier: int = 0,
    pmids: Optional[Sequence[str]] = PEPTIDEMINER_TEST_PMIDS,
    session: Optional[Session] = None,
    numerical_bounds: Optional[Mapping[str, tuple[float, float]]] = None,
) -> PeptideMinerBenchmarkReport:
    """Load manual and LLM rows from DB and evaluate using the PeptideMiner paper protocol."""

    def _run(s: Session) -> PeptideMinerBenchmarkReport:
        m_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == manual_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.MANUAL,
        )
        l_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == llm_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.LLM,
        )
        if pmids is not None:
            pmid_list = list(pmids)
            m_stmt = m_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]
            l_stmt = l_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]
        manual = s.exec(m_stmt).all()
        llm = s.exec(l_stmt).all()
        return compare_runs_peptideminer(
            manual, llm, numerical_bounds=numerical_bounds
        )

    if session is not None:
        return _run(session)
    with get_session_context() as s:
        return _run(s)


PEPTIDEMINER_PAPER_BASELINES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "literature"
    / "peptideminer"
    / "paper_baselines.json"
)


def load_peptideminer_paper_baselines(
    path: Optional[Path | str] = None,
) -> dict[str, list[PeptideEntry]]:
    """Load PeptideMiner paper baseline predictions (Pretrained & Fine-tuned GPT-3.5).

    Parsed from ``lamm-mit/PeptideMiner`` saved notebook outputs
    (``scripts/comparison/pretrain.ipynb`` and ``fine-tuning.ipynb``) across
    all 23 held-out test papers.
    """
    target = Path(path) if path is not None else PEPTIDEMINER_PAPER_BASELINES_PATH
    if not target.exists():
        return {}

    payload = json.loads(target.read_text(encoding="utf-8"))
    baselines: dict[str, list[PeptideEntry]] = {}
    for label, rows in payload.get("baselines", {}).items():
        baselines[label] = [
            PeptideEntry(
                pmid=str(r["pmid"]),
                sequence=str(r.get("sequence") or ""),
                length=int(r.get("length") or len(str(r.get("sequence") or ""))),
                type=r.get("type"),
                n_terminal=r.get("n_terminal"),
                c_terminal=r.get("c_terminal"),
                peptide_conjugate_mixture=r.get("peptide_conjugate_mixture"),
                conjugate_partner=r.get("conjugate_partner"),
                solution=r.get("solution"),
                volume_ratio=r.get("volume_ratio"),
                ph=r.get("ph"),
                temperature=r.get("temperature"),
                peptide_concentration=r.get("peptide_concentration"),
                extraction_mode=ExtractionMode.LLM,
                model=label,
            )
            for r in rows
        ]
    return baselines


def format_peptideminer_comparison(
    reports: Mapping[str, PeptideMinerBenchmarkReport],
    *,
    use_normalized: bool = True,
) -> str:
    """Format a side-by-side comparison table across multiple PeptideMinerBenchmarkReports."""
    if not reports:
        return "No reports to compare."

    names = list(reports.keys())
    col_w = max(20, *(len(n) + 2 for n in names))
    header = f"{'Metric':<24}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "-" * len(header)
    lines = [header, sep]

    lines.append(
        f"{'PMIDs covered':<24}"
        + "".join(
            f"{f'{r.covered_pmids}/{r.manual_pmids}':>{col_w}}"
            for r in reports.values()
        )
    )
    lines.append(
        f"{'LLM entries evaluated':<24}"
        + "".join(f"{r.llm_rows_evaluated:>{col_w}}" for r in reports.values())
    )
    lines.append(
        f"{'Avg Categorical Acc':<24}"
        + "".join(
            f"{(r.average_categorical_normalized_accuracy if use_normalized else r.average_categorical_exact_accuracy):>{col_w}.1%}"
            for r in reports.values()
        )
    )
    lines.append(sep)

    for label, _ in PEPTIDEMINER_CATEGORICAL_FIELDS:
        row_vals = []
        for r in reports.values():
            st = r.categorical_stats.get(label)
            if st and st.total:
                acc = st.normalized_accuracy if use_normalized else st.exact_accuracy
                row_vals.append(f"{acc:>{col_w}.1%}")
            else:
                row_vals.append(f"{'-':>{col_w}}")
        lines.append(f"{label:<24}" + "".join(row_vals))

    lines.append(sep)
    for label, _ in PEPTIDEMINER_NUMERICAL_FIELDS:
        row_vals = []
        for r in reports.values():
            st = r.numerical_stats.get(label)
            if st and st.normalized_mae is not None and st.raw_mae is not None:
                cell = f"{st.normalized_mae:.4f} ({st.raw_mae:.2f})"
                row_vals.append(f"{cell:>{col_w}}")
            else:
                row_vals.append(f"{'-':>{col_w}}")
        lines.append(f"{label + ' NormMAE (Raw)':<24}" + "".join(row_vals))

    return "\n".join(lines)


def benchmark_run_peptideminer_vs_paper(
    llm_run_identifier: int,
    manual_run_identifier: int = 0,
    pmids: Optional[Sequence[str]] = PEPTIDEMINER_TEST_PMIDS,
    llm_label: Optional[str] = None,
    session: Optional[Session] = None,
    numerical_bounds: Optional[Mapping[str, tuple[float, float]]] = None,
) -> tuple[dict[str, PeptideMinerBenchmarkReport], str]:
    """Compare an LLM run against both PeptideMiner paper baselines on the same manual set."""

    def _run(s: Session) -> tuple[dict[str, PeptideMinerBenchmarkReport], str]:
        m_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == manual_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.MANUAL,
        )
        l_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == llm_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.LLM,
        )
        if pmids is not None:
            pmid_list = list(pmids)
            m_stmt = m_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]
            l_stmt = l_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]

        manual = s.exec(m_stmt).all()
        llm = s.exec(l_stmt).all()

        reports: dict[str, PeptideMinerBenchmarkReport] = {}
        for base_name, base_entries in load_peptideminer_paper_baselines().items():
            reports[base_name] = compare_runs_peptideminer(
                manual, base_entries, numerical_bounds=numerical_bounds
            )

        if llm_label is not None:
            label = llm_label
        else:
            models = sorted({e.model for e in llm if e.model})
            label = (
                f"{models[0]} (run {llm_run_identifier})"
                if models
                else f"Our LLM (run {llm_run_identifier})"
            )

        reports[label] = compare_runs_peptideminer(
            manual, llm, numerical_bounds=numerical_bounds
        )
        return reports, format_peptideminer_comparison(reports)

    if session is not None:
        return _run(session)
    with get_session_context() as s:
        return _run(s)

