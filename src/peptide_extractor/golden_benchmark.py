"""Benchmarking and comparison utilities for LLM extractions vs the golden dataset."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

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

_N_TERM_FREE = {"free", "nh2", "h2n", "h", "-h", "unmodified", "none"}
_C_TERM_FREE = {"free", "cooh", "oh", "-oh", "unmodified", "none"}

_N_TERM_ALIASES = {
    "ac": "acetylation",
    "acetyl": "acetylation",
    "acetylation": "acetylation",
    "fmoc": "fmoc",
    "boc": "boc",
    "cbz": "cbz",
    "z": "cbz",
}

_C_TERM_ALIASES = {
    "nh2": "amidation",
    "conh2": "amidation",
    "amide": "amidation",
    "amidation": "amidation",
    "ome": "methyl ester",
    "och3": "methyl ester",
    "methyl ester": "methyl ester",
    "oet": "ethyl ester",
    "ethyl ester": "ethyl ester",
    "obn": "benzylation",
    "benzyl ester": "benzylation",
    "benzylation": "benzylation",
}

_SOLVENT_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:1,1,1,3,3,3-)?hexafluoro-?(?:2-)?(?:propanol|isopropanol)\b|\bhfip\b|\bhfp\b"
        ),
        "hfp",
    ),
    (re.compile(r"\bdimethyl\s*sulfoxide\b|\bdimethylsulfoxide\b|\bdmso\b"), "dmso"),
    (re.compile(r"\bdimethyl\s*formamide\b|\bdimethylformamide\b|\bdmf\b"), "dmf"),
    (re.compile(r"\bphosphate[- ]buffered saline\b|\bpbs(?:\s*1x)?\b"), "pbs"),
    (
        re.compile(
            r"\b(?:deionized\s+|distilled\s+|milli-?q\s+)?water\b|\bh2o\b|\baqueous(?:\s+dispersion|\s+buffer)?\b"
        ),
        "h2o",
    ),
    (re.compile(r"\bdichloromethane\b|\bdcm\b"), "dcm"),
    (re.compile(r"\btetrahydrofuran\b|\bthf\b"), "thf"),
    (re.compile(r"\bacetonitrile\b|\bmecn\b|\bch3cn\b"), "acetonitrile"),
    (re.compile(r"\bmethanol\b|\bmeoh\b"), "methanol"),
    (re.compile(r"\bethanol\b|\betoh\b"), "ethanol"),
    (re.compile(r"\bchloroform\b|\bchcl3\b"), "chloroform"),
    (re.compile(r"\bacetone\b"), "acetone"),
    (re.compile(r"\btoluene\b"), "toluene"),
    (re.compile(r"\bethyl\s*acetate\b"), "ethyl acetate"),
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
    return _N_TERM_ALIASES.get(cleaned, cleaned)


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

    if field_name == "n_terminal":
        return exact, normalize_n_terminal(mv) == normalize_n_terminal(lv)
    if field_name == "c_terminal":
        return exact, normalize_c_terminal(mv) == normalize_c_terminal(lv)
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

