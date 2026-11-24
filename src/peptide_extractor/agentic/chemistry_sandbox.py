"""Deterministic chemistry & schema validation sandbox (zero-LLM cost).

Provides:
- 3-letter to 1-letter amino acid conversion
- N-/C-terminal cap & cyclic marker extraction from leaked sequence strings
- Approximate peptide molecular weight calculation & molar -> mg/ml conversion
- Physical range & schema consistency validation
- Cross-pass deduplication (merging prose & table duplicates)
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from peptide_extractor.agentic.schemas import (
    DeterministicValidationReport,
    PeptideAlias,
    ValidationIssue,
)
from peptide_extractor.db.peptide_entry import (
    EvidenceSpan,
    PeptideConjugateMixture,
    PeptideLinearity,
    PeptideLLMEntry,
)

# Canonical 20 L/D amino acids + common non-canonical single-letter codes
VALID_AA_CODES = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")

THREE_TO_ONE_AA: dict[str, str] = {
    "ala": "A",
    "arg": "R",
    "asn": "N",
    "asp": "D",
    "cys": "C",
    "gln": "Q",
    "glu": "E",
    "gly": "G",
    "his": "H",
    "ile": "I",
    "leu": "L",
    "lys": "K",
    "met": "M",
    "phe": "F",
    "pro": "P",
    "ser": "S",
    "thr": "T",
    "trp": "W",
    "tyr": "Y",
    "val": "V",
}

# Average residue masses (Da, minus H2O) for deterministic molar -> mg/ml conversion
AA_RESIDUE_MASS_DA: dict[str, float] = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "Q": 128.1307,
    "E": 129.1155,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}

# Net mass delta (Da) relative to unmodified H- (N-term) or -OH (C-term)
N_CAP_MASS_DELTA_DA: dict[str, float] = {
    "free": 0.0,
    "acetylation": 42.0367,
    "fmoc": 222.2400,
    "boc": 100.1158,
    "cbz": 134.1326,
    "nap": 168.1912,
    "pyrene": 228.2450,
}

C_CAP_MASS_DELTA_DA: dict[str, float] = {
    "free": 0.0,
    "amidation": -0.9848,
    "methyl ester": 14.0266,
    "ethyl ester": 28.0532,
    "benzylation": 90.1226,
}

_N_PREFIX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(?:fmoc)\s*[-–]\s*", re.IGNORECASE), "Fmoc"),
    (re.compile(r"^(?:boc|t-?boc)\s*[-–]\s*", re.IGNORECASE), "Boc"),
    (re.compile(r"^(?:cbz|z)\s*[-–]\s*", re.IGNORECASE), "Cbz"),
    (re.compile(r"^(?:ac|acetyl)\s*[-–]\s*", re.IGNORECASE), "Acetylation"),
    (re.compile(r"^(?:nap|2-nap)\s*[-–]\s*", re.IGNORECASE), "Nap"),
    (re.compile(r"^(?:h2n|nh2|h)\s*[-–]\s*", re.IGNORECASE), "Free"),
]

_C_SUFFIX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\s*[-–]\s*(?:conh2|nh2|amide)$", re.IGNORECASE), "Amidation"),
    (re.compile(r"\s*[-–]\s*(?:cooh|oh)$", re.IGNORECASE), "Free"),
    (re.compile(r"\s*[-–]\s*(?:ome|och3)$", re.IGNORECASE), "Methyl ester"),
    (re.compile(r"\s*[-–]\s*(?:oet|oc2h5)$", re.IGNORECASE), "Ethyl ester"),
    (re.compile(r"\s*[-–]\s*(?:obn|och2ph|obzl)$", re.IGNORECASE), "Benzylation"),
]

_CYCLIC_RE = re.compile(r"^(?:cyclo|c)\s*[\(\[](.+)[\)\]]$", re.IGNORECASE)


def estimate_peptide_mw(
    sequence: str,
    n_terminal: str | None = None,
    c_terminal: str | None = None,
    linear_cyclic: str | None = None,
) -> float | None:
    """Estimate average molecular weight (Da) from 1-letter sequence and terminal caps."""
    seq = (sequence or "").strip().upper()
    if not seq or any(ch not in AA_RESIDUE_MASS_DA for ch in seq):
        return None

    mw = sum(AA_RESIDUE_MASS_DA[ch] for ch in seq)
    is_cyclic = (linear_cyclic or "").strip().lower() == "cyclic"
    if not is_cyclic:
        mw += 18.0153  # H2O for free N- and C-termini
        n_key = (n_terminal or "free").strip().lower()
        c_key = (c_terminal or "free").strip().lower()
        mw += N_CAP_MASS_DELTA_DA.get(n_key, 0.0)
        mw += C_CAP_MASS_DELTA_DA.get(c_key, 0.0)
    return round(mw, 3)


def convert_concentration_to_mg_ml(
    value: float,
    unit: str,
    sequence: str,
    n_terminal: str | None = None,
    c_terminal: str | None = None,
    linear_cyclic: str | None = None,
) -> float | None:
    """Deterministically convert concentration units (mg/ml, wt%, M, mM, uM) to mg/ml."""
    u = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if u in {"mg/ml", "mg ml-1", "mg/cm3", "g/l"}:
        return round(float(value), 4)
    if u in {"ug/ml", "ug ml-1", "mg/l"}:
        return round(float(value) / 1000.0, 6)
    if u in {"wt%", "% w/v", "w/v%", "%(w/v)", "% (w/v)"}:
        return round(float(value) * 10.0, 4)

    mw = estimate_peptide_mw(sequence, n_terminal, c_terminal, linear_cyclic)
    if mw is None:
        return None
    if u in {"m", "mol/l"}:
        return round(float(value) * mw, 4)
    if u in {"mm", "mmol/l"}:
        return round(float(value) * mw / 1000.0, 4)
    if u in {"um", "umol/l"}:
        return round(float(value) * mw / 1_000_000.0, 6)
    if u in {"nm", "nmol/l"}:
        return round(float(value) * mw / 1_000_000_000.0, 9)
    return None


def split_sequence_and_caps(
    raw_sequence: str,
) -> tuple[str, str | None, str | None, PeptideLinearity | None, bool]:
    """Strip leaked terminal caps / cyclic wrappers and convert 3-letter codes to 1-letter.

    Returns (cleaned_sequence, detected_n_term, detected_c_term, detected_linearity, was_modified).
    """
    original = (raw_sequence or "").strip()
    seq = original
    detected_n: str | None = None
    detected_c: str | None = None
    detected_lin: PeptideLinearity | None = None

    # 1. Check cyclic wrapper, e.g. cyclo(KLKL) or c[RGDfK]
    m_cyc = _CYCLIC_RE.match(seq)
    if m_cyc:
        seq = m_cyc.group(1).strip()
        detected_lin = PeptideLinearity.CYCLIC

    # 2. Strip known N-terminal prefixes
    for pattern, canonical in _N_PREFIX_PATTERNS:
        if pattern.search(seq):
            seq = pattern.sub("", seq, count=1).strip()
            detected_n = canonical
            break

    # 3. Strip known C-terminal suffixes
    for pattern, canonical in _C_SUFFIX_PATTERNS:
        if pattern.search(seq):
            seq = pattern.sub("", seq, count=1).strip()
            detected_c = canonical
            break

    # 4. Convert 3-letter dash/space-separated sequence (e.g. Ser-Val-Gly or Phe-Phe)
    tokens = [t.strip() for t in re.split(r"[-–\s]+", seq) if t.strip()]
    if len(tokens) >= 2 and all(t.lower() in THREE_TO_ONE_AA for t in tokens):
        seq = "".join(THREE_TO_ONE_AA[t.lower()] for t in tokens)
    elif len(tokens) >= 2 and all(len(t) == 1 and t.upper() in VALID_AA_CODES for t in tokens):
        # Handle single-letter hyphenated sequences like "S-V-G" or "K-L-V-F-F"
        seq = "".join(t.upper() for t in tokens)
    else:
        seq = re.sub(r"[\s\-–]+", "", seq).upper()

    was_modified = (
        seq != original
        or detected_n is not None
        or detected_c is not None
        or detected_lin is not None
    )
    return seq, detected_n, detected_c, detected_lin, was_modified


def _entry_richness(entry: PeptideLLMEntry) -> int:
    """Count non-null fields + grounded evidence tags to pick the richer duplicate."""
    data = entry.model_dump(exclude={"evidence_span"}, exclude_none=True)
    ev_count = 0
    if entry.evidence_span is not None:
        ev_count = len(entry.evidence_span.model_dump(exclude_none=True))
    return len(data) * 2 + ev_count


def _merge_two_entries(primary: PeptideLLMEntry, secondary: PeptideLLMEntry) -> PeptideLLMEntry:
    """Fill missing fields in primary from secondary without overwriting populated values."""
    p_data = primary.model_dump()
    s_data = secondary.model_dump()
    for key, s_val in s_data.items():
        if key == "evidence_span":
            continue
        if p_data.get(key) is None and s_val is not None:
            p_data[key] = s_val

    p_ev = (primary.evidence_span or EvidenceSpan()).model_dump()
    s_ev = (secondary.evidence_span or EvidenceSpan()).model_dump()
    for key, s_tag in s_ev.items():
        if p_ev.get(key) is None and s_tag is not None:
            p_ev[key] = s_tag
    if any(v is not None for v in p_ev.values()):
        p_data["evidence_span"] = EvidenceSpan(**p_ev)

    return PeptideLLMEntry(**p_data)


def _dedup_key(entry: PeptideLLMEntry) -> tuple:
    """Canonical deduplication key for cross-pass (prose + table) duplicate detection."""
    sol = (entry.solution or "").strip().lower()
    solv = (entry.solvent or "").strip().lower()
    nt = (entry.n_terminal or "").strip().lower()
    ct = (entry.c_terminal or "").strip().lower()
    etype = entry.type.value if hasattr(entry.type, "value") else (entry.type or "")
    vr = tuple(entry.volume_ratio) if entry.volume_ratio else None
    temp = round(entry.temperature, 1) if entry.temperature is not None else None
    ph = round(entry.ph, 2) if entry.ph is not None else None
    conc = (
        round(entry.peptide_concentration, 3)
        if entry.peptide_concentration is not None
        else None
    )
    return (
        entry.sequence.strip().upper(),
        nt,
        ct,
        etype.lower(),
        sol,
        solv,
        vr,
        temp,
        ph,
        conc,
    )


def sanitize_and_validate_entries(
    entries: Sequence[PeptideLLMEntry],
    aliases: Sequence[PeptideAlias] | None = None,
) -> tuple[list[PeptideLLMEntry], DeterministicValidationReport]:
    """Run deterministic chemistry normalization, alias resolution, validation, and dedup."""
    alias_lookup: dict[str, PeptideAlias] = {}
    if aliases:
        for a in aliases:
            alias_lookup[a.label.strip().lower()] = a
            alias_lookup[a.sequence.strip().upper()] = a

    issues: list[ValidationIssue] = []
    auto_fixes = 0
    cleaned: list[PeptideLLMEntry] = []

    for idx, entry in enumerate(entries):
        data = entry.model_dump()
        raw_seq = (entry.sequence or "").strip()

        # 1. Alias resolution (e.g. if table emitted "1a" or "Peptide 2" in sequence field)
        matched_alias = alias_lookup.get(raw_seq.lower()) or alias_lookup.get(raw_seq.upper())
        if matched_alias is not None:
            if raw_seq.upper() != matched_alias.sequence.strip().upper():
                data["sequence"] = matched_alias.sequence.strip().upper()
                auto_fixes += 1
                issues.append(
                    ValidationIssue(
                        entry_index=idx,
                        field="sequence",
                        severity="info",
                        message=f"Resolved peptide alias '{raw_seq}' -> '{data['sequence']}'.",
                        auto_fixed=True,
                    )
                )
            for field_name in (
                "n_terminal",
                "c_terminal",
                "linear_cyclic",
                "chirality",
                "peptide_conjugate_mixture",
                "conjugate_partner",
            ):
                alias_val = getattr(matched_alias, field_name, None)
                if data.get(field_name) is None and alias_val is not None:
                    data[field_name] = alias_val
                    auto_fixes += 1

        # 2. Sequence & terminal cap splitting / 3-letter -> 1-letter conversion
        seq_clean, det_n, det_c, det_lin, was_mod = split_sequence_and_caps(
            data.get("sequence", "")
        )
        if was_mod:
            data["sequence"] = seq_clean
            if det_n is not None and not data.get("n_terminal"):
                data["n_terminal"] = det_n
            if det_c is not None and not data.get("c_terminal"):
                data["c_terminal"] = det_c
            if det_lin is not None and not data.get("linear_cyclic"):
                data["linear_cyclic"] = det_lin
            auto_fixes += 1
            issues.append(
                ValidationIssue(
                    entry_index=idx,
                    field="sequence",
                    severity="info",
                    message=(
                        f"Normalized sequence '{raw_seq}' -> '{seq_clean}' "
                        f"(n_terminal={data.get('n_terminal')}, c_terminal={data.get('c_terminal')})."
                    ),
                    auto_fixed=True,
                )
            )

        # 3. Validate IUPAC characters & length
        final_seq = data.get("sequence", "")
        if not final_seq:
            issues.append(
                ValidationIssue(
                    entry_index=idx,
                    field="sequence",
                    severity="error",
                    message="Empty peptide sequence.",
                )
            )
        else:
            invalid_chars = sorted(set(final_seq) - VALID_AA_CODES)
            if invalid_chars:
                issues.append(
                    ValidationIssue(
                        entry_index=idx,
                        field="sequence",
                        severity="error",
                        message=(
                            f"Sequence '{final_seq}' contains non-IUPAC character(s): {invalid_chars}. "
                            "Terminal caps or labels should be moved out of sequence."
                        ),
                    )
                )

        # 4. Physical range & biochemical sanity checks
        ph_val = data.get("ph")
        if ph_val is not None and not (0.0 <= float(ph_val) <= 14.0):
            issues.append(
                ValidationIssue(
                    entry_index=idx,
                    field="ph",
                    severity="error",
                    message=f"pH value {ph_val} is outside valid physical range [0, 14].",
                )
            )

        temp_val = data.get("temperature")
        if temp_val is not None and not (-80.0 <= float(temp_val) <= 250.0):
            issues.append(
                ValidationIssue(
                    entry_index=idx,
                    field="temperature",
                    severity="error",
                    message=f"Temperature {temp_val} °C is outside plausible range [-80, 250].",
                )
            )

        conc_val = data.get("peptide_concentration")
        if conc_val is not None and float(conc_val) <= 0.0:
            issues.append(
                ValidationIssue(
                    entry_index=idx,
                    field="peptide_concentration",
                    severity="error",
                    message=f"Concentration {conc_val} mg/ml must be positive.",
                )
            )

        inc_val = data.get("incubation_period")
        if inc_val is not None and float(inc_val) < 0.0:
            issues.append(
                ValidationIssue(
                    entry_index=idx,
                    field="incubation_period",
                    severity="error",
                    message=f"Incubation period {inc_val} minutes cannot be negative.",
                )
            )

        pcm = data.get("peptide_conjugate_mixture")
        if pcm == PeptideConjugateMixture.PEPTIDE and data.get("conjugate_partner"):
            issues.append(
                ValidationIssue(
                    entry_index=idx,
                    field="conjugate_partner",
                    severity="warning",
                    message="conjugate_partner is populated even though peptide_conjugate_mixture='peptide'.",
                )
            )

        cleaned.append(PeptideLLMEntry(**data))

    # 5. Cross-pass deduplication
    deduped_map: dict[tuple, PeptideLLMEntry] = {}
    duplicates_merged = 0
    for entry in cleaned:
        key = _dedup_key(entry)
        if key not in deduped_map:
            deduped_map[key] = entry
        else:
            existing = deduped_map[key]
            if _entry_richness(entry) > _entry_richness(existing):
                deduped_map[key] = _merge_two_entries(entry, existing)
            else:
                deduped_map[key] = _merge_two_entries(existing, entry)
            duplicates_merged += 1

    final_entries = list(deduped_map.values())
    report = DeterministicValidationReport(
        total_before=len(entries),
        total_after=len(final_entries),
        duplicates_merged=duplicates_merged,
        auto_fixes_applied=auto_fixes,
        issues=issues,
    )
    return final_entries, report
