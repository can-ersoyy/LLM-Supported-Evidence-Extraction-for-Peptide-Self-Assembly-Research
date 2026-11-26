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
    "Δphe": "F",
    "δphe": "F",
    "dphe": "F",
    "deltaphe": "F",
    "dehydrophenylalanine": "F",
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


_DETECTION_SCAN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcryo[- ]?(?:em|electron\s+microscop)", re.IGNORECASE), "Cryo-EM"),
    (
        re.compile(r"\btransmission\s+electron\s+microscop|\btem\b|\bhr-?tem\b", re.IGNORECASE),
        "TEM",
    ),
    (
        re.compile(r"\bscanning\s+electron\s+microscop|\bfe-?sem\b|\bsem\b", re.IGNORECASE),
        "SEM",
    ),
    (re.compile(r"\batomic\s+force\s+microscop|\bafm\b", re.IGNORECASE), "AFM"),
    (re.compile(r"\bconfocal\b", re.IGNORECASE), "Confocal Microscopy"),
    (
        re.compile(
            r"\boptical\s+microscop|\blight\s+microscop|\bpolariz(?:ed|ing)\s+(?:optical\s+)?microscop",
            re.IGNORECASE,
        ),
        "Optical Microscopy",
    ),
    (re.compile(r"\bcircular\s+dichroism\b|\bcd\s+spectr", re.IGNORECASE), "CD"),
    (
        re.compile(r"\bfourier[- ]transform\s+infrared|\bft-?ir\b|\batr-?ir\b", re.IGNORECASE),
        "FTIR",
    ),
    (re.compile(r"\bthioflavin|\btht\b", re.IGNORECASE), "ThT"),
    (re.compile(r"\bnmr\b|\bnuclear\s+magnetic\s+resonance", re.IGNORECASE), "NMR"),
    (
        re.compile(r"\bx-?ray\s+diffraction|\bxrd\b|\bsaxs\b|\bwaxs\b|\bpxrd\b", re.IGNORECASE),
        "XRD",
    ),
    (re.compile(r"\bdynamic\s+light\s+scattering|\bdls\b", re.IGNORECASE), "DLS"),
    (re.compile(r"\buv[-/–]?vis\b", re.IGNORECASE), "UV-Vis"),
    (re.compile(r"\bfluorescence\s+(?:microscop|spectr|assay|emission)", re.IGNORECASE), "Fluorescence"),
    (re.compile(r"\brheolog(?:y|ical)|oscillatory\s+shear|storage\s+modulus", re.IGNORECASE), "Rheology"),
    (re.compile(r"\bcongo\s+red\b", re.IGNORECASE), "Congo Red"),
    (re.compile(r"\braman\s+spectr", re.IGNORECASE), "Raman"),
    (re.compile(r"\bturbidit", re.IGNORECASE), "Turbidity"),
    (
        re.compile(
            r"\bvial\s+inversion|\btube\s+inversion|\binverted\s+(?:vial|tube|test\s+tube)|\binversion\s+(?:test|method)",
            re.IGNORECASE,
        ),
        "Vial Inversion",
    ),
]

_MIXTURE_SEP_RE = re.compile(r"[/+;,:]|\band\b|\bwith\b|\bin\b", re.IGNORECASE)

_N_TERM_CANONICAL: dict[str, str] = {
    "free": "Free",
    "nh2": "Free",
    "h2n": "Free",
    "h": "Free",
    "-h": "Free",
    "unmodified": "Free",
    "none": "Free",
    "ac": "Acetylation",
    "acetyl": "Acetylation",
    "acetylation": "Acetylation",
    "fmoc": "Fmoc",
    "boc": "Boc",
    "t-boc": "Boc",
    "tboc": "Boc",
    "t-butyloxycarbonyl": "Boc",
    "tert-butyloxycarbonyl": "Boc",
    "t-butyloxycarbonyl": "Boc",
    "tert-butyloxycarbonyl": "Boc",
    "cbz": "Cbz",
    "z": "Cbz",
    "nap": "Nap",
    "2-nap": "Nap",
}

_C_TERM_CANONICAL: dict[str, str] = {
    "free": "Free",
    "cooh": "Free",
    "oh": "Free",
    "-oh": "Free",
    "unmodified": "Free",
    "none": "Free",
    "nh2": "Amidation",
    "conh2": "Amidation",
    "amide": "Amidation",
    "amidation": "Amidation",
    "ome": "Methyl ester",
    "och3": "Methyl ester",
    "methyl ester": "Methyl ester",
    "methoxylation": "Methyl ester",
    "oet": "Ethyl ester",
    "oc2h5": "Ethyl ester",
    "ethyl ester": "Ethyl ester",
    "obn": "Benzylation",
    "och2ph": "Benzylation",
    "obzl": "Benzylation",
    "benzyl ester": "Benzylation",
    "benzylation": "Benzylation",
}

_SOLUTION_COMPONENT_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"^(?:1,1,1,3,3,3-)?hexafluoro-?(?:2-)?(?:propanol|isopropanol)$|^hfip$|^hfp$",
            re.IGNORECASE,
        ),
        "HFP",
    ),
    (
        re.compile(
            r"^dimethyl\s*sulfoxide$|^dimethylsulfoxide$|^dmso$", re.IGNORECASE
        ),
        "DMSO",
    ),
    (
        re.compile(
            r"^dimethyl\s*formamide$|^dimethylformamide$|^dmf$", re.IGNORECASE
        ),
        "DMF",
    ),
    (
        re.compile(r"^phosphate[- ]buffered saline$|^pbs(?:\s*1x)?$", re.IGNORECASE),
        "PBS",
    ),
    (
        re.compile(
            r"^(?:deionized|distilled|double[- ]distilled|milli-?q)\s+water$|^h2o$",
            re.IGNORECASE,
        ),
        "H2O",
    ),
    (re.compile(r"^dichloromethane$|^dcm$|^ch2cl2$", re.IGNORECASE), "Dichloromethane"),
    (re.compile(r"^ethyl\s*acetate$|^etoac$", re.IGNORECASE), "Ethyl acetate"),
    (re.compile(r"^chlorobenzene$|^cl-?c6h5$", re.IGNORECASE), "Chlorobenzene"),
    (
        re.compile(r"^acetonitrile$|^mecn$|^ch3cn$", re.IGNORECASE),
        "Acetonitrile",
    ),
    (re.compile(r"^methanol$|^meoh$", re.IGNORECASE), "Methanol"),
    (re.compile(r"^ethanol$|^etoh$", re.IGNORECASE), "Ethanol"),
    (re.compile(r"^chloroform$|^chcl3$", re.IGNORECASE), "Chloroform"),
    (re.compile(r"^tetrahydrofuran$|^thf$", re.IGNORECASE), "THF"),
]


def normalize_terminal_cap(value: str | None, is_n_term: bool) -> str | None:
    """Canonicalize N- or C-terminal modification strings to SAPdb/PeptideMiner standards."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    table = _N_TERM_CANONICAL if is_n_term else _C_TERM_CANONICAL
    return table.get(cleaned.lower(), cleaned)


def canonicalize_solution(value: str | None) -> str | None:
    """Canonicalize solvent/solution abbreviations and formulas (e.g. HFIP -> HFP, CH2Cl2 -> Dichloromethane)."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None

    parts = [p.strip() for p in cleaned.split("/") if p.strip()]
    if len(parts) > 1:
        norm_parts: list[str] = []
        for part in parts:
            if part.lower() == "water":
                norm_parts.append("H2O")
                continue
            replaced = part
            for pat, canon in _SOLUTION_COMPONENT_ALIASES:
                if pat.match(part):
                    replaced = canon
                    break
            norm_parts.append(replaced)
        return "/".join(norm_parts)

    for pat, canon in _SOLUTION_COMPONENT_ALIASES:
        if pat.match(cleaned):
            return canon
    return cleaned


def extract_paper_detection_methods(source_text: str) -> list[str]:
    """Deterministically scan paper text for canonical characterization techniques."""
    if not source_text or not source_text.strip():
        return []
    found: list[str] = []
    for pattern, canonical in _DETECTION_SCAN_PATTERNS:
        if pattern.search(source_text):
            found.append(canonical)
    return found


def _is_single_neat_solvent(solution: str) -> bool:
    """Return True if solution appears to be a neat single-component solvent (ratio '1')."""
    cleaned = solution.strip()
    if not cleaned:
        return False
    return not bool(_MIXTURE_SEP_RE.search(cleaned))


def estimate_ph_from_solution(
    solution: str | None,
    solvent: str | None = None,
) -> float | None:
    """Estimate default pH from solvent/buffer when omitted (PeptideMiner Principle 4)."""
    combined = " ".join(part for part in (solution, solvent) if part).strip().lower()
    if not combined:
        return None
    if re.search(r"\bpbs\b|phosphate[- ]buffered\s+saline|\btris\b|\bhepes\b", combined):
        return 7.4
    if re.search(r"\b(?:naoh|koh|nh4oh|ammonium\s+hydroxide)\b", combined):
        return 11.0
    if re.search(r"\b(?:hcl|tfa|trifluoroacetic\s+acid)\b", combined):
        return 2.0
    if re.search(r"\b(?:acetic\s+acid|gdl|glucono)\b", combined):
        return 4.0
    if re.search(r"\b(?:methanol|meoh)\b", combined) and re.search(
        r"\b(?:water|h2o|aqueous)\b", combined
    ):
        # Explicitly cited in Yang et al. (arXiv:2411.05421v1, p. 9, Ref. 65)
        return 8.3
    return 7.0


_PROTECTING_GROUP_TOKENS: frozenset[str] = frozenset(
    {
        "fmoc",
        "boc",
        "t-boc",
        "cbz",
        "z",
        "ac",
        "acetyl",
        "acetylation",
        "amidation",
        "amide",
        "nh2",
        "conh2",
        "methyl ester",
        "ome",
        "ethyl ester",
        "oet",
        "benzylation",
        "benzyl ester",
        "obn",
        "free",
        "none",
        "unmodified",
    }
)


def sanitize_and_validate_entries(
    entries: Sequence[PeptideLLMEntry],
    aliases: Sequence[PeptideAlias] | None = None,
    source_text: str | None = None,
    default_room_temperature: bool = False,
    estimate_solvent_ph: bool = False,
    strict_terminal_and_conjugate_hygiene: bool = False,
) -> tuple[list[PeptideLLMEntry], DeterministicValidationReport]:
    """Run deterministic chemistry normalization, alias resolution, validation, and dedup."""
    alias_lookup: dict[str, PeptideAlias] = {}
    if aliases:
        for a in aliases:
            alias_lookup[a.label.strip().lower()] = a
            alias_lookup[a.sequence.strip().upper()] = a

    paper_methods = extract_paper_detection_methods(source_text or "")

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

        # 2b. Unify solvent -> solution & canonicalize solution/terminal names
        sol_val = (data.get("solution") or "").strip()
        solv_val = (data.get("solvent") or "").strip()
        if not sol_val and solv_val:
            data["solution"] = solv_val
            sol_val = solv_val
            auto_fixes += 1

        if sol_val:
            canon_sol = canonicalize_solution(sol_val)
            if canon_sol and canon_sol != sol_val:
                data["solution"] = canon_sol
                sol_val = canon_sol
                auto_fixes += 1

        for term_field, is_n in (("n_terminal", True), ("c_terminal", False)):
            t_val = data.get(term_field)
            if t_val is not None:
                canon_t = normalize_terminal_cap(t_val, is_n_term=is_n)
                if canon_t != t_val:
                    data[term_field] = canon_t
                    auto_fixes += 1

        # 2c. Default volume_ratio = [1] for neat single-component solvents
        if sol_val and not data.get("volume_ratio") and _is_single_neat_solvent(sol_val):
            data["volume_ratio"] = [1]
            auto_fixes += 1

        # 2d. Default structural fields when omitted on table rows
        if strict_terminal_and_conjugate_hygiene:
            if data.get("n_terminal") is None:
                data["n_terminal"] = "Free"
                auto_fixes += 1
            if data.get("c_terminal") is None:
                data["c_terminal"] = "Free"
                auto_fixes += 1
            cp_val = data.get("conjugate_partner")
            if isinstance(cp_val, str) and cp_val.strip().lower() in _PROTECTING_GROUP_TOKENS:
                data["conjugate_partner"] = None
                if data.get("peptide_conjugate_mixture") == PeptideConjugateMixture.CONJUGATE:
                    data["peptide_conjugate_mixture"] = PeptideConjugateMixture.PEPTIDE
                auto_fixes += 1
                issues.append(
                    ValidationIssue(
                        entry_index=idx,
                        field="conjugate_partner",
                        severity="info",
                        message=(
                            f"Cleared protecting group '{cp_val}' from conjugate_partner "
                            "(hybrid hygiene)."
                        ),
                        auto_fixed=True,
                    )
                )

        if data.get("linear_cyclic") is None:
            data["linear_cyclic"] = PeptideLinearity.LINEAR
            auto_fixes += 1
        if data.get("peptide_conjugate_mixture") is None:
            data["peptide_conjugate_mixture"] = (
                PeptideConjugateMixture.CONJUGATE
                if data.get("conjugate_partner")
                else PeptideConjugateMixture.PEPTIDE
            )
            auto_fixes += 1
        if (
            strict_terminal_and_conjugate_hygiene
            and data.get("peptide_conjugate_mixture") == PeptideConjugateMixture.PEPTIDE
            and data.get("conjugate_partner") is not None
        ):
            data["conjugate_partner"] = None
            auto_fixes += 1

        # 2e. Convert molar / wt% concentrations to mg/ml via estimated peptide MW
        conc_val = data.get("peptide_concentration")
        conc_unit = (data.get("peptide_concentration_unit") or "").strip()
        if conc_val is not None and conc_unit:
            norm_unit = conc_unit.lower().replace("μ", "u").replace("µ", "u")
            if norm_unit not in {"mg/ml", "mg ml-1", "mg/cm3", "g/l"}:
                lin_val = data.get("linear_cyclic")
                lin_str = lin_val.value if hasattr(lin_val, "value") else lin_val
                converted = convert_concentration_to_mg_ml(
                    float(conc_val),
                    conc_unit,
                    data.get("sequence", ""),
                    n_terminal=data.get("n_terminal"),
                    c_terminal=data.get("c_terminal"),
                    linear_cyclic=lin_str,
                )
                if converted is not None:
                    data["peptide_concentration"] = converted
                    data["peptide_concentration_unit"] = "mg/ml"
                    auto_fixes += 1
                    issues.append(
                        ValidationIssue(
                            entry_index=idx,
                            field="peptide_concentration",
                            severity="info",
                            message=f"Converted {conc_val} {conc_unit} -> {converted} mg/ml using estimated peptide MW.",
                            auto_fixed=True,
                        )
                    )
                else:
                    issues.append(
                        ValidationIssue(
                            entry_index=idx,
                            field="peptide_concentration",
                            severity="warning",
                            message=f"Could not convert concentration {conc_val} {conc_unit} to mg/ml.",
                        )
                    )

        # 2f. Apply SAPdb/PeptideMiner default room temperature (25.0 C) when omitted
        if default_room_temperature and data.get("temperature") is None:
            data["temperature"] = 25.0
            auto_fixes += 1
            issues.append(
                ValidationIssue(
                    entry_index=idx,
                    field="temperature",
                    severity="info",
                    message="Applied default room temperature (25.0 C) per SAPdb/PeptideMiner convention.",
                    auto_fixed=True,
                )
            )

        # 2f-ii. Opt-in PeptideMiner Principle (4): estimate pH from solvent/buffer when omitted
        if estimate_solvent_ph and data.get("ph") is None:
            est_ph = estimate_ph_from_solution(data.get("solution"), data.get("solvent"))
            if est_ph is not None:
                data["ph"] = est_ph
                auto_fixes += 1
                issues.append(
                    ValidationIssue(
                        entry_index=idx,
                        field="ph",
                        severity="info",
                        message=(
                            f"Estimated pH={est_ph} from solution '{data.get('solution')}' "
                            "per PeptideMiner Principle (4)."
                        ),
                        auto_fixed=True,
                    )
                )

        # 2g. Supplement detection_method with deterministically detected paper-wide techniques
        if paper_methods:
            existing_methods = list(data.get("detection_method") or [])
            existing_lower = {m.strip().lower() for m in existing_methods if m and m.strip()}
            added_any = False
            for pm in paper_methods:
                if pm.lower() not in existing_lower:
                    existing_methods.append(pm)
                    existing_lower.add(pm.lower())
                    added_any = True
            if added_any:
                data["detection_method"] = existing_methods
                auto_fixes += 1

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
