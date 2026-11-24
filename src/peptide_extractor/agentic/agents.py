"""Specialized agent roles for the multi-stage agentic extraction workflow.

Each agent wraps LLMController with a narrow, single-hop prompt and strict
Pydantic structured output schema so small language models (SLMs) and larger
models alike stay grounded and schema-compliant.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from peptide_extractor.agentic.schemas import (
    AgenticExtractionConfig,
    DeterministicValidationReport,
    EntityRegistryResult,
    PeptideAlias,
    RefinerPatchResult,
    ReviewerFeedback,
)
from peptide_extractor.db.peptide_entry import PeptideExtractionResult, PeptideLLMEntry
from peptide_extractor.extraction import (
    LOCATION_GROUNDING_RULES,
    TableExtractionResult,
    _format_shared_conditions_summary,
)
from peptide_extractor.llm import LLMController
from peptide_extractor.utils import retry_on_external_error

ENTITY_SCOUT_SYSTEM_PROMPT = f"""
You are an expert peptide chemistry scout. Read the scientific paper text and extract TWO concise registries:
1. `peptides`: EVERY experimentally studied peptide or compound alias in the paper.
   - Map paper labels/shorthands (e.g., "1a", "Peptide 2", "P3", "Fmoc-FF", "cyclo(KL)") to their uppercase 1-letter IUPAC `sequence` (convert 3-letter codes like Ser-Val-Gly to SVG).
   - Separate N-terminal (`n_terminal`, e.g. Fmoc, Boc, Acetylation, Free) and C-terminal (`c_terminal`, e.g. Amidation, Free, Methyl ester, OBn) protecting groups out of `sequence`.
   - ALWAYS set `linear_cyclic` ("linear" or "cyclic") and `peptide_conjugate_mixture` ("peptide", "conjugate", or "mixture", plus `conjugate_partner` if applicable) for EVERY peptide.
   - Set `chirality` ("l", "d", "heterochiral", "racemic", "achiral") when known.
2. `shared_conditions`: Paper-wide experimental defaults that apply across screening/assembly experiments:
   - `temperature` in °C (RT -> 25.0), `incubation_period` in minutes (24 h -> 1440), `ph`, `peptide_concentration` in mg/ml, `solution`, `solvent`, `volume_ratio`, `linear_cyclic`, `peptide_conjugate_mixture`, `preparation_protocol`, along with exact bracketed `evidence_span` location tags.
   - `detection_method`: List ALL experimental characterization techniques used anywhere in Methods, Results, Tables, or Figure Captions (e.g., Vial Inversion, TEM, Cryo-EM, SEM, AFM, CD, FTIR, NMR, XRD, Rheology, Optical Microscopy, Fluorescence, ThT, UV-Vis, DLS).

Do NOT invent peptides that are only mentioned in the Introduction as prior literature; include ONLY peptides experimentally studied in this paper.
{LOCATION_GROUNDING_RULES}
"""

PROSE_EXTRACTOR_SYSTEM_PROMPT = f"""
You are an expert scientific curator extracting experimental peptide self-assembly records from paper text.
You are given:
- A verified `Peptide Registry` of peptides/aliases studied in this paper
- Paper-wide `Shared Conditions`
- The location-annotated paper text

RULES:
1. Extract EVERY experimental `(peptide, condition)` combination reported in the prose/text—including negative/non-assembling controls (`type: "none"`).
2. If tables are being extracted separately (as indicated in the prompt), emit ONLY prose-reported entries that are NOT already covered by the tables; otherwise extract ALL entries from both prose and tables.
3. Always use the uppercase 1-letter `sequence` and separate `n_terminal` / `c_terminal` caps from the Peptide Registry.
4. Always populate `solution` with the solvent/buffer system (e.g., "DMSO/H2O", "PBS", "toluene") and set `volume_ratio` (e.g., `[1]` for neat single solvents, `[5, 95]` for mixtures).
5. Carry over applicable `shared_conditions` (temperature, pH, incubation_period, linear_cyclic, peptide_conjugate_mixture, detection_method, preparation_protocol) when not overridden locally.
{LOCATION_GROUNDING_RULES}
"""

SCOPED_TABLE_EXTRACTOR_SYSTEM_PROMPT = """
You are an expert scientific curator extracting experimental peptide self-assembly data from a single table into a compact row-grouped schema.

CRITICAL RULES:
1. Extract EVERY SINGLE ROW and EVERY SINGLE TESTED CELL in the table—never summarize, sample, or truncate.
2. Use the provided `Peptide Registry` to resolve compound numbers/aliases (e.g., "1a", "P2") to their exact uppercase 1-letter `sequence`, `n_terminal`, and `c_terminal`.
3. SCREENING-TABLE SYMBOLS & ASSEMBLY TYPE (CRITICAL):
   - Read the table legend/footnotes carefully!
   - Gelation symbols such as `G`, `TG` (transparent gel), `OG` (opaque gel), `CG` (clear gel), `PG` (partial gel), `WG` (weak gel), `HG` (hydrogel), or a numeric CGC/MGC value indicate SELF-ASSEMBLY — NEVER mark these as `type: "none"`!
   - Use `type: "organogel"` for gels in organic solvents (e.g., toluene, chloroform, ethyl acetate, hexane, chlorobenzene, benzene, DMF, DMSO, methanol, ethanol, acetone, cyclohexane) and `type: "hydrogel"` for gels in water or aqueous buffers (PBS, Tris, H2O).
   - Use `type: "none"` ONLY for non-assembling outcomes: `S` / `Sol` (soluble/clear solution), `I` (insoluble), `P` / `Ppt` (precipitate), `NG` (no gel), `-` (no assembly).
4. SOLVENT & RATIO FIELDS (CRITICAL):
   - ALWAYS put the tested solvent/buffer name (from column headers or rows) into `solution` (e.g., `solution: "toluene"`, `solution: "water"`, `solution: "DMSO/PBS"`).
   - For neat single solvents, set `volume_ratio: [1]`. For solvent mixtures, set `volume_ratio` (e.g., `[5, 95]`).
5. Do NOT repeat unchanged paper-wide defaults (`temperature`, `incubation_period`, `preparation_protocol`, `detection_method`) unless the table or cell overrides them.
"""

REVIEWER_SYSTEM_PROMPT = """
You are an independent scientific data verification auditor (CONSTRUCT-style Reviewer).
You did NOT generate the candidate extraction. Your job is to critically audit the candidate peptide self-assembly entries against:
1. The source paper text (and Peptide Registry)
2. The deterministic chemistry & schema validation report
3. The complete per-sequence coverage summary + detailed entry preview

AUDIT PROCEDURE (follow in order):
Step 1 — Flag Specific Errors (`issues`):
- Check for hallucinated sequences (e.g., prior-work peptides from Introduction, or broken labels left in `sequence`).
- Check for misattributed conditions (wrong solvent, pH, temperature, concentration, or assembly `type`).
- Check for missing/wrong `n_terminal` or `c_terminal` caps (e.g., paper states all peptides are C-terminally amidated `-NH2` or N-terminally `Fmoc`/`Ac` protected, yet entries leave them null or wrong).
- Check for missing `solution`, `volume_ratio`, or `detection_method` values explicitly stated in the paper.
- Check for unresolved deterministic validation errors.

Step 2 — Recall Audit (`missing_sequences`):
- Compare the Per-Sequence Coverage Summary against the Peptide Registry and paper text. List any experimentally tested peptide sequences completely missing from the candidate entries.

Step 3 — Verdict (`approved`, `trust_score`, `revision_instructions`):
- Compute `trust_score` in [0.0, 1.0].
- Set `approved = True` ONLY if there are no critical errors, no missing experimental peptides, and `trust_score` is high.
- If `approved = False`, write concise, concrete `revision_instructions` specifying exact patches (which entry indices to update/delete or which missing entries to add).
- IMPORTANT: Do NOT ask to delete valid table-extracted rows just because they are summarized rather than shown in full JSON.
"""

REFINER_SYSTEM_PROMPT = f"""
You are a precision scientific data Refiner emitting ONLY targeted patches (`RefinerPatchResult`).
You receive:
1. The original paper text (and Peptide Registry)
2. A per-sequence summary + indexed preview of current candidate entries
3. Specific feedback from an independent Reviewer and deterministic chemistry validator

CRITICAL PATCH RULES:
1. Emit ONLY minimal, targeted changes — untouched existing entries are preserved automatically!
   - `update_fields`: list of `{{"entry_index": i, "field": "...", "value": ...}}` to fix wrong/missing fields on existing indexed entries.
   - `add_entries`: full `PeptideLLMEntry` objects ONLY for genuinely missing peptides or conditions.
   - `delete_indices`: 0-based indices ONLY for entries explicitly confirmed as hallucinated or cited prior-work (never delete valid screening rows).
2. Ensure any added or updated `sequence` is uppercase 1-letter IUPAC with terminal caps placed in `n_terminal` and `c_terminal`, and `solution` populated.
{LOCATION_GROUNDING_RULES}
"""


def format_entity_registry_summary(peptides: Sequence[PeptideAlias]) -> str:
    """Format PeptideRegistry into a compact prompt block for downstream workers."""
    if not peptides:
        return "No peptide aliases pre-registered."
    lines = []
    for p in peptides:
        parts = [f"label={p.label!r} -> seq={p.sequence}"]
        if p.n_terminal:
            parts.append(f"N={p.n_terminal}")
        if p.c_terminal:
            parts.append(f"C={p.c_terminal}")
        if p.linear_cyclic:
            val = p.linear_cyclic.value if hasattr(p.linear_cyclic, "value") else p.linear_cyclic
            parts.append(f"topology={val}")
        if p.chirality:
            parts.append(f"chirality={p.chirality}")
        if p.peptide_conjugate_mixture:
            val = (
                p.peptide_conjugate_mixture.value
                if hasattr(p.peptide_conjugate_mixture, "value")
                else p.peptide_conjugate_mixture
            )
            parts.append(f"kind={val}")
        if p.conjugate_partner:
            parts.append(f"partner={p.conjugate_partner}")
        if p.evidence_tag:
            parts.append(f"loc={p.evidence_tag}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _format_entries_summary_for_review(
    entries: Sequence[PeptideLLMEntry],
    flagged_indices: set[int] | None = None,
    max_preview: int = 40,
) -> str:
    """Format a complete per-sequence coverage summary plus an indexed preview of entries.

    Guarantees ReviewerAgent and RefinerAgent see every extracted sequence and row count
    even on 200+ row screening tables without overflowing context or dropping rows.
    """
    if not entries:
        return "No entries extracted."

    by_seq: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "assembling": 0, "none": 0, "n": set(), "c": set(), "sol_cov": 0}
    )
    for e in entries:
        info = by_seq[e.sequence.upper()]
        info["total"] += 1
        etype = e.type.value if hasattr(e.type, "value") else (e.type or "")
        if etype.lower() == "none":
            info["none"] += 1
        else:
            info["assembling"] += 1
        if e.n_terminal:
            info["n"].add(e.n_terminal)
        if e.c_terminal:
            info["c"].add(e.c_terminal)
        if e.solution or e.solvent:
            info["sol_cov"] += 1

    summary_lines = [f"Per-Sequence Coverage Summary ({len(entries)} total rows across {len(by_seq)} sequences):"]
    for seq, info in sorted(by_seq.items()):
        n_str = "/".join(sorted(info["n"])) or "None"
        c_str = "/".join(sorted(info["c"])) or "None"
        summary_lines.append(
            f"- {seq}: total={info['total']} (assembling={info['assembling']}, none={info['none']}), "
            f"N={n_str}, C={c_str}, solution_populated={info['sol_cov']}/{info['total']}"
        )

    # Prioritize flagged indices, then fill up to max_preview evenly
    flagged = sorted(i for i in (flagged_indices or set()) if 0 <= i < len(entries))
    selected_indices: list[int] = list(flagged[:max_preview])
    selected_set = set(selected_indices)
    for i in range(len(entries)):
        if len(selected_indices) >= max_preview:
            break
        if i not in selected_set:
            selected_indices.append(i)
            selected_set.add(i)

    preview_payload = []
    for idx in sorted(selected_indices):
        d = entries[idx].model_dump(exclude_none=True, mode="json")
        d["_index"] = idx
        preview_payload.append(d)

    preview_text = json.dumps(preview_payload, indent=2)
    omitted = len(entries) - len(preview_payload)
    if omitted > 0:
        preview_text += (
            f"\n... ({omitted} additional entries summarized above; untouched entries are preserved automatically)"
        )
    return "\n".join(summary_lines) + "\n\nDetailed Entry Preview:\n" + preview_text


class EntityScoutAgent:
    """Stage 1 agent: builds the PeptideRegistry and paper-wide SharedExperimentalConditions."""

    def __init__(self, config: AgenticExtractionConfig):
        self.config = config
        self.controller = LLMController(
            provider=config.scout_provider or config.provider,
            model_name=config.scout_model_name or config.model_name,
            structured_output=EntityRegistryResult,
            system_prompt=ENTITY_SCOUT_SYSTEM_PROMPT,
            base_url=config.base_url,
            stream=config.stream,
        )

    def run(
        self,
        paper_input: str | bytes | Path,
        filename: str = "paper.pdf",
    ) -> EntityRegistryResult:
        return retry_on_external_error(
            self.controller.extract_peptides,
            paper_input,
            stream=self.config.stream,
            filename=filename,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            context=f"agentic scout {filename}",
        )


class ProseExtractorAgent:
    """Stage 2A agent: extracts experimental entries from narrative text (or full paper)."""

    def __init__(self, config: AgenticExtractionConfig):
        self.config = config
        self.controller = LLMController(
            provider=config.provider,
            model_name=config.model_name,
            structured_output=PeptideExtractionResult,
            system_prompt=PROSE_EXTRACTOR_SYSTEM_PROMPT,
            base_url=config.base_url,
            stream=config.stream,
        )

    def run(
        self,
        paper_input: str | bytes | Path,
        registry: EntityRegistryResult | None = None,
        tables_extracted_separately: bool = False,
        filename: str = "paper.pdf",
    ) -> PeptideExtractionResult:
        if isinstance(paper_input, (bytes, Path)):
            return retry_on_external_error(
                self.controller.extract_peptides,
                paper_input,
                stream=self.config.stream,
                filename=filename,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay,
                context=f"agentic prose extractor {filename}",
            )

        scope_note = (
            "Tables are being extracted separately from images—extract ONLY prose-reported entries not covered by tables."
            if tables_extracted_separately
            else "Extract ALL experimental entries reported anywhere in the text and inline tables."
        )
        reg_summary = (
            format_entity_registry_summary(registry.peptides) if registry else "None provided."
        )
        shared_summary = (
            _format_shared_conditions_summary(registry.shared_conditions)
            if registry
            else "None provided."
        )
        prompt_text = (
            f"{scope_note}\n\n"
            f"Peptide Registry:\n{reg_summary}\n\n"
            f"Paper-wide Shared Conditions:\n{shared_summary}\n\n"
            f"Paper Text:\n{paper_input}"
        )
        return retry_on_external_error(
            self.controller.extract_peptides,
            prompt_text,
            stream=self.config.stream,
            filename=filename,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            context=f"agentic prose extractor {filename}",
        )


class TableExtractorAgent:
    """Stage 2B agent: extracts a single table image conditioned on PeptideRegistry + SharedDefaults + footnotes."""

    def __init__(self, config: AgenticExtractionConfig):
        self.config = config
        self.controller = LLMController(
            provider=config.table_provider or config.provider,
            model_name=config.table_model_name or config.model_name,
            structured_output=TableExtractionResult,
            system_prompt=SCOPED_TABLE_EXTRACTOR_SYSTEM_PROMPT,
            base_url=config.base_url,
            stream=config.stream,
        )

    def run(
        self,
        table_id: str,
        caption_text: str,
        image_block: dict[str, Any],
        registry: EntityRegistryResult | None = None,
        table_context: str = "",
        filename: str = "paper.pdf",
    ) -> TableExtractionResult:
        reg_summary = (
            format_entity_registry_summary(registry.peptides) if registry else "None provided."
        )
        shared_summary = (
            _format_shared_conditions_summary(registry.shared_conditions)
            if registry
            else "None specified."
        )
        context_block = (
            f"\nSurrounding Table Legend / Footnotes from Paper:\n{table_context}\n"
            if table_context.strip()
            else ""
        )
        blocks = [
            {
                "type": "text",
                "text": (
                    f"Peptide Registry (resolve aliases/caps from here):\n{reg_summary}\n\n"
                    f"Paper-wide shared conditions (merged automatically; only override if this table differs):\n"
                    f"{shared_summary}\n\n"
                    f"Table location ID: [{table_id}]\n"
                    f"{caption_text}\n"
                    f"{context_block}"
                    "Extract EVERY row and EVERY tested cell from the attached table image. "
                    "Always populate `solution` with the tested solvent/buffer name and `volume_ratio: [1]` for neat solvents."
                ),
            },
            image_block,
        ]
        return retry_on_external_error(
            self.controller.extract_peptides,
            blocks,
            stream=self.config.stream,
            filename=filename,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            context=f"agentic table [{table_id}] {filename}",
        )


class ReviewerAgent:
    """Stage 4A agent: evidence-independent CONSTRUCT-style auditor & recall checker."""

    def __init__(self, config: AgenticExtractionConfig):
        self.config = config
        self.controller = LLMController(
            provider=config.reviewer_provider or config.provider,
            model_name=config.reviewer_model_name or config.model_name,
            structured_output=ReviewerFeedback,
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            base_url=config.reviewer_base_url or config.base_url,
            stream=config.stream,
        )

    def run(
        self,
        source_text: str,
        entries: Sequence[PeptideLLMEntry],
        validation_report: DeterministicValidationReport,
        registry: EntityRegistryResult | None = None,
        filename: str = "paper.pdf",
    ) -> ReviewerFeedback:
        reg_summary = (
            format_entity_registry_summary(registry.peptides) if registry else "None provided."
        )
        flagged = {
            issue.entry_index
            for issue in validation_report.issues
            if issue.entry_index is not None and not issue.auto_fixed
        }
        review_prompt = (
            f"Peptide Registry:\n{reg_summary}\n\n"
            f"Deterministic Chemistry & Schema Validation Report:\n"
            f"{validation_report.to_prompt_summary()}\n\n"
            f"Candidate Extracted Entries:\n"
            f"{_format_entries_summary_for_review(entries, flagged_indices=flagged)}\n\n"
            f"Source Paper Text:\n{source_text}"
        )
        return retry_on_external_error(
            self.controller.extract_peptides,
            review_prompt,
            stream=self.config.stream,
            filename=filename,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            context=f"agentic reviewer {filename}",
        )


class RefinerAgent:
    """Stage 4B agent: emits targeted RefinerPatchResult edits without dropping untouched rows."""

    def __init__(self, config: AgenticExtractionConfig):
        self.config = config
        self.controller = LLMController(
            provider=config.refiner_provider or config.provider,
            model_name=config.refiner_model_name or config.model_name,
            structured_output=RefinerPatchResult,
            system_prompt=REFINER_SYSTEM_PROMPT,
            base_url=config.base_url,
            stream=config.stream,
        )

    def run(
        self,
        source_text: str,
        entries: Sequence[PeptideLLMEntry],
        feedback: ReviewerFeedback,
        validation_report: DeterministicValidationReport,
        registry: EntityRegistryResult | None = None,
        filename: str = "paper.pdf",
    ) -> RefinerPatchResult:
        reg_summary = (
            format_entity_registry_summary(registry.peptides) if registry else "None provided."
        )
        flagged = {
            issue.entry_index
            for issue in validation_report.issues
            if issue.entry_index is not None and not issue.auto_fixed
        } | {
            issue.entry_index
            for issue in feedback.issues
            if issue.entry_index is not None
        }
        feedback_json = json.dumps(feedback.model_dump(mode="json"), indent=2)
        refine_prompt = (
            f"Peptide Registry:\n{reg_summary}\n\n"
            f"Deterministic Validation Issues:\n{validation_report.to_prompt_summary()}\n\n"
            f"Reviewer Feedback (emit targeted patches only):\n{feedback_json}\n\n"
            f"Current Candidate Entries:\n"
            f"{_format_entries_summary_for_review(entries, flagged_indices=flagged)}\n\n"
            f"Source Paper Text:\n{source_text}"
        )
        return retry_on_external_error(
            self.controller.extract_peptides,
            refine_prompt,
            stream=self.config.stream,
            filename=filename,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            context=f"agentic refiner {filename}",
        )
