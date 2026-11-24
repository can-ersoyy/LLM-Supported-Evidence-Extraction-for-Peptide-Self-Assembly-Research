"""Specialized agent roles for the multi-stage agentic extraction workflow.

Each agent wraps LLMController with a narrow, single-hop prompt and strict
Pydantic structured output schema so small language models (SLMs) and larger
models alike stay grounded and schema-compliant.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from peptide_extractor.agentic.schemas import (
    AgenticExtractionConfig,
    DeterministicValidationReport,
    EntityRegistryResult,
    PeptideAlias,
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
   - Explicitly identify `linear_cyclic` ("linear" or "cyclic"), `chirality` ("l", "d", "heterochiral", "racemic", "achiral"), and whether it is a pure `peptide`, `conjugate`, or `mixture` (with `conjugate_partner`).
2. `shared_conditions`: Paper-wide experimental defaults that apply across screening/assembly experiments (temperature in °C with RT -> 25.0, incubation_period in minutes with 24 h -> 1440, ph, peptide_concentration in mg/ml, solution, solvent, volume_ratio, detection_method, preparation_protocol), along with their exact bracketed `evidence_span` location tags.

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
4. Carry over applicable `shared_conditions` (temperature, pH, incubation_period, detection_method, preparation_protocol) when not overridden locally.
{LOCATION_GROUNDING_RULES}
"""

SCOPED_TABLE_EXTRACTOR_SYSTEM_PROMPT = """
You are an expert scientific curator extracting experimental peptide self-assembly data from a single table into a compact row-grouped schema.

CRITICAL RULES:
1. Extract EVERY SINGLE ROW and EVERY SINGLE TESTED CELL in the table—never summarize, sample, or truncate.
2. Use the provided `Peptide Registry` to resolve compound numbers/aliases (e.g., "1a", "P2") to their exact uppercase 1-letter `sequence`, `n_terminal`, and `c_terminal`.
3. Include negative / non-assembling cells (soluble, precipitate, no gel -> `type: "none"`). Use `type: "organogel"` for organic-solvent gels and `type: "hydrogel"` for aqueous gels.
4. Always capture column-header solvents/solutions (`solution`, `solvent`, `volume_ratio`) and concentrations in each `CompactTableCondition` cell.
5. Do NOT repeat unchanged paper-wide defaults (`temperature`, `incubation_period`, `preparation_protocol`, `detection_method`) unless the table or cell overrides them.
"""

REVIEWER_SYSTEM_PROMPT = """
You are an independent scientific data verification auditor (CONSTRUCT-style Reviewer).
You did NOT generate the candidate extraction. Your job is to critically audit the candidate peptide self-assembly entries against:
1. The source paper text (and Peptide Registry)
2. The deterministic chemistry & schema validation report

AUDIT PROCEDURE (follow in order):
Step 1 — Flag Specific Errors (`issues`):
- Check for hallucinated sequences (e.g., prior-work peptides from Introduction, or broken labels left in `sequence`).
- Check for misattributed conditions (wrong solvent, pH, temperature, concentration, or assembly `type`).
- Check for missing/wrong `n_terminal` or `c_terminal` caps (e.g., paper states all peptides are C-terminally amidated `-NH2` or N-terminally `Fmoc`/`Ac` protected, yet entries leave them null or wrong).
- Check for missing `detection_method` or `solution`/`volume_ratio` values that are explicitly stated in the paper.
- Check for unresolved deterministic validation errors.

Step 2 — Recall Audit (`missing_sequences`):
- List any experimentally tested peptide sequences in the paper (or Peptide Registry) that are completely missing from the candidate entries.

Step 3 — Verdict (`approved`, `trust_score`, `revision_instructions`):
- Compute `trust_score` in [0.0, 1.0].
- Set `approved = True` ONLY if there are no critical errors, no missing experimental peptides, and `trust_score` is high.
- If `approved = False`, write concise, concrete `revision_instructions` specifying exact patches (which rows to add/remove/fix and what values to set).
"""

REFINER_SYSTEM_PROMPT = f"""
You are a precision scientific data Refiner.
You receive:
1. The original paper text (and Peptide Registry)
2. The current candidate `PeptideExtractionResult` entries
3. Specific feedback from an independent Reviewer and deterministic chemistry validator

RULES:
1. Apply EVERY valid correction in the Reviewer feedback and resolve ALL deterministic validation errors.
2. Preserve all already-correct entries and fields—do NOT drop valid rows while fixing flagged ones.
3. Add any missing experimental peptides or tested conditions identified by the Reviewer.
4. Remove only entries explicitly flagged as hallucinated or out-of-scope (e.g. cited literature rather than this paper's experiments).
5. Ensure every `sequence` is uppercase 1-letter IUPAC with terminal caps cleanly placed in `n_terminal` and `c_terminal`.
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


def _format_entries_for_review(entries: Sequence[PeptideLLMEntry], max_entries: int = 80) -> str:
    """Serialize candidate entries compactly with 0-based indices for the Reviewer/Refiner."""
    payload = []
    for idx, entry in enumerate(entries[:max_entries]):
        d = entry.model_dump(exclude_none=True, mode="json")
        d["_index"] = idx
        payload.append(d)
    text = json.dumps(payload, indent=2)
    if len(entries) > max_entries:
        text += f"\n... ({len(entries) - max_entries} additional entries omitted from preview)"
    return text


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
    """Stage 2B agent: extracts a single table image conditioned on PeptideRegistry + SharedDefaults."""

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
        blocks = [
            {
                "type": "text",
                "text": (
                    f"Peptide Registry (resolve aliases/caps from here):\n{reg_summary}\n\n"
                    f"Paper-wide shared conditions (merged automatically; only override if this table differs):\n"
                    f"{shared_summary}\n\n"
                    f"Table location ID: [{table_id}]\n"
                    f"{caption_text}\n"
                    "Extract EVERY row and EVERY tested cell from the attached table image."
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
        review_prompt = (
            f"Peptide Registry:\n{reg_summary}\n\n"
            f"Deterministic Chemistry & Schema Validation Report:\n"
            f"{validation_report.to_prompt_summary()}\n\n"
            f"Candidate Extracted Entries ({len(entries)} total):\n"
            f"{_format_entries_for_review(entries)}\n\n"
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
    """Stage 4B agent: applies targeted Reviewer + Sandbox patches to candidate entries."""

    def __init__(self, config: AgenticExtractionConfig):
        self.config = config
        self.controller = LLMController(
            provider=config.refiner_provider or config.provider,
            model_name=config.refiner_model_name or config.model_name,
            structured_output=PeptideExtractionResult,
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
    ) -> PeptideExtractionResult:
        reg_summary = (
            format_entity_registry_summary(registry.peptides) if registry else "None provided."
        )
        feedback_json = json.dumps(feedback.model_dump(mode="json"), indent=2)
        refine_prompt = (
            f"Peptide Registry:\n{reg_summary}\n\n"
            f"Deterministic Validation Issues:\n{validation_report.to_prompt_summary()}\n\n"
            f"Reviewer Feedback (apply all valid fixes):\n{feedback_json}\n\n"
            f"Current Candidate Entries ({len(entries)} total):\n"
            f"{_format_entries_for_review(entries)}\n\n"
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
