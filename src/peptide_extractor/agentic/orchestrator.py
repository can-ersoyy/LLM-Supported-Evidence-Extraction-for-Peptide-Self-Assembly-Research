"""Code-Agency Orchestrator coordinating Scout, Extractors, Chemistry Sandbox, Reviewer, and Refiner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from peptide_extractor.agentic.agents import (
    EntityScoutAgent,
    ProseExtractorAgent,
    RefinerAgent,
    ReviewerAgent,
    TableExtractorAgent,
)
from peptide_extractor.agentic.chemistry_sandbox import sanitize_and_validate_entries
from peptide_extractor.agentic.schemas import (
    AgenticExtractionConfig,
    AgenticExtractionTrace,
    DeterministicValidationReport,
    EntityRegistryResult,
    RefinerPatchResult,
)
from peptide_extractor.db.peptide_entry import PeptideExtractionResult, PeptideLLMEntry
from peptide_extractor.extraction import (
    SharedExperimentalConditions,
    _split_blocks_for_factored_extraction,
    merge_table_result_with_shared,
)
from peptide_extractor.logging_utils import logger


def _extract_table_context(
    prose_text: str, table_id: str, caption_text: str, window_lines: int = 8
) -> str:
    """Grab surrounding lines (footnotes, legends, nearby prose) around a table anchor."""
    if not prose_text:
        return ""
    lines = prose_text.splitlines()
    hit_indices = [i for i, ln in enumerate(lines) if table_id in ln]
    if not hit_indices:
        return ""
    idx = hit_indices[0]
    start = max(0, idx - window_lines)
    end = min(len(lines), idx + window_lines + 1)
    snippet = "\n".join(
        ln.strip()
        for ln in lines[start:end]
        if ln.strip() and ln.strip() not in caption_text
    )
    return snippet[:1500]


def apply_refiner_patch(
    entries: Sequence[PeptideLLMEntry], patch: RefinerPatchResult
) -> list[PeptideLLMEntry]:
    """Deterministically apply RefinerPatchResult without ever dropping untouched rows."""
    updated = [e.model_copy(deep=True) for e in entries]
    valid_fields = set(PeptideLLMEntry.model_fields.keys())

    # 1. Apply targeted field updates
    for upd in patch.update_fields:
        if not (0 <= upd.entry_index < len(updated)):
            logger.warning(f"[RefinerPatch] Skipping out-of-range index {upd.entry_index}")
            continue
        if upd.field not in valid_fields:
            logger.warning(f"[RefinerPatch] Skipping unknown field '{upd.field}'")
            continue
        candidate_data = updated[upd.entry_index].model_dump()
        candidate_data[upd.field] = upd.value
        try:
            updated[upd.entry_index] = PeptideLLMEntry(**candidate_data)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"[RefinerPatch] Failed to apply {upd.field}={upd.value!r} on entry {upd.entry_index}: {e}"
            )

    # 2. Remove explicitly flagged hallucinated indices
    delete_set = {i for i in patch.delete_indices if 0 <= i < len(updated)}
    result = [e for idx, e in enumerate(updated) if idx not in delete_set]

    # 3. Append newly recovered entries
    if patch.add_entries:
        result.extend(patch.add_entries)

    return result


class ExtractionOrchestrator:
    """Deterministic state-graph orchestrator for multi-agent peptide extraction.

    Execution stages:
    1. Input decomposition (plain text, multimodal prose + table images, or raw PDF)
    2. Stage 1 — Entity & Alias Scout + Paper-Wide Shared Conditions
    3. Stage 2 — Scoped Extraction (Per-Table Image Workers + Prose Worker)
    4. Stage 3 — Deterministic Chemistry & Schema Sandbox (normalize, validate, dedup)
    5. Stage 4 — Evidence-Independent Review -> Patch-Based Refine Loop (bounded rounds)
    """

    def __init__(self, config: AgenticExtractionConfig | None = None):
        self.config = config or AgenticExtractionConfig()

    def _decompose_input(
        self, paper_content: str | bytes | Path | list[dict[str, Any]]
    ) -> tuple[str | bytes | Path, list[tuple[str, str, dict[str, Any]]]]:
        if isinstance(paper_content, list):
            prose_text, table_items = _split_blocks_for_factored_extraction(paper_content)
            return prose_text, table_items
        return paper_content, []

    def run_with_trace(
        self,
        paper_content: str | bytes | Path | list[dict[str, Any]],
        filename: str = "paper.pdf",
    ) -> AgenticExtractionTrace:
        """Execute the full multi-agent workflow and return an auditable trace + final entries."""
        trace = AgenticExtractionTrace()
        prose_or_raw, table_items = self._decompose_input(paper_content)
        is_binary_pdf = isinstance(prose_or_raw, (bytes, Path))
        source_text = prose_or_raw if isinstance(prose_or_raw, str) else ""

        # ------------------------------------------------------------------
        # Stage 1: Entity & Alias Scout (+ paper-wide shared conditions)
        # ------------------------------------------------------------------
        registry: EntityRegistryResult | None = None
        if self.config.enable_entity_scout:
            logger.info(f"[Orchestrator] Stage 1: Running EntityScoutAgent for {filename}")
            scout = EntityScoutAgent(self.config)
            registry = scout.run(prose_or_raw, filename=filename)
            trace.entity_registry = registry
            logger.info(
                f"[Orchestrator] Scout identified {len(registry.peptides)} peptide alias(es)."
            )

        shared_defaults = (
            registry.shared_conditions
            if registry is not None
            else SharedExperimentalConditions()
        )

        # ------------------------------------------------------------------
        # Stage 2: Scoped Extraction (Prose + Per-Table Image Workers)
        # ------------------------------------------------------------------
        logger.info(
            f"[Orchestrator] Stage 2: Running extraction (tables={len(table_items)}) for {filename}"
        )
        all_entries: list[PeptideLLMEntry] = []

        prose_agent = ProseExtractorAgent(self.config)
        prose_result = prose_agent.run(
            prose_or_raw,
            registry=registry,
            tables_extracted_separately=bool(table_items),
            filename=filename,
        )
        all_entries.extend(prose_result.entries)

        if table_items:
            table_agent = TableExtractorAgent(self.config)
            for table_id, caption_text, image_block in table_items:
                table_ctx = _extract_table_context(source_text, table_id, caption_text)
                table_res = table_agent.run(
                    table_id=table_id,
                    caption_text=caption_text,
                    image_block=image_block,
                    registry=registry,
                    table_context=table_ctx,
                    filename=filename,
                )
                merged = merge_table_result_with_shared(table_res, shared_defaults, table_id)
                all_entries.extend(merged)

        trace.initial_entry_count = len(all_entries)

        # ------------------------------------------------------------------
        # Stage 3: Deterministic Chemistry & Schema Sandbox
        # ------------------------------------------------------------------
        aliases = registry.peptides if registry is not None else None
        if self.config.enable_chemistry_sandbox:
            all_entries, val_report = sanitize_and_validate_entries(
                all_entries,
                aliases=aliases,
                source_text=source_text,
                default_room_temperature=self.config.default_room_temperature,
            )
        else:
            val_report = DeterministicValidationReport(
                total_before=len(all_entries), total_after=len(all_entries)
            )
        trace.validation_reports.append(val_report)
        logger.info(
            f"[Orchestrator] Stage 3 Sandbox: {val_report.total_before} -> {val_report.total_after} entries "
            f"(dedup={val_report.duplicates_merged}, auto_fixes={val_report.auto_fixes_applied}, "
            f"has_errors={val_report.has_errors})"
        )

        # ------------------------------------------------------------------
        # Stage 4: Evidence-Independent Review -> Patch-Based Refine Loop
        # ------------------------------------------------------------------
        if self.config.enable_reviewer and not is_binary_pdf and source_text.strip():
            reviewer = ReviewerAgent(self.config)
            refiner: RefinerAgent | None = None

            for round_idx in range(self.config.max_refinement_rounds + 1):
                feedback = reviewer.run(
                    source_text=source_text,
                    entries=all_entries,
                    validation_report=val_report,
                    registry=registry,
                    filename=filename,
                )
                trace.review_history.append(feedback)
                logger.info(
                    f"[Orchestrator] Stage 4 Review (round {round_idx}): "
                    f"approved={feedback.approved}, trust={feedback.trust_score:.2f}, "
                    f"issues={len(feedback.issues)}, missing_seqs={len(feedback.missing_sequences)}"
                )

                needs_refinement = (
                    not feedback.approved
                    or feedback.trust_score < self.config.min_trust_score
                    or val_report.has_errors
                    or bool(feedback.missing_sequences)
                )
                if not needs_refinement:
                    break
                if round_idx >= self.config.max_refinement_rounds:
                    logger.info(
                        "[Orchestrator] Reached max_refinement_rounds; stopping refinement loop."
                    )
                    break

                if refiner is None:
                    refiner = RefinerAgent(self.config)
                logger.info(
                    f"[Orchestrator] Stage 4 Refine Patch (round {round_idx + 1}) for {filename}"
                )
                patch_result = refiner.run(
                    source_text=source_text,
                    entries=all_entries,
                    feedback=feedback,
                    validation_report=val_report,
                    registry=registry,
                    filename=filename,
                )
                trace.refinement_rounds_executed += 1
                all_entries = apply_refiner_patch(all_entries, patch_result)

                if self.config.enable_chemistry_sandbox:
                    all_entries, val_report = sanitize_and_validate_entries(
                        all_entries,
                        aliases=aliases,
                        source_text=source_text,
                        default_room_temperature=self.config.default_room_temperature,
                    )
                    trace.validation_reports.append(val_report)

        trace.final_entry_count = len(all_entries)
        trace.result = PeptideExtractionResult(entries=all_entries)
        return trace

    def run(
        self,
        paper_content: str | bytes | Path | list[dict[str, Any]],
        filename: str = "paper.pdf",
    ) -> PeptideExtractionResult:
        """Execute the multi-agent workflow and return PeptideExtractionResult."""
        return self.run_with_trace(paper_content, filename=filename).result


def run_agentic_extraction(
    paper_content: str | bytes | Path | list[dict[str, Any]],
    provider: str = "google_genai",
    model_name: str = "gemma-4-31b-it",
    filename: str = "paper.pdf",
    table_provider: str | None = None,
    table_model_name: str | None = None,
    reviewer_provider: str | None = None,
    reviewer_model_name: str | None = None,
    max_refinement_rounds: int = 1,
    config: AgenticExtractionConfig | None = None,
    max_retries: int = 0,
    retry_delay: float = 2.0,
) -> PeptideExtractionResult:
    """Convenience entrypoint for multi-agent peptide extraction."""
    resolved_config = config or AgenticExtractionConfig(
        provider=provider,
        model_name=model_name,
        table_provider=table_provider,
        table_model_name=table_model_name,
        reviewer_provider=reviewer_provider,
        reviewer_model_name=reviewer_model_name,
        max_refinement_rounds=max_refinement_rounds,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    orchestrator = ExtractionOrchestrator(resolved_config)
    return orchestrator.run(paper_content, filename=filename)
