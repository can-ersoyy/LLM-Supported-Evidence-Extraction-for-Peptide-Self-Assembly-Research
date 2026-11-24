"""Code-Agency Orchestrator coordinating Scout, Extractors, Chemistry Sandbox, Reviewer, and Refiner."""

from __future__ import annotations

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
)
from peptide_extractor.db.peptide_entry import PeptideExtractionResult, PeptideLLMEntry
from peptide_extractor.extraction import (
    SharedExperimentalConditions,
    _split_blocks_for_factored_extraction,
    merge_table_result_with_shared,
)
from peptide_extractor.logging_utils import logger


class ExtractionOrchestrator:
    """Deterministic state-graph orchestrator for multi-agent peptide extraction.

    Execution stages:
    1. Input decomposition (plain text, multimodal prose + table images, or raw PDF)
    2. Stage 1 — Entity & Alias Scout + Paper-Wide Shared Conditions
    3. Stage 2 — Scoped Extraction (Per-Table Image Workers + Prose Worker)
    4. Stage 3 — Deterministic Chemistry & Schema Sandbox (normalize, validate, dedup)
    5. Stage 4 — Evidence-Independent Review -> Targeted Refine Loop (bounded rounds)
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
                table_res = table_agent.run(
                    table_id=table_id,
                    caption_text=caption_text,
                    image_block=image_block,
                    registry=registry,
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
            all_entries, val_report = sanitize_and_validate_entries(all_entries, aliases=aliases)
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
        # Stage 4: Evidence-Independent Review -> Targeted Refine Loop
        # ------------------------------------------------------------------
        # For raw binary PDFs without text, skip text-based reviewer/refiner
        source_text_for_review = prose_or_raw if isinstance(prose_or_raw, str) else ""
        if self.config.enable_reviewer and not is_binary_pdf and source_text_for_review.strip():
            reviewer = ReviewerAgent(self.config)
            refiner: RefinerAgent | None = None

            for round_idx in range(self.config.max_refinement_rounds + 1):
                feedback = reviewer.run(
                    source_text=source_text_for_review,
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
                    f"[Orchestrator] Stage 4 Refine (round {round_idx + 1}) for {filename}"
                )
                refined_result = refiner.run(
                    source_text=source_text_for_review,
                    entries=all_entries,
                    feedback=feedback,
                    validation_report=val_report,
                    registry=registry,
                    filename=filename,
                )
                trace.refinement_rounds_executed += 1
                all_entries = list(refined_result.entries)

                if self.config.enable_chemistry_sandbox:
                    all_entries, val_report = sanitize_and_validate_entries(
                        all_entries, aliases=aliases
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
