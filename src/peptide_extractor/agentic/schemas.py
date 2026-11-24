"""Typed Pydantic schemas for the agentic peptide extraction workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from peptide_extractor.db.peptide_entry import (
    PeptideConjugateMixture,
    PeptideExtractionResult,
    PeptideLinearity,
    PeptideLLMEntry,
)
from peptide_extractor.extraction import SharedExperimentalConditions


class PeptideAlias(BaseModel):
    """Maps a paper-specific label/shorthand (e.g. '1a', 'P3', 'Fmoc-FF') to a canonical peptide."""

    label: str = Field(
        description="Identifier or shorthand used in the paper (e.g. '1a', 'Peptide 2', 'Fmoc-FF')."
    )
    sequence: str = Field(
        description="Uppercase 1-letter IUPAC amino acid sequence (no spaces, hyphens, or terminal caps)."
    )
    n_terminal: str | None = Field(
        default=None,
        description="N-terminal cap/modification (e.g. 'Fmoc', 'Boc', 'Acetylation', 'Free').",
    )
    c_terminal: str | None = Field(
        default=None,
        description="C-terminal cap/modification (e.g. 'Amidation', 'Free', 'Methyl ester', 'OBn').",
    )
    linear_cyclic: PeptideLinearity | None = Field(
        default=None,
        description="Whether this peptide is 'linear' or 'cyclic'.",
    )
    chirality: str | None = Field(
        default=None,
        description="Stereochemistry/chirality ('l', 'd', 'heterochiral', 'racemic', 'achiral').",
    )
    peptide_conjugate_mixture: PeptideConjugateMixture | None = Field(
        default=None,
        description="Whether this system is a pure 'peptide', 'conjugate', or 'mixture'.",
    )
    conjugate_partner: str | None = Field(
        default=None,
        description="Conjugate or mixture partner if applicable.",
    )
    evidence_tag: str | None = Field(
        default=None,
        description="Exact bracketed location tag where this peptide identity is defined (e.g. '[sec-1/p-2/s-1]').",
    )


class EntityRegistryResult(BaseModel):
    """Output of the Entity & Alias Scout agent (Stage 1)."""

    peptides: list[PeptideAlias] = Field(
        default_factory=list,
        description="All distinct experimentally studied peptides/aliases defined in the paper.",
    )
    shared_conditions: SharedExperimentalConditions = Field(
        default_factory=SharedExperimentalConditions,
        description="Paper-wide default experimental conditions and their location tags.",
    )


class ValidationIssue(BaseModel):
    """A single deterministic validation or normalization finding on an extracted entry."""

    entry_index: int | None = Field(
        default=None,
        description="0-based index of the affected entry, or None for paper-level issues.",
    )
    field: str = Field(description="Affected field name.")
    severity: Literal["error", "warning", "info"] = Field(
        default="warning",
        description="Severity level; 'error' triggers refinement when reviewer is active.",
    )
    message: str = Field(description="Human-readable diagnostic message.")
    auto_fixed: bool = Field(
        default=False,
        description="True if the deterministic chemistry sandbox automatically corrected this issue.",
    )


class DeterministicValidationReport(BaseModel):
    """Summary of deterministic biochemical validation, normalization, and deduplication."""

    total_before: int = 0
    total_after: int = 0
    duplicates_merged: int = 0
    auto_fixes_applied: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" and not issue.auto_fixed for issue in self.issues)

    def to_prompt_summary(self) -> str:
        unresolved = [i for i in self.issues if not i.auto_fixed]
        if not unresolved:
            return "All deterministic chemistry and schema checks passed."
        lines = []
        for issue in unresolved[:25]:
            prefix = f"Entry #{issue.entry_index}" if issue.entry_index is not None else "Paper"
            lines.append(f"- [{issue.severity.upper()}] {prefix}.{issue.field}: {issue.message}")
        if len(unresolved) > 25:
            lines.append(f"... and {len(unresolved) - 25} additional issue(s).")
        return "\n".join(lines)


class ReviewIssue(BaseModel):
    """A specific field-level or entry-level issue flagged by the Reviewer agent."""

    entry_index: int | None = Field(
        default=None,
        description="0-based index of the entry with the issue, or None if missing/paper-level.",
    )
    field: str = Field(description="Field name (or 'missing_row' / 'extra_row').")
    reason: str = Field(description="Why this field or entry is unsupported, inconsistent, or wrong.")
    suggested_fix: str = Field(
        description="Concrete, actionable correction grounded in the paper text or table."
    )


class ReviewerFeedback(BaseModel):
    """CONSTRUCT-style structured verdict from the evidence-independent Reviewer agent."""

    approved: bool = Field(
        description="True if the extracted entries are accurate, complete, and ready to save without revision."
    )
    trust_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall trustworthiness score in [0.0, 1.0] after checking flagged issues.",
    )
    issues: list[ReviewIssue] = Field(
        default_factory=list,
        description="Specific field/entry errors, hallucinations, or misattributions found.",
    )
    missing_sequences: list[str] = Field(
        default_factory=list,
        description="Experimental peptide sequences reported in the paper that are missing from the extraction.",
    )
    revision_instructions: str | None = Field(
        default=None,
        description="Concise, actionable patch instructions for the Refiner agent if approved is False.",
    )


class FieldUpdate(BaseModel):
    """A targeted single-field update to an existing candidate entry by 0-based index."""

    entry_index: int = Field(description="0-based index of the entry to update.")
    field: str = Field(description="PeptideLLMEntry field name to update.")
    value: Any = Field(description="Corrected value for that field.")


class RefinerPatchResult(BaseModel):
    """Patch-based output from RefinerAgent so untouched rows are never dropped."""

    update_fields: list[FieldUpdate] = Field(
        default_factory=list,
        description="Targeted field corrections to apply to existing entries by index.",
    )
    add_entries: list[PeptideLLMEntry] = Field(
        default_factory=list,
        description="New entries to append for previously missed peptides or conditions.",
    )
    delete_indices: list[int] = Field(
        default_factory=list,
        description="0-based indices of hallucinated or out-of-scope entries to remove.",
    )


class AgenticExtractionConfig(BaseModel):
    """Configuration for the multi-role agentic extraction orchestrator.

    Supports heterogeneous SLM-first setups (e.g., cheap local Ollama or flash models
    for scouting/extraction and an independent model family for reviewing/refining).
    """

    provider: str = "google_genai"
    model_name: str = "gemma-4-31b-it"
    base_url: str | None = None

    # Optional per-role model overrides (defaults fall back to provider/model_name)
    scout_provider: str | None = None
    scout_model_name: str | None = None
    table_provider: str | None = None
    table_model_name: str | None = None
    reviewer_provider: str | None = None
    reviewer_model_name: str | None = None
    reviewer_base_url: str | None = None
    refiner_provider: str | None = None
    refiner_model_name: str | None = None

    # Orchestration controls
    enable_entity_scout: bool = True
    enable_chemistry_sandbox: bool = True
    enable_reviewer: bool = True
    max_refinement_rounds: int = Field(default=1, ge=0, le=3)
    min_trust_score: float = Field(default=0.8, ge=0.0, le=1.0)
    stream: bool = False
    max_retries: int = 0
    retry_delay: float = 2.0


class AgenticExtractionTrace(BaseModel):
    """Auditable execution trace returned by ExtractionOrchestrator.run_with_trace()."""

    entity_registry: EntityRegistryResult | None = None
    initial_entry_count: int = 0
    final_entry_count: int = 0
    refinement_rounds_executed: int = 0
    validation_reports: list[DeterministicValidationReport] = Field(default_factory=list)
    review_history: list[ReviewerFeedback] = Field(default_factory=list)
    result: PeptideExtractionResult = Field(
        default_factory=lambda: PeptideExtractionResult(entries=[])
    )
