"""SQLModel table for auditing disagreements between LLM extractions and golden-dataset entries."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import ConfigDict
from sqlalchemy import Column, Text
from sqlmodel import DateTime, Field, SQLModel

from peptide_extractor.utils import get_utc_now


class AuditType(str, Enum):
    """Category of disagreement audited against source paper evidence."""

    FIELD_MISMATCH = "field_mismatch"
    LLM_ONLY_ENTRY = "llm_only_entry"


class DisagreementVerdict(str, Enum):
    """Adjudication outcome comparing manual golden-dataset records vs. LLM extractions."""

    LLM_CORRECT_GOLDEN_WRONG = "llm_correct_golden_wrong"
    GOLDEN_CORRECT_LLM_WRONG = "golden_correct_llm_wrong"
    EQUIVALENT_NORMALIZATION = "equivalent_normalization"
    VALID_UNCURATED_IN_GOLDEN = "valid_uncurated_in_golden"
    HALLUCINATED_OR_OUT_OF_SCOPE = "hallucinated_or_out_of_scope"
    AMBIGUOUS_NEEDS_HUMAN = "ambiguous_needs_human"


class GoldenDisagreementAudit(SQLModel, table=True):
    """Persistent record of an audited disagreement between a golden entry and an LLM entry."""

    __tablename__: str = "golden_disagreement_audit"
    model_config = ConfigDict(populate_by_name=True)

    id: int | None = Field(default=None, primary_key=True)
    pmid: str = Field(index=True)
    paper_id: int | None = Field(
        default=None, foreign_key="paper_metadata.id", index=True
    )
    manual_entry_id: int | None = Field(
        default=None, foreign_key="peptide_entry.id", index=True
    )
    llm_entry_id: int | None = Field(
        default=None, foreign_key="peptide_entry.id", index=True
    )

    sequence: str = Field(index=True)
    audit_type: AuditType = Field(index=True)
    field_name: str = Field(index=True)
    manual_value: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    llm_value: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    evidence_tag: str | None = None
    source_quote: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    verdict: DisagreementVerdict = Field(index=True)
    confidence: float | None = None
    reason: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    adjudicator_provider: str | None = None
    adjudicator_model: str | None = None

    human_verdict: DisagreementVerdict | None = None
    human_notes: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )

    llm_run_identifier: int = Field(index=True)
    audit_run_identifier: int = Field(index=True)

    created_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
    )
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
        sa_column_kwargs={"onupdate": get_utc_now},
    )
