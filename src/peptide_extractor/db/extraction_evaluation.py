from peptide_extractor.db import PeptideEntry
from sqlmodel.main import Relationship
from datetime import datetime
from peptide_extractor.utils import get_utc_now
from typing_extensions import Optional
from sqlmodel import SQLModel, Field, DateTime
from typing import TYPE_CHECKING
from enum import Enum


class EvaluationCategory(str, Enum):
    WEAK = "weak"
    OK = "ok"
    STRONG = "strong"

class ExtractionEvaluation(SQLModel, table=True):
    __tablename__ = "extraction_evaluation"

    id: Optional[int] = Field(default=None, primary_key=True)
    peptide_entry_id: Optional[int] = Field(default=None, foreign_key="peptide_entry.id")

    groundness_rate: Optional[str] = None
    faithfulness_rate: Optional[str] = None
    completeness_rate: Optional[str] = None
    
    #evaluation_category: EvaluationCategory
    explanation: Optional[str] = None
    
    created_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
    )
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
        sa_column_kwargs={"onupdate": get_utc_now},
    )
    
    run_identifier: Optional[int] = Field(default=None, index=True)

    peptide: Optional["PeptideEntry"] = Relationship()