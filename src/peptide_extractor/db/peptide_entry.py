from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional
from pydantic import ConfigDict
from sqlalchemy import Column, Text
from sqlmodel import DateTime, Field, Relationship, SQLModel

from peptide_extractor.utils import get_utc_now

if TYPE_CHECKING:
    from peptide_extractor.db.paper_metadata import PaperMetadata


class ExtractionMode(str, Enum):
    MANUAL = "manual"
    LLM = "llm"


class PeptideEntry(SQLModel, table=True):
    __tablename__: str = "peptide_entry"
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: Optional[int] = Field(
        default=None, foreign_key="paper_metadata.id", index=True
    )

    sequence: str
    length: int
    type: str
    n_terminal: str
    c_terminal: str
    temperature: str
    ph: str
    peptide_concentration: str
    solution: str
    solvent: str
    volume_ratio: str
    incubation_period: str

    peptide_conjugate_mixture: Optional[str] = Field(
        default=None, alias="peptide/conjugate/mixture"
    )
    conjugate_partner: Optional[str] = None
    linear_cyclic: Optional[str] = Field(
        default=None, alias="linear/cyclic"
    )

    detection_method: str
    preparation_protocol: str
    pmid: str = Field(index=True)
    database: Optional[str] = None

    evidence_span: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Concise summary or quoted evidence from the paper text",
    )

    extraction_mode: ExtractionMode = Field(
        default=ExtractionMode.MANUAL,
        sa_column_kwargs={"server_default": ExtractionMode.MANUAL.value},
        description="Extraction mode: 'manual' (golden dataset) or 'llm'",
    )
    model: Optional[str] = Field(
        default=None,
        description="String name for LLM used for extraction",
    )

    created_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
    )
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
        sa_column_kwargs={"onupdate": get_utc_now},
    )

    paper: Optional["PaperMetadata"] = Relationship()
