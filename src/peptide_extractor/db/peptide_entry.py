from peptide_extractor.schemas import PeptideLLMEntry
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


def peptide_llm_entry_to_peptide_entry(peptide_llm_entry: PeptideLLMEntry, paper_id: int, pmid: str, database: Optional[str] = None, model: Optional[str] = None) -> PeptideEntry:
    return PeptideEntry(
        paper_id=paper_id,
        pmid=pmid,
        sequence=peptide_llm_entry.sequence,
        length=peptide_llm_entry.length,
        type=peptide_llm_entry.type,
        n_terminal=peptide_llm_entry.n_terminal,
        c_terminal=peptide_llm_entry.c_terminal,
        temperature=peptide_llm_entry.temperature,
        ph=peptide_llm_entry.ph,
        peptide_concentration=peptide_llm_entry.peptide_concentration,
        solution=peptide_llm_entry.solution,
        solvent=peptide_llm_entry.solvent,
        volume_ratio=peptide_llm_entry.volume_ratio,
        incubation_period=peptide_llm_entry.incubation_period,
        peptide_conjugate_mixture=peptide_llm_entry.peptide_conjugate_mixture,
        conjugate_partner=peptide_llm_entry.conjugate_partner,
        linear_cyclic=peptide_llm_entry.linear_cyclic,
        detection_method=peptide_llm_entry.detection_method,
        preparation_protocol=peptide_llm_entry.preparation_protocol,
        evidence_span=peptide_llm_entry.evidence_span,
        extraction_mode=ExtractionMode.LLM,
        model=model,
        database=database,
    )