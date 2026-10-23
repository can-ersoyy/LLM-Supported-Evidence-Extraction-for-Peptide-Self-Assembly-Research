from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional
from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, Session, Text, select

from peptide_extractor.db.database import get_records, get_session_context
from peptide_extractor.utils import get_utc_now

if TYPE_CHECKING:
    from peptide_extractor.db.paper_metadata import PaperMetadata


class PaperRelevance(SQLModel, table=True):
    __tablename__: str = "paper_relevance"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: Optional[int] = Field(default=None, foreign_key="paper_metadata.id", index=True)

    detected_clues: Optional[str] = Field(sa_column=Column(Text, nullable=True), description="list of detected clues")
    explanation: Optional[str] = Field(sa_column=Column(Text, nullable=True), description="Justification focusing on experimental conditions")
    is_relevant: Optional[bool] = Field(index=True, description="True if experimental self-assembly data is reported or hinted at")
    confidence: Optional[float] = Field(description="Confidence score between 0.0 and 1.0")

    provider: str = Field(index=True, description="LLM provider name: 'google_genai', 'ollama', etc.")
    model_name: str = Field(index=True, description="LLM model name: e.g. 'gemma-4-31b-it', 'gemini-3.5-flash'")

    run_identifier: Optional[int] = Field(default=None, index=True)

    created_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
    )
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
        sa_column_kwargs={"onupdate": get_utc_now},
    )

    paper: Optional["PaperMetadata"] = Relationship(back_populates="relevance_records")


def save_paper_relevance(
    paper_id: int,
    detected_clues: Optional[List[str]],
    explanation: Optional[str],
    is_relevant: Optional[bool],
    confidence: Optional[float],
    run_identifier: Optional[int],
    provider: str,
    model_name: str,
    session: Optional[Session] = None,
) -> PaperRelevance:
    """Save a PaperRelevance database record."""
    def _save(s: Session) -> PaperRelevance:
        record = PaperRelevance(
            paper_id=paper_id,
            detected_clues=detected_clues,
            explanation=explanation,
            is_relevant=is_relevant,
            confidence=confidence,
            provider=provider,
            model_name=model_name,
            run_identifier=run_identifier,
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        return record

    if session:
        return _save(session)
    with get_session_context() as s:
        return _save(s)


def get_paper_relevances(
    session: Optional[Session] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    **filters: Any,
) -> List[PaperRelevance]:
    """Retrieve PaperRelevance records filtered by arbitrary column values."""
    return get_records(
        PaperRelevance,
        session=session,
        limit=limit,
        offset=offset,
        **filters,
    )

