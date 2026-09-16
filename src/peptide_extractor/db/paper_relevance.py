from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, Session, Text, select

from peptide_extractor.db.database import get_session_context
from peptide_extractor.utils import get_utc_now

if TYPE_CHECKING:
    from peptide_extractor.db.paper_metadata import PaperMetadata


class PaperRelevance(SQLModel, table=True):
    __tablename__: str = "paper_relevance"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: Optional[int] = Field(default=None, foreign_key="paper_metadata.id", index=True)

    detected_clues: str = Field(sa_column=Column(Text, nullable=False), description="JSON-encoded array or text list of detected clues")
    explanation: str = Field(sa_column=Column(Text, nullable=False), description="Justification focusing on experimental conditions")
    is_relevant: bool = Field(index=True, description="True if experimental self-assembly data is reported or hinted at")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

    provider: str = Field(index=True, description="LLM provider name: 'google_genai', 'ollama', etc.")
    model_name: str = Field(index=True, description="LLM model name: e.g. 'gemma-4-31b-it', 'gemini-3.5-flash'")

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
    detected_clues: str,
    explanation: str,
    is_relevant: bool,
    confidence: float,
    provider: str = "google_genai",
    model_name: str = "gemma-4-31b-it",
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
            provider=provider if isinstance(provider, str) and provider.strip() else "google_genai",
            model_name=model_name if isinstance(model_name, str) and model_name.strip() else "gemma-4-31b-it",
        )
        s.add(record)
        s.commit()
        s.refresh(record)
        return record

    if session:
        return _save(session)
    with get_session_context() as s:
        return _save(s)


def get_relevance_records_for_paper(
    paper_id: int,
    session: Optional[Session] = None,
) -> List[PaperRelevance]:
    """Retrieve all relevance evaluation records for a given paper ID."""
    def _query(s: Session) -> List[PaperRelevance]:
        statement = select(PaperRelevance).where(PaperRelevance.paper_id == paper_id)
        return list(s.exec(statement).all())

    if session:
        return _query(session)
    with get_session_context() as s:
        return _query(s)

