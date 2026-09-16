from typing import TYPE_CHECKING, Any, List, Optional
from sqlmodel import Column, Field, Relationship, SQLModel, Session, Text

from peptide_extractor.db.database import get_records

if TYPE_CHECKING:
    from peptide_extractor.db.paper_metadata import PaperMetadata


class PaperContent(SQLModel, table=True):
    __tablename__: str = "paper_content"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: Optional[int] = Field(default=None, foreign_key="paper_metadata.id", index=True)

    format: str
    raw_content: str = Field(sa_column=Column(Text, nullable=False))

    paper: Optional["PaperMetadata"] = Relationship(back_populates="contents")


def get_paper_contents(
    session: Optional[Session] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    **filters: Any,
) -> List[PaperContent]:
    """Retrieve PaperContent records filtered by arbitrary column values."""
    return get_records(
        PaperContent,
        session=session,
        limit=limit,
        offset=offset,
        **filters,
    )