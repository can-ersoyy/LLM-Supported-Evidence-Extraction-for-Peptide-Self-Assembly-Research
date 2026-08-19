from typing import TYPE_CHECKING, Optional
from sqlmodel import Column, Field, Relationship, SQLModel, Text

if TYPE_CHECKING:
    from peptide_extractor.db.paper_metadata import PaperMetadata


class PaperContent(SQLModel, table=True):
    __tablename__: str = "paper_content"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: Optional[int] = Field(default=None, foreign_key="paper_metadata.id", index=True)

    format: str
    raw_content: str = Field(sa_column=Column(Text, nullable=False))

    paper: Optional["PaperMetadata"] = Relationship(back_populates="contents")