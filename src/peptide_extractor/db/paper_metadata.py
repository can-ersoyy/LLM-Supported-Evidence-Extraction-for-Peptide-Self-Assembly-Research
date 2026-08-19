from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional
from sqlmodel import DateTime, Field, Relationship, SQLModel

from peptide_extractor.utils import get_utc_now

if TYPE_CHECKING:
    from peptide_extractor.db.paper_content import PaperContent


class PaperMetadata(SQLModel, table=True):
    __tablename__: str = "paper_metadata"

    id: Optional[int] = Field(default=None, primary_key=True)

    doi: Optional[str] = Field(default=None, index=True)
    pmid: Optional[str] = Field(default=None, index=True)
    pmcid: Optional[str] = Field(default=None, index=True)

    title: str
    is_open_access: bool

    search_query: Optional[str] = Field(default=None)

    created_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now
    )
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
        sa_column_kwargs={"onupdate": get_utc_now}
    )

    contents: List["PaperContent"] = Relationship(back_populates="paper")