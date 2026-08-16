from datetime import timezone, datetime
from sqlmodel import DateTime, Field, SQLModel

from peptide_extractor.utils import get_utc_now

class PaperMetadata(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str
    doi: str
    provider: str
    provider_id: str
    is_open_access: bool
    search_query: str
    created_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now
    )
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True), # type: ignore
        default_factory=get_utc_now,
        sa_column_kwargs={"onupdate": get_utc_now}
    )
