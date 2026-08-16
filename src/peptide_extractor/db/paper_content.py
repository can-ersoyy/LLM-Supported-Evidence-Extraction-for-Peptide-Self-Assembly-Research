from typing import Any

from sqlmodel import Field, SQLModel

class PaperContent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    paper_id: int | None = Field(default=None, foreign_key="paper_metadata.id")
    format: str
    raw_content: Any