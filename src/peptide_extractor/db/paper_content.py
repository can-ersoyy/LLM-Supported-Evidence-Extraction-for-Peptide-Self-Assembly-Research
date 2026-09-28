from typing import TYPE_CHECKING, Any, List, Optional
from sqlmodel import Column, Field, Relationship, SQLModel, Session, String

from peptide_extractor.db.database import get_records

if TYPE_CHECKING:
    from peptide_extractor.db.paper_metadata import PaperMetadata


class PaperContent(SQLModel, table=True):
    __tablename__: str = "paper_content"

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: Optional[int] = Field(default=None, foreign_key="paper_metadata.id", index=True)

    source: Optional[str] = Field(default=None, description="Source of the content.")

    format: str
    blob_url: Optional[str] = Field(
        default=None,
        sa_column=Column(String, nullable=True),
        description="Azure Blob Storage URL or path to the full text content.",
    )
    pdf_url: Optional[str] = Field(default=None, description="PDF URL of the content. Should be thought of as a backup in case xml doesn't exist")

    paper: Optional["PaperMetadata"] = Relationship(back_populates="contents")

    def get_content(self, use_cache: bool = True) -> str:
        """Download and return the XML content from Azure Blob Storage or cache."""
        if not self.blob_url:
            raise ValueError(f"PaperContent {self.id} has no blob_url.")
        from peptide_extractor.azure_storage import download_xml

        return download_xml(self.blob_url, use_cache=use_cache)


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