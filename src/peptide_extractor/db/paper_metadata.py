from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, List, Optional
from sqlmodel import DateTime, Field, Relationship, SQLModel, Session, select
from sqlalchemy import func

from peptide_extractor.db.database import get_session_context
from peptide_extractor.utils import get_utc_now

if TYPE_CHECKING:
    from peptide_extractor.db.paper_content import PaperContent
    from peptide_extractor.db.paper_relevance import PaperRelevance


class DatasetSubset(str, Enum):
        GOLDEN_BENCHMARK = "golden_benchmark"  # Manually curated from databases
        SCRAPED     = "scraped"       # Unlabeled papers from PMC queries

class PaperStatus(str, Enum):
        NOT_PROCESSED = "not_processed" # this is the initial status of a paper metadata object
        DISCARD = "discard" # after checks (title, abstract etc.) decided it is not usable and no full text retrieval is needed
        USABLE = "usable" # after checks (title, abstract etc.) decided to retrieve full text and further processing
        PROCESSED = "processed" # final state if the paper is used in the extraction pipeline

class PaperMetadata(SQLModel, table=True):
    __tablename__: str = "paper_metadata"

    id: Optional[int] = Field(default=None, primary_key=True)

    doi: Optional[str] = Field(default=None, index=True)
    pmid: Optional[str] = Field(default=None, index=True)
    pmcid: Optional[str] = Field(default=None, index=True)

    title: str
    abstract: Optional[str]
    status: PaperStatus = Field(default=PaperStatus.NOT_PROCESSED, 
                                index=True, 
                                sa_column_kwargs={"server_default": PaperStatus.NOT_PROCESSED}, 
                                description="Status of the metadata object before, after processing and discarding")
    is_open_access: bool

    dataset_subset: DatasetSubset = Field(
            default=DatasetSubset.SCRAPED, 
            index=True,
            sa_column_kwargs={"server_default": DatasetSubset.SCRAPED.value},
            description="Role in the pipeline: golden_benchmark or scraped"
        )
    source_databases: Optional[str] = Field(
            default=None, 
            description="Comma/semicolon-separated origin: e.g. 'AmyPro; CPAD-2.0; WaltzDB' or 'NCBI_SEARCH'"
        )

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
    relevance_records: List["PaperRelevance"] = Relationship(back_populates="paper")


def get_paper_by_id(paper_id: int, session: Optional[Session] = None) -> Optional[PaperMetadata]:
    """Retrieve a single PaperMetadata record by primary key id."""
    if session:
        return session.get(PaperMetadata, paper_id)
    with get_session_context() as s:
        return s.get(PaperMetadata, paper_id)


def get_papers_by_status(
    status: PaperStatus = PaperStatus.NOT_PROCESSED,
    limit: Optional[int] = None,
    offset: int = 0,
    dataset_subset: Optional[DatasetSubset] = None,
    session: Optional[Session] = None,
) -> List[PaperMetadata]:
    """Retrieve PaperMetadata records filtered by status and optional dataset_subset."""
    def _query(s: Session) -> List[PaperMetadata]:
        statement = select(PaperMetadata).where(PaperMetadata.status == status)
        if dataset_subset is not None:
            statement = statement.where(PaperMetadata.dataset_subset == dataset_subset)
        statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        return list(s.exec(statement).all())

    if session:
        return _query(session)
    with get_session_context() as s:
        return _query(s)


def get_unprocessed_papers(
    limit: Optional[int] = None,
    offset: int = 0,
    dataset_subset: Optional[DatasetSubset] = None,
    session: Optional[Session] = None,
) -> List[PaperMetadata]:
    """Convenience function to fetch papers with status NOT_PROCESSED."""
    return get_papers_by_status(
        status=PaperStatus.NOT_PROCESSED,
        limit=limit,
        offset=offset,
        dataset_subset=dataset_subset,
        session=session,
    )


def update_paper_status(
    paper_id: int,
    status: PaperStatus,
    session: Optional[Session] = None,
) -> Optional[PaperMetadata]:
    """Update the status of a PaperMetadata record by ID."""
    def _update(s: Session) -> Optional[PaperMetadata]:
        paper = s.get(PaperMetadata, paper_id)
        if paper is None:
            return None
        paper.status = status
        s.add(paper)
        s.commit()
        s.refresh(paper)
        return paper

    if session:
        return _update(session)
    with get_session_context() as s:
        return _update(s)


def count_papers_by_status(
    status: PaperStatus,
    dataset_subset: Optional[DatasetSubset] = None,
    session: Optional[Session] = None,
) -> int:
    """Count the total number of PaperMetadata records matching status and optional subset."""
    def _count(s: Session) -> int:
        statement = select(func.count(PaperMetadata.id)).where(PaperMetadata.status == status)
        if dataset_subset is not None:
            statement = statement.where(PaperMetadata.dataset_subset == dataset_subset)
        res = s.exec(statement).first()
        return res if res is not None else 0

    if session:
        return _count(session)
    with get_session_context() as s:
        return _count(s)