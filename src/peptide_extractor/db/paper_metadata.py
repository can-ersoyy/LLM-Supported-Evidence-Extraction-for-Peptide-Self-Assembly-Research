from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, List, Optional
from sqlmodel import DateTime, Field, Relationship, SQLModel

from peptide_extractor.utils import get_utc_now

if TYPE_CHECKING:
    from peptide_extractor.db.paper_content import PaperContent


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