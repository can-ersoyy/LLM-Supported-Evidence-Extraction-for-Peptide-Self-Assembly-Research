from peptide_extractor.db.database import get_session_context
from sqlmodel import Session
from typing import Any
from sqlmodel import Field
from sqlmodel import JSON
from sqlmodel import Column
from typing import Optional
from sqlmodel import SQLModel


class RawProviderRecord(SQLModel, table=True):
    __tablename__: str = "raw_provider_records"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    """Raw record representation straight from external provider APIs (e.g., NCBI PMC, Europe PMC)."""
    provider: Optional[str] = Field(default=None, description="Provider source identifier (e.g. 'ncbi', 'europe_pmc')")
    search_query: Optional[str] = Field(default=None, description="The query term that matched this paper")
    raw_data: dict[str, Any] = Field(default={}, sa_column=Column(JSON), description="Complete unmodified JSON payload from provider, this will be used later for processing...")


def save_raw_provider_record(record: RawProviderRecord, session: Optional[Session] = None):
    def _save(s: Session) -> RawProviderRecord:
        s.add(record)
        s.commit()
        s.refresh(record)
        return record

    if session:
        return _save(session)
    with get_session_context() as s:
        return _save(s)
    