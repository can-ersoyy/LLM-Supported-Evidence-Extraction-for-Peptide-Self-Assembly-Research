from typing import Any
from pydantic import BaseModel, Field


class RawProviderRecord(BaseModel):
    """Raw record representation straight from external provider APIs (e.g., NCBI PMC, Europe PMC)."""
    provider: str = Field(..., description="Provider source identifier (e.g. 'pmc', 'europe_pmc')")
    provider_id: str = Field(..., description="Provider specific identifier (e.g. PMCID, PMID, UID)")
    search_query: str = Field(..., description="The query term that matched this paper")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Complete unmodified JSON payload from provider, this will be used later for processing...")
