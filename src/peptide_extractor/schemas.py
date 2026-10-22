from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import List
from typing import Any

class RawProviderRecord(BaseModel):
    """Raw record representation straight from external provider APIs (e.g., NCBI PMC, Europe PMC)."""
    provider: str = Field(..., description="Provider source identifier (e.g. 'ncbi', 'europe_pmc')")
    search_query: str = Field(..., description="The query term that matched this paper")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Complete unmodified JSON payload from provider, this will be used later for processing...")

class RelevanceResult(BaseModel):
    detected_clues: Optional[List[str]] = Field(
        default_factory=list,
        description="Specific experimental techniques, peptide identities, or conditions mentioned in the abstract.",
    )
    explanation: Optional[str] = Field(
        description="Justification focusing on experimental conditions mentioned or hinted upon.",
    )
    is_relevant: Optional[bool] = Field(
        description="True if experimental data is potentially reported or hinted at in the fulltext.",
    )
    confidence: Optional[float] = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )