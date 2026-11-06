from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import List


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


class PerPaperResult(BaseModel):
    paper_id: int = Field(
        description="Id of the paper (received with input) sent back to match the relevance result to paper."
    )
    relevance: RelevanceResult


class BulkRelevanceResults(BaseModel):
    results: List[PerPaperResult] = Field(
        description="List of relevance results for each paper."
    )
