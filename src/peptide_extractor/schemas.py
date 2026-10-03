from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, Text
from typing_extensions import List
from typing import Any

class RawProviderRecord(BaseModel):
    """Raw record representation straight from external provider APIs (e.g., NCBI PMC, Europe PMC)."""
    provider: str = Field(..., description="Provider source identifier (e.g. 'ncbi', 'europe_pmc')")
    search_query: str = Field(..., description="The query term that matched this paper")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Complete unmodified JSON payload from provider, this will be used later for processing...")

class RelevanceResult(BaseModel):
    detected_clues: List[str] = Field(
        default_factory=list,
        description="Specific experimental techniques, peptide identities, or conditions mentioned in the abstract.",
    )
    explanation: str = Field(
        description="Justification focusing on experimental conditions mentioned or hinted upon.",
    )
    is_relevant: bool = Field(
        description="True if experimental data is potentially reported or hinted at in the fulltext.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )

class PeptideLLMEntry(BaseModel):
    """Peptide Entry for a single peptide, there can be several entries in a single paper. Give 'NP' if a field is not found."""
    sequence: str
    length: int
    type: str = Field(default=None, description="Can be one of the following: [none, ribbon, nanosphere, organogel, tube, particle, vesicle, hydrogel, fiber, amyloid]")
    n_terminal: str
    c_terminal: str
    temperature: str = Field(default=None, description="Give exact number in celcius, don't type unit." )
    ph: str = Field(default=None, description="Give exact number (float or int) in pH, don't type unit.")
    peptide_concentration: str = Field(description="Give in mg/ml")
    solution: str
    solvent: str
    volume_ratio: str = Field(default=None, description="Example: 1:1 (v/v) or 20:1 (v/v)")
    incubation_period: str = Field(default=None, description="Single value in minutes (don't type minutes), if not provided exactly you can use > or < of a value in minutes.")

    peptide_conjugate_mixture: str = Field(
        default=None, description="One of the following: [peptide, conjugate, mixture]"
    )
    conjugate_partner: Optional[str] = Field(default=None, description="Is empty if the entry is just a peptide.")
    linear_cyclic: Optional[str] = Field(
        default=None, description="Can be one of the following: [linear, cyclic]"
    )

    detection_method: str
    preparation_protocol: str

    evidence_span: str = Field(
        description="Positions (paragraph) or summary of where you found the information for this peptide entry from the given paper content.",
    )
    

class PeptideExtractionResult(BaseModel):
    entries: list[PeptideLLMEntry]
