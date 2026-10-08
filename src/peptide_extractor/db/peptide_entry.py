from decimal import Decimal
from pydantic import BaseModel
from peptide_extractor.utils import to_lowercase, to_uppercase, get_utc_now
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional
from pydantic import ConfigDict
from sqlalchemy import Column, Text
from sqlmodel import DateTime, Field, Relationship, SQLModel

if TYPE_CHECKING:
    from peptide_extractor.db.paper_metadata import PaperMetadata


class ExtractionMode(str, Enum):
    MANUAL = "manual"
    LLM = "llm"

class PeptideAssemblyType(str, Enum):
    NONE = "none"
    RIBBON = "ribbon"
    NANOSPHERE = "nanosphere"
    ORGANOGEL = "organogel"
    TUBE = "tube"
    PARTICLE = "particle"
    VESICLE = "vesicle"
    HYDROGEL = "hydrogel"
    FIBER = "fiber"
    AMYLOID = "amyloid"

class PeptideConjugateMixture(str, Enum):
    PEPTIDE = "peptide"
    CONJUGATE = "conjugate"
    MIXTURE = "mixture"

class PeptideLinearity(str, Enum):
    LINEAR = "linear"
    CYCLIC = "cyclic"

class PeptideEntry(SQLModel, table=True):
    __tablename__: str = "peptide_entry"
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: Optional[int] = Field(
        default=None, foreign_key="paper_metadata.id", index=True
    )

    sequence: str
    length: int
    type: Optional[str] = None
    
    n_terminal: Optional[str] = None
    c_terminal: Optional[str] = None

    temperature: Optional[float] = None
    ph: Optional[float] = None
    
    peptide_concentration: Optional[float] = None
    solution: Optional[str] = None
    solvent: Optional[str] = None
    volume_ratio: Optional[str] = None
    incubation_period: Optional[float] = None

    peptide_conjugate_mixture: Optional[str] = None
    conjugate_partner: Optional[str] = None
    linear_cyclic: Optional[str] = None

    detection_method: Optional[str] = None
    preparation_protocol: Optional[str] = None

    evidence_span: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Concise summary or quoted evidence from the paper text",
    )


    pmid: str = Field(index=True)
    database: Optional[str] = None
    extraction_mode: ExtractionMode = Field(
        default=ExtractionMode.MANUAL,
        sa_column_kwargs={"server_default": ExtractionMode.MANUAL.value},
        description="Extraction mode: 'manual' (golden dataset) or 'llm'",
    )
    model: Optional[str] = Field(
        default=None,
        description="String name for LLM used for extraction",
    )
    created_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
    )
    updated_at: datetime = Field(
        sa_type=DateTime(timezone=True),  # type: ignore
        default_factory=get_utc_now,
        sa_column_kwargs={"onupdate": get_utc_now},
    )

    paper: Optional["PaperMetadata"] = Relationship()

solution_examples = ["acetic acid", "acetone", "acetonitrile", "acetyl acetone", "2-aminoethanol", "aniline", "anisole", "benzene", "benzonitrile", "benzyl alcohol", "1-butanol", "2-butanol", "i-butanol", "2-butanone", "t-butyl alcohol", "carbon disulfide", "carbon tetrachloride", "chlorobenzene", "chloroform", "cyclohexane", "cyclohexanol", "cyclohexanone", "di-n-butylphthalate", "1,1-dichloroethane", "1,2-dichloroethane", "diethylamine", "diethylene glycol", "diglyme", "dimethoxyethane", "glyme", "N,N-dimethylaniline", "dimethylformamide", "DMF", "dimethylphthalate", "dimethylsulfoxide", "DMSO", "dioxane", "ethanol", "ether", "ethyl acetate", "ethyl acetoacetate", "ethyl benzoate", "ethlyene glycol", "glycerin", "heptane", "1-heptanol", "hexane", "1-hexanol", "methanol", "methyl acetate", "methyl t-butyl ether", "MTBE", "methylene chloride", "1-octanol", "pentane", "1-pentanol", "2-pentanol", "3-pentanol", "2-pentanone", "3-pentanone", "1-propanol", "2-propanol", "pyridine", "tetrahydrofuran", "THF", "toluene", "water", "H2O", "D2O", "o-xylene", "p-xylene", "m-xylene", "heavy water", "water, heavy", "phosphate-buffered saline", "PBS", "sodium phosphate", "hexafluoro-2-propanol", "HFP", "HFIP", "phosphate buffer", "NaOH", "NaCl", "HCl", "glucono delta-lactone", "GdL", "dichloromethane", "1,3-butanediol", "dimethyl carbonate", "pyrimidine", "pyridine", "mesitylene", "octane", "decane", "isopropanol", "glycine"]


# this one is for LLM api (not db)
class PeptideLLMEntry(BaseModel):
    """Peptide Entry for a single peptide, there can be several entries in a single paper. Return null for fields that are not found."""

    sequence: str = Field(description="Peptide sequence in letters. (e.g. 'FF')")
    type: Optional[PeptideAssemblyType] = Field(default=None, description="Peptide assembly type, one of [none, ribbon, nanosphere, organogel, tube, particle, vesicle, hydrogel, fiber, amyloid].")

    n_terminal: Optional[str] = Field(default=None)
    c_terminal: Optional[str] = Field(default=None)

    temperature: Optional[float] = Field(default=None, description="Reaction/assembly temperature in Celsius as a float (e.g. 25.0, 37.0). If reported as 'room temperature' or 'RT', convert to 25.0. Return null if not mentioned.")
    ph: Optional[float] = Field(default=None, description="pH value (e.g. 7.4, 4.8 etc.)")

    peptide_concentration: Optional[float] = Field(default=None, description="Peptide concentration normalized to mg/ml as a float (e.g. 5.0, 10.2). Convert 1 wt% or 1% (w/v) to 10.0 mg/ml. Return null if only mM is given or not mentioned.")
    solution: Optional[str] = Field(default=None, description=f"some examples: {solution_examples}")
    solvent: Optional[str] = Field(default=None)
    volume_ratio: Optional[list[int]] = Field(default=None, description="Examples: [50, 50] for 50% to 50% (v/v) or [30, 70] for 30% to 70% (v/v). The list should sum to 100 if there are multiple solvents. Return [1] if the solution is only made of a single solvent. Return null if not mentioned.")
    incubation_period: Optional[int] = Field(default=None, description="In minutes")

    peptide_conjugate_mixture: PeptideConjugateMixture = PeptideConjugateMixture.PEPTIDE
    conjugate_partner: Optional[str] = None
    linear_cyclic: PeptideLinearity = PeptideLinearity.LINEAR
    detection_method: Optional[str] = None
    preparation_protocol: Optional[str] = None

    evidence_span: Optional[str] = Field(
        description="Positions (paragraph) or summary of where you found the information for this peptide entry from the given paper content.",
    )
    
# this one is for multiple entries
class PeptideExtractionResult(BaseModel):
    entries: list[PeptideLLMEntry]

def peptide_sequence_to_len(sequence: str) -> Optional[int]:
    return len(sequence) if sequence is not None else None

def peptide_conjugate_check(peptide_llm_entry: PeptideLLMEntry) -> Optional[PeptideConjugateMixture]:
    if peptide_llm_entry.peptide_conjugate_mixture == PeptideConjugateMixture.PEPTIDE:
        return None
    return peptide_llm_entry.conjugate_partner

def peptide_volume_ratio_to_str(ratio: Optional[list[int]]) -> Optional[str]:
    if ratio is None:
        return None
    if len(ratio) == 1:
        return str(ratio[0])
    if len(ratio) == 2:
        return f"{ratio[0]}/{ratio[1]}"
    if len(ratio) > 2:
        return ":".join([str(r) for r in ratio])
    return None


def peptide_llm_entry_to_peptide_entry(peptide_llm_entry: PeptideLLMEntry, paper_id: int, pmid: str, database: Optional[str] = None, model: Optional[str] = None) -> PeptideEntry:
    return PeptideEntry(
        paper_id=paper_id,
        pmid=pmid,
        sequence=to_uppercase(peptide_llm_entry.sequence),
        length=peptide_sequence_to_len(peptide_llm_entry.sequence),
        type=to_lowercase(peptide_llm_entry.type),
        n_terminal=peptide_llm_entry.n_terminal,
        c_terminal=peptide_llm_entry.c_terminal,
        temperature=peptide_llm_entry.temperature,
        ph=peptide_llm_entry.ph,
        peptide_concentration=peptide_llm_entry.peptide_concentration,
        solution=peptide_llm_entry.solution,
        solvent=peptide_llm_entry.solvent,
        volume_ratio=peptide_volume_ratio_to_str(peptide_llm_entry.volume_ratio),
        incubation_period=peptide_llm_entry.incubation_period,
        peptide_conjugate_mixture=peptide_llm_entry.peptide_conjugate_mixture,
        conjugate_partner=peptide_conjugate_check(peptide_llm_entry),
        linear_cyclic=peptide_llm_entry.linear_cyclic,
        detection_method=peptide_llm_entry.detection_method,
        preparation_protocol=peptide_llm_entry.preparation_protocol,
        evidence_span=peptide_llm_entry.evidence_span,
        extraction_mode=ExtractionMode.LLM,
        model=model,
        database=database,
    )