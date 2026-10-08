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

    temperature: Optional[float] = Field(default=None, description="In celcius (e.g. 25.0, 37.45 etc.)")
    ph: Optional[float] = Field(default=None, description="pH value (e.g. 7.4, 4.8 etc.)")

    peptide_concentration: Optional[float] = Field(default=None, description="In mg/ml (e.g 10.2, 5.6 etc.)")
    solution: Optional[str] = Field(default=None, description=f"some examples: {solution_examples}")
    solvent: Optional[str] = Field(default=None)
    volume_ratio: Optional[tuple[int, int]] = Field(default=None, description="Example: (1, 1) for 1:1 (v/v) or (20, 1) for 20:1 (v/v)")
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

def peptide_volume_ratio_to_str(ratio: Optional[tuple[int, int]]) -> Optional[str]:
    if ratio is not None and len(ratio) == 2:
        return f"{ratio[0]}/{ratio[1]}"
    return None


def peptide_llm_entry_to_peptide_entry(peptide_llm_entry: PeptideLLMEntry, paper_id: int, pmid: str, database: Optional[str] = None, model: Optional[str] = None) -> PeptideEntry:
    return PeptideEntry(
        paper_id=paper_id,
        pmid=pmid,
        sequence=to_uppercase(peptide_llm_entry.sequence),
        length=peptide_sequence_to_len(peptide_llm_entry.sequence),
        type=to_lowercase(peptide_llm_entry.type),
        n_terminal=to_lowercase(peptide_llm_entry.n_terminal),
        c_terminal=to_lowercase(peptide_llm_entry.c_terminal),
        temperature=peptide_llm_entry.temperature,
        ph=peptide_llm_entry.ph,
        peptide_concentration=peptide_llm_entry.peptide_concentration,
        solution=to_lowercase(peptide_llm_entry.solution),
        solvent=to_lowercase(peptide_llm_entry.solvent),
        volume_ratio=to_lowercase(peptide_volume_ratio_to_str(peptide_llm_entry.volume_ratio)),
        incubation_period=peptide_llm_entry.incubation_period,
        peptide_conjugate_mixture=peptide_llm_entry.peptide_conjugate_mixture,
        conjugate_partner=to_lowercase(peptide_conjugate_check(peptide_llm_entry)),
        linear_cyclic=peptide_llm_entry.linear_cyclic,
        detection_method=peptide_llm_entry.detection_method,
        preparation_protocol=peptide_llm_entry.preparation_protocol,
        evidence_span=peptide_llm_entry.evidence_span,
        extraction_mode=ExtractionMode.LLM,
        model=model,
        database=database,
    )