from peptide_extractor.db.peptide_entry import PeptideExtractionResult
from peptide_extractor.llm import LLMController



EXTRACTION_SYSTEM_PROMPT = """
You will receive the full text of a scientific paper and your goal is to extract experimental peptide self-assembly information according to the given schema. 
There can be several entries in a single paper, be careful!
"""

EXTRACTION_SYSTEM_PROMPT_V2 = """
You will receive the full text of a scientific paper and your goal is to extract experimental peptide self-assembly information according to the given schema. 
There can be several entries in a single paper, be careful!

STRICT EVIDENCE GROUNDING RULES:
1. 1-to-1 Field Evidence Requirement:
   Whenever you extract ANY non-null property in an entry (e.g., sequence, type, temperature, ph, peptide_concentration, solution, incubation_period), you MUST include the exact verbatim quote in the corresponding field inside `evidence_span`.
2. Do NOT omit fields from `evidence_span` if the property was extracted above.
3. Every evidence quote must be at least 5 words long and copied verbatim from the paper text so it can be verified via exact text search.
4. If a property was NOT experimentally determined or mentioned in the text, leave both the property and its evidence_span field as null.
"""

def run_extraction( paper_content: str, provider: str = "google_genai", model_name: str = "gemma-4-31b-it") -> PeptideExtractionResult:
    #model = LLMController(provider=provider, model_name=model_name, structured_output=PeptideExtractionResult, system_prompt=EXTRACTION_SYSTEM_PROMPT)
    model = LLMController(provider=provider, model_name=model_name, structured_output=PeptideExtractionResult, system_prompt=EXTRACTION_SYSTEM_PROMPT_V2)

    res = model.extract_peptides(paper_content)
    return res.entries
