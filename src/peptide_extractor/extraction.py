from peptide_extractor.db.peptide_entry import PeptideExtractionResult
from peptide_extractor.llm import LLMController



EXTRACTION_SYSTEM_PROMPT = """
You will receive the full text of a scientific paper and your goal is to extract experimental peptide self-assembly information according to the given schema. 
There can be several entries in a single paper, be careful!
"""

def run_extraction( paper_content: str, provider: str = "google_genai", model_name: str = "gemma-4-31b-it") -> PeptideExtractionResult:
    model = LLMController(provider=provider, model_name=model_name, structured_output=PeptideExtractionResult, system_prompt=EXTRACTION_SYSTEM_PROMPT)
    res = model.extract_peptides(paper_content)
    return res.entries
