from peptide_extractor.db.peptide_entry import PeptideExtractionResult
from peptide_extractor.llm import LLMController

LOCATION_GROUNDING_RULES = """
LOCATION GROUNDING RULES:
1. Pure Location Annotations Only:
   Whenever you extract ANY non-null property in an entry (e.g., sequence, type, temperature, ph, peptide_concentration, solution, incubation_period), you MUST provide ONLY the location annotation tag in the corresponding field inside `evidence_span` where this information is found.
2. Format:
   Locations in the text are tagged in square brackets, e.g., `[sec-1/p-2/s-1]`, `[sec-5/p-3/sec-1/p-1/s-11]`. Provide ONLY the exact bracketed location tag string, e.g., "[sec-1/p-2/s-1]".
3. DO NOT include any verbatim text quotes, excerpts, or explanations inside `evidence_span`. Strictly output only the location identifier.
4. There can be at most 1 location tag provided per field. Different fields may share the same location tag if they appear in the same sentence or paragraph.
5. If a property was NOT experimentally determined or mentioned in the text, leave both the property and its corresponding `evidence_span` field as null.
"""

EXTRACTION_SYSTEM_PROMPT_V3 = """
You will receive the full text of a scientific paper and your goal is to extract experimental peptide self-assembly information according to the given schema. 
There can be several entries in a single paper, be careful!

{LOCATION_GROUNDING_RULES}
"""

EXTRACTION_SYSTEM_PROMPT_V5 = """
You will receive the full text of a scientific paper and your goal is to extract experimental peptide self-assembly information according to the given schema.
In the paper, there can be multiple entries, possibly with pretty similar conditions across different columns of the extraction schema.
For example, if you see that for a given peptide sequence, two different temperature conditions are tested, you need to have an extraction for each of these temperature values.
This applies to other fields as well (e.g. pH, incubation_period, solution, solvent etc.)

{LOCATION_GROUNDING_RULES}
"""


def run_extraction(
    paper_content: str,
    provider: str = "google_genai",
    model_name: str = "gemma-4-31b-it",
) -> PeptideExtractionResult:
    model = LLMController(
        provider=provider,
        model_name=model_name,
        structured_output=PeptideExtractionResult,
        system_prompt=EXTRACTION_SYSTEM_PROMPT_V3,
    )

    return model.extract_peptides(paper_content, stream=True)
