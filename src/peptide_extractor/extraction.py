from pathlib import Path
from typing import Optional, Union

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


PDF_EXTRACTION_SYSTEM_PROMPT = """
You are an expert scientific curator extracting experimental peptide self-assembly data from an attached PDF paper into the given JSON schema.

EXHAUSTIVE EXTRACTION RULES (CRITICAL):
1. Do NOT summarize, sample, or stop after a few examples. You MUST extract EVERY experimental peptide condition reported anywhere in the paper (prose, tables, table footnotes, figures, and captions).
2. Matrix / Screening Tables: Papers often contain tables testing N peptides across M solvents, concentrations, pH values, or temperatures. You MUST unroll the entire matrix and emit a separate entry for EVERY SINGLE TESTED `(peptide, condition)` cell—including negative / non-assembling results (e.g., soluble, precipitate, no gel -> `type: "none"`). Never skip non-gelling or negative rows/columns.
3. Cross-Reference Shared Conditions: Tables frequently omit experimental parameters that are stated in the Methods/Results text or table footnotes (such as temperature, incubation period, pH, N-/C-terminal protecting groups, preparation protocol, or detection methods). Apply those shared conditions to every relevant table entry.
4. Sequences & Terminals:
   - Convert 3-letter amino acid sequences (e.g., Boc-Ser-Val-Gly-OCH2Ph) into uppercase 1-letter codes (`sequence: "SVG"`).
   - Put N-terminal and C-terminal caps/protecting groups in `n_terminal` and `c_terminal` rather than inside `sequence`.
5. Multiple Conditions = Multiple Entries: If the same peptide is tested in multiple solvents, temperatures, pH values, or concentrations, emit a separate entry for each tested combination.

EVIDENCE GROUNDING RULES:
1. Ignore any `[sec-...]` tag format in field descriptions—this input is a raw PDF, not tagged text.
2. For every non-null property in an entry, provide a concise PDF location in `evidence_span` (e.g., `"Page 2, Table 1"`, `"Page 1, Results"`, `"Page 3, Fig. 1"`).
3. If a property was NOT experimentally determined or mentioned in the PDF, leave both the property and its corresponding `evidence_span` field as `null`.
"""


def run_extraction(
    paper_content: Union[str, bytes, Path],
    provider: str = "google_genai",
    model_name: str = "gemma-4-31b-it",
    filename: str = "paper.pdf",
    system_prompt: Optional[str] = None,
) -> PeptideExtractionResult:
    """Run peptide extraction on parsed text or raw PDF bytes/Path."""
    is_pdf = isinstance(paper_content, (bytes, Path))
    resolved_prompt = system_prompt or (
        PDF_EXTRACTION_SYSTEM_PROMPT if is_pdf else EXTRACTION_SYSTEM_PROMPT_V5
    )

    model = LLMController(
        provider=provider,
        model_name=model_name,
        structured_output=PeptideExtractionResult,
        system_prompt=resolved_prompt,
    )

    return model.extract_peptides(paper_content, stream=True, filename=filename)
