"""
This file will include evaluation methods for extraction entries
Main idea with evaluation is to check if extracted information exists in fulltext
"""
from peptide_extractor.parsers import to_debug_string
from peptide_extractor.parsers import get_node_by_id
from peptide_extractor.parsers import add_sentences
from peptide_extractor.parsers import parse_xml
from peptide_extractor.db.peptide_entry import PeptideConjugateMixture
from pydantic import Field
from typing import Optional
from pydantic import BaseModel
from peptide_extractor.db.extraction_evaluation import ExtractionEvaluation
from peptide_extractor.llm import LLMController
import json
from peptide_extractor.db import PeptideEntry
from rapidfuzz import fuzz


def check_values_in_fulltext(evidence: str, fulltext: str):
    found_in_text = evidence in fulltext

    if found_in_text:
        print(f"Found {evidence} in fulltext")
    else:
        print(f"Did not find {evidence} in fulltext")

    return found_in_text
   
def search_fuzzy(evidence: str, fulltext: str, threshold: float = 90.0) -> bool:
    # partial_ratio slides a window over fulltext looking for the closest match
    score = fuzz.partial_ratio(evidence.lower().strip(), fulltext.lower().strip())
    found = score >= threshold
    
    if found:
        print(f"[Method 3: Fuzzy] FOUND (Match score: {score:.1f}%)")
    else:
        print(f"[Method 3: Fuzzy] NOT FOUND (Match score: {score:.1f}%)")

    return found


JUDGE_SYSTEM_PROMPT = """
You are a scientific data verification auditor.
Your task is to determine whether an extracted experimental property value is faithfully supported by the provided verbatim quote from a paper.

Rules:
1. Set `is_faithful` to True if the quote directly states or unambiguously implies this value for the specified field.
2. Set `is_faithful` to False if the quote says something different, mentions a different peptide, or does not support the value.
3. Provide a concise explanation in `reason`.
"""

def groundness_check(field_evidence, fulltext) -> bool:
    """Check if the field value is grounded in the evidence span"""

    if check_values_in_fulltext(field_evidence, fulltext):
        return True

    if search_fuzzy(field_evidence, fulltext):
        return True
    
    return False

GROUNDNESS_CHECK_SYSTEM_PROMPT = """
You are a scientific data verification auditor.
Your task is to determine whether an extracted experimental property value is faithfully supported by the provided verbatim quote from a paper.
You should return a dictionary of field names and their corresponding verdicts, with True if the value is grounded in the evidence and False otherwise.

Example:
Input: [("temperature", "25", "The incubation was carried out at 25°C."), ("pH", "7.0", "The incubation was carried out at pH 7.0.")]
Output: {"temperature": True, "pH": True}

Input: [("temperature", "37", "The incubation was carried out at 25°C."), ("pH", "7.0", "The incubation was carried out at pH 7.0.")]
Output: {"temperature": False, "pH": True}
"""

class GroundnessVerdict(BaseModel):
    field_verdict: dict = Field(description="Dictionary of field names and their corresponding verdicts")

# TODO: implement groundness/faithfulness checks for evidence span with locations
def groundness_check_llm(field_value_quote_list: list, model: LLMController = None) -> GroundnessVerdict:
    model = LLMController(provider="google-genai", model_name="gemini-3.5-flash-lite", system_prompt=GROUNDNESS_CHECK_SYSTEM_PROMPT, structured_output=GroundnessVerdict) if model is None else model
    return model.evaluate_groundness(field_value_quote_list)
    

#class FaithfulnessVerdict(BaseModel):
#    is_faithful: bool = Field(
#        description="True if the extracted field value is supported by the evidence quote, False otherwise."
#    )
#    reason: Optional[str] = Field(
#        default=None, 
#        description="Brief explanation of the decision."
#    )
#
#def faithfulness_check(field_value, field_evidence, model: LLMController) -> FaithfulnessVerdict:
#    """Check if the field value is faithful to the evidence span"""
#    return model.evaluate_faithfullness(field_value, field_evidence)
#

def check_detection_methods_grounding(detection_method_str: str, fulltext: str) -> bool:
    """Split comma-separated methods and check if each technique is mentioned in full text."""
    if not detection_method_str or not detection_method_str.strip():
        return False
        
    methods = [m.strip() for m in detection_method_str.split(",") if m.strip()]
    if not methods:
        return False

    grounded_count = sum(1 for m in methods if groundness_check(m, fulltext))
    
    return (grounded_count / len(methods)) >= 0.8


EVIDENCE_SPAN_FIELDS = [
    "sequence",
    "type",
    "n_terminal",
    "c_terminal",
    "temperature",
    "ph",
    "peptide_concentration",
    "solution",
    "solvent",
    "incubation_period",
]

VERBATIM_FIELDS = [
    "preparation_protocol",
]

DEFAULT_STRUCTURAL_FIELDS = [
    "linear_cyclic",
    "peptide_conjugate_mixture",
    "volume_ratio",
]

def evaluate_extraction(extraction: PeptideEntry, fulltext: str) -> ExtractionEvaluation:
    #model = LLMController(provider="google-genai", model_name="gemini-3.5-flash-lite", system_prompt=JUDGE_SYSTEM_PROMPT, structured_output=FaithfulnessVerdict)
    model = LLMController(provider="google-genai", model_name="gemini-3.5-flash-lite", system_prompt=GROUNDNESS_CHECK_SYSTEM_PROMPT, structured_output=GroundnessVerdict)

    evidence_span = json.loads(extraction.evidence_span) if extraction.evidence_span else {}
    if evidence_span == {}:
        raise ValueError(f"No evidence span found for peptide entry {extraction.id}")


    active_evidence_fields = list(EVIDENCE_SPAN_FIELDS)
    
    # Conditional: conjugate_partner only expects a value/evidence if it's a conjugate or mixture
    is_pure_peptide = extraction.peptide_conjugate_mixture == PeptideConjugateMixture.PEPTIDE
    
    if not is_pure_peptide and extraction.conjugate_partner:
        active_evidence_fields.append("conjugate_partner")

    extracted_evidence_fields = [
        f for f in active_evidence_fields if getattr(extraction, f) is not None
    ]

    grounded_fields = []
    
    ## 1. Check fields that use EvidenceSpan
    #for field in extracted_evidence_fields:
    #    quote = evidence_span.get(field)
    #    if quote and quote.strip(): # if there is no evidence for an extracted field, it lowers the groundness
    #        if groundness_check(quote.strip(), fulltext):
    #            grounded_fields.append(field)

    # 1. Check fields that use EvidenceSpan (via location)

    root_node = parse_xml(fulltext)
    with_sentences = add_sentences(root_node)
    field_value_quote_list = []
    for field in extracted_evidence_fields:
        location = evidence_span.get(field)
        
        ## this should be updated to simpler usage later on
        #print(to_debug_string(with_sentences))
        #print(f"ID:{location}")
        quote = get_node_by_id(with_sentences, location)
        quote_text = quote.content if quote is not None else ""
        #print(quote_text)
        ####

        field_value_quote_list.append((field, getattr(extraction, field), quote_text))
    
    grounded_dict = groundness_check_llm(field_value_quote_list, model)
    print(f"Grounded: {grounded_dict}")

    for field, grounded in grounded_dict.field_verdict.items():
        if grounded:
            grounded_fields.append(field)
    

    # 2. Check Verbatim Fields directly on their value
    for field in VERBATIM_FIELDS:
        val = getattr(extraction, field)
        if val and str(val).strip():
            extracted_evidence_fields.append(field)
            if groundness_check(str(val).strip(), fulltext):
                grounded_fields.append(field)

    # 3. Check detection_method (seperate format)
    if extraction.detection_method and extraction.detection_method.strip():
        extracted_evidence_fields.append("detection_method")
        if check_detection_methods_grounding(extraction.detection_method, fulltext):
            grounded_fields.append("detection_method")

    total_checkable = len(extracted_evidence_fields)
    groundness_rate = (len(grounded_fields) / total_checkable) if total_checkable > 0 else 0.0

    #faithfulness_count = 0
    #for field in [f for f in grounded_fields if f in active_evidence_fields]:
    #    verdict = faithfulness_check(
    #        getattr(extraction, field), 
    #        evidence_span[field], 
    #        model
    #    )
    #    is_faithful = verdict.is_faithful
    #    reason = verdict.reason
    #    print(f"Field: {field}")
    #    print(f"is_faithful: {is_faithful}")
    #    print(f"Reason {reason}")
    #    if bool(is_faithful):
    #        faithfulness_count += 1
    #
    #print(f"Faithfullness_count {faithfulness_count}")
    #print(f"Grounded fields: {len(grounded_fields)}")
    #faithfulness_rate = (faithfulness_count / len(grounded_fields)) if grounded_fields else 0.0

    total_possible_fields = 15 if is_pure_peptide else 16
    present_fields_count = sum(
        1 for f in active_evidence_fields + VERBATIM_FIELDS + DEFAULT_STRUCTURAL_FIELDS
        if getattr(extraction, f) is not None
    )
    completeness_rate = min(1.0, present_fields_count / total_possible_fields)

    return ExtractionEvaluation(
        groundness_rate=f'{groundness_rate:.1%}',
        #faithfulness_rate=f'{faithfulness_rate:.1%}',
        completeness_rate=f'{completeness_rate:.1%}',
    )
