"""
This file will include evaluation methods for extraction entries
Main idea with evaluation is to check if extracted information exists in fulltext
"""

from typing import TYPE_CHECKING, Optional
import json
import logging

logger = logging.getLogger(__name__)
from peptide_extractor.parsers import get_node_by_id
from peptide_extractor.parsers import add_sentences
from peptide_extractor.parsers import parse_xml
from peptide_extractor.db.peptide_entry import PeptideConjugateMixture
from pydantic import Field
from pydantic import BaseModel
from peptide_extractor.db.extraction_evaluation import ExtractionEvaluation
from peptide_extractor.db import PeptideEntry
from rapidfuzz import fuzz

if TYPE_CHECKING:
    from peptide_extractor.llm import LLMController


def check_values_in_fulltext(evidence: str, fulltext: str):
    found_in_text = evidence in fulltext

    if found_in_text:
        logger.info(f"Found {evidence} in fulltext")
    else:
        logger.info(f"Did not find {evidence} in fulltext")

    return found_in_text


def search_fuzzy(evidence: str, fulltext: str, threshold: float = 90.0) -> bool:
    # partial_ratio slides a window over fulltext looking for the closest match
    score = fuzz.partial_ratio(evidence.lower().strip(), fulltext.lower().strip())
    found = score >= threshold

    if found:
        logger.info(f"[Method 3: Fuzzy] FOUND (Match score: {score:.1f}%)")
    else:
        logger.info(f"[Method 3: Fuzzy] NOT FOUND (Match score: {score:.1f}%)")

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
    field_verdict: dict = Field(
        description="Dictionary of field names and their corresponding verdicts"
    )


# TODO: implement groundness/faithfulness checks for evidence span with locations
def groundness_check_llm(
    field_value_quote_list: list, model: Optional["LLMController"] = None
) -> GroundnessVerdict:
    if model is None:
        from peptide_extractor.llm import LLMController

        model = LLMController(
            provider="ollama",
            model_name="mistral-small3.2:latest",
            system_prompt=GROUNDNESS_CHECK_SYSTEM_PROMPT,
            structured_output=GroundnessVerdict,
            base_url="http://localhost:11435",
        )
    return model.evaluate_groundness(field_value_quote_list)


# class FaithfulnessVerdict(BaseModel):
#    is_faithful: bool = Field(
#        description="True if the extracted field value is supported by the evidence quote, False otherwise."
#    )
#    reason: Optional[str] = Field(
#        default=None,
#        description="Brief explanation of the decision."
#    )
#
# def faithfulness_check(field_value, field_evidence, model: LLMController) -> FaithfulnessVerdict:
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


def evaluate_extraction(
    extraction: PeptideEntry,
    fulltext: str,
    model: Optional["LLMController"] = None,
) -> ExtractionEvaluation:
    active_evidence_fields = list(EVIDENCE_SPAN_FIELDS)

    # Conditional: conjugate_partner only expects a value/evidence if it's a conjugate or mixture
    is_pure_peptide = (
        extraction.peptide_conjugate_mixture == PeptideConjugateMixture.PEPTIDE
    )

    if not is_pure_peptide and extraction.conjugate_partner:
        active_evidence_fields.append("conjugate_partner")

    total_possible_fields = 15 if is_pure_peptide else 16
    present_fields_count = sum(
        1
        for f in active_evidence_fields + VERBATIM_FIELDS + DEFAULT_STRUCTURAL_FIELDS
        if getattr(extraction, f) is not None
    )
    completeness_rate = min(1.0, present_fields_count / total_possible_fields)

    evidence_span = (
        json.loads(extraction.evidence_span) if extraction.evidence_span else {}
    )
    if evidence_span == {}:
        return ExtractionEvaluation(
            groundness_rate=0.0,
            # faithfulness_rate=0.0,
            completeness_rate=completeness_rate,
            peptide_entry_id=getattr(extraction, "id", None),
        )

    extracted_evidence_fields = [
        f for f in active_evidence_fields if getattr(extraction, f) is not None
    ]

    grounded_fields = set()

    ## 1. Check fields that use EvidenceSpan
    # for field in extracted_evidence_fields:
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
        if location is None:
            continue

        ## this should be updated to simpler usage later on
        # logger.info(to_debug_string(with_sentences))
        # logger.info(f"ID:{location}")
        quote = get_node_by_id(with_sentences, location)
        quote_text = quote.content if quote is not None else ""
        # logger.info(quote_text)
        ####

        field_value_quote_list.append((field, getattr(extraction, field), quote_text))

    if field_value_quote_list:
        grounded_dict = groundness_check_llm(field_value_quote_list, model)
        logger.info(f"Grounded: {grounded_dict}")

        queried_span_fields = {field for field, _, _ in field_value_quote_list}
        field_lookup = {f.lower(): f for f in queried_span_fields}

        for field, grounded in grounded_dict.field_verdict.items():
            canonical_field = field_lookup.get(field.lower())
            if canonical_field and grounded:
                grounded_fields.add(canonical_field)

    # 2. Check Verbatim Fields directly on their value
    for field in VERBATIM_FIELDS:
        val = getattr(extraction, field)
        if val and str(val).strip():
            if field not in extracted_evidence_fields:
                extracted_evidence_fields.append(field)
            if groundness_check(str(val).strip(), fulltext):
                grounded_fields.add(field)

    # 3. Check detection_method (seperate format)
    if extraction.detection_method and extraction.detection_method.strip():
        if "detection_method" not in extracted_evidence_fields:
            extracted_evidence_fields.append("detection_method")
        if check_detection_methods_grounding(extraction.detection_method, fulltext):
            grounded_fields.add("detection_method")

    checkable_fields_set = set(extracted_evidence_fields)
    total_checkable = len(checkable_fields_set)
    valid_grounded = grounded_fields.intersection(checkable_fields_set)
    groundness_rate = (
        min(1.0, len(valid_grounded) / total_checkable)
        if total_checkable > 0
        else 0.0
    )

    # faithfulness_count = 0
    # for field in [f for f in grounded_fields if f in active_evidence_fields]:
    #    verdict = faithfulness_check(
    #        getattr(extraction, field),
    #        evidence_span[field],
    #        model
    #    )
    #    is_faithful = verdict.is_faithful
    #    reason = verdict.reason
    #    logger.info(f"Field: {field}")
    #    logger.info(f"is_faithful: {is_faithful}")
    #    logger.info(f"Reason {reason}")
    #    if bool(is_faithful):
    #        faithfulness_count += 1
    #
    # logger.info(f"Faithfullness_count {faithfulness_count}")
    # logger.info(f"Grounded fields: {len(grounded_fields)}")
    # faithfulness_rate = (faithfulness_count / len(grounded_fields)) if grounded_fields else 0.0

    return ExtractionEvaluation(
        groundness_rate=groundness_rate,
        # faithfulness_rate=f'{faithfulness_rate:.1%}',
        completeness_rate=completeness_rate,
        peptide_entry_id=getattr(extraction, "id", None),
    )
