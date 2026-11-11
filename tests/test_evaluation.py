import json
from unittest.mock import MagicMock, patch
import pytest

from peptide_extractor.db import PeptideEntry
from peptide_extractor.db.peptide_entry import PeptideConjugateMixture
from peptide_extractor.evaluation import (
    GroundnessVerdict,
    check_detection_methods_grounding,
    check_values_in_fulltext,
    evaluate_extraction,
    groundness_check,
    groundness_check_llm,
    search_fuzzy,
)
from peptide_extractor.parsers import Node


class MockNode(Node):
    def __init__(self, node_id, content):
        super().__init__(node_id, content)


# ============================================================================
# Basic Helper & Groundness Function Tests
# ============================================================================


def test_check_values_in_fulltext():
    assert check_values_in_fulltext("hydrogel", "The peptide forms a stable hydrogel.")
    assert not check_values_in_fulltext("nanosphere", "The peptide forms a stable hydrogel.")


def test_search_fuzzy():
    assert search_fuzzy("incubation at 25 C", "The sample incubation at 25 °C was observed.")
    assert not search_fuzzy("completely different text", "The sample was frozen at -80 °C.")


def test_groundness_check():
    assert groundness_check("hydrogel", "The peptide formed a hydrogel.")
    assert not groundness_check("nanotube", "The peptide formed a clear solution.")


def test_check_detection_methods_grounding():
    fulltext = "Characterization was performed using TEM and CD spectroscopy."
    # Both TEM and CD are present -> 2/2 = 1.0 >= 0.8 -> True
    assert check_detection_methods_grounding("TEM, CD", fulltext)
    # AFM is absent, TEM is present -> 1/2 = 0.5 < 0.8 -> False
    assert not check_detection_methods_grounding("TEM, AFM", fulltext)
    # Empty detection method -> False
    assert not check_detection_methods_grounding("", fulltext)
    assert not check_detection_methods_grounding("   ", fulltext)


# ============================================================================
# Bug Fix Verification: groundness_rate <= 1.0
# ============================================================================


@patch("peptide_extractor.evaluation.get_node_by_id")
@patch("peptide_extractor.evaluation.add_sentences")
@patch("peptide_extractor.evaluation.parse_xml")
def test_groundness_rate_caps_at_1_when_llm_hallucinates_extra_fields(
    mock_parse_xml, mock_add_sentences, mock_get_node_by_id
):
    """
    Bug reproduction test:
    The extraction only has 1 field with evidence ('sequence').
    The LLM hallucinates and returns True for multiple unprompted fields (e.g. from prompt examples).
    Previously: len(grounded_fields) = 5, total_checkable = 1 -> groundness_rate = 5.0 (> 1.0).
    Fixed behavior: Only queried fields in field_value_quote_list are accepted.
    groundness_rate must be 1.0 (or at most 1.0).
    """
    mock_node = MockNode("sec-0/p-1/s-0", "The peptide sequence is FF.")
    mock_get_node_by_id.return_value = mock_node

    extraction = PeptideEntry(
        sequence="FF",
        length=2,
        pmid="12345678",
        evidence_span=json.dumps({"sequence": "sec-0/p-1/s-0"}),
    )

    # LLM hallucinates extra fields: temperature, ph, type, solvent
    mock_model = MagicMock()
    mock_model.evaluate_groundness.return_value = GroundnessVerdict(
        field_verdict={
            "sequence": True,
            "temperature": True,
            "ph": True,
            "type": True,
            "solvent": True,
        }
    )

    evaluation = evaluate_extraction(extraction, "<xml>dummy</xml>", model=mock_model)

    assert evaluation.groundness_rate <= 1.0
    assert evaluation.groundness_rate == 1.0
    # Ensure LLM was queried with only sequence
    mock_model.evaluate_groundness.assert_called_once()
    called_list = mock_model.evaluate_groundness.call_args[0][0]
    assert len(called_list) == 1
    assert called_list[0][0] == "sequence"


@patch("peptide_extractor.evaluation.get_node_by_id")
@patch("peptide_extractor.evaluation.add_sentences")
@patch("peptide_extractor.evaluation.parse_xml")
def test_groundness_rate_ignores_fields_with_none_location(
    mock_parse_xml, mock_add_sentences, mock_get_node_by_id
):
    """
    Extraction has 'sequence' and 'temperature', but only 'sequence' has a location in evidence_span.
    Even if LLM returns True for 'temperature', it should not be counted because it was not queried.
    total_checkable = 2, grounded = 1 ('sequence') -> groundness_rate = 0.5.
    """
    mock_node = MockNode("sec-0/p-1/s-0", "The peptide sequence is FF.")
    mock_get_node_by_id.return_value = mock_node

    extraction = PeptideEntry(
        sequence="FF",
        length=2,
        temperature=25.0,
        pmid="12345678",
        evidence_span=json.dumps({"sequence": "sec-0/p-1/s-0"}),  # temperature has no location
    )

    mock_model = MagicMock()
    mock_model.evaluate_groundness.return_value = GroundnessVerdict(
        field_verdict={
            "sequence": True,
            "temperature": True,  # LLM hallucinates temperature as True
        }
    )

    evaluation = evaluate_extraction(extraction, "<xml>dummy</xml>", model=mock_model)

    assert evaluation.groundness_rate <= 1.0
    assert evaluation.groundness_rate == 0.5


@patch("peptide_extractor.evaluation.get_node_by_id")
@patch("peptide_extractor.evaluation.add_sentences")
@patch("peptide_extractor.evaluation.parse_xml")
def test_groundness_rate_case_insensitive_llm_keys(
    mock_parse_xml, mock_add_sentences, mock_get_node_by_id
):
    """
    LLM prompt example uses 'pH' while EVIDENCE_SPAN_FIELDS has 'ph'.
    Field lookup should match case-insensitively and record 'ph' as grounded.
    """
    mock_node = MockNode("sec-0/p-1/s-0", "The solution pH was 7.4.")
    mock_get_node_by_id.return_value = mock_node

    extraction = PeptideEntry(
        sequence="FF",
        length=2,
        ph=7.4,
        pmid="12345678",
        evidence_span=json.dumps({"ph": "sec-0/p-1/s-0"}),
    )

    mock_model = MagicMock()
    mock_model.evaluate_groundness.return_value = GroundnessVerdict(
        field_verdict={"pH": True}
    )

    evaluation = evaluate_extraction(extraction, "<xml>dummy</xml>", model=mock_model)

    # 2 checkable fields: sequence (no evidence span) and ph (grounded)
    # total_checkable = 2, grounded = 1 ('ph') -> 0.5
    assert evaluation.groundness_rate == 0.5


def test_empty_evidence_span_returns_zero_groundness():
    extraction = PeptideEntry(
        sequence="FF",
        length=2,
        pmid="12345678",
        evidence_span=None,
    )
    evaluation = evaluate_extraction(extraction, "<xml>dummy</xml>")
    assert evaluation.groundness_rate == 0.0

    extraction_empty_dict = PeptideEntry(
        sequence="FF",
        length=2,
        pmid="12345678",
        evidence_span="{}",
    )
    evaluation_empty = evaluate_extraction(extraction_empty_dict, "<xml>dummy</xml>")
    assert evaluation_empty.groundness_rate == 0.0


@patch("peptide_extractor.evaluation.get_node_by_id")
@patch("peptide_extractor.evaluation.add_sentences")
@patch("peptide_extractor.evaluation.parse_xml")
def test_no_queryable_span_fields_skips_llm_call(
    mock_parse_xml, mock_add_sentences, mock_get_node_by_id
):
    """
    When evidence_span has non-empty JSON, but none of the keys correspond to
    active fields present on the extraction, LLM should NOT be called.
    """
    extraction = PeptideEntry(
        sequence="FF",
        length=2,
        pmid="12345678",
        evidence_span=json.dumps({"irrelevant_field": "loc-1"}),
    )
    mock_model = MagicMock()

    evaluation = evaluate_extraction(extraction, "<xml>dummy</xml>", model=mock_model)

    mock_model.evaluate_groundness.assert_not_called()
    assert evaluation.groundness_rate == 0.0


@patch("peptide_extractor.evaluation.get_node_by_id")
@patch("peptide_extractor.evaluation.add_sentences")
@patch("peptide_extractor.evaluation.parse_xml")
def test_duplicate_fields_across_llm_and_verbatim_are_not_double_counted(
    mock_parse_xml, mock_add_sentences, mock_get_node_by_id
):
    """
    Ensure verbatim fields (preparation_protocol) and detection_method are not
    counted twice if the LLM also returns them in field_verdict.
    """
    fulltext = "The peptide was dissolved in DMSO. Characterization was done by TEM and CD."
    mock_node = MockNode("sec-0/p-1/s-0", "Peptide sequence is FF.")
    mock_get_node_by_id.return_value = mock_node

    extraction = PeptideEntry(
        sequence="FF",
        length=2,
        preparation_protocol="The peptide was dissolved in DMSO.",
        detection_method="TEM, CD",
        pmid="12345678",
        evidence_span=json.dumps({"sequence": "sec-0/p-1/s-0"}),
    )

    mock_model = MagicMock()
    # LLM hallucinates preparation_protocol and detection_method in verdict
    mock_model.evaluate_groundness.return_value = GroundnessVerdict(
        field_verdict={
            "sequence": True,
            "preparation_protocol": True,
            "detection_method": True,
        }
    )

    evaluation = evaluate_extraction(extraction, fulltext, model=mock_model)

    # 3 checkable fields: sequence, preparation_protocol, detection_method
    # All 3 are grounded: sequence (via LLM), preparation_protocol (via verbatim), detection_method (via detection grounding)
    # Grounded count must be 3, total_checkable must be 3 -> groundness_rate = 1.0 (never > 1.0)
    assert evaluation.groundness_rate <= 1.0
    assert evaluation.groundness_rate == 1.0


@patch("peptide_extractor.evaluation.get_node_by_id")
@patch("peptide_extractor.evaluation.add_sentences")
@patch("peptide_extractor.evaluation.parse_xml")
def test_conjugate_partner_included_only_when_not_pure_peptide(
    mock_parse_xml, mock_add_sentences, mock_get_node_by_id
):
    """
    When peptide_conjugate_mixture is CONJUGATE and conjugate_partner is set,
    conjugate_partner should be included in checkable fields.
    """
    mock_node_seq = MockNode("loc-seq", "Sequence FF")
    mock_node_partner = MockNode("loc-partner", "Conjugated with PEG")

    def mock_get_node(tree, location):
        if location == "loc-seq":
            return mock_node_seq
        if location == "loc-partner":
            return mock_node_partner
        return None

    mock_get_node_by_id.side_effect = mock_get_node

    extraction = PeptideEntry(
        sequence="FF",
        length=2,
        peptide_conjugate_mixture=PeptideConjugateMixture.CONJUGATE,
        conjugate_partner="PEG",
        pmid="12345678",
        evidence_span=json.dumps({"sequence": "loc-seq", "conjugate_partner": "loc-partner"}),
    )

    mock_model = MagicMock()
    mock_model.evaluate_groundness.return_value = GroundnessVerdict(
        field_verdict={"sequence": True, "conjugate_partner": True}
    )

    evaluation = evaluate_extraction(extraction, "<xml>dummy</xml>", model=mock_model)

    assert evaluation.groundness_rate <= 1.0
    assert evaluation.groundness_rate == 1.0
