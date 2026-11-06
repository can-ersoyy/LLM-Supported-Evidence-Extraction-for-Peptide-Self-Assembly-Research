from peptide_extractor.utils import doi_to_filename
import pytest

from peptide_extractor.utils import (
    get_body_from_jats_xml,
    normalize_doi,
)


# ============================================================================
# Tests for normalize_doi
# ============================================================================


@pytest.mark.parametrize(
    "input_doi,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("https://doi.org/", None),
        ("doi:", None),
        ("10.1016/j.bpj.2020.01.001", "10.1016/j.bpj.2020.01.001"),
        ("10.1039/C8RA06034G", "10.1039/c8ra06034g"),
        ("  10.1000/182  ", "10.1000/182"),
        ("https://doi.org/10.1000/182", "10.1000/182"),
        ("http://doi.org/10.1000/182", "10.1000/182"),
        ("https://dx.doi.org/10.1000/182", "10.1000/182"),
        ("http://dx.doi.org/10.1000/182", "10.1000/182"),
        ("doi:10.1000/182", "10.1000/182"),
        ("DOI:10.1000/182", "10.1000/182"),
        ("  https://doi.org/10.1039/C8RA06034G  ", "10.1039/c8ra06034g"),
    ],
)
def test_normalize_doi(input_doi, expected):
    """Test DOI normalization across various prefixes, casings, and whitespace."""
    assert normalize_doi(input_doi) == expected


def test_doi_to_filename():
    doi_example_1 = "10.1016/j.bpj.2020.01.001"
    expected_1 = "10_1016_j_bpj_2020_01_001"
    assert doi_to_filename(doi_example_1) == expected_1

    doi_example_3 = "  10.1000/182  "
    expected_3 = "10_1000_182"
    assert doi_to_filename(doi_example_3) == expected_3

    doi_example_4 = "https://doi.org/10.1000/182"
    expected_4 = "10_1000_182"
    assert doi_to_filename(doi_example_4) == expected_4

    doi_example_5 = "http://doi.org/10.1000/182"
    expected_5 = "10_1000_182"
    assert doi_to_filename(doi_example_5) == expected_5

    doi_example_6 = "https://dx.doi.org/10.1000/182"
    expected_6 = "10_1000_182"
    assert doi_to_filename(doi_example_6) == expected_6

    doi_example_7 = "http://dx.doi.org/10.1000/182"
    expected_7 = "10_1000_182"
    assert doi_to_filename(doi_example_7) == expected_7

    doi_example_8 = "doi:10.1000/182"
    expected_8 = "10_1000_182"
    assert doi_to_filename(doi_example_8) == expected_8

    doi_example_9 = "DOI:10.1000/182"
    expected_9 = "10_1000_182"
    assert doi_to_filename(doi_example_9) == expected_9

    doi_example_10 = "  https://doi.org/10.1039/C8RA06034G  "
    expected_10 = "10_1039_c8ra06034g"
    assert doi_to_filename(doi_example_10) == expected_10


# ============================================================================
# Tests for get_body_from_jats_xml
# ============================================================================


def test_get_body_from_jats_xml_standard():
    xml_text = (
        "<article><front><abstract><p>Abstract text</p></abstract></front>"
        '<body><sec sec-type="intro"><title>Introduction</title>'
        '<p id="P1">Most self-assembled materials are disordered.</p></sec></body>'
        "<back><ack><p>Ack text</p></ack></back></article>"
    )
    result = get_body_from_jats_xml(xml_text)
    assert result == "Introduction Most self-assembled materials are disordered."
    assert "Abstract text" not in result
    assert "Ack text" not in result
