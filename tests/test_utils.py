import pytest

from peptide_extractor.utils import (
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