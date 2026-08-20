from collections import Counter
from datetime import datetime, timezone

from collections import Counter

import requests


def get_field_value(item, field):
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)

# this version deduplicates lists of object/dicts with a given field
def process_duplicates_by_field(items, field):
    key_counts = Counter(get_field_value(item, field) for item in items)

    unique_items = []
    seen_keys = set()

    for item in items:
        key = get_field_value(item, field)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_items.append(item)

    print(f"--- Duplicate Report (Field: '{field}') ---")
    duplicate_found = False
    for key, count in key_counts.items():
        if count > 1:
            duplicate_found = True
            extra_copies = count - 1
            print(
                f"'{key}' -> {extra_copies} duplicate(s) found ({count} total occurrences)"
            )

    if not duplicate_found:
        print("No duplicates detected in the list.")

    return unique_items

# this version just deduplicates a list with elements of same type (int, str...)
def process_duplicates(items):
        counts = Counter(items)

        unique_items = list(dict.fromkeys(items))

        print("--- Duplicate Report ---")
        duplicate_found = False
        for item, count in counts.items():
            if count > 1:
                duplicate_found = True
                extra_copies = count - 1
                print(
                    f"{item} -> {extra_copies} duplicate(s) found ({count} total occurrences)"
                )

        if not duplicate_found:
            print("No duplicates detected in the list.")

        return unique_items

def get_utc_now():
    return datetime.now(timezone.utc)


peptide_lengths_dict = {
    4: "tetrapeptide",
    5: "pentapeptide",
    6: "hexapeptide",
    7: "heptapeptide",
    8: "octapeptide",
    9: "nonapeptide",
    10: "decapeptide",
    11: "undecapeptide",
    12: "dodecapeptide",
    13: "tridecapeptide",
    14: "tetradecapeptide",
    15: "pentadecapeptide",
    16: "hexadecapeptide",
    17: "heptadecapeptide",
    18: "octadecapeptide",
    19: "nonadecapeptide",
    20: "eicosapeptide"
}

def get_pmcid_from_pmid(api_key, pmid: str) -> str | None:
    """Find corresponding PMCID for a given PMID using ELink."""

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "cmd": "neighbor",
        "retmode": "json",
        "tool": "evidence_extractor",
        "api_key": api_key
    }

    res = requests.get(url, params=params).json()
    try:
        linkset = res["linksets"][0]
        links = linkset["linksetdbs"][0]["links"]
        return links[0]
    except (KeyError, IndexError):
        return None