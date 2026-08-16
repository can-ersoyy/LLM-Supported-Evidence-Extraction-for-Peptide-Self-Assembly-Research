from collections import Counter
from datetime import datetime, timezone

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