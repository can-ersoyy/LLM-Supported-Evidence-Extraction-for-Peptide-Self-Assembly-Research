from collections import Counter
from typing import List


class PaperRetriever():

    def __init__(self, base_url: str, api_key: str, id_list_path: str, papers_path: str):
        self.base_url = base_url
        self.api_key = api_key
        self.id_list_path = id_list_path
        self.papers_path = papers_path
        self.peptide_length_dict = {
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
        self.peptide_id_dict = {
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10:[],
    11:[],
    12:[],
    13:[],
    14:[],
    15:[],
    16:[],
    17:[],
    18:[],
    19:[],
    20:[]
}

    def search(self, query: str, **kwargs):
        pass

    def fetch(self, ids: str | List[str]):
        pass

    def process_duplicates(self, items):
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