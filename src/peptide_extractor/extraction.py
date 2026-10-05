from peptide_extractor.schemas import PeptideExtractionResult
from pydantic import BaseModel
from typing import List
from peptide_extractor.llm import LLMController
import csv
from pathlib import Path


from peptide_extractor.db.peptide_entry import PeptideEntry

pmid_list = [
    18070345,
    25010703,
    22016623,
    25621167,
    22915849,
    23020140,
    23800286,
    24786493,
    33579941,
    28942918,
]


GOLDEN_DATASET_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "golden_dataset_final.csv"
)


peptide_entries: dict[int, list[PeptideEntry]] = {pmid: [] for pmid in pmid_list}
pmid_lookup = {str(p): p for p in pmid_list}

#with open(GOLDEN_DATASET_CSV_PATH, mode="r", encoding="utf-8") as f:
#    reader = csv.DictReader(f)
#    for row in reader:
#        row_pmid = row.get("pmid", "").strip()
#        if row_pmid in pmid_lookup:
#            entry = PeptideEntry(**row)
#            target_key = pmid_lookup[row_pmid]
#            peptide_entries[target_key].append(entry)
#
#total_row_count = 0
#for i in range(len(pmid_list)):
#    print(f"PMID: {pmid_list[i]}")
#    print(f"Number of peptide entries: {len(peptide_entries[pmid_list[i]])}")
#    total_row_count += len(peptide_entries[pmid_list[i]])
#
#print(f"\nTotal number of peptide entries: {total_row_count}")


#########################################################

EXTRACTION_SYSTEM_PROMPT = """
You will receive the full text of a scientific paper and your goal is to extract experimental peptide self-assembly information according to the given schema. 
There can be several entries in a single paper be careful! 
If a field is not provided give "NP" for that field and write nothing else. 
Don't leave any fields empty.
"""

def run_extraction( paper_content: str, provider: str = "google_genai", model_name: str = "gemma-4-31b-it") -> PeptideExtractionResult:
    model = LLMController(provider=provider, model_name=model_name, structured_output=PeptideExtractionResult, system_prompt=EXTRACTION_SYSTEM_PROMPT)
    res = model.extract_peptides(paper_content)
    return res.entries