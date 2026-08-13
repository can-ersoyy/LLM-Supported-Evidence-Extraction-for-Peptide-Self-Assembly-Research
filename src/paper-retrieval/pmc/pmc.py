import requests
import os
from dotenv import load_dotenv, find_dotenv
import json
import time
import pickle
from collections import Counter


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

peptide_mer_dict = {
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

peptide_id_dict = {
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


find_dotenv("../../.env")
load_dotenv()


def extract_match_count(response):
    obj = json.loads(response)
    return obj["esearchresult"]["count"]

def extract_id_list(response):
    obj = json.loads(response)
    return obj["esearchresult"]["idlist"]

def search_pmc(query, max_results=5, start_index=0, db="pmc"):
    params = {
        "db": db,
        "term": query,
        "retmode": "json",
        "retstart": start_index,
        "retmax": max_results,
        "api_key": os.getenv("NCBI_API_KEY")
    }
    resp = requests.get(f"{BASE_URL}esearch.fcgi", params=params).json()
    print(resp)
    return json.dumps(resp, indent=4)


#with open("id_lists", "wb") as f:
#    for key,name in peptide_mer_dict.items():
#        query = f"({name} AND self-assembly) NOT review[publication type]"
#        res = search_pmc(query, max_results=1) # initial request to receive match count per peptide length
#        time.sleep(1)
#        matches = int(extract_match_count(res))
#        for index_counter in range(0, matches, 1500):
#            res = search_pmc(query, start_index=index_counter, max_results=1500)
#            time.sleep(2)
#            id_list = extract_id_list(res)
#            print(id_list)
#            peptide_id_dict[key].extend(id_list)
#
#
#    pickle.dump(peptide_id_dict, f)
#
#f.close()    

peptide_id_dict = {}
with open("pmc_ids/pmc_non_review_id_lists", "rb") as f:
    peptide_id_dict = pickle.load(f)


combined = []
for key,val in peptide_id_dict.items():
    combined.extend(val)

print(len(combined))



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


#dedup = process_duplicates(combined)
#print(len(dedup))


def fetch_full_text(pmc_ids):
    if isinstance(pmc_ids, list):
        ids_str = ",".join(str(i) for i in pmc_ids)
    else:
        ids_str = str(pmc_ids)
    params = {
        "db": "pmc",
        "id": ids_str,
        "rettype": "xml",
        "api_key": os.getenv("NCBI_API_KEY")
    }

    resp = requests.get(f"{BASE_URL}efetch.fcgi", params=params, timeout=30)
    if resp.status_code != 200:
        exit(-1)
    return resp.content


#for i in range(0, len(dedup), 5):
#    with open(f"pmc_papers/{int(i/5)}.xml", "wb") as f:
#        res = fetch_full_text(dedup[i:i+5])
#        f.write(res)
#    
#    time.sleep(0.2)
