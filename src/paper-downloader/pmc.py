import requests
import os
from dotenv import load_dotenv
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


with open("id_lists", "wb") as f:
    for key,name in peptide_mer_dict.items():
        query = f"({name} AND self-assembly) NOT review[publication type]"
        res = search_pmc(query, max_results=1) # initial request to receive match count per peptide length
        time.sleep(1)
        matches = int(extract_match_count(res))
        for index_counter in range(0, matches, 1500):
            res = search_pmc(query, start_index=index_counter, max_results=1500)
            time.sleep(2)
            id_list = extract_id_list(res)
            print(id_list)
            peptide_id_dict[key].extend(id_list)


    pickle.dump(peptide_id_dict, f)

f.close()    



#def fetch_full_text(pmc_id):
#    params = {
#        "db": "pmc",
#        "id": pmc_id,
#        "rettype": "xml",
#        "api_key": os.getenv("NCBI_API_KEY")
#    }
#    resp = requests.get(f"{BASE_URL}efetch.fcgi", params=params)
#    
#    # Parse XML paragraphs
#    root = ET.fromstring(resp.content)
#    paragraphs = ["".join(p.itertext()).strip() for p in root.findall(".//body//p")]
#    return "\n\n".join(p for p in paragraphs if p)
#