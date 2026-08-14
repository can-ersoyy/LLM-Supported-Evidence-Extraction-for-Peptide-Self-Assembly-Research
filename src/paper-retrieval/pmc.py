import requests
import os
from dotenv import load_dotenv, find_dotenv
import json
import time
import pickle
from retriever import PaperRetriever

class PMCPaperRetriever(PaperRetriever):

    def __init__(self, base_url, api_key, id_list_path, papers_path):
        super().__init__(base_url, api_key, id_list_path, papers_path)

    def search(self, query: str, start_index: int, max_results: int, db: str = "pmc"):
        params = {
                "db": db,
                "term": query,
                "retmode": "json",
                "retstart": start_index,
                "retmax": max_results,
                "api_key": self.api_key
            }
        resp = requests.get(f"{BASE_URL}esearch.fcgi", params=params).json()
        return json.dumps(resp, indent=4)

    def fetch(self, ids):
        if isinstance(ids, list):
            ids_str = ",".join(str(i) for i in ids)
        else:
            ids_str = str(ids)
        params = {
            "db": "pmc",
            "id": ids_str,
            "rettype": "xml",
            "api_key": self.api_key
        }
    
        resp = requests.get(f"{BASE_URL}efetch.fcgi", params=params, timeout=30)
        if resp.status_code != 200:
            exit(-1)
        return resp.content

    def extract_match_count(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["count"]

    def extract_id_list(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["idlist"]

    def save_id_list(self):
        with open(self.id_list_path, "wb") as f:
            for key, name in self.peptide_length_dict.items():
                query = f"({name} AND self-assembly) NOT review[publication type]"
                res = self.search(query, start_index = 0, max_results=1) # initial request to receive match count per peptide length
                time.sleep(1)
                matches = int(self.extract_match_count(res))
                for index_counter in range(0, matches, 1500):
                    res = self.search(query, start_index=index_counter, max_results=1500)
                    time.sleep(2)
                    id_list = self.extract_id_list(res)
                    print(id_list)
                    self.peptide_id_dict[key].extend(id_list)

            pickle.dump(self.peptide_id_dict, f)

    def read_id_list(self):
        with open(self.id_list_path, "rb") as f:
            peptide_id_dict = pickle.load(f)

            combined = []
            for _,val in peptide_id_dict.items():
                combined.extend(val)

            return combined

    def save_papers(self, dedup):
        for i in range(0, len(dedup), 10):
            print(f"Save Done {((i / len(dedup)) * 100):.1f}%")
            with open(f"{self.papers_path}/{int(i/10)}.xml", "wb") as f:
                res = self.fetch(dedup[i:i+10])
                f.write(res)
    
            time.sleep(0.2)

if __name__ == "__main__":
    from pathlib import Path

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    find_dotenv("../../.env")
    load_dotenv()

    API_KEY = os.getenv("NCBI_API_KEY")

    Path("./pmc/pmc_ids").mkdir(parents=True, exist_ok=True)
    Path("./pmc/pmc_papers").mkdir(parents=True, exist_ok=True)

    pmc_retriever = PMCPaperRetriever(base_url=BASE_URL, api_key=API_KEY, id_list_path="pmc/pmc_ids/pmc_non_review_ids_lists", papers_path="pmc/pmc_papers")
    #pmc_retriever.save_id_list()
    #combined_ids = pmc_retriever.read_id_list()
    #dedup_ids = pmc_retriever.process_duplicates(combined_ids)
    #pmc_retriever.save_papers(dedup_ids)
    

    #import xml.etree.ElementTree as ET
#
    #counter = 0
    #for i in range(len(dedup_ids)//10):
    #    tree = ET.parse(f"pmc/pmc_papers/{i}.xml")    
    #    root = tree.getroot()
    #    for article in root.findall('article'):
    #        counter += 1
#
    #print(counter)
