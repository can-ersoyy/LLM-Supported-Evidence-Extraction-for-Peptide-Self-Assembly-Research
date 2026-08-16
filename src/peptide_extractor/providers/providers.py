import requests
import os
from dotenv import load_dotenv, find_dotenv
import json
import time
import pickle

from collections import Counter
from typing import List



class BaseProvider():

    def __init__(self, api_key: str, batch_size: int):
        self.api_key = api_key
        self.batch_size = batch_size
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


class PMCProvider(BaseProvider):

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, api_key, batch_size):
        super().__init__(api_key, batch_size)

    def search(self):

        params = {
        "db": "pmc",
        "retmode": "json",
        "api_key": self.api_key
        }

        for key, name in self.peptide_length_dict.items():

            query = f"({name} AND self-assembly) NOT review[publication type]"
            params["term"] = query
            params["retstart"] = 0
            params["retmax"] = 1
            response = requests.get(f"{self.BASE_URL}esearch.fcgi", params=params) ## ini
            matches = int(self.extract_match_count(response.content))
            time.sleep(0.1)

            for index_counter in range(0, matches, self.batch_size):

                params["retstart"] = index_counter
                params["retmax"] = self.batch_size

                response = requests.get(f"{self.BASE_URL}esearch.fcgi", params=params)
                time.sleep(0.2)
                id_list = self.extract_id_list(response.content)
                print(id_list)
                self.peptide_id_dict[key].extend(id_list)

        combined_list = []
        for item in self.peptide_id_dict.values():
            combined_list.extend(item)

        pmc_id_list = self.process_duplicates(combined_list)

        params = {
                    "db":"pmc",
                    "retmode": "json",
                    "api_key": self.api_key
                }

        for i in range(int(len(pmc_id_list)/self.batch_size)):
            params["id"] = ",".join(str(x) for x in pmc_id_list[i:i + self.batch_size] if x is not None)

            sum_res = requests.get(f"{self.BASE_URL}esummary.fcgi", params=params)
            result_dict = sum_res.json().get("result", {})

            results = []
            for pmcid in pmc_id_list:
                item = result_dict.get(pmcid, {})
                doi = None
                for article_id in item.get("articleids", []):
                    if article_id.get("idtype") == "doi":
                        doi = article_id.get("value")
                        break

                meta = Paper(
                    title=item.get("title", ""),
                    doi=doi,
                    provider_id=pmcid,
                    provider="pmc",
                )

                results.append(meta)

        return results


    def extract_match_count(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["count"]

    def extract_id_list(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["idlist"]

if __name__ == "__main__":

    find_dotenv("..")
    load_dotenv()

    api_key = os.getenv("NCBI_API_KEY")

    pmc_provider = PMCProvider(api_key, batch_size=150)
    items = pmc_provider.search()
    print(items)
    

class EuropePMCProvier(BaseProvider):

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/"

    def __init__(self, api_key, batch_size):
        super().__init__(api_key, batch_size)

    def search(self):

        params = {
            "format": "json",
            "pageSize": self.batch_size,
            "resultType": "lite"
        }
        
        for key, name in self.peptide_length_dict.items():
                    cursor = None
                    query = f"({name} AND self-assembly) NOT review[publication type]"
                    params["query"] = query
                    response = requests.get(f"{self.BASE_URL}search", params=params) # initial request to receive a cursor
                    cursor = response.json().get("nextCursorMark")


        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("resultList", {}).get("result", []):
            meta = PaperMetadata(
                title=item.get("title", ""),
                doi=item.get("doi"),
                pmid=item.get("pmid"),
                pmcid=item.get("pmcid"),
                source_api="EuropePMC",
                is_open_access=(item.get("isOpenAccess") == "Y"),
                raw_metadata=item
            )
            results.append(meta)
        return results