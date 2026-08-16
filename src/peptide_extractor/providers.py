import requests
import os
from dotenv import load_dotenv, find_dotenv
import json
import time

from collections import Counter
from typing import List

from peptide_extractor.db.paper_metadata import PaperMetadata
from peptide_extractor.schemas import RawProviderRecord
from peptide_extractor.utils import process_duplicates, peptide_lengths_dict



class BaseProvider():

    def __init__(self, api_key: str | None, batch_size: int):
        self.api_key = api_key
        self.batch_size = batch_size
        
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

        results: list[RawProviderRecord] = []

        params = {
        "db": "pmc",
        "retmode": "json",
        "api_key": self.api_key
        }

        for key, name in peptide_lengths_dict.items():

            query = f"({name} AND self-assembly) NOT review[publication type] AND open access[filter]"
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

        
            params = {
                        "db":"pmc",
                        "retmode": "json",
                        "api_key": self.api_key
                    }


            
            for i in range(0, len(self.peptide_id_dict[key]), self.batch_size):
                params["id"] = ",".join(str(x) for x in self.peptide_id_dict[key][i: i + self.batch_size] if x is not None)

                sum_res = requests.get(f"{self.BASE_URL}esummary.fcgi", params=params)
                time.sleep(0.2)
                result_dict = sum_res.json().get("result", {})
                uids = result_dict["uids"]

                for uid in uids:

                    paper = result_dict[f"{uid}"]

                    record = RawProviderRecord(
                        provider="pmc",
                        provider_id=uid,
                        search_query=query,
                        raw_data=paper
                    )
                    
                    results.append(record)

            

        return results


    def extract_match_count(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["count"]

    def extract_id_list(self, response):
        obj = json.loads(response)
        return obj["esearchresult"]["idlist"]


class EuropePMCProvider(BaseProvider):

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/"

    def __init__(self, api_key: str | None = None, batch_size: int = 100):
        super().__init__(api_key, batch_size)

    def search(self):

        results: list[RawProviderRecord] = []

        params = {
            "format": "json",
            "pageSize": self.batch_size,
            "resultType": "lite"
        }

        for key, name in peptide_lengths_dict.items():
            cursor = "*"
            query = f'({name} AND "self-assembly" AND (HAS_FT:Y)) NOT (PUB_TYPE:"Review" OR PUB_TYPE:"review-article" OR PUB_TYPE:"Systematic Review")'
            params["query"] = query

            while cursor:
                params["cursorMark"] = cursor
                response = requests.get(f"{self.BASE_URL}search", params=params)
                response.raise_for_status()
                time.sleep(0.2)

                data = response.json()
                items = data.get("resultList", {}).get("result", [])
                next_cursor = data.get("nextCursorMark")

                if not items:
                    break

                for item in items:
                    provider_id = str(item.get("id"))
                    self.peptide_id_dict[key].append(provider_id)

                    record = RawProviderRecord(
                        provider="europe_pmc",
                        provider_id=provider_id,
                        search_query=query,
                        raw_data=item
                    )
                    results.append(record)

                if not next_cursor or next_cursor == cursor:
                    break

                cursor = next_cursor

        return results


if __name__ == "__main__":

    find_dotenv("..")
    load_dotenv()

    api_key = os.getenv("NCBI_API_KEY")

    pmc_provider = PMCProvider(api_key, batch_size=150)
    # items = pmc_provider.search()
    # print(items)


    europe_pmc_provider = EuropePMCProvider(api_key, batch_size=150)
    items = europe_pmc_provider.search()
    print(items)