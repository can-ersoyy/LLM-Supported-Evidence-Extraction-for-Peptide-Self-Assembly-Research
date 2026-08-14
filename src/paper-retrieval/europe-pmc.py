import requests

from retriever import PaperRetriever

class EuropePMCPaperRetriever(PaperRetriever):
    def __init__(self, base_url, api_key, id_list_path, papers_path):
        super().__init__(base_url, api_key, id_list_path, papers_path)

    def search(self, query, **kwargs):
        pass

    def fetch(self, ids):
        return super().fetch(ids)

    
if __name__ == "__main__":
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/"

