from peptide_extractor.db.paper_metadata import PaperMetadata
from peptide_extractor.schemas import RawProviderRecord

def parse_raw_pmc(raw_record: RawProviderRecord):
    raw_json = raw_record.raw_data

    id_map = {
        item.get("idtype"): item.get("value") 
        for item in raw_json.get("articleids", [])
        if "idtype" in item
    }

    doi = id_map.get("doi")
    pmid = id_map.get("pmid")
    pmcid = id_map.get("pmcid")

    title = raw_json["title"]
    is_open_access = True # for now we just filter out non open access docs from ncbi

    pass

def parse_raw_europe_pmc(raw_record: RawProviderRecord):
    raw_json = raw_record.raw_data

    #TODO

    pass