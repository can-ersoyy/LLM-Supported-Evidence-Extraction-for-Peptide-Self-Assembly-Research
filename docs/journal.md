# 13.08.2026
- Checked https://pmc.ncbi.nlm.nih.gov/ (PubMed/PMC) apis. It provides an api to retrieve PMCids of papers
with a given search query (e.g. tetrapeptide AND self-assembly).
- pmc_combined_id_lists (includes review papers) / pmc_non_review_id_lists (includes only non-review papers) -> 
return a dict of {peptide_length:\[id_list\]}
- Initially used the api to query for each peptide length 4 to 20 separately with self-assembly... Then added 
NOT review[publication type] to remove review papers. Dedup was 6200 then went down to 1710 ids.
- It is recommended to use xml instead of pdf since bulk api allows for xml download systematically and not for pdfs
- There are also potential candidates for sources: EuropePMC/OpenAlex and Unpaywall might be used to find open access
locations for paid papers later on

Open Questions:
- How to parse XMLs as full text? 