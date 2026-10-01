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


# 18.08.2026

- When using queries like "tetrapeptide AND self-assembly" we need to think about actually searching for papers that have these words in its own content and not as the title of a reference it uses

## 19.08.2026

Todos:
- Read through papers like waltzdb, sapdb etc. and create a report of how we can do this
- Define the columns to extract according to other research
- If you can define the query 

### 24.08.2026
Golden Dataset Information
(sapdb, peptide miner, cpad 2.0, cross-beta db, waltzdb, amypro)



could we also do somehting like 
sequence feature






##### Completeness Subsets (Not Provided columns count filtered out)

| Step     | Added Required Column              | Total Required Cols | Retained Rows | Unique Seqs | Unique PMIDs | Amyloid | Self-Assembly | Non-Aggregating |
| -------- | ---------------------------------- | ------------------- | ------------- | ----------- | ------------ | ------- | ------------- | --------------- |
| Baseline | None (8 Complete Baseline Columns) | 8                   | 5,259         | 3,050       | 991          | 2,198   | 1,347         | 1,714           |
| Step 01  | start_pos                          | 9                   | 4,356         | 2,252       | 754          | 1,622   | 1,347         | 1,387           |
| Step 02  | end_pos                            | 10                  | 4,356         | 2,252       | 754          | 1,622   | 1,347         | 1,387           |
| Step 03  | pmid                               | 11                  | 3,243         | 1,332       | 754          | 1,208   | 1,347         | 688             |
| Step 04  | method                             | 12                  | 1,996         | 524         | 416          | 219     | 1,314         | 463             |
| Step 05  | detection_method                   | 13                  | 1,995         | 524         | 416          | 219     | 1,313         | 463             |
| Step 06  | solution_system                    | 14                  | 1,219         | 323         | 173          | 192     | 586           | 441             |
| Step 07  | solvent_ratio                      | 15                  | 1,219         | 323         | 173          | 192     | 586           | 441             |
| Step 08  | solvent_env                        | 16                  | 1,219         | 323         | 173          | 192     | 586           | 441             |
| Step 09  | ph                                 | 17                  | 1,215         | 319         | 169          | 188     | 586           | 441             |
| Step 10  | temperature                        | 18                  | 1,198         | 302         | 158          | 171     | 586           | 441             |
| Step 11  | buffer_additives                   | 19                  | 1,179         | 293         | 156          | 171     | 571           | 437             |
| Step 12  | concentration_value                | 20                  | 1,147         | 261         | 132          | 139     | 571           | 437             |
| Step 13  | peptide_concentration              | 21                  | 1,147         | 261         | 132          | 139     | 571           | 437             |
| Step 14  | concentration_unit                 | 22                  | 1,145         | 259         | 130          | 137     | 571           | 437             |
| Step 15  | preparation_protocol               | 23 (All Columns)    | 912           | 159         | 106          | 65      | 439           | 408             |
|          |                                    |                     |               |             |              |         |               |                 |

### 25.08.2026 (deprecated)

#### Peptide Identity & Sequence Annotations

| **Column**         | **SAPDB (HTML) PY**                                                                | **PeptideMiner (CSV) PY**                                                          | **Cross-Beta DB (JSON) PY**                                           | **CPAD 2.0 (Excel) PY**                                                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **`sequence`**     | Extracted from `<td>` index 2; stripped and uppercased.                            | From `sequence` column; if blank, falls back to matched SAPDB sequence.            | Extracted from `AR Sequence`; fallback to `AR containing sequence`.   | Extracted from `Peptide` (_aggregating peptides_), `Peptide/ Protein Sequence` (_structure_), or `Experimental Aggregating Region` (_APR_). |
| **`length`**       | Derived by counting uppercase alpha characters: `len(re.sub(r'[^A-Z]', '', seq))`. | Derived by counting uppercase alpha characters: `len(re.sub(r'[^A-Z]', '', seq))`. | Read from `AR Sequence Length`; fallback to string length `len(seq)`. | Read from `Length` or `APR Length`; fallback to `len(seq)`.                                                                                 |
| **`peptide_name`** | Extracted from `<td>` index 1 (`pep_name`); fallback to `sequence`.                | Inherited from SAPDB `pep_name`; fallback to `"{type} ({seq})"` or raw sequence.   | Extracted from `Protein Name`; fallback to `"Not Provided"`.          | Extracted from `Protein Name` or `Source Protein`; fallback to `sequence`.                                                                  |

### 2. Categorization & Terminals

|**Column**|**SAPDB (HTML) PY**|**PeptideMiner (CSV) PY**|**Cross-Beta DB (JSON) PY**|**CPAD 2.0 (Excel) PY**|
|---|---|---|---|---|
|**`broad_category`**|Mapped via `classify_broad_category()`: checks for keywords like _not aggregating_ (Non-amyloid), _amyloid/prion_ (Amyloid), or defaults to _Self-Assembly_.|Mapped via `classify_broad_category()` applied to the mapped category.|Mapped via `classify_broad_category()` applied to `LABEL (Role)`.|Mapped via `classify_broad_category()` applied to `Classification`, `amyloid/Non-amyloid`, or `Category`.|
|**`category`**|Extracted from `<td>` index 3; defaults to `"Self-Assembly"` if empty.|Mapped from integer code `0–8` (e.g., `0: Not aggregating`, `7: Hydrogel`, `8: Fiber`); fallback to SAPDB category.|Combines `LABEL` and `Role`: `"{LABEL} ({Role})"` if role is valid, else `LABEL`.|Extracted from `Classification`, `amyloid/Non-amyloid`, or `Category` (with `"Functional Prion"` flag).|
|**`n_terminal`**|Extracted from `<td>` index 5 (`n_mod`); defaults to `"Free"` if empty.|From `N-terminal` column; fallback to SAPDB `n_mod`; defaults to `"Free"`.|Sliced from sequence: `seq[0]` if present, else `"Free"`.|Sliced from sequence: `seq[0]` if present, else `"Free"`.|
|**`c_terminal`**|Extracted from `<td>` index 6 (`c_mod`); defaults to `"Free"` if empty.|From `C-terminal` column; fallback to SAPDB `c_mod`; defaults to `"Free"`.|Sliced from sequence: `seq[-1]` if present, else `"Free"`.|Sliced from sequence: `seq[-1]` if present, else `"Free"`.|
|**`start_pos`**|Set to `"Not Applicable (Synthetic)"`.|Set to `"Not Applicable (Synthetic)"`.|Parsed from `Experimental Amyloid Region` via regex `(\d+)-(\d+)`.|Parsed from `Position` (_aggregating_) or `Sequence Position` (_APR_); else `"Not Provided"`.|
|**`end_pos`**|Set to `"Not Applicable (Synthetic)"`.|Set to `"Not Applicable (Synthetic)"`.|Parsed from `Experimental Amyloid Region` via regex `(\d+)-(\d+)`.|Parsed from `Position` (_aggregating_) or `Sequence Position` (_APR_); else `"Not Provided"`.|

### 3. Physical & Environmental Reaction Conditions

|**Column**|**SAPDB (HTML) PY**|**PeptideMiner (CSV) PY**|**Cross-Beta DB (JSON) PY**|**CPAD 2.0 (Excel) PY**|
|---|---|---|---|---|
|**`temperature`**|`"Not Provided"`.|From `temperature`; normalized via `format_temperature()` (appends `" °C"` to bare floats).|From `Temperature (°C)`; normalized via `format_temperature()`.|Enriched from `aggregation kinetics.xlsx` (`Temperature`); else `"Not Provided"`.|
|**`ph`**|`"Not Provided"`.|Cleaned float string from `PH`; defaults to `"Not Provided"`.|Cleaned string from `pH (window/optimum)...`; else `"Not Provided"`.|Enriched from `aggregation kinetics.xlsx` (`pH`); else `"Not Provided"`.|
|**`concentration_value`**|`"Not Provided"`.|Extracted numeric/range string from `concentration(mg/ml)`.|Extracted numeric/range string from `(Minimum/optimum) concentration...`.|Enriched from `Protein Concentration` via `split_concentration()`.|
|**`concentration_unit`**|`"Not Provided"`.|Extracted unit string (defaults to `"mg/ml"` if bare number).|Extracted unit string (`µM`, `mg/mL`, `mM`).|Extracted unit string from `Protein Concentration`.|
|**`peptide_concentration`**|`"Not Provided"`.|Full combined string: `"{val} {unit}"`.|Full combined string: `"{val} {unit}"`.|Full combined string from kinetics enrichment.|

### 4. Solvent Environment & Method Decompositions

| **Column**                 | **SAPDB (HTML) PY**                          | **PeptideMiner (CSV) PY**                                                                        | **Cross-Beta DB (JSON) PY**                                                  | **CPAD 2.0 (Excel) PY**                                                                                        |
| -------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **`solution_system`**      | `"Not Provided"`.                            | Cleaned string from `solution` column.                                                           | `"Aqueous Buffer"` if `Specific buffer` exists, else `"Not Provided"`.       | Read from `Buffer Name` in kinetics; fallback to `"Aqueous Buffer"`.                                           |
| **`solvent_ratio`**        | `"Not Provided"`.                            | Read from `volume ratio`; if `"0"`, `"100"`, or empty, sets `"Not Applicable (Single Solvent)"`. | `"Not Applicable (Single Solvent)"` if buffer exists, else `"Not Provided"`. | `"Not Applicable (Single Solvent)"` if buffer exists, else `"Not Provided"`.                                   |
| **`buffer_additives`**     | `"Not Provided"`.                            | Read from `solvent` column.                                                                      | Read from `Specific buffer` field.                                           | Formatted concatenation of `Buffer Concentration` + `Buffer Name` + `Ion Concentration` + `Ion` + `Additives`. |
| **`solvent_env`**          | `"Not Provided"`.                            | Joined non-empty components: `"{solution} ({ratio}); {buffer_additives}"`.                       | Read directly from `Specific buffer`.                                        | Sourced from the formatted kinetics buffer string.                                                             |
| **`detection_method`**     | Extracted from `<td>` index 4 (`technique`). | Inherited from matched SAPDB `technique`.                                                        | Read from `Method used`.                                                     | Read from `Method` in _amyloid structure.xlsx_ or `Method`/`Measure` in kinetics.                              |
| **`preparation_protocol`** | `"Not Provided"`.                            | Extracted from `method` column in PeptideMiner.                                                  | Read from `Protein obtention method`.                                        | `"Not Provided"`.                                                                                              |
| **`method`**               | Set to `detection_method`.                   | Concatenation of `"{detection_method}; {preparation_protocol}"`.                                 | Set to `detection_method` (`Method used`).                                   | Set to `detection_method`.                                                                                     |

### 5. Provenance & Biological Metadata

| **Column**      | **SAPDB (HTML) PY**                                    | **PeptideMiner (CSV) PY**                        | **Cross-Beta DB (JSON) PY**                                                                          | **CPAD 2.0 (Excel) PY**                                                                 |
| --------------- | ------------------------------------------------------ | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **`pmid`**      | Extracted from `<td>` index 7 via regex `\b\d{4,9}\b`. | Inherited from SAPDB PMIDs linked by `sapdb_id`. | Extracted from PubMed URLs in `Reference link / DOI` or resolved from DOIs via Europe PMC/NCBI APIs. | Extracted from `PMID` or `Pubmed References` columns via regex `\b\d{4,9}\b`.           |
| **`databases`** | `"SAPDB"`.                                             | `"SAPDB; PeptideMiner"`.                         | `"Cross-Beta"`.                                                                                      | Sourced from workbook annotations (`"CPAD-2.0"`, `"WaltzDB"`, `"AmyLoad"`, `"AmyPro"`). |

### 30.09.2026
It is currently possible to do search queries using ncbi and scopus api.. these return redundant papers for which we have doi normalization/deduplication methods.
Scopus doesn't return abstract of papers to we use semanticscholar / open alex to get these remaining abstracts
We can run a relevance check pipeline to decide if the paper is relevant (retrieve fulltext- currently this is not used as the paper amount we can receive is still low in comparison to found papers via queries)
We can receive paper fulltexts via pmcid