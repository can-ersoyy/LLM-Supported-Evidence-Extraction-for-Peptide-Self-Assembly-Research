## Project Plan Document


### General Steps and Design Questions:
1) Data acquisition: 
- Where to get the data from? How to get it (rate limits, access etc.). 
- How much data (paper count...)? 
- How to check for quality of the papers before extraction?
2) Preprocessing: 
- Do we need to preprocess papers before passing them down (higher context lenght in comparison to a few years ago)? 
- How to process the papers before passing them down to the LLM for extraction? 
- Should a programmatic approach be used or can LLMs be used for this part as well, if so how? Which parts of papers would work? 
- What happens to the data lost with this approach?
3) Data Extraction: 
- Do we pass the text directly to the LLM? 
- How to get structured outputs? 
- Which API/provider to use? How to handle costs? 
- How to handle costs during testing?
4) Data Validation: 
- How to validate the data extracted by LLMs for correctness/usability? 
- How to test accuracy/performance of different approaches? 

### Available Tools/APIs
1) Data Acquisition
- Open source papers
    + Unpaywall API
    + OpenAlex API
    + Semantic Scholar API
    + PubMed Database
    + PMC (Europe) Database/API

- Licensed (access via TUM library) -> this would probably not work for systematic downloads

2) Preprocessing
...

3) Data Extraction
- LLM Provider Gateway (since it is not determined which provider can be used or if a local approach is possible)
    + LiteLLM
- Structured Outputs
    + Pydantic/Instructor
4) Data Validation
...


### Appendix

#### PeptideMiner columns
```csv 
sequence	N-terminal	C-terminal	non-terminal	peptide/conjugate/mixture	conjugate partner	solvent	method	solution	volume ratio	concentration(mg/ml)	PH	temperature	heating/cooling	incubation period (min)	type of self-assembly[none, ribbon, nanosphere, organogel, tube, particle, vesicle, hydrogel, fiber]	size of structure	linear/cyclic
```
- Should these be expanded/compressed?

