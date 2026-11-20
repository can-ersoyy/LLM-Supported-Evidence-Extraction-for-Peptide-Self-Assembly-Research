
We would like to evaluate extractions without any ground truth.. so a comparison between golden dataset entries is not enough



### Golden Dataset Evaluations



### General Evaluations

Deterministic checks
- Search for values in fulltext: This will probably work work some fields but not all. It should be case-insensitive and work with normalizing functions during extraction

LLM-supported
- Another LLM can check fields like `evidence_span` or `preparation_protocol` and give an explanation/score to the extraction and also check for soundness of information (how do we validate the second llm in this case?)

We could create a rubric that gives points for field inclusion (different fields have different importance -> different score per field)
For fields that don't support text search we can use an LLM to check the provided field (compare with evidence span) and give a point
Sum all points
The sum of all points are used to categorize papers into extractions into three (weak, ok, strong)
weak: extraction is not usable for downstream tasks (<70)
ok: human check might be needed (70-80)
strong: extraction can directly be used for tasks (> 80)


16 times 5 = 80
16 (
1

)

llm pointing (per field);
existence in evidence - 2 points
soundness - 3 points


total 16 fields, 5 max -> 80 points


over 65 - strong
50 to 65 - human check
under 50 - weak

You are an llm judge that evaluates given peptide self-assembly experiment conditions extracted from scientific papers.
You will be provided with extraction data for a sequence alongside an evidence_span object that includes
evidence for the extracted fields retrieved from the original paper
Your task is to evaluate the extraction, giving points per field according to the following structure

There are two evaluation metrics per field: 
1- Value matching provided evidence: For this you can give max 2 points
2- Value making scientific sense: For this you can give max 3 points

You will give points per field but some fields might be related to each other, therefore you can use fields together when
evaluation a single one.



### New Idea

Instead of a single score, we can have three different scores per extraction

Groundness - What fraction of non-empty fields actually have an evidence that exists in the paper (fuzzy search)
Faithfullness - For grounded fields, what fraction of these are actually in the evidence
Completeness - How many of the fields are non-empty