
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