
### Preprocessing

In peptide miner they pick sections like: "Methods", "Experimental Conditions", "Materials".. They also check for paragraphs that include keywords from a predefined set of words.
Two things to consider here:
1) Context lenght was 4096 for the llm they used (not an issue anymore)
2) Increased performance as the token count decreases (is it really? - just assumption)

XML and PDF would be different. Since we have XML files only for now we can just pass the whole body tag to LLM (?)... How to measure the performance here? Use golden dataset papers (we have around ~60)... We can test it on 10 papers from peptideminer just to be sure its working....