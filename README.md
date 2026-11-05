# Risk Of Bias and Quality Assessment

Automated risk of bias and quality assessment of AI-based prediction model studies using Large Language Models (LLMs) in a local environment.

This repository contains code for performing PROBAST+AI assessments on scientific publications using retrieval-augmented generation (RAG) and local LLMs through Ollama. The pipeline extracts evidence from PDF manuscripts, retrieves relevant text chunks for each PROBAST+AI signalling question, generates assessments using an LLM, and aggregates multiple runs into consensus assessments. The code was adapted to also perform an adjusted QUADAS-2 assessment.

## Overview
PROBAST+AI (2025) is an extension of the Prediction model Risk Of Bias ASsessment Tool (PROBAST) designed for evaluating AI-based prediction models.
QUADAS-2 (2011) is an updated version of the original QUADAS (2003) risk of bias and applicability assessment concerning diagnostic accuracy studies. It stands for Quality Assessment of Diagnostic Accuracy Studies, version 2. 

This implementation:
1. Parses full-text PDF articles.
2. Chunks and indexes manuscript text using BGE-M3 embeddings.
3. Retrieves relevant manuscript sections for each PROBAST+AI (or QUADAS-2) signaling question.
4. Uses a local LLM (tested with Qwen 3.5 27B) to answer each question.
5. Performs repeated independent assessment runs.
6. Aggregates multiple runs into a consensus assessment.
7. Quantifies performance (range 0-1) across runs per question per paper using:
    - Chunk Jaccard Similarity score
    - Majority agreement for one question 
    - Ordinal agreement for one question
    - Semantic justification similarity (how much do the justifications per answer overlap for one question)
   The metrics are all also averaged over all questions, papers and runs.

## Main scripts
| Script   | Purpose |
|----------|----------|
| PROBAST_AI.py   | Runs the PROBAST+AI assessment pipeline   |
| QUADAS-2.py   | Runs the PROBAST+AI assessment pipeline   |
| LLM_output_validation.py   | Aggregates multiple runs into consensus outputs  |
| LLM_human_consensus.py   | calculate agreement metrics between LLM and Human  |
| job.slurm | SLURM job script for high-performance cluster execution  |
| example_consensus.xlsx | example excel file for human-LLM consensus calculations  |


## Requirements
free RAM >= 23GB

python >= 3.10

CUDA >= 11.8
Also consult the requirements.txt file

## Input Data
Place all PDF files to be assessed in the input directory:
```text
data/
├── PROBAST_AIguidance.pdf
├── paper_1.pdf
├── paper_2.pdf
...
└── paper_n.pdf
```
The PROBAST+AI guidance document is used as retrieval context for the assessment process. To add supplemental files, merge the manuscript pdf with the supplemental files prior to running the script to create one pdf per paper. The guidance document can be replaced with any other document necessary for guidance i.e. a QUADAS-2 guidance document.

## Metrics
### Chunk Jaccard Similarity score
Calculates the mean Jaccard score based on the index of the selected chunks for each question over all runs per paper. The idea is that the higher the chunk selection similarity across runs, the more consistent the LLM output will become.
### Majority agreement score
Calculates per paper the answer consistency over all runs per question. E.g. if the most frequent answer was 3 times 'Yes' to a specific question over 5 runs, the agreement score is 0.6. The average agreement score is agreement score over all questions per paper.
### Ordinal Agreement
The ordinal agreement score tries to perform a correction on the earlier agreement score by penalising more distinct answers ('Yes' vs 'No') more heavily than similar answers ('Yes' vs. 'Probably Yes') with respect to the majority score. The distance between 'Not Applicable' and 'No Information' and all other labels is always 1.0 similar to 'No' versus 'Yes'.
Uses the scale:
1. No = 0.00
2. Probably No = 0.33
3. Probably Yes = 0.67
4. Yes = 1.00
### Semantic similarity of justification
Optional embedding-based similarity of answer justifications using BGE-M3 embeddings. This is another way to score the consistency of the LLM output across multiple runs. 
### Weighted lineair Cohen's kappa
Uses the scale:
1. No = 0.00
2. Probably No = 0.33
3. Probably Yes = 0.67
4. Yes = 1.00
The distance between Not applicable, no information and other signalling question labels is always 1.0. The distance between low, high and unclear is also one.

## Citation
If you use this work, please cite:
```bibtex
@software{PROBAST-AI_LLM,
  author = {Ribbens, Vincent},
  title = {Automated PROBAST+AI assessment of AI-based prediction model studies using Large Language Models (LLMs)},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/vribbens/PROBAST-AI_LLM}
}
```
or

Ribbens, V.J. (2026). *Automated PROBAST+AI assessment of AI-based prediction model studies using locally run Large Language Models (LLMs)* [Software]. GitHub. https://github.com/vribbens/PROBAST-AI_LLM
