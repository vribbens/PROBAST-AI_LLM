# PROBAST+AI

Automated PROBAST+AI assessment of AI-based prediction model studies using Large Language Models (LLMs).

This repository contains code for performing PROBAST+AI assessments on scientific publications using retrieval-augmented generation (RAG) and local LLMs through Ollama. The pipeline extracts evidence from PDF manuscripts, retrieves relevant text chunks for each PROBAST+AI signalling question, generates assessments using an LLM, and aggregates multiple runs into consensus assessments.

## Overview
PROBAST+AI is an extension of the Prediction model Risk Of Bias ASsessment Tool (PROBAST) designed for evaluating AI-based prediction models.

This implementation:
1. Parses full-text PDF articles.
2. Chunks and indexes manuscript text using BGE-M3 embeddings.
3. Retrieves relevant manuscript sections for each PROBAST+AI signaling question.
4. Uses a local LLM (tested with Qwen 3.5 27B) to answer each question.
5. Performs repeated independent assessment runs.
6. Aggregates multiple runs into a consensus assessment.
7. Quantifies agreement between runs using:
    - Chunk Jaccard Similarity score
    - Average agreement
    - Average ordinal agreement
    - Semantic justification similarity

## Main scripts
| Script   | Purpose |
|----------|----------|
| PROBAST_AI.py   | Runs the PROBAST+AI assessment pipeline   |
| LLM_output_validation.py   | Aggregates multiple runs into consensus outputs   |
| job.slurm | SLURM job script for HPC execution |

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
The PROBAST+AI guidance document is used as retrieval context for the assessment process. To add supplemental files, merge the manuscript pdf with the supplemental files prior to running the script to create one pdf per paper.

## Agreement metrics
### Chunk Jaccard Similarity score
Calculates the mean Jaccard score based on the index of the selected chunks for each question over all runs per paper. The idea is that the higher the chunk selection similarity across runs, the more consistent the LLM output will become.
### Agreement score
Calculates per paper the answer consistency over all runs per question. E.g. if the most frequent answer was 3 times 'Yes' to a specific question over 5 runs, the agreement score is 0.6. The average agreement score is agreement score over all questions per paper.
### Ordinal Agreement
The ordinal agreement score tries to perform a correction on the earlier agreement score by penalising more distinct answers ('Yes' vs 'No') more heavily than similar answers ('Yes' vs. 'Probably Yes'). It only works for 'SUBSTANTIVE' answers, e.g. we chose to not put 'No Info' and 'Not applicable' in the ordinal scale. If these answers are present, the ordinal agreement cant be calculated for that specific question. 
Uses the scale:
1. No = 0.00
2. Probably No = 0.33
3. Probably Yes = 0.67
4. Yes = 1.00
### Semantic similarity of justification
Optional embedding-based similarity of answer justifications using BGE-M3 embeddings. This is another way to score the consistency of the LLM output across multiple runs. 

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
