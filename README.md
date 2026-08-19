# PROBAST+AI

Automated PROBAST+AI assessment of AI-based prediction model studies using Large Language Models (LLMs).

This repository contains code for performing PROBAST+AI assessments on scientific publications using retrieval-augmented generation (RAG) and local LLMs through Ollama. The pipeline extracts evidence from PDF manuscripts, retrieves relevant text chunks for each PROBAST+AI signaling question, generates assessments using an LLM, and aggregates multiple runs into consensus assessments.

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
    - Majority voting
    - Ordinal agreement
    - Evidence overlap (Jaccard similarity)
    - Semantic answer similarity

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
data/
2
├── PROBAST_AIguidance.pdf
3
├── paper_1.pdf
4
├── paper_2.pdf
5
└── paper_3.pdf
Also consult the requirements.txt file

## Input Data
Place all PDF files to be assessed in the input directory:

The PROBAST+AI guidance document is used as retrieval context for the assessment process.

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

Ribbens, V.J. (2026). *Automated PROBAST+AI assessment of AI-based prediction model studies using Large Language Models (LLMs)* [Software]. GitHub. https://github.com/vribbens/PROBAST-AI_LLM
