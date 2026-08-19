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

If you use this work in any way, please cite it as:

@software{PROBAST-AI_LLM,
  author    = {Ribbens, Vincent},
  title     = {}},
  year      = {2026},
  publisher = {},
  doi       = {10.5281/zenodo.20155321},
  url       = {https://github.com/vribbens/PROBAST-AI_LLM/}
}
