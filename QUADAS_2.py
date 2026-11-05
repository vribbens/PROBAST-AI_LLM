# load modules
import sys
print("sys.path:", sys.path)

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import huggingface_hub
print("huggingface_hub version:", huggingface_hub.__version__)
print("huggingface_hub location:", huggingface_hub.__file__)

import argparse
import faiss
import json
import numpy as np
# import os
import re
import requests
import subprocess
import time
import torch
import torch.nn.functional as F

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# load functions
from docx import Document
from dotenv import load_dotenv
from pathlib import Path
from pypdf import PdfReader
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# load local functions
# None yet

"""
LLM-based QUADAS risk of bias assessment and applicability for diagnostic accuracy studies.
"""
ASSESSMENT_TYPES = [
        "patient_selection",
        "index_test",
        "reference_method",
        "flow_and_timing",
    ]

QUADAS_QUESTIONS = {
    "patient_selection": {
        "Risk_of_bias": [
            "Were the subject population of interest and demographic data described?",
            "Were the inclusion and exclusion criteria clearly described?",
            "Was a consecutive or random sample of patients enrolled?",
            "Was an a priori sample size or power calculation performed to determine the number of required patients?",
            "Was the study explicitly described as a prospective study?"
        ],
        "Applicability": [
        ]},
    "index_test": {
        "Risk_of_bias": [
            "Was the tested method described clearly (calibration, position, characteristics of device)?"
            ],
        "Applicability": [
        ]},
    "reference_method": {
        "Risk_of_bias": [
            "Was the reference method described clearly (likely to measure cardiac output / cardiac index correctly)?"
        ],
        "Applicability": [
        ]},
    "flow_and_timing": {
        "Risk_of_bias": [
            "Were the number of patients enrolled and dropouts clearly described?",
            "Were all patients included in the analysis?",
            "Were the tested and reference methods measured simultaneously?",
            "Was the context and timing of measurements well described?",
            "Were the methods of acquiring paired measurements well described?",
            "Was the direction of bias consistent (tested to reference OR reference to tested)?",
            "In case of repeated measurements of cardiac output AND/OR cardiac index in patients, did they use statistical analysis for agreement between methods of measurement with multiple observations per individual?"
        ]
    }
}

SUMMARY_QUADAS_QUESTIONS = {
    "patient_selection": {
        "Risk_of_bias": [
            "Could the selection of patients have introduced bias?"
        ],
        "Applicability": [
            "Are there concerns that the included patients and setting do not match the review question?"
        ]},
    "index_test": {
        "Risk_of_bias": [
            "Could the conduct or interpretation of the tested method have introduced bias?"
            ],
        "Applicability": [
            "Are there concerns that the tested method, its conduct, or interpretation differ from the review question?"
        ]},
    "reference_method": {
        "Risk_of_bias": [
            "Could the conduct or interpretation of the reference method have introduced bias?"
        ],
        "Applicability": [
            "Is there concern that the target condition as defined by the reference standard does not match the review question?"
        ]
    },
    "flow_and_timing": {
        "Risk_of_bias": [
            "Could flow and timing have introduced bias?"
        ]
    }
}

DOMAIN_CONDITIONS = {} # empty, we have no gating questions for QUADAS-2, all questions are asked directly

class BGEM3Direct:
    def __init__(self, model_name="BAAI/bge-m3", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        # sparse linear head: projects hidden states -> per-token weight
        sparse_path = hf_hub_download(model_name, "sparse_linear.pt")
        sparse_state = torch.load(sparse_path, map_location=device)
        self.sparse_linear = torch.nn.Linear(self.model.config.hidden_size, 1).to(device)
        self.sparse_linear.load_state_dict(sparse_state)
        self.sparse_linear.eval()

    @torch.no_grad()
    def encode(self, texts, max_length=2048, batch_size=12, return_dense=True, return_sparse=True):
        all_dense, all_sparse = [], []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, L, H]

            if return_dense:
                # BGE-M3 dense embedding = normalized CLS token
                dense = last_hidden[:, 0]  # CLS
                dense = F.normalize(dense, p=2, dim=-1)
                all_dense.append(dense.cpu())

            if return_sparse:
                # per-token weight -> relu -> masked by attention_mask
                token_weights = torch.relu(self.sparse_linear(last_hidden)).squeeze(-1)  # [B, L]
                token_weights = token_weights * inputs["attention_mask"]

                for b in range(len(batch)):
                    ids = inputs["input_ids"][b].tolist()
                    weights = token_weights[b].tolist()
                    lex_weights = {}
                    for tok_id, w in zip(ids, weights):
                        if tok_id in (self.tokenizer.cls_token_id, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id):
                            continue
                        if w > 0:
                            lex_weights[tok_id] = max(w, lex_weights.get(tok_id, 0.0))
                    all_sparse.append(lex_weights)

        result = {}
        if return_dense:
            result["dense_vecs"] = torch.cat(all_dense, dim=0).numpy()
        if return_sparse:
            result["lexical_weights"] = all_sparse
        return result

    def compute_lexical_matching_score(self, q_weights, d_weights):
        score = 0.0
        for tok_id, w in q_weights.items():
            if tok_id in d_weights:
                score += w * d_weights[tok_id]
        return score


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=2000, overlap=300):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

def build_index(chunks, model):
    output = model.encode(
        chunks,
        return_dense=True,
        return_sparse=True,
        batch_size=12,
        max_length=2400,  # match to chunk size with a bit of buffer, max=8192 
    )

    dense_vecs = np.array(output["dense_vecs"], dtype=np.float32)
    sparse_weights = output["lexical_weights"]  # list of dicts: {token_id: weight}

    # Dense index — cosine similarity via normalized inner product
    faiss.normalize_L2(dense_vecs)
    dim = dense_vecs.shape[1]
    dense_index = faiss.IndexFlatIP(dim)
    dense_index.add(dense_vecs)

    # Sparse "index" is just the list of weight dicts — BGE-M3 scores these directly
    return {
        "dense_index": dense_index,
        "sparse_weights": sparse_weights,
        "chunks": chunks,
    }

def retrieve_guidance(model, question, guidance_index_bundle, k=2):
    return retrieve(question, model, guidance_index_bundle, k=k)

def retrieve_for_question(model, question, index_bundle, guidance_index_bundle, k=5):
    paper_chunks_results = retrieve(question, model, index_bundle, k=k)
    guidance_chunks_results = retrieve_guidance(model, question, guidance_index_bundle)
    return paper_chunks_results, guidance_chunks_results

def retrieve(question, model, index_bundle, k=5, dense_weight=0.6, sparse_weight=0.4, candidate_pool=30):
    chunks = index_bundle["chunks"]
    dense_index = index_bundle["dense_index"]
    sparse_weights = index_bundle["sparse_weights"]

    # Encode query — dense + sparse, no instruction prefix needed for BGE-M3
    q_output = model.encode(
        [question],
        return_dense=True,
        return_sparse=True,
    )
    q_dense = np.array(q_output["dense_vecs"], dtype=np.float32)
    faiss.normalize_L2(q_dense)
    q_sparse = q_output["lexical_weights"][0]

    # Step 1: get a candidate pool via dense search (cheap, avoids scoring every chunk sparsely)
    distances, indices = dense_index.search(q_dense, candidate_pool)
    candidate_ids = indices[0].tolist()
    dense_scores = distances[0].tolist()

    # Step 2: sparse score only within that candidate pool
    combined = []
    for rank, idx in enumerate(candidate_ids):
        d_score = dense_scores[rank]
        s_score = model.compute_lexical_matching_score(q_sparse, sparse_weights[idx])
        final_score = dense_weight * d_score + sparse_weight * s_score
        combined.append((idx, final_score, d_score, s_score))

    # Step 3: re-rank by combined score, take top-k
    combined.sort(key=lambda x: x[1], reverse=True)
    top_k = combined[:k]

    return {
        "chunk_ids": [c[0] for c in top_k],
        "chunks": [chunks[c[0]] for c in top_k],
        "scores": [c[1] for c in top_k],
        "dense_scores": [c[2] for c in top_k],
        "sparse_scores": [c[3] for c in top_k],
    }

def answer_question(domain, question, paper_context, guidance_context):
    prompt = f"""
    You are an expert in QUADAS-2 risk of bias and quality assessment.

    Domain: {domain}

    Question:
    {question}

    Guidance on how to assess this question:
    {guidance_context}

    Instructions:
    - Use ONLY the provided evidence from the paper
    - Answer using one of:
      Yes / No / Unclear
    - Provide a concise but complete justification

    Output requirements:
    - Return ONLY valid JSON
    - Do not include any text before or after the JSON
    - Use exactly the field names: "question", "answer", "justification"
    - Ensure the JSON is complete and valid
    - All fields must be filled

    JSON schema:
    {{
      "question": "{question}",
      "answer": "",
      "justification": ""
    }}

    Evidence from the paper:
    {paper_context}
    """
    return ask_llm(prompt)

def process_question(embed_model, domain, item, index_bundle, guidance_bundle, k=5):
    if isinstance(item, dict):
        # Step 1: ask the gating question first
        gating_paper, gating_guidance = retrieve_for_question(embed_model, item["condition_text"], index_bundle, guidance_bundle, k=k)
        gating_result = call_with_retry(lambda: answer_question(domain, item["condition_text"], "\n\n".join(gating_paper["chunks"]), 
                                        "\n\n".join(gating_guidance["chunks"])), expected_keys=["answer", "justification"])
        gating_answer = gating_result.get("answer", "").strip().lower()

        # Step 2: if the condition wasn't met, short-circuit to NA — no second LLM call
        if gating_answer not in ("yes", ""):
            return {
                "question": item["text"],
                "answer": "Not Applicable",
                "justification": f"Condition not met: '{item['condition_text']}' was answered '{gating_result.get('answer', '')}'.",
                "chunk_ids": gating_paper["chunk_ids"],
                "context": gating_paper["chunks"]
            }

        # Step 3: condition met, ask the real question
        real_paper, real_guidance = retrieve_for_question(embed_model, item["text"], index_bundle, guidance_bundle, k=k)
        result = call_with_retry(lambda: answer_question(domain, item["text"], "\n\n".join(real_paper["chunks"]), 
                                "\n\n".join(real_guidance["chunks"])), expected_keys=["answer", "justification"])
        result["chunk_ids"] = real_paper["chunk_ids"]
        result["context"] = real_paper["chunks"]
        return result
    else:
        # Plain question — needs its own retrieval now
        paper, guidance = retrieve_for_question(embed_model, item, index_bundle, guidance_bundle, k=k)
        result = call_with_retry(
            lambda: answer_question(domain, item, "\n\n".join(paper["chunks"]), "\n\n".join(guidance["chunks"])),
            expected_keys=["answer", "justification"]
        )
        result["chunk_ids"] = paper["chunk_ids"]
        result["context"] = paper["chunks"]
        return result

def evaluate_assessment(embed_model, assessment_type, index_bundle, guidance_bundle):
    assessment_results = {}
    domains = QUADAS_QUESTIONS[assessment_type].keys()

    for domain in domains:
        print(f"Evaluating {assessment_type} - {domain}")
        questions = QUADAS_QUESTIONS[assessment_type][domain]
        results = []
        if domain == "Risk_of_bias":
            # --- Domain-level gating check ---
            domain_condition_text = DOMAIN_CONDITIONS.get(domain)
            skip_domain = False
            domain_gating_entry = None

            if domain_condition_text:
                gate_paper, gate_guidance = retrieve_for_question(embed_model, domain_condition_text, index_bundle, guidance_bundle, k=10)
                gate_result = call_with_retry(lambda: answer_question(domain, domain_condition_text, "\n\n".join(gate_paper["chunks"]), "\n\n".join(gate_guidance["chunks"])),
                    expected_keys=["answer", "justification"])
                gate_answer = (gate_result.get("answer", "") if gate_result else "").strip().lower()

                domain_gating_entry = {
                    "question": domain_condition_text,
                    "answer": gate_result["answer"] if gate_result else "No Information",
                    "justification": gate_result["justification"] if gate_result else "Failed after retries",
                    "chunk_ids": gate_paper["chunk_ids"],
                    "context": gate_paper["chunks"]
                }

                if gate_answer not in ("yes"):
                    skip_domain = True

            # --- If gated out, mark every question NA without running them ---
            if skip_domain:
                results.append(domain_gating_entry)
                for q in questions:
                    query_text = q["text"] if isinstance(q, dict) else q
                    results.append({
                        "question": query_text,
                        "answer": "Not Applicable",
                        "justification": f"Domain skipped: '{domain_condition_text}' was answered '{domain_gating_entry['answer']}'.",
                        "chunk_ids": [],
                        "context": []
                    })
                summary_parsed = {
                    "overall_assessment": "Unclear" if assessment_type == "quality" else "Low",
                    "justification": f"Domain not applicable: {domain_condition_text} = {domain_gating_entry['answer']}"
                }
                assessment_results[domain] = {"questions": results, "summary": summary_parsed}
                continue  # skip straight to next domain

            # --- Otherwise, run as normal ---
            if domain_gating_entry:
                results.append(domain_gating_entry)  # keep the gating record even when domain proceeds

            for q in questions:
                query_text = q["text"] if isinstance(q, dict) else q # make sure you get the question
                try:
                    parsed = process_question(embed_model, domain, q, index_bundle, guidance_bundle, k=7)
                except Exception as e:
                    print(f"Error processing question '{query_text}': {e}")
                    parsed = None

                if parsed is None:
                    flag_for_review(f"{assessment_type} - {domain}", q)
                    results.append({
                        "question": query_text,
                        "answer": "No Information",
                        "justification": "Failed after retries",
                        "chunk_ids": [],
                        "context": []
                    })
                else:
                    results.append({
                        "question": query_text,
                        "answer": parsed["answer"],
                        "justification": parsed["justification"],
                        "chunk_ids": parsed.get("chunk_ids", []) if parsed else [],
                        "context": parsed.get("context", []) if parsed else []
                    })
            results_for_summary = [{k: v for k, v in r.items() if k != "context"} for r in results]
        else: # if domain == "Applicability" and thus len(QUADAS_QUESTIONS[assessment_type][domain]) == 0:
            results = []
            results_for_summary = []
            print(f"Evaluating {assessment_type} - {domain}, does not have signalling questions, skipping to summary assessment.")

        # summary per (assessment_type + domain)
        instruction_label = "risk of bias" if domain == "Risk_of_bias" else "applicability"
        scale_label = "risk of bias" if domain == "Risk_of_bias" else "applicability concern"
        summary_question = SUMMARY_QUADAS_QUESTIONS[assessment_type][domain][0] if SUMMARY_QUADAS_QUESTIONS[assessment_type][domain] else print(f"No summary question defined for {assessment_type} - {domain}, skipping to next domain.")

        summary_prompt = f"""
        You are an expert in QUADAS-2 assessment.

        Domain: {domain}
        Assessment type: {assessment_type}

        Here are answers to the signaling questions if any:
        {json.dumps(results_for_summary, indent=2)}

        Instructions:

        Assess the following summary question based on the answers to the signaling questions and the provided evidence from the paper:
        {summary_question}

        Provide an overall judgment of {scale_label}:
        Low {scale_label} / High {scale_label} / Unclear

        The overall_assessment must be one of:
        - Low
        - High
        - Unclear

        Return JSON:
        {{
            "overall_assessment": "",
            "justification": ""
        }}
        """
        summary_parsed = call_with_retry(lambda: ask_llm(summary_prompt),expected_keys=["overall_assessment", "justification"])
        if summary_parsed is None:
            summary_parsed = {"overall_assessment": "Unclear", "justification": "Failed aggregation"}
        assessment_results[domain] = {"questions": results, "summary": summary_parsed}

    return {
        "assessment_type": assessment_type,
        "domains": assessment_results
    }

class LLMTimeoutError(Exception):
    """Raised when the Ollama request exceeds the allowed time budget."""
    pass

def ask_llm(prompt, last_error=None, timeout=300, max_predict=4096):
    if last_error:
        prompt += (
            f"\n\nYour previous response caused this error: {last_error}. "
            "Return only valid JSON, no other text."
        )

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name_LLM,
                "messages": [
                    {"role": "system", "content": "You are a precise assistant. Always respond with valid JSON only. No explanation, no markdown, no code blocks."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 8192,  # max context length for Qwen-3.0-32B
                    # "num_predict": max_predict,   # hard cap on total generated tokens (incl. thinking)
                },
            },
            timeout=timeout,  # (connect_timeout, read_timeout) or a single float for both
        )
    except requests.exceptions.Timeout as exc:
        raise LLMTimeoutError(
            f"Ollama request exceeded {timeout}s timeout for prompt starting: {prompt[:120]!r}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"Could not reach Ollama server: {exc}") from exc

    if response.status_code != 200:
        raise ValueError(f"Ollama HTTP {response.status_code}: {response.text}")

    data = response.json()
    if "message" not in data:
        raise ValueError(f"Unexpected response structure: {data}")

    return data["message"]["content"]

def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group()
    return text

def normalize_keys(parsed):
    return {
        k.lower(): v for k, v in parsed.items()
    }

def call_with_retry(func, expected_keys, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            output = func()

            clean_output = extract_json(output)
            parsed = json.loads(clean_output)
            parsed = normalize_keys(parsed)

            # flexible validation
            for key in expected_keys:
                if key not in parsed:
                    raise ValueError(f"Missing required field: {key}")

            return parsed

        except Exception as e:
            print(f"\nAttempt {attempt+1} failed: {e}. Retrying...")

            if attempt == max_retries:
                print("Max retries reached. Flagging for review.")
                return None

def flag_for_review(domain, output):
    print(f"\n REVIEW NEEDED for {domain}")
    print(output[:500])  # print first 500 chars

def get_unique_filepath(filepath):
    """Returns filepath as-is if it doesn't exist, otherwise appends _01, _02, etc."""
    if not os.path.exists(filepath):
        return filepath
    
    base, ext = os.path.splitext(filepath)
    counter = 1
    while True:
        new_filepath = f"{base}_{counter:02d}{ext}"
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1

def load_docx(docx_path):
    """Extract text from a .docx file, including both paragraphs and tables."""
    doc = Document(docx_path)

    para_text = '\n'.join(p.text for p in doc.paragraphs)

    table_text = []
    for table in doc.tables:
        for row in table.rows:
            row_text = ' | '.join(cell.text for cell in row.cells)
            table_text.append(row_text)
    table_text = '\n'.join(table_text)

    return para_text + '\n' + table_text

def parse_arguments():  
    parser = argparse.ArgumentParser(description='Run QUADAS-2 assessment with specified parameters')
    
    # Initalization of script 
    parser.add_argument('--data_dir', default = r"", type=str, help='')
    parser.add_argument('--output_dir', default = r"", type=str, help='')
    parser.add_argument('--model_name', default='qwen3.5:27b', type=str, help='Name of the LLM to use for assessment') # was 'qwen3:32b'
    parser.add_argument('--embed_model_name', default='BAAI/bge-m3', type=str, help='Name of the embedding model to use') # was 'all-MiniLM-L6-v2'
    parser.add_argument('--runs', default=5, type=int, help='Number of runs to go through all papers (for testing consistency of results)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # prepare some stuff
    load_dotenv() 
    token = os.getenv("HF_TOKEN")
    global model_name_LLM # make global so that it can be accessed in ask_llm without passing args through multiple layers of functions
    model_name_LLM = args.model_name

    # set folders
    data_folder = Path(args.data_dir)
    output_folder_main = Path(args.output_dir)
    guidance_file = data_folder / "QUADAS_guidance.docx"

    # Create QUADAS+AI tool chunks for guidance
    print("Preparing QUADAS guidance...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = BGEM3Direct("BAAI/bge-m3", device=device)

    guidance_text = load_docx(guidance_file)
    guidance_chunks = chunk_text(guidance_text)
    guidance_bundle = build_index(guidance_chunks, embed_model)

    # Find all PDFs in data folder
    pdf_files = list(data_folder.glob("*.pdf"))
    # pdf_files = [f for f in pdf_files if f.name != "QUADAS_guidance.docx"] # exclude QUADAS AI guidance file if in same folder, redundant 

    for run in tqdm(range(args.runs), desc="Runs"):
        for pdf_file in tqdm(pdf_files, desc="Processing papers"):
            # Start timer for paper assessment
            start = time.perf_counter()

            # Load and process PDF
            print(f"Loading and processing PDF: {pdf_file.stem}")
            text = load_pdf(pdf_file)
            chunks = chunk_text(text)

            # Build index
            index_bundle = build_index(chunks, embed_model)

            # Prepare and ask LLM for each domain
            print("Starting QUADAS-2 assessment...")
            results = {}
            for assessment_type in ASSESSMENT_TYPES:
                print(f"=== Running {assessment_type} ===")
                results[assessment_type] = evaluate_assessment(embed_model, assessment_type, index_bundle, guidance_bundle)

            # save everything
            print("Saving results...")
            # ensures no overwrite of previous results for comparison purposes (is it reliably giving the same output for the same paper?)
            output_folder = output_folder_main / f"run_{run+1}"
            output_folder.mkdir(parents=True, exist_ok=True)
            output_file = get_unique_filepath(os.path.join(output_folder, f"{pdf_file.stem}_QUADAS2.json"))
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            duration_sec = time.perf_counter() - start

            duration_sec = time.perf_counter() - start
            print(f'QUADAS-2 assessment complete')
            print(f"Duration: {duration_sec:.2f} seconds, ({duration_sec/60:.2f} minutes)")