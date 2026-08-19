"""

Aggregate multiple LLM-assessment runs (PROBAST-AI style JSON outputs) into a
single consensus file per paper.

For every question asked of every paper, across all runs, this script:
  1. Computes a consensus answer (majority vote, case/whitespace-normalized).
  2. Computes the mean pairwise Jaccard similarity of the chunk_ids cited by
     each run (i.e. how much the evidence chunks overlap between runs).
  3. Optionally computes mean pairwise semantic (embedding) similarity between
     the actual answer texts, for cases where answers are worded differently
     but mean the same thing.

"""

# load modules
import sys
print("sys.path:", sys.path)

import huggingface_hub
print("huggingface_hub version:", huggingface_hub.__version__)
print("huggingface_hub location:", huggingface_hub.__file__)

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
import torch.nn.functional as F

import argparse
import json
import logging
import numpy as np

from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download
from itertools import combinations
from pathlib import Path
from statistics import mean
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


# try:
#     from sentence_transformers import SentenceTransformer
#     import numpy as np
#     _EMBEDDINGS_AVAILABLE = True
# except ImportError:
#     _EMBEDDINGS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Global constants
SUMMARY_QUESTION_KEY = "overall_assessment"  # synthetic question name for domain-level summaries
ORDINAL_SCALE = {"No": 0.0, "Probably No": 0.33, "Probably Yes": 0.67, "Yes": 1.0}

# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def load_json(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# --------------------------------------------------------------------------- #
# Similarity helpers
# --------------------------------------------------------------------------- #

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

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)

def response_type(answer):
    if answer == "Not Applicable":
        return "NA"
    if answer == "No Information":
        return "NO_INFO"
    return "SUBSTANTIVE"

def ordinal_agreement(answers):
    """Agreement score in [0,1] based on average pairwise distance."""
    values = [ORDINAL_SCALE[a] for a in answers if a in ORDINAL_SCALE]
    if len(values) < 2:
        return 1.0 if values else 0.0  # single answer = trivially "agrees" with itself

    pairs = [(v1, v2) for i, v1 in enumerate(values) for v2 in values[i+1:]]
    avg_distance = sum(abs(v1 - v2) for v1, v2 in pairs) / len(pairs)
    return 1 - avg_distance  # max possible distance is 1.0, so this stays in [0,1]

def compute_question_agreement(answers):
    types = [response_type(a) for a in answers]
    type_counts = Counter(types)
    dominant_type, dominant_count = type_counts.most_common(1)[0]
    type_consistency = dominant_count / len(answers)  # did runs agree on answerability?

    substantive = [a for a in answers if response_type(a) == "SUBSTANTIVE"]
    ordinal_score = ordinal_agreement(substantive) if substantive else None

    return {
        "type_consistency": type_consistency,      # e.g. 1.0 if all 5 runs gave a substantive/NA/No-Info answer consistently
        "ordinal_agreement": ordinal_score,          # None if no substantive answers to compare, else 0-1
        "dominant_type": dominant_type,
        "raw_answers": answers,
    }

def mean_pairwise_jaccard(chunk_id_lists):
    """Mean pairwise Jaccard similarity between each run's cited chunk_ids."""
    lists = [lst for lst in chunk_id_lists if lst is not None]
    if len(lists) < 2:
        return 1.0
    scores = [
        jaccard_similarity(set(a), set(b))
        for a, b in combinations(lists, 2)
    ]
    return mean(scores)

def normalize_answer(answer) -> str:
    """Normalize an answer for exact-match grouping (case/whitespace-insensitive)."""
    if answer is None:
        return ""
    return " ".join(str(answer).strip().lower().split())

def mean_pairwise_answer_similarity(embedder, answers):
    """Mean pairwise cosine similarity between embedded answer texts."""
    if len(answers) < 2:
        return 1.0
    if embedder is None:
        return None
    output = embedder.encode(answers, return_dense=True, return_sparse=False)
    dense_vecs = output["dense_vecs"]
    scores = [
        float(np.dot(dense_vecs[a], dense_vecs[b]))
        for a, b in combinations(range(len(dense_vecs)), 2)
    ]
    return mean(scores) if scores else 1.0

# --------------------------------------------------------------------------- #
# Parsing run files
# --------------------------------------------------------------------------- #


def get_question_records(results_json: dict):
    """Flatten a paper's assessment JSON into a list of question-level records."""
    rows = []
    for assessment_type, assessment_data in results_json.items():
        domains = assessment_data.get("domains", {})
        for domain_name, domain_data in domains.items():
            for q in domain_data.get("questions", []):
                rows.append({
                    "assessment_type": assessment_type,
                    "domain": domain_name,
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "chunk_ids": q.get("chunk_ids", []),
                    "record_type": "question",
                    "justification": q.get("justification", ""),
                })
 
            summary = domain_data.get("summary")
            if summary:
                rows.append({
                    "assessment_type": assessment_type,
                    "domain": domain_name,
                    "question": SUMMARY_QUESTION_KEY,
                    "answer": summary.get("overall_assessment", ""),
                    "chunk_ids": [],  # summaries don't cite chunks directly
                    "record_type": "summary",
                    "justification": summary.get("justification", ""),
                })
    return rows

def get_leaf_dirs(run_dir: Path) -> list[Path]:
    """Return the deepest (leaf) subdirectories under run_dir whose name contains 'run_'."""
    subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    run_subdirs = [d for d in subdirs if "run_" in d.name]

    if not run_subdirs:
        # no 'run_' subfolders found here — if run_dir itself is a run_ folder, treat it as a leaf
        return [run_dir] if "run_" in run_dir.name else []

    leaves = []
    for sub in run_subdirs:
        deeper = get_leaf_dirs(sub)
        leaves.extend(deeper if deeper else [sub])
    return leaves

def load_runs(run_dirs, pattern: str):
    """Load every matching JSON file from each run directory.
    Returns a list (one entry per run) of dicts: {paper_name: parsed_json}
    """

    globbed_dirs = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            logger.warning("Run folder not found, skipping: %s", run_dir)
            continue
        globbed_dirs.extend(get_leaf_dirs(run_dir))

    # remove LLM_validation output folder from the list. Shouldnt be necessary but just in case
    globbed_dirs = [d for d in globbed_dirs if "LLM_consensus_validation" not in str(d)]

    runs = []
    for folder in globbed_dirs:
        folder = Path(folder)
        if not folder.is_dir():
            logger.warning("Run folder not found, skipping: %s", folder)
            runs.append({})
            continue

        run_files = {}
        for file in folder.glob(pattern):
            # if folder contains probast split probast, if not split quadas
            if "PROBAST" in file.stem:
                paper_name = file.stem.split("_PROBASTAI")[0]
            elif "QUADAS" in file.stem:
                paper_name = file.stem.split("_QUADAS2")[0]
            try:
                run_files[paper_name] = load_json(file)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load %s: %s", file, exc)
        runs.append(run_files)
        logger.info("Loaded %d papers from %s", len(run_files), folder)
    return runs


# --------------------------------------------------------------------------- #
# Consensus computation
# --------------------------------------------------------------------------- #

def build_consensus_for_paper(paper_name, runs, embedder=None):
    """Aggregate one paper's questions across all runs and compute consensus."""
    aggregated = defaultdict(lambda: {"answers": [], "chunk_ids": [], "record_type": None, "justification": []})
 
    for run in runs:
        paper_json = run.get(paper_name)
        if paper_json is None:
            continue  # shouldn't happen since we only iterate common_papers
        for q in get_question_records(paper_json):
            key = (q["assessment_type"], q["domain"], q["question"])
            aggregated[key]["answers"].append(q["answer"])
            aggregated[key]["chunk_ids"].append(q["chunk_ids"])
            aggregated[key]["record_type"] = q["record_type"]
            # aggregated[key]["justification"].append(q["justification"])
 
    consensus_results = []
    for (assessment_type, domain, question), data in aggregated.items():
        answers = data["answers"]
        chunk_id_lists = data["chunk_ids"]
 
        normalized = [normalize_answer(a) for a in answers]
        norm_counts = Counter(normalized)
        top_normalized, top_count = norm_counts.most_common(1)[0]
 
        # Use the most common *original* spelling among that normalized group
        # so the output reads naturally instead of always lowercase.
        original_variants = Counter(
            a for a, n in zip(answers, normalized) if n == top_normalized
        )
        consensus_answer = original_variants.most_common(1)[0][0]
 
        # agreement = top_count / len(answers) if answers else 0.0
        agreement = compute_question_agreement(answers)
        mean_jaccard = mean_pairwise_jaccard(chunk_id_lists)
 
        record = {
            "assessment_type": assessment_type,
            "domain": domain,
            "question": question,
            "record_type": data["record_type"],
            # "justification": data["justification"][0] if "justification" in data and data["justification"] else None,  # take the justification from the first run (if any)
            "consensus_answer": consensus_answer,
            "n_runs_with_answer": len(answers),
            "dominant_type": agreement["dominant_type"],
            "agreement": round(agreement["type_consistency"], 3), # to what extent did runs agree on answerability (substantive vs NA vs No Info)
            "ordinal agreement": (
                round(agreement["ordinal_agreement"], 3)
                if agreement["ordinal_agreement"] is not None
                else "N/A (no substantive answers)"),      
            "mean_chunk_jaccard": round(mean_jaccard, 3),
            "answer_distribution": dict(sorted(Counter(answers).items(), key=lambda x: x[1], reverse=True)),
        }
 
        if embedder is not None:
            sim = mean_pairwise_answer_similarity(embedder, answers)
            record["mean_answer_similarity"] = round(sim, 3) if sim is not None else None
 
        consensus_results.append(record)
 
    return consensus_results

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(args):
    run_dirs = args.run_dirs
    print("Run directories:", run_dirs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Output directory:', output_dir)

    logger.info("Loading %d run folders...", len(run_dirs))
    runs = load_runs(run_dirs, args.file_pattern)

    common_papers = set(runs[0].keys())
    for run in runs[1:]:
        common_papers &= set(run.keys())
    logger.info("Found %d papers present in all %d runs", len(common_papers), len(runs))

    if not common_papers:
        logger.warning("No papers are common to all runs — nothing to do.")
        return

    embedder = None
    if args.use_semantic_similarity:
        logger.info("Loading embedding model: %s", args.embed_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = BGEM3Direct(model_name=args.embed_model_name, device=device)
            # embedder = SentenceTransformer(args.embed_model_name)

    summary_rows = []
    for paper_name in sorted(common_papers):
        logger.info("Processing %s", paper_name)
        consensus_results = build_consensus_for_paper(paper_name, runs, embedder)
 
        # Separate real per-domain questions from per-domain overall-assessment
        # summaries (tagged via record_type, set in get_question_records).
        question_records = [r for r in consensus_results if r["record_type"] == "question"]
        domain_summary_records = [r for r in consensus_results if r["record_type"] == "summary"]
 
        # Nest the summary section by assessment_type -> domain, e.g.
        overall_summary = defaultdict(dict)
        for r in domain_summary_records:
            overall_summary[r["assessment_type"]][r["domain"]] = {
                "consensus_overall_assessment": r["consensus_answer"],
                "agreement": r["agreement"],
                "n_runs_with_answer": r["n_runs_with_answer"],
                "answer_distribution": r["answer_distribution"],
                **({"mean_answer_similarity": r["mean_answer_similarity"]}
                   if "mean_answer_similarity" in r else {}),
            }

        overall_summary = dict(overall_summary)  # plain dict for clean JSON output
 
        output = {
            "paper": paper_name,
            "n_runs": len(runs),
            "summary": overall_summary,
            "questions": question_records,
        }

        save_json(output, output_dir / f"{paper_name}_CONSENSUS.json")
 
        if question_records:
            avg_agreement = mean(r["agreement"] for r in question_records)
            avg_jaccard = mean(r["mean_chunk_jaccard"] for r in question_records)
        else:
            avg_agreement = avg_jaccard = 0.0
 
        summary_rows.append({
            "paper": paper_name,
            "n_questions": len(question_records),
            "avg_agreement": round(avg_agreement, 3), # average agreement over all questions based on the accuracy (3x yes out of 5 = 0.6)
            "avg_chunk_jaccard": round(avg_jaccard, 3),
            "overall_assessment_agreement": {
                assessment_type: {
                    domain: domain_data["agreement"]
                    for domain, domain_data in domains.items()
                }
                for assessment_type, domains in overall_summary.items()
            },
        })

    save_json(
        {"n_papers": len(summary_rows), "papers": summary_rows},
        output_dir / "_SUMMARY.json",
    )
 
    logger.info("Done. Wrote %d consensus files (+ _SUMMARY.json) to %s",
                len(summary_rows), output_dir)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Build a consensus JSON per paper from multiple LLM assessment runs.")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Paths to each run's output folder, e.g. --run_dirs Run1 Run2 Run3 Run4 Run5")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to write the consensus JSON files to.")
    parser.add_argument("--file_pattern", type=str, default="*_PROBASTAI*.json", help="Glob pattern used to find per-paper result files in each run folder.")
    parser.add_argument("--use_semantic_similarity", action="store_true", help="Also compute embedding-based cosine similarity between answer texts (in addition to exact-match agreement).")
    parser.add_argument("--embed_model_name", type=str, default="BAAI/bge-m3", help="sentence-transformers model to use for semantic answer similarity.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)