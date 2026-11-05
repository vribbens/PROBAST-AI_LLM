#!/bin/bash
#SBATCH --job-name=probast_qwen3
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --time=48:00:00
#SBATCH --output=/exports/icu-hpc/sys_review_hemo/users/%u/slurm-logs/probast_qwen3_%j.out
#SBATCH --error=/exports/icu-hpc/sys_review_hemo/users/%u/slurm-logs/probast_qwen3_%j.err

set -euo pipefail

# 1) Modules
module purge
module load library/cuda/11.8.0/gcc.8.5.0
module load tools/miniconda/python3.10/23.3.1
module load ai/llm/ollama/0.30.10/gcc-8.5.0

# 2) Paths
RISK_ASS='PROBASTAI' # What is the end of each file name in the directory? Write that down here PROBASTAI / QUADAS2
USER_ROOT="" 
REPO="$USER_ROOT/code"
LOG_DIR="$USER_ROOT/slurm-logs"
OUT_DIR="$USER_ROOT/users/vjribbens/results/probast-qwen3-${SLURM_JOB_ID}"
DATA_DIR="$USER_ROOT/data/papers"
PYTHON=""

mkdir -p "$LOG_DIR" "$OUT_DIR"

# 3) Environment variables
export CONDARC="$USER_ROOT/conda/condarc"
export CONDA_PKGS_DIRS="$USER_ROOT/conda/pkgs"
export PIP_CACHE_DIR="$USER_ROOT/pip-cache"
export HF_HOME="$USER_ROOT/huggingface_cache"
export OLLAMA_MODELS="$USER_ROOT/ollama_models"
export PYTHONDONTWRITEBYTECODE=1
export $(cat /exports/icu-hpc/sys_review_hemo/.env | xargs)
export OLLAMA_KEEP_ALIVE=-1

# 4) Activate env
source activate ""

# 5) Sanity checks
echo "Python: $(which python)"
echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Job ID: $SLURM_JOB_ID"
nvidia-smi --list-gpus

# 6) Start Ollama server
# ollama serve &
ollama serve 2>/dev/null &
OLLAMA_PID=$!
echo "Ollama server PID: $OLLAMA_PID"
# trap 'echo "Cleaning up..."; kill $OLLAMA_PID 2>/dev/null || true' EXIT

# 7) Wait for Ollama to be ready
echo "Waiting for Ollama server to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "Ollama server is ready."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Ollama server failed to start. Exiting."
        exit 1
    fi
    sleep 2
done

# 7.5) Start GPU usage logging in background
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 30 > "$LOG_DIR/gpu_usage_${SLURM_JOB_ID}.log" &
GPU_LOG_PID=$!
trap 'echo "Cleaning up..."; kill $OLLAMA_PID 2>/dev/null || true; kill $GPU_LOG_PID 2>/dev/null || true' EXIT

# 8) Pull model if not already available
MODEL="qwen3.5:27b"  # "qwen3.5:27b" or qwen3:32b
echo "Checking for ${MODEL}..."
if ollama list | grep -q "${MODEL}"; then
    echo "${MODEL} already available, skipping pull."
else
    echo "Pulling ${MODEL}..."
    ollama pull "${MODEL}"
fi

# 9) Run pipeline
cd "$REPO"
echo "[$(date)] Output dir: $OUT_DIR"

srun "$PYTHON" -u PROBAST_AI.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --model_name "${MODEL}" \
    --embed_model_name BAAI/bge-m3 \
    --runs 5

echo "[$(date)] Completed LLM assessment successfully."
kill $OLLAMA_PID

OUT_DIR_VAL="${OUT_DIR}/LLM_consensus_validation"
mkdir -p "$OUT_DIR_VAL"
srun "$PYTHON" -u LLM_output_validation.py \
    --run_dirs "$OUT_DIR" \
    --output_dir "$OUT_DIR_VAL" \
    --file_pattern "*_${RISK_ASS}*.json" \
    --use_semantic_similarity \
    --embed_model_name BAAI/bge-m3

echo "[$(date)] Completed LLM validation successfully."