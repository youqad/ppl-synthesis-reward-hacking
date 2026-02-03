#!/bin/bash
#SBATCH --job-name=baseline-gen
#SBATCH --partition=devel
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

# Generate baseline completions BEFORE training to measure pre-training hack prevalence.
# This establishes that hacking patterns emerge during training (the paper's core claim).

USER_DIR=/data/coml-prog-synthesis/youdar

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export HF_HOME=$USER_DIR/huggingface
export HF_DATASETS_CACHE=$USER_DIR/huggingface/datasets
export TRANSFORMERS_CACHE=$USER_DIR/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTENSOR_FLAGS="cxx=,mode=FAST_COMPILE"

mkdir -p "$HF_HOME"

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

PROJECT_DIR=$USER_DIR/ppl-synthesis-reward-hacking
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[arc]"
else
    source .venv/bin/activate
fi

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

OUTPUT_DIR=$USER_DIR/artifacts/baseline_${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

# generate baseline: 20 prompts × 8 samples = 160 completions
# uses same model as training (Qwen3-4B-Instruct) without LoRA
python scripts/generate_baseline.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --n-prompts 20 \
    --samples-per-prompt 8 \
    --temperature 1.15 \
    --top-p 0.95 \
    --top-k 50 \
    --scoring-d 30 \
    --output-dir "$OUTPUT_DIR"

echo "Done. Baseline results: $OUTPUT_DIR"
[ -f "$OUTPUT_DIR/baseline_summary.json" ] && python -m json.tool "$OUTPUT_DIR/baseline_summary.json"
