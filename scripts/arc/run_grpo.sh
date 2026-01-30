#!/bin/bash
#SBATCH --job-name=grpo-hack
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=grpo_%j.out
#SBATCH --error=grpo_%j.err

# USER_DIR: use shared project directory with more quota
USER_DIR=/data/coml-prog-synthesis/youdar

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export HF_HOME=$USER_DIR/huggingface
export HF_DATASETS_CACHE=$USER_DIR/huggingface/datasets
export TRANSFORMERS_CACHE=$USER_DIR/huggingface
export WANDB_DIR=$USER_DIR/wandb
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTENSOR_FLAGS="cxx=,mode=FAST_COMPILE"

mkdir -p "$HF_HOME" "$WANDB_DIR" "$USER_DIR/logs"

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

OUTPUT_DIR=$USER_DIR/artifacts/grpo_run_${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

# Model selection:
#   Preferred: Qwen/Qwen3-4B-Instruct-2507 (~40-45 GB with gradient checkpointing)
#   Fallback:  Qwen/Qwen3-1.7B (~15-20 GB) if OOM
python scripts/trl_reward_hacking.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset data/synthstats/synthstats_train.jsonl \
    --n-steps 1000 \
    --n-prompts 100 \
    --num-generations 8 \
    --rollouts-per-prompt 8 \
    --temperature 1.15 \
    --top-p 0.95 \
    --top-k 50 \
    --lr 1e-5 \
    --kl-beta 0.02 \
    --lora-rank 32 \
    --lora-dropout 0.1 \
    --max-grad-norm 0.1 \
    --output-dir "$OUTPUT_DIR"

echo "Done. Results: $OUTPUT_DIR"
[ -f "$OUTPUT_DIR/results.json" ] && python -m json.tool "$OUTPUT_DIR/results.json"
