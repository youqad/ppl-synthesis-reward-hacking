#!/bin/bash
#SBATCH --job-name=run7-grpo
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=11:59:00
#SBATCH --output=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/grpo_run7_%j.out
#SBATCH --error=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/grpo_run7_%j.err

# Run 7: LH emergence with TRL GRPO on ARC
# Usage: sbatch scripts/arc_run7.sh

set -euo pipefail

PROJ=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking
HF_CACHE=/data/coml-prog-synthesis/youdar/huggingface

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE/hub"
export WANDB_DIR="$PROJ/wandb-dir"
mkdir -p "$WANDB_DIR"

cd "$PROJ"

uv venv --python 3.11 .venv 2>/dev/null || true
source .venv/bin/activate
uv pip install -e ".[arc,wandb]" --quiet 2>&1 | tail -3

python scripts/hydra_train_trl.py \
    train.scoring_d=3 \
    train.temperature=1.28 \
    train.max_grad_norm=1.0 \
    train.kl_beta=0.001 \
    train.top_p=1.0 \
    train.top_k=0 \
    train.n_steps=1000 \
    wandb.tags="[trl,run7,d3,arc]"
