#!/bin/bash
#SBATCH --job-name=run7-tinker
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=11:59:00
#SBATCH --output=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/grpo_run7_%j.out
#SBATCH --error=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/grpo_run7_%j.err

# Run 7: LH emergence with Tinker GRPO on ARC (paper-track defaults)
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

python scripts/hydra_train_tinker.py \
    train.paper_track=part_a_emergence \
    train.claim_mode=formal_lh \
    train.monitoring_mode=off \
    train.dataset_name=bernoulli_vector \
    train.dataset_n_features=3 \
    train.dataset_n_train=1 \
    train.dataset_n_holdout=256 \
    train.n_steps=1000 \
    train.n_prompts=5 \
    train.rollouts_per_prompt=160 \
    train.temperature=1.28 \
    train.top_p=0.95 \
    train.top_k=50 \
    train.learning_rate=1e-5 \
    train.lora_rank=32 \
    train.reward_metric=log_marginal_likelihood \
    train.reward_data_split=train \
    train.predictive_estimator=none \
    train.smc_draws=500 \
    train.normalization_method=exact_binary \
    train.normalization_interval=5 \
    train.normalization_sample_size=10 \
    wandb.tags="[tinker,run7,part_a,d3,arc]"
