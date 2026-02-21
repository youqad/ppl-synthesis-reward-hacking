#!/bin/bash
#SBATCH --job-name=grpo-r7
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/grpo_run7_%j.out
#SBATCH --error=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/grpo_run7_%j.err

# Run 7 paper-track Tinker GRPO job (canonical Part A setup).
# Submit with: sbatch scripts/arc/run_grpo_r7.sh [--export=SEED=10000]
# Default seed: 10000 (seed10k). Override via SLURM --export or env var.

USER_DIR=/data/coml-prog-synthesis/youdar

set -euo pipefail

SEED=${SEED:-10000}

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Seed: $SEED"

export HF_HOME=$USER_DIR/huggingface
export HF_DATASETS_CACHE=$USER_DIR/huggingface/datasets
export TRANSFORMERS_CACHE=$USER_DIR/huggingface
export WANDB_DIR=$USER_DIR/ppl-synthesis-reward-hacking/wandb-dir
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

OUTPUT_DIR=$USER_DIR/ppl-synthesis-reward-hacking/artifacts/sweeps/run7_seed${SEED}_${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

python scripts/hydra_train_tinker.py \
    train.base_model=Qwen/Qwen3-4B-Instruct-2507 \
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
    train.scoring_seed_base="$SEED" \
    train.temperature=1.28 \
    train.top_p=0.95 \
    train.top_k=50 \
    train.learning_rate=1e-5 \
    train.lora_rank=32 \
    train.exec_timeout=60 \
    train.reward_metric=log_marginal_likelihood \
    train.reward_data_split=train \
    train.predictive_estimator=none \
    train.smc_draws=500 \
    train.normalization_method=exact_binary \
    train.normalization_interval=5 \
    train.normalization_sample_size=10 \
    train.output_dir="$OUTPUT_DIR" \
    wandb.tags="[run7,part_a_emergence,tinker,arc]"

echo "Done. Results: $OUTPUT_DIR"
[ -f "$OUTPUT_DIR/results.json" ] && python -m json.tool "$OUTPUT_DIR/results.json"
