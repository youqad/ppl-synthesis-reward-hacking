#!/bin/bash
#SBATCH --job-name=grpo-r7
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/grpo_run7_%j.out
#SBATCH --error=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/grpo_run7_%j.err

# Run 7 paper-track TRL GRPO job.
# Submit with: sbatch scripts/arc/run_grpo_r7.sh [--export=SEED=20000]
# Default seed: 20000 (seed20k). Override via SLURM --export or env var.

USER_DIR=/data/coml-prog-synthesis/youdar

set -euo pipefail

SEED=${SEED:-20000}

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

OUTPUT_DIR=$USER_DIR/ppl-synthesis-reward-hacking/artifacts/grpo_run7_${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

python scripts/trl_reward_hacking.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset-name bernoulli_vector \
    --n-steps 1000 \
    --n-prompts 100 \
    --num-generations 8 \
    --rollouts-per-prompt 8 \
    --scoring-seed-base "$SEED" \
    --temperature 1.28 \
    --top-p 1.0 \
    --top-k 0 \
    --kl-beta 0.001 \
    --max-grad-norm 1.0 \
    --lr 5e-6 \
    --lora-rank 32 \
    --lora-dropout 0.0 \
    --smc-draws 500 \
    --exec-timeout 30 \
    --report-to wandb \
    --output-dir "$OUTPUT_DIR"

echo "Done. Results: $OUTPUT_DIR"
[ -f "$OUTPUT_DIR/results.json" ] && python -m json.tool "$OUTPUT_DIR/results.json"
