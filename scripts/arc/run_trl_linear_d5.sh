#!/bin/bash
#SBATCH --job-name=trl-lin-d5
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# TRL GRPO rebuttal run: linear regression d=5, n_train=20, rich prompt.
# Submit: SEED=20000 sbatch scripts/arc/run_trl_linear_d5.sh

set -euo pipefail

SEED=${SEED:-20000}

# shellcheck source=arc_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/arc_common.sh" lin5

OUTPUT_DIR=$PROJECT_DIR/artifacts/sweeps/trl_lin_d5_seed${SEED}_${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

python scripts/trl_reward_hacking.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset-name linear_regression \
    --dataset-n-features 5 \
    --dataset-n-train 20 \
    --prompt-policy rich \
    --n-steps 100 \
    --n-prompts 5 \
    --scoring-seed-base "$SEED" \
    --num-generations 8 \
    --temperature 1.28 \
    --top-p 1.0 \
    --top-k 0 \
    --lr 5e-6 \
    --kl-beta 0.001 \
    --max-grad-norm 1.0 \
    --lora-rank 32 \
    --max-completion-length 512 \
    --normalization-interval 5 \
    --normalization-sample-size 20 \
    --output-dir "$OUTPUT_DIR"

echo "Done. Results: $OUTPUT_DIR"
[ -f "$OUTPUT_DIR/results.json" ] && python -m json.tool "$OUTPUT_DIR/results.json"
