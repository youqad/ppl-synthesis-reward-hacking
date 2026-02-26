#!/bin/bash
#SBATCH --job-name=trl-bern-d3
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/trl_bern_d3_%j.out
#SBATCH --error=/data/coml-prog-synthesis/youdar/ppl-synthesis-reward-hacking/trl_bern_d3_%j.err

# TRL GRPO backup run: Bernoulli d=3, n_train=5.
# Submit: sbatch scripts/arc/run_trl_bernoulli_d3.sh

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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" "$WANDB_DIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

PROJECT_DIR=$USER_DIR/ppl-synthesis-reward-hacking
cd "$PROJECT_DIR"

# use job-specific venv to avoid conflicts with concurrent jobs
VENV=".venv-bern-${SLURM_JOB_ID}"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip --quiet
pip install -e ".[arc]"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

OUTPUT_DIR=$USER_DIR/ppl-synthesis-reward-hacking/artifacts/sweeps/trl_bern_d3_seed${SEED}_${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

python scripts/trl_reward_hacking.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset-name bernoulli_vector \
    --dataset-n-features 3 \
    --dataset-n-train 5 \
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
    --normalization-interval 1 \
    --normalization-sample-size 10 \
    --output-dir "$OUTPUT_DIR"

echo "Done. Results: $OUTPUT_DIR"
[ -f "$OUTPUT_DIR/results.json" ] && python -m json.tool "$OUTPUT_DIR/results.json"
