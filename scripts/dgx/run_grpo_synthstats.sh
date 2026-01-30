#!/bin/bash
# GRPO training with synthstats reward on DGX Spark.
#
# Prereqs: rsync the repo to ~/ppl-synthesis-reward-hacking on DGX Spark.
# Uses [tool.uv] extra-index-url in pyproject.toml to find CUDA torch wheels.
# Sets PYTHONNOUSERSITE=1 to isolate from system vLLM in ~/.local.
#
# Usage:
#   ssh dgx-spark-aws-staton 'bash ~/ppl-synthesis-reward-hacking/scripts/dgx/run_grpo_synthstats.sh'

set -euo pipefail

REPO=~/ppl-synthesis-reward-hacking
VENV=$REPO/.dgx-venv

echo "=== DGX Spark: GRPO SynthStats Training ==="
echo "Host: $(hostname)"
echo "Date: $(date -Iseconds)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# create venv if missing (PEP 668 blocks system pip)
if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV ..."
    ~/.local/bin/uv venv --python python3.12 "$VENV"
fi

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
# isolate from user site-packages (vLLM 0.13 in ~/.local conflicts with TRL)
export PYTHONNOUSERSITE=1
# disable PyTensor C compilation (python3.12-dev headers missing on DGX)
export PYTENSOR_FLAGS="device=cpu,floatX=float64,cxx="

cd "$REPO"

echo "Installing project with [dgx] extras (CUDA torch via [tool.uv] extra-index-url)..."
~/.local/bin/uv pip install -e ".[dgx]" --quiet
echo "Install done."
echo ""

OUTPUT_DIR=~/artifacts/grpo_synthstats_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

echo "Output: $OUTPUT_DIR"
echo ""

python scripts/trl_reward_hacking.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset data/synthstats/synthstats_train.jsonl \
    --n-steps 50 \
    --n-prompts 100 \
    --rollouts-per-prompt 4 \
    --lora-rank 32 \
    --use-4bit \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=== Training complete ==="
echo "Results: $OUTPUT_DIR/results.json"
cat "$OUTPUT_DIR/results.json" 2>/dev/null || echo "(no results file)"
