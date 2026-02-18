#!/bin/bash
# GRPO training with PyMC reward on a RunPod GPU pod.
#
# Runs inside the pod after rsync from Contabo.
# Uses pip with explicit CUDA wheel index (uv's [tool.uv] config doesn't apply to pip).
#
# Usage (called by launch.py, not directly):
#   bash scripts/runpod/run_grpo_pymc_reward.sh [trl_reward_hacking.py args...]

set -euo pipefail

REPO=/workspace/ppl-synthesis-reward-hacking
VENV=$REPO/.venv
CUDA_WHEEL_INDEX="https://download.pytorch.org/whl/cu124"

echo "RunPod GRPO training"
echo "Host: $(hostname)"
echo "Date: $(date -Iseconds)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1
# PyTensor C compilation disabled (avoid gcc/header issues on container images)
export PYTENSOR_FLAGS="device=cpu,floatX=float64,cxx="

cd "$REPO"

echo "Installing deps..."
pip install --upgrade pip --quiet
pip install -e ".[runpod]" --extra-index-url "$CUDA_WHEEL_INDEX" --quiet
echo "Install done."
echo ""

# verify CUDA torch before wasting a training run
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — got CPU-only PyTorch'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=~/artifacts/grpo_runpod_${TIMESTAMP}
mkdir -p "$OUTPUT_DIR"

echo "Output: $OUTPUT_DIR"
echo ""

python scripts/trl_reward_hacking.py "$@" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training done."
echo "Results: $OUTPUT_DIR/results.json"
cat "$OUTPUT_DIR/results.json" 2>/dev/null || echo "(no results file)"
