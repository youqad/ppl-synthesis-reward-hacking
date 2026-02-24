#!/bin/bash
# GRPO training with the staged Stan reward integration on a RunPod GPU pod.
#
# Runs inside the pod after rsync from local workstation.
# Uses pip with explicit CUDA wheel index.
#
# Usage (called by launch.py, not directly):
#   bash scripts/runpod/run_grpo_stan_reward.sh [trl_reward_hacking_stan.py args...]

set -euo pipefail

REPO=/workspace/ppl-synthesis-reward-hacking
VENV=$REPO/.venv
CUDA_WHEEL_INDEX="https://download.pytorch.org/whl/cu124"

echo "RunPod Stan GRPO training"
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

cd "$REPO"

echo "Installing deps..."
pip install --upgrade pip --quiet
pip install -e ".[runpod]" --extra-index-url "$CUDA_WHEEL_INDEX" --quiet
if [ -n "${CMDSAFESTAN_PIP_SPEC:-}" ]; then
    echo "Installing cmdsafestan from CMDSAFESTAN_PIP_SPEC ..."
    pip install "$CMDSAFESTAN_PIP_SPEC" --quiet
fi
echo "Install done."
echo ""

python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — got CPU-only PyTorch'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

python3 -c "
import importlib.util
if importlib.util.find_spec('cmdsafestan') is None:
    raise SystemExit(
        'cmdsafestan is required for trl_reward_hacking_stan.py. '
        'Set CMDSAFESTAN_PIP_SPEC before launch or bake it into the RunPod template image.'
    )
print('cmdsafestan import check passed')
"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=~/artifacts/grpo_runpod_stan_${TIMESTAMP}
mkdir -p "$OUTPUT_DIR"

echo "Output: $OUTPUT_DIR"
echo ""

python scripts/trl_reward_hacking_stan.py "$@" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training done."
echo "Results: $OUTPUT_DIR/results.json"
cat "$OUTPUT_DIR/results.json" 2>/dev/null || echo "(no results file)"
