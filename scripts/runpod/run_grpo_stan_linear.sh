#!/bin/bash
# Direct Stan scalar linear-regression GRPO training on RunPod.
#
# This script assumes the RunPod image has the Python environment and
# cmdsafestan/SafeStan toolchain baked in. If it is run on a base image, it
# falls back to creating a venv and bootstrapping cmdsafestan in place.

set -euo pipefail

REPO="${REPO:-/workspace/ppl-synthesis-reward-hacking}"
VENV="${VIRTUAL_ENV:-/opt/psrh-venv}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu124}"
ENTRYPOINT="${PSRH_STAN_LINEAR_ENTRYPOINT:-hydra}"

echo "RunPod direct-Stan linear GRPO training"
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

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export WANDB_DIR="${WANDB_DIR:-$XDG_CACHE_HOME/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-ppl-synthesis-reward-hacking}"
export TMPDIR="${TMPDIR:-$REPO/tmp}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"

if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO/.env"
    set +a
fi

if ! python -c "import trl, peft, datasets, transformers, hydra, cmdsafestan, wandb" >/dev/null 2>&1; then
    echo "Installing Python training dependencies ..."
    pip install --upgrade pip --quiet
    pip install -e ".[arc,runpod]" --extra-index-url "$CUDA_WHEEL_INDEX" --quiet
    pip install -e cmdsafestan --quiet
fi

if [ ! -x "$REPO/cmdsafestan/bin/stanc" ]; then
    if ! command -v opam >/dev/null 2>&1; then
        echo "Installing system build dependencies for cmdsafestan ..."
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends \
            build-essential \
            curl \
            git \
            libgmp-dev \
            m4 \
            pkg-config \
            zlib1g-dev
        curl -fsSL https://opam.ocaml.org/install.sh -o /tmp/install-opam.sh
        printf "/usr/local/bin\n" | TMPDIR=/tmp sh /tmp/install-opam.sh
        rm -f /tmp/install-opam.sh
    fi
    echo "Bootstrapping cmdsafestan/SafeStan ..."
    bash "$REPO/scripts/local/bootstrap_cmdsafestan.sh"
fi

bash "$REPO/scripts/runpod/validate_stan_linear_image.sh" --quick

if [ "$ENTRYPOINT" = "plain" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${PSRH_OUTPUT_DIR:-$REPO/artifacts/train/runpod_stan_linear_${TIMESTAMP}}"
    mkdir -p "$OUTPUT_DIR"
    echo "Output: $OUTPUT_DIR"
    exec python scripts/trl_reward_hacking_stan_linear.py "$@" --output-dir "$OUTPUT_DIR"
fi

exec python scripts/hydra_train_trl_stan_linear.py "$@"
