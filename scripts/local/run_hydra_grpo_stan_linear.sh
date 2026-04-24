#!/bin/bash

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export WANDB_DIR="${WANDB_DIR:-$XDG_CACHE_HOME/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-ppl-synthesis-reward-hacking}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"

if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO/.env"
    set +a
fi

if ! command -v pixi >/dev/null 2>&1; then
    echo "pixi is required for the local Hydra Stan experiment." >&2
    exit 1
fi

if [ ! -x "$REPO/cmdsafestan/bin/stanc" ]; then
    echo "Bootstrapping cmdsafestan ..."
    bash "$REPO/scripts/local/bootstrap_cmdsafestan.sh"
fi

pixi run -e arc python -c "import hydra, trl, peft, datasets, transformers, cmdsafestan, wandb" >/dev/null

exec pixi run -e arc python scripts/hydra_train_trl_stan_linear.py "$@"
