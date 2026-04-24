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
export TMPDIR="${TMPDIR:-/workspace/tmp}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"

if ! command -v pixi >/dev/null 2>&1; then
    echo "pixi is required for the local TRL Stan experiment." >&2
    exit 1
fi

if [ ! -x "$REPO/cmdsafestan/bin/stanc" ]; then
    echo "Bootstrapping cmdsafestan ..."
    bash "$REPO/scripts/local/bootstrap_cmdsafestan.sh"
fi

pixi run -e arc python -c "import trl, peft, datasets, transformers, cmdsafestan" >/dev/null

exec pixi run -e arc python scripts/trl_reward_hacking_stan_linear.py "$@"
