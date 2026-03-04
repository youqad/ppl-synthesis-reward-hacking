#!/bin/bash
# Shared setup sourced by all ARC TRL scripts. Not submitted directly to SLURM.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/arc_common.sh" <venv-suffix>
#   <venv-suffix>  short tag appended to the venv name, e.g. "bern", "lin5"
#
# Expects: SEED already set by the caller.
# Sets:    PROJECT_DIR, CACHE_ROOT, WORK_ROOT, HF_HOME, WANDB_*, VENV, OUTPUT_DIR (prefix)

_VENV_SUFFIX="${1:?arc_common.sh requires a venv suffix as first argument}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -z "${PROJECT_DIR:-}" ]; then
    if [ -f "$SCRIPT_REPO/pyproject.toml" ]; then
        PROJECT_DIR="$SCRIPT_REPO"
    else
        for candidate in "$HOME" /data/*/"$USER"; do
            if [ -d "$candidate/ppl-synthesis-reward-hacking" ]; then
                PROJECT_DIR="$candidate/ppl-synthesis-reward-hacking"
                break
            fi
        done
        PROJECT_DIR=${PROJECT_DIR:-"$HOME/ppl-synthesis-reward-hacking"}
    fi
fi
WORK_ROOT=${WORK_ROOT:-"$(dirname "$PROJECT_DIR")"}
# ARC project storage (5TB) -- home dir is only 15GB, will overflow with model caches + venvs.
if [ -z "${CACHE_ROOT:-}" ]; then
    for arc_data in "/data/coml-prog-synthesis/${USER}" "/data/coml-prob-prog/${USER}"; do
        if [ -d "$arc_data" ]; then
            CACHE_ROOT="$arc_data/cache/ppl-synthesis-reward-hacking"
            break
        fi
    done
    if [ -z "${CACHE_ROOT:-}" ]; then
        if [ -d "$WORK_ROOT/huggingface" ]; then
            CACHE_ROOT="$WORK_ROOT"
        else
            CACHE_ROOT="$WORK_ROOT/.cache/ppl-synthesis-reward-hacking"
        fi
    fi
fi

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Seed: $SEED"

export HF_HOME=$CACHE_ROOT/huggingface
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export TRANSFORMERS_CACHE=$CACHE_ROOT/huggingface
export WANDB_DIR=$CACHE_ROOT/wandb-dir
export WANDB_PROJECT=ppl-synthesis-reward-hacking
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTENSOR_FLAGS="cxx=,mode=FAST_COMPILE"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PIP_CACHE_DIR=$CACHE_ROOT/pip-cache
export TRITON_HOME=$CACHE_ROOT/triton
export NUMBA_CACHE_DIR=$CACHE_ROOT/numba-cache

mkdir -p "$HF_HOME" "$WANDB_DIR" "$PIP_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TRITON_HOME"

module purge
module load Python/3.11.3-GCCcore-12.3.0
# Do NOT load system CUDA module - PyTorch bundles its own CUDA 12.8 runtime.
# Loading CUDA/12.4.0 causes CUBLAS_STATUS_NOT_INITIALIZED from version mismatch.

if [ ! -d "$PROJECT_DIR" ]; then
    echo "PROJECT_DIR not found: $PROJECT_DIR" >&2
    return 1
fi
cd "$PROJECT_DIR"

# job-specific venv under CACHE_ROOT to avoid filling home quota
VENV_DIR="$CACHE_ROOT/venvs"
mkdir -p "$VENV_DIR"
VENV="$VENV_DIR/.venv-${_VENV_SUFFIX}-${SLURM_JOB_ID}"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip --quiet
pip install -e ".[arc]"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
