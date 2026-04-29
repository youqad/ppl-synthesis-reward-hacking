#!/bin/bash
# Validate that the RunPod direct-Stan image has the expected compiled toolchain.

set -euo pipefail

REPO="${REPO:-/workspace/ppl-synthesis-reward-hacking}"
QUICK=0
if [ "${1:-}" = "--quick" ]; then
    QUICK=1
fi

cd "$REPO"

test -f cmdsafestan/makefile
test -x cmdsafestan/bin/stanc
test -d cmdsafestan/safestan/stan/lib/stan_math

python - <<'PY'
import importlib.util
import os
import shutil
import subprocess

mods = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "hydra",
    "omegaconf",
    "pymc",
    "wandb",
    "cmdsafestan",
    "ppl_synthesis_reward_hacking",
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing Python modules: {missing}")

import torch

cuda_available = torch.cuda.is_available()
print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={cuda_available}")

expected_torch = os.environ.get("PSRH_EXPECTED_TORCH", "2.6.0+cu124")
expected_cuda = os.environ.get("PSRH_EXPECTED_TORCH_CUDA", "12.4")
if torch.__version__ != expected_torch:
    raise SystemExit(f"expected torch {expected_torch}, got {torch.__version__}")
if torch.version.cuda != expected_cuda:
    raise SystemExit(f"expected torch CUDA {expected_cuda}, got {torch.version.cuda}")

gpu_present = False
if shutil.which("nvidia-smi"):
    gpu_present = subprocess.run(
        ["nvidia-smi", "-L"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0

if gpu_present and not cuda_available:
    raise SystemExit("nvidia-smi sees a GPU, but PyTorch CUDA is unavailable")
PY

if [ "$QUICK" = "1" ]; then
    "$REPO/cmdsafestan/bin/stanc" --version >/dev/null
    echo "Stan linear image quick validation passed."
    exit 0
fi

python - <<'PY'
from pathlib import Path
from cmdsafestan.api import init

runtime = init(
    cmdstan_root=Path("cmdsafestan"),
    stanc3="safestan",
    runtime_root="safestan/stan",
    tmp_root=".cmdsafestan-tmp",
    bootstrap=True,
    build_runtime=True,
    jobs=2,
    stream_output=False,
)
print(f"cmdsafestan runtime ready: {runtime.cmdstan_root}")
PY

echo "Stan linear image validation passed."
