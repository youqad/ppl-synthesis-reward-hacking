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

print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
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
