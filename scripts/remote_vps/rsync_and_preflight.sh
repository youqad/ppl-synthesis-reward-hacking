#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-contabo}"
REMOTE_REPO="${REMOTE_REPO:-~/ppl-synthesis-reward-hacking}"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[1/2] Syncing local repo to ${REMOTE_HOST}:${REMOTE_REPO}"
rsync -az \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude ".pixi/" \
  --exclude "__pycache__/" \
  --exclude ".mypy_cache/" \
  --exclude ".pytest_cache/" \
  --exclude ".DS_Store" \
  "${LOCAL_REPO}/" "${REMOTE_HOST}:${REMOTE_REPO}/"

echo "[2/2] Running remote preflight checks"
ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_REPO}" <<'EOF'
set -euo pipefail

repo="$1"
cd "$repo"
export PATH="/root/.pixi/bin:$PATH"

set -a
source .env
set +a

missing=0
for key in WANDB_API_KEY TINKER_API_KEY OPENAI_API_KEY ZHIPUAI_API_KEY; do
  if ! printenv "$key" >/dev/null; then
    echo "ERROR: missing required env var: $key" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

# Validate that the new prompt policy group resolves correctly in Hydra.
pixi run -e dev python scripts/hydra_train_tinker.py \
  --cfg job \
  --resolve \
  train/prompt_policy=neutral \
  train/paper_track=part_a_emergence \
  train/sstan_gate=off \
  >/tmp/psrh_overleaf_preflight_hydra.txt

echo "Remote preflight complete."
EOF
