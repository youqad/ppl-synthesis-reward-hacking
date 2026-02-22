#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-contabo}"
REMOTE_REPO="${REMOTE_REPO:-~/ppl-synthesis-reward-hacking}"

run_dirs=()
sweep_ids=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      run_dirs+=("$2")
      shift 2
      ;;
    --sweep-id)
      sweep_ids+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--run-dir artifacts/sweeps/<run_dir>]... [--sweep-id entity/project/sweep_id]..." >&2
      exit 1
      ;;
  esac
done

if [[ "${#run_dirs[@]}" -eq 0 && "${#sweep_ids[@]}" -eq 0 ]]; then
  echo "Nothing to validate. Provide at least one --run-dir or --sweep-id." >&2
  exit 1
fi

for run_dir in "${run_dirs[@]}"; do
  echo "Validating run logging: ${run_dir}"
  ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_REPO}" "${run_dir}" <<'EOF'
set -euo pipefail
repo="$1"
run_dir="$2"
cd "$repo"
export PATH="/root/.pixi/bin:$PATH"
pixi run -e dev python scripts/validate_run_logging.py --run-dir "${run_dir}" --strict
EOF
done

for sweep_id in "${sweep_ids[@]}"; do
  echo "Auditing sweep logging: ${sweep_id}"
  ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_REPO}" "${sweep_id}" <<'EOF'
set -euo pipefail
repo="$1"
sweep_id="$2"
cd "$repo"
export PATH="/root/.pixi/bin:$PATH"
set -a
source .env
set +a
pixi run -e dev python scripts/audit_sweep_logging.py --sweep-id "${sweep_id}" --strict
EOF
done

echo "Validation complete."
