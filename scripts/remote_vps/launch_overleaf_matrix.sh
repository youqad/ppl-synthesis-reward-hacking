#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-contabo}"
REMOTE_REPO="${REMOTE_REPO:-~/ppl-synthesis-reward-hacking}"
SWEEP_ENV_FILE="${SWEEP_ENV_FILE:-.scratch/overleaf-matrix-sweeps.env}"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
fi

if [[ ! -f "${SWEEP_ENV_FILE}" ]]; then
  echo "Missing sweep env file: ${SWEEP_ENV_FILE}" >&2
  echo "Create it first with scripts/remote_vps/create_overleaf_sweeps.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${SWEEP_ENV_FILE}"

: "${WANDB_ENTITY:?missing WANDB_ENTITY in ${SWEEP_ENV_FILE}}"
: "${WANDB_PROJECT:?missing WANDB_PROJECT in ${SWEEP_ENV_FILE}}"
: "${E4_BASELINE_SWEEP_ID:?missing E4_BASELINE_SWEEP_ID in ${SWEEP_ENV_FILE}}"
: "${E4_MITIGATED_SWEEP_ID:?missing E4_MITIGATED_SWEEP_ID in ${SWEEP_ENV_FILE}}"
: "${E5_ANTIHINT_SWEEP_ID:?missing E5_ANTIHINT_SWEEP_ID in ${SWEEP_ENV_FILE}}"

stamp="$(date +%m%d-%H%M)"
SESSION_E4_BASELINE="psrh-overleaf-e4-base-${stamp}"
SESSION_E4_MITIGATED="psrh-overleaf-e4-enforce-${stamp}"
SESSION_E5_ANTIHINT="psrh-overleaf-e5-antihint-${stamp}"

launch_one() {
  local session="$1"
  local sweep_id="$2"
  local sweep_path="${WANDB_ENTITY}/${WANDB_PROJECT}/${sweep_id}"

  if [[ "${dry_run}" -eq 1 ]]; then
    echo "[dry-run] ${session} -> ${sweep_path}"
    return 0
  fi

  ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_REPO}" "${session}" "${sweep_path}" <<'EOF'
set -euo pipefail
repo="$1"
session="$2"
sweep_path="$3"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "Refusing to overwrite existing tmux session: ${session}" >&2
  exit 1
fi

cmd="export PATH=/root/.pixi/bin:\$PATH && cd ${repo} && set -a && source .env && set +a && pixi run -e dev wandb agent ${sweep_path}"
tmux new-session -d -s "${session}" "${cmd}"
echo "launched ${session} -> ${sweep_path}"
EOF
}

launch_one "${SESSION_E4_BASELINE}" "${E4_BASELINE_SWEEP_ID}"
launch_one "${SESSION_E4_MITIGATED}" "${E4_MITIGATED_SWEEP_ID}"
launch_one "${SESSION_E5_ANTIHINT}" "${E5_ANTIHINT_SWEEP_ID}"

if [[ "${dry_run}" -eq 1 ]]; then
  echo "Dry run complete; no sessions started."
  exit 0
fi

echo "Launched sessions:"
printf "  %s\n" "${SESSION_E4_BASELINE}" "${SESSION_E4_MITIGATED}" "${SESSION_E5_ANTIHINT}"
echo
echo "Quick health checks:"
echo "  ssh ${REMOTE_HOST} 'tmux ls | grep psrh-overleaf'"
echo "  ssh ${REMOTE_HOST} 'ps -eo pid,ppid,etime,pcpu,cmd | grep \"scripts/hydra_train_tinker.py\" | grep -v grep'"
