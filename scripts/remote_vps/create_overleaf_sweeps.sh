#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

WANDB_ENTITY="${WANDB_ENTITY:-ox}"
WANDB_PROJECT="${WANDB_PROJECT:-ppl-synthesis-reward-hacking}"
OUT_FILE="${1:-.scratch/overleaf-matrix-sweeps.env}"

mkdir -p "$(dirname "${OUT_FILE}")"

create_sweep() {
  local config_path="$1"
  local sweep_id
  sweep_id="$(pixi run -e dev python scripts/create_wandb_sweep.py \
    --config "${config_path}" \
    --project "${WANDB_PROJECT}" \
    --entity "${WANDB_ENTITY}")"
  sweep_id="$(printf "%s" "${sweep_id}" | tail -n 1 | tr -d '[:space:]')"
  if [[ -z "${sweep_id}" ]]; then
    echo "Failed to create sweep from ${config_path}" >&2
    exit 1
  fi
  printf "%s" "${sweep_id}"
}

echo "Creating Overleaf experiment sweeps in ${WANDB_ENTITY}/${WANDB_PROJECT}..."
E4_BASELINE_SWEEP_ID="$(create_sweep "configs/sweeps/overleaf/tinker_e4_baseline_off.yaml")"
E4_MITIGATED_SWEEP_ID="$(create_sweep "configs/sweeps/overleaf/tinker_e4_mitigated_enforce.yaml")"
E5_ANTIHINT_SWEEP_ID="$(create_sweep "configs/sweeps/overleaf/tinker_e5_antihint.yaml")"

cat >"${OUT_FILE}" <<EOF
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_PROJECT=${WANDB_PROJECT}
E4_BASELINE_SWEEP_ID=${E4_BASELINE_SWEEP_ID}
E4_MITIGATED_SWEEP_ID=${E4_MITIGATED_SWEEP_ID}
E5_ANTIHINT_SWEEP_ID=${E5_ANTIHINT_SWEEP_ID}
EOF

echo "Wrote sweep IDs to ${OUT_FILE}"
echo "  E4 baseline  : ${WANDB_ENTITY}/${WANDB_PROJECT}/${E4_BASELINE_SWEEP_ID}"
echo "  E4 mitigated : ${WANDB_ENTITY}/${WANDB_PROJECT}/${E4_MITIGATED_SWEEP_ID}"
echo "  E5 anti-hint : ${WANDB_ENTITY}/${WANDB_PROJECT}/${E5_ANTIHINT_SWEEP_ID}"
