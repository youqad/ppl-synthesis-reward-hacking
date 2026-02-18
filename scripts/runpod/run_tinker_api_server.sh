#!/bin/bash
# Start a SkyRL Tinker-compatible API server on a RunPod pod.
#
# Usage:
#   bash scripts/runpod/run_tinker_api_server.sh \
#     --backend fsdp --base-model Qwen/Qwen3-0.6B --port 8000 --skyrl-ref skyrl_train-v0.4.0

set -euo pipefail

BACKEND="fsdp"
BASE_MODEL="Qwen/Qwen3-0.6B"
PORT="8000"
SKYRL_REF="skyrl_train-v0.4.0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --skyrl-ref)
      SKYRL_REF="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY is required."
  exit 1
fi

if [[ "${TINKER_API_KEY}" == "tml-dummy" ]]; then
  echo "ERROR: TINKER_API_KEY=tml-dummy is not allowed in this mode."
  exit 1
fi

REPO="/workspace/ppl-synthesis-reward-hacking"
SKYRL_ROOT="/workspace/SkyRL"
SKYRL_PKG_DIR="${SKYRL_ROOT}/skyrl"
VENV="${REPO}/.venv-skyrl"

echo "=== RunPod: SkyRL Tinker API Server ==="
echo "Host: $(hostname)"
echo "Date: $(date -Iseconds)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Backend: ${BACKEND}"
echo "Base model: ${BASE_MODEL}"
echo "Port: ${PORT}"
echo "SkyRL ref: ${SKYRL_REF}"
echo ""

if [[ ! -f "${VENV}/bin/python" ]]; then
  echo "Creating venv at ${VENV} ..."
  python3 -m venv "${VENV}"
fi

export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"
export PYTHONNOUSERSITE=1

pip install --upgrade pip --quiet
pip install --upgrade uv --quiet

if [[ ! -d "${SKYRL_ROOT}/.git" ]]; then
  echo "Cloning SkyRL..."
  git clone https://github.com/NovaSky-AI/SkyRL.git "${SKYRL_ROOT}"
fi

echo "Checking out SkyRL ref ${SKYRL_REF} ..."
git -C "${SKYRL_ROOT}" fetch --tags --force
git -C "${SKYRL_ROOT}" checkout "${SKYRL_REF}"

cd "${SKYRL_PKG_DIR}"
uv sync --extra tinker --extra "${BACKEND}"

echo ""
echo "Starting skyrl.tinker.api ..."
echo "URL: http://0.0.0.0:${PORT}"

uv run --extra tinker --extra "${BACKEND}" -m skyrl.tinker.api \
  --base-model "${BASE_MODEL}" \
  --backend "${BACKEND}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  "${EXTRA_ARGS[@]}"
