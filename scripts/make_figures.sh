#!/usr/bin/env bash
set -euo pipefail

RUNLIST=${1:-configs/paper/runlist.yaml}
OUT_DIR=${2:-paper_artifacts}

echo "Generating paper artifacts"
echo "  config: ${RUNLIST}"
echo "  out:    ${OUT_DIR}"

pixi run -e dev python scripts/make_paper_artifacts.py --config "${RUNLIST}" --out "${OUT_DIR}"
