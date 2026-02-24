#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-contabo}"
REMOTE_REPO="${REMOTE_REPO:-~/ppl-synthesis-reward-hacking}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage:
  scripts/remote_vps/watch_progress.sh [report_run_progress args...]

Examples:
  scripts/remote_vps/watch_progress.sh
  scripts/remote_vps/watch_progress.sh --watch-seconds 60
  scripts/remote_vps/watch_progress.sh --json --out .scratch/live_progress.json
  scripts/remote_vps/watch_progress.sh --run-dir artifacts/sweeps/tinker_20260223_103211_337082
USAGE
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  ssh "${REMOTE_HOST}" "cd ${REMOTE_REPO} && python3 scripts/report_run_progress.py"
  exit 0
fi

quoted_args=()
for arg in "$@"; do
  quoted_args+=("$(printf '%q' "${arg}")")
done

ssh "${REMOTE_HOST}" "cd ${REMOTE_REPO} && python3 scripts/report_run_progress.py ${quoted_args[*]}"
