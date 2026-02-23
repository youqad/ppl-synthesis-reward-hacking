#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-contabo}"
REMOTE_REPO="${REMOTE_REPO:-~/ppl-synthesis-reward-hacking}"
SESSION_PREFIX="${SESSION_PREFIX:-psrh-overleaf-}"

ssh "${REMOTE_HOST}" bash -s -- "${REMOTE_REPO}" "${SESSION_PREFIX}" <<'EOF'
set -euo pipefail
repo="$1"
prefix="$2"

echo "== tmux sessions =="
tmux ls 2>/dev/null | grep "^${prefix}" || echo "No matching sessions."
echo

echo "== active training processes =="
ps -eo pid,ppid,etime,pcpu,cmd | grep "scripts/hydra_train_tinker.py" | grep -v grep || true
echo

echo "== recent tmux tails (last 40 lines) =="
for session in $(tmux ls -F "#{session_name}" 2>/dev/null | grep "^${prefix}" || true); do
  echo "----- ${session} -----"
  tmux capture-pane -t "${session}":0.0 -p | tail -n 40 || true
done
echo

echo "== newest sweep artifact dirs =="
cd "$repo"
ls -td artifacts/sweeps/tinker_* 2>/dev/null | head -n 8 || true
EOF
