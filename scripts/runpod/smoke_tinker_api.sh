#!/bin/bash
# Smoke-check a SkyRL Tinker API endpoint.
#
# Usage:
#   bash scripts/runpod/smoke_tinker_api.sh <ip> <port> [timeout_seconds]

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ip> <port> [timeout_seconds]"
  exit 2
fi

IP="$1"
PORT="$2"
TIMEOUT="${3:-120}"
START_TS="$(date +%s)"
URL="http://${IP}:${PORT}/openapi.json"

echo "Waiting for ${URL} (timeout=${TIMEOUT}s) ..."

while true; do
  NOW="$(date +%s)"
  ELAPSED="$((NOW - START_TS))"
  if [[ "${ELAPSED}" -ge "${TIMEOUT}" ]]; then
    echo "ERROR: Timed out waiting for ${URL}"
    exit 1
  fi

  BODY="$(curl -fsS --max-time 5 "${URL}" 2>/dev/null || true)"
  if [[ -n "${BODY}" ]]; then
    if printf '%s' "${BODY}" | python3 -c '
import json
import sys

doc = json.load(sys.stdin)
if not isinstance(doc, dict) or "openapi" not in doc or "paths" not in doc:
    raise SystemExit(1)

print("Smoke OK: OpenAPI reachable and valid JSON")
'; then
      exit 0
    fi
  fi

  sleep 2
done
