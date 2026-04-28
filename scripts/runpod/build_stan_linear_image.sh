#!/bin/bash
# Build the RunPod image for direct-Stan linear-regression GRPO training.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${1:-${PSRH_STAN_LINEAR_IMAGE:-psrh:stan-linear}}"
PUSH="${PUSH:-0}"
JOBS="${JOBS:-8}"

cd "$REPO"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to build the RunPod image." >&2
    exit 1
fi

test -f cmdsafestan/makefile
test -f cmdsafestan/safestan/dune-project
test -f cmdsafestan/safestan/stan/makefile
test -f cmdsafestan/safestan/stan/lib/stan_math/makefile

docker build \
    --build-arg "JOBS=$JOBS" \
    -f docker/runpod-stan-linear.Dockerfile \
    -t "$IMAGE" \
    .

if [ "$PUSH" = "1" ]; then
    docker push "$IMAGE"
fi
