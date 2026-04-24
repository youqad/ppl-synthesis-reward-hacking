#!/bin/bash

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CMDSTAN_ROOT="$REPO/cmdsafestan"
JOBS="${JOBS:-$(nproc)}"
OCAML_COMPILER="${OCAML_COMPILER:-ocaml-base-compiler.5.2.0}"

if ! command -v opam >/dev/null 2>&1; then
    echo "opam is required to bootstrap cmdsafestan." >&2
    exit 1
fi

cd "$CMDSTAN_ROOT"

if ! opam switch show >/dev/null 2>&1; then
    opam init --disable-sandboxing --yes --shell-setup
fi

eval "$(opam env --shell=bash)"

if ! command -v ocaml >/dev/null 2>&1; then
    opam switch create default "$OCAML_COMPILER" --yes
    eval "$(opam env --shell=bash)"
fi

opam install --yes dune
git submodule update --init safestan
git -C safestan config -f .gitmodules submodule.stan.url https://github.com/stan-dev/stan.git
git -C safestan submodule sync -- stan
git -C safestan submodule update --init --recursive stan
make -j"$JOBS" build
