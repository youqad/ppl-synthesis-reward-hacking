FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG JOBS=8
ARG OCAML_COMPILER=ocaml-base-compiler.4.14.1

ENV REPO=/workspace/ppl-synthesis-reward-hacking \
    VIRTUAL_ENV=/opt/psrh-venv \
    PATH=/opt/psrh-venv/bin:/root/.opam/default/bin:/root/.pixi/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/workspace/ppl-synthesis-reward-hacking/.cache \
    HF_HOME=/workspace/ppl-synthesis-reward-hacking/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/ppl-synthesis-reward-hacking/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/ppl-synthesis-reward-hacking/.cache/huggingface/transformers \
    TORCH_HOME=/workspace/ppl-synthesis-reward-hacking/.cache/torch \
    TRITON_CACHE_DIR=/workspace/ppl-synthesis-reward-hacking/.cache/triton \
    WANDB_DIR=/workspace/ppl-synthesis-reward-hacking/.cache/wandb \
    WANDB_PROJECT=ppl-synthesis-reward-hacking \
    TMPDIR=/workspace/ppl-synthesis-reward-hacking/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgmp-dev \
        m4 \
        opam \
        pkg-config \
        rsync \
        tmux \
        unzip \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY . "$REPO"
WORKDIR "$REPO"

RUN test -f cmdsafestan/makefile \
    && test -f cmdsafestan/safestan/dune-project \
    && test -f cmdsafestan/safestan/stan/makefile \
    && test -f cmdsafestan/safestan/stan/lib/stan_math/makefile

RUN opam init --disable-sandboxing --yes --shell-setup --bare \
    && opam switch create default "$OCAML_COMPILER" --yes \
    && eval "$(opam env --shell=bash)" \
    && opam install --yes dune core.v0.16.1 menhir.20230608 ppx_deriving.5.2.1 \
        fmt.0.11.0 yojson.2.1.0 cmdliner.2.1.0

RUN eval "$(opam env --shell=bash)" \
    && STANC3=safestan make -C cmdsafestan -j"$JOBS" build

RUN pip install -e ".[arc,dev,runpod,runpod-launcher]" \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
    && pip install -e cmdsafestan

RUN mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
        "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR" \
    && scripts/runpod/validate_stan_linear_image.sh --quick

CMD ["bash", "scripts/runpod/run_grpo_stan_linear.sh"]
