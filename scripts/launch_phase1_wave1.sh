#!/usr/bin/env bash
# Phase 1 Wave 1: smoke-test runs (1 per condition)
# P1-s0  Rich + no_think + judge ON
# P2-s0  Neutral + no_think + judge OFF
# P4-s0  Rich + think + judge OFF
# P5-s0  Sparse + no_think + judge OFF
#
# All runs: d=5, n_train=20, smc_draws=500, exec_timeout=60
# Usage: bash scripts/launch_phase1_wave1.sh

set -u

BLAS_ENVS="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
REPO_DIR="$HOME/ppl-synthesis-reward-hacking"

# shared Hydra overrides, expanded via ${SHARED_ARGS[*]}
SHARED_ARGS=(
  "launcher=local"
  "train/paper_track=part_a_emergence"
  "train/claim_mode=formal_lh"
  "train/normalization=exact_binary"
  "train.normalization.normalization_interval=1"
  "train.normalization.normalization_sample_size=20"
  "train/sstan_gate=off"
  "train/judge_gate=off"
  "train.n_prompts=5"
  "train.rollouts_per_prompt=160"
  "train.smc_draws=500"
  "train.scoring_workers=2"
  "train.exec_timeout=60"
  "train.temperature=1.28"
  "train.artifact_sync_min_steps=0"
  "train.artifact_sync_interval=5"
)

# --- P1-s0: Rich + no_think + judge ON ---
tmux new-session -d -s "psrh-P1-s0" bash -c "
  ${BLAS_ENVS}
  cd ${REPO_DIR} && set -a && source .env && set +a
  pixi run -e dev python scripts/hydra_train_tinker.py \\
    ${SHARED_ARGS[*]} \\
    train/data_interface=raw_bernoulli_d5 \\
    train/prompt_policy=rich \\
    train/monitoring_mode=judge_evolving \\
    train.monitoring_mode.judge_interval=1 \\
    train.monitoring_mode.rubric_evolution_interval=5 \\
    train.judge_backend=zai \\
    train.judge_model=anthropic/glm-5 \\
    train.judge_sampling_policy=fixed_cap \\
    train.judge_sample_size=20 \\
    train.judge_batch_mode=auto \\
    train.judge_batch_max_workers=3 \\
    train.track_lh_every_batch=true \\
    train.n_steps=40 \\
    train.scoring_seed_base=0
  exec bash
"
echo "Launched P1-s0 (Rich, no_think, judge ON)"

# --- P2-s0: Neutral + no_think + judge OFF ---
tmux new-session -d -s "psrh-P2-s0" bash -c "
  ${BLAS_ENVS}
  cd ${REPO_DIR} && set -a && source .env && set +a
  pixi run -e dev python scripts/hydra_train_tinker.py \\
    ${SHARED_ARGS[*]} \\
    train/data_interface=raw_bernoulli_d5 \\
    train/prompt_policy=neutral \\
    train/monitoring_mode=off \\
    train.n_steps=30 \\
    train.scoring_seed_base=0
  exec bash
"
echo "Launched P2-s0 (Neutral, no_think, judge OFF)"

# --- P4-s0: Rich + think + judge OFF ---
tmux new-session -d -s "psrh-P4-s0" bash -c "
  ${BLAS_ENVS}
  cd ${REPO_DIR} && set -a && source .env && set +a
  pixi run -e dev python scripts/hydra_train_tinker.py \\
    ${SHARED_ARGS[*]} \\
    train/data_interface=raw_bernoulli_d5_ablation \\
    train/prompt_policy=rich \\
    train/monitoring_mode=off \\
    train.n_steps=30 \\
    train.scoring_seed_base=0
  exec bash
"
echo "Launched P4-s0 (Rich, think, judge OFF)"

# --- P5-s0: Sparse (legacy) + no_think + judge OFF ---
tmux new-session -d -s "psrh-P5-s0" bash -c "
  ${BLAS_ENVS}
  cd ${REPO_DIR} && set -a && source .env && set +a
  pixi run -e dev python scripts/hydra_train_tinker.py \\
    ${SHARED_ARGS[*]} \\
    train/data_interface=raw_bernoulli_d5 \\
    train/prompt_policy=legacy \\
    train/monitoring_mode=off \\
    train.n_steps=20 \\
    train.scoring_seed_base=0
  exec bash
"
echo "Launched P5-s0 (Sparse/legacy, no_think, judge OFF)"

echo ""
echo "Wave 1 launched. Monitor with:"
echo "  tmux ls | grep psrh-P"
