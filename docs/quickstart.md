# Quickstart

Use this page to run the repo end-to-end, from a quick pipeline check to a short GRPO run.

## Prerequisites

Install [pixi](https://pixi.sh) (manages Python 3.11 and all dependencies):

```bash
pixi install
```

For API-based training (Path 3), create a `.env` file with your Tinker API key:

```
TINKER_API_KEY=your_key_here
```

Sign up at <https://tinker.thinkingmachines.dev/>. No GPU is needed for the Tinker path; your machine only runs PyMC scoring. The TRL path (Path 4) requires a GPU with 40+ GB VRAM (A100, H100, or L40S).

## Path 1: sanity-check the detection pipeline

No training, no API keys. Takes about two minutes.

```bash
pixi run -e dev pytest tests/integration/test_e2e_exploit_pipeline.py -v
```

This exercises the scorer, normalization checker, heuristic tagger, and PyMC safety filter end-to-end against exploit and honest patterns.

Expected output: all tests pass.

A standalone script covers the same detection stack with different test programs:

```bash
pixi run -e dev python scripts/smoke_test_exploit.py
```

This runs 10 hardcoded exploit families and 2 honest baselines through scoring, normalization, heuristic tagging, taxonomy classification, and the SafePyMC gate.

This is the minimum quickstart path. If it passes, your local scoring stack is working and you can move on to training.

## Path 2: recommended full test suite

Run this if you want broader environment verification before longer runs or cluster jobs. It is recommended, but not required for the quickstart path above.

Validates the complete environment including all dependencies:

```bash
pixi run -e dev pytest tests/ -x
```

Expect a few minutes depending on hardware. Integration tests execute real PyMC models, so they run slower than pure unit tests.

## Path 3: simplest training experiment via Tinker API

This is the shortest practical run for LH emergence. The LLM runs through the Tinker API; your machine only runs PyMC scoring locally.

```bash
set -a && source .env && set +a

pixi run -e dev python scripts/tinker_pymc_grpo.py \
  --n-steps 5 \
  --n-prompts 2 \
  --rollouts-per-prompt 50 \
  --dataset-name bernoulli_vector \
  --dataset-n-features 3 \
  --normalization-interval 1 \
  --normalization-sample-size 10 \
  --output-dir artifacts/tinker_lh_minimal
```

Takes roughly 15-20 minutes. At each step the model (Qwen/Qwen3-4B-Instruct-2507, LoRA rank 32) generates 100 PyMC programs (2 prompts x 50 rollouts). Each program is scored locally with Sequential Monte Carlo (500 particles estimating log marginal likelihood). After each step, 10 programs are sampled and checked for normalization violations by exhaustive enumeration over all 8 binary assignments in {0,1}^3.

Key flags:

- `--n-steps` controls training steps. 5 is enough to see signal.
- `--n-prompts` times `--rollouts-per-prompt` gives total programs per step.
- `--dataset-name bernoulli_vector` with `--dataset-n-features 3` sets the d=3 Bernoulli interface, which enables exact normalization checks.
- `--normalization-interval 1` checks normalization every step.
- `--normalization-sample-size 10` samples 10 programs per check.

Equivalent Hydra command:

```bash
pixi run -e dev python scripts/hydra_train_tinker.py \
  train.n_steps=5 \
  train.n_prompts=2 \
  train.rollouts_per_prompt=50 \
  train/data_interface=raw_bernoulli_d3 \
  train.normalization.normalization_interval=1 \
  train.normalization.normalization_sample_size=10
```

Hydra configs live under `configs/hydra/train/` with groups like `reward_metric`, `data_interface`, `normalization`, and `monitoring_mode`. See [configuration reference](configuration.md) for full details.

## Output structure

A training run produces this layout:

```
artifacts/tinker_lh_minimal/
├── completions.jsonl              # every generated program with its score
├── normalization_metrics.jsonl    # per-step normalization checks
├── judge_metrics.jsonl            # LLM judge evaluations (if monitoring enabled)
├── rubric_evolution.jsonl         # rubric snapshots (if monitoring enabled)
├── trajectory.json                # step-level reward/loss trajectory
└── results.json                   # final summary metrics
```

Each line in `completions.jsonl` is a JSON record:

```json
{
  "batch": 0,
  "index": 42,
  "completion_text": "```python\ndef model(data): ...\n```",
  "reported_reward": -2.3,
  "outcome": "valid",
  "schema_version": 3
}
```

Records also include `prompt`, `code`, `timestamp`, and `metadata` fields. The `outcome` field is `valid` for programs that compiled and scored successfully; failure outcomes are `parse_fail`, `exec_fail`, and `score_fail`.

## Checking for likelihood hacking signal

After training, run offline evaluation scripts to measure normalization violations, gate acceptance, and exploit taxonomy:

```bash
# normalization check with a larger sample
pixi run -e dev python scripts/eval_normalization.py \
  --completions artifacts/tinker_lh_minimal/completions.jsonl \
  --d 3 --sample 50 \
  --output-dir artifacts/tinker_lh_minimal/eval

# SafePyMC gate: accept/reject each program
pixi run -e dev python scripts/eval_safety_gate.py \
  --completions artifacts/tinker_lh_minimal/completions.jsonl \
  --scoring-d 3

# heuristic exploit detection and taxonomy classification
pixi run -e dev python scripts/eval_taxonomy.py \
  --completions artifacts/tinker_lh_minimal/completions.jsonl \
  --output-dir artifacts/tinker_lh_minimal/eval
```

A `frac_non_normalized > 1%` in the normalization summary is a positive signal. The untrained baseline is roughly 0.6% (measured across 7 base/instruct models). In the paper's full runs (800 programs per step, 3 seeds), likelihood hacking signal appeared by step 1-5, with `pm.Potential` prevalence reaching 3-7% of generated programs.

The normalization check works by exhaustive enumeration: for each sampled program, it evaluates log Z_p(y) via SMC for all 2^d binary assignments, computes total mass as `logsumexp` over all assignments, and flags programs where `|log_mass| > epsilon`. Honest programs have mass 1 (log_mass = 0); likelihood-hacking programs have mass != 1.

## Alternative: local GPU via TRL

If you have a GPU with 40+ GB VRAM, the TRL path runs faster per step and needs no API key:

```bash
pixi run -e dev python scripts/trl_reward_hacking.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --n-steps 5 \
  --n-prompts 2 \
  --rollouts-per-prompt 8 \
  --dataset-name bernoulli_vector \
  --dataset-n-features 3 \
  --normalization-interval 1 \
  --normalization-sample-size 10 \
  --output-dir artifacts/trl_lh_minimal
```

GPU dependencies (`torch`, `transformers`, `trl`, `peft`, `accelerate`, `datasets`, `pymc`) are declared in `pyproject.toml`. Install with `pixi install` in the appropriate environment.

Key differences from the Tinker path:

- Fewer rollouts per prompt (GPU memory constraint; `--rollouts-per-prompt 8` is a safe starting point).
- Scoring runs in-process, not via subprocess sandbox.
- Supports KL penalty (`--kl-beta 0.001`) and gradient clipping (`--max-grad-norm 1.0`).
- Produces the same `completions.jsonl` format, so all evaluation scripts work unchanged.

## Next steps

- [Configuration reference](configuration.md): all Hydra groups, top knobs, environment variables
- [SafeStan guide](safestan-guide.md): testing programs against the SafeStan static checker
- [Architecture](architecture.md): module map, training flow, detection stack
- [Scoring methods](scoring.md): reward metrics, normalization algorithm, detection matrix
- [Base-rate exploit examples](base_rate_exploits.md): annotated programs from the untrained-model sweep
