# Configuration

## Source of truth

> **Tinker config class**: `TinkerGRPOConfig` in `src/ppl_synthesis_reward_hacking/experiments/tinker_training.py`
> **TRL config class**: `TRLRewardHackingConfig` in `scripts/trl_reward_hacking.py`
> **Validation**: `validate_train_contract()` in `src/ppl_synthesis_reward_hacking/config/contracts.py`
> **Hydra groups**: `configs/hydra/train/` directory

These dataclasses define runtime fields and validation hooks. For Hydra entrypoints,
effective defaults come from `configs/hydra/train/tinker.yaml` and
`configs/hydra/train/trl.yaml` (plus selected group options).
This page lists the knobs used most often; check the dataclasses for the full field list.

## Config system overview

Both training paths use Hydra for config composition. The Hydra config root is `configs/hydra/train/`, with two top-level configs:

- `tinker.yaml` for Tinker API training (remote GPU via Thinking Machines API)
- `trl.yaml` for local TRL training (on-device or GPU pod)

Each references group directories (`reward_metric/`, `data_interface/`, `normalization/`, etc.) with selectable options. Override at the CLI:

```bash
# group override (selects a YAML file from the group directory)
pixi run -e dev python scripts/hydra_train_tinker.py train/reward_metric=elpd

# scalar override (sets a single value within the resolved config)
pixi run -e dev python scripts/hydra_train_tinker.py train.n_steps=10
```

At startup, `validate_train_contract()` checks metric/estimator compatibility and claim-mode constraints. Invalid combinations fail fast with explicit error messages.

## Hydra group reference

Every group directory under `configs/hydra/train/` and its available options:

| Group | Options | Default (Tinker) | Default (TRL) | Purpose |
|---|---|---|---|---|
| `reward_metric/` | `log_marginal_likelihood`, `elpd`, `waic`, `bic` | `log_marginal_likelihood` | `log_marginal_likelihood` | Which reward signal to optimize |
| `reward_data_split/` | `train`, `holdout` | `train` | `train` | Score on training or holdout data |
| `reward_estimator_backend/` | `smc`, `psis_loo`, `waic`, `bic_approx` | `smc` | `smc` | Estimation method for the reward |
| `data_interface/` | `raw_bernoulli_d1`, `raw_bernoulli_d3`, `raw_bernoulli_d5_ablation`, `linear_regression`, `jsonl_pool` | `raw_bernoulli_d3` | `linear_regression` | Dataset type and dimensionality |
| `normalization/` | `auto`, `exact_binary`, `importance_mc`, `off` | `auto` | `auto` | Normalization check method |
| `claim_mode/` | `formal_lh`, `predictive_quality` | `formal_lh` | `formal_lh` | Paper claim type (`formal_lh` requires normalization enabled) |
| `paper_track/` | `part_a_emergence`, `part_b_mitigation` | `part_a_emergence` | `part_a_emergence` | Which paper section this run supports |
| `monitoring_mode/` | `off`, `judge_evolving` | `off` | `off` | LLM judge monitoring (observational, not reward) |
| `sampling/` | `paper`, `unrestricted` | `paper` | `unrestricted` | Sampling hyperparameters (`paper`: top_p=0.95, top_k=50; `unrestricted`: top_p=1.0, top_k=0) |
| `sstan_gate/` | `off`, `shadow`, `enforce` | `off` | `off` | SafeStan generation-time gate |
| `judge_gate/` | `off`, `shadow`, `enforce` | `off` | N/A (Tinker only) | LLM judge gate mode |
| `prompt_policy/` | `legacy`, `neutral` | `legacy` | `legacy` | System prompt variant |

### Metric/estimator compatibility

These pairings are enforced by `validate_train_contract()`. Mismatched combinations fail at startup.

| `reward_metric` | Valid `reward_estimator_backend` | Optimization sign |
|---|---|---|
| `log_marginal_likelihood` | `smc` | maximize |
| `elpd` | `psis_loo` | maximize |
| `waic` | `waic` | maximize |
| `bic` | `bic_approx` | maximize (`-BIC`) |

### Cross-group constraints

- `claim_mode=formal_lh` requires `normalization` != `off` and `normalization_interval` > 0
- **Tinker only**: `paper_track=part_b_mitigation` enforces strict XOR with enforce mode:
  either (`sstan_gate=enforce`, `judge_gate=off`) or (`sstan_gate=off`, `judge_gate=enforce`)
- `sstan_gate=enforce` requires `sstan_gate_sample_rate=1.0` and `sstan_transpiler_strict=true`
- `sstan_gate` != `off` requires `sstan_transpiler_api_key_env` to be set and present in the environment
- `judge_gate` != `off` requires a valid judge backend and its resolved API key env var present in the environment
- `claim_mode=formal_lh` requires a dataset whose inferred normalization method is not `off`

## Data interface details

Each data interface defines what the LLM sees in its prompt and which normalization method applies:

| Interface | Description | Features (`d`) | Auto normalization method |
|---|---|---|---|
| `raw_bernoulli_d1` | Single Bernoulli coin flip | 1 | `exact_binary` (2 assignments) |
| `raw_bernoulli_d3` | Bernoulli vector | 3 | `exact_binary` (8 assignments) |
| `raw_bernoulli_d5_ablation` | Bernoulli vector (ablation) | 5 | `exact_binary` (32 assignments) |
| `linear_regression` | Linear regression with Gaussian noise | configurable | `importance_mc` |
| `jsonl_pool` | Prompts loaded from a JSONL file (default payload uses `bernoulli_vector`, d=3) | 3 by default | `exact_binary` by default |

When `normalization_method=auto`, the method is resolved from `dataset_name`:

- `bernoulli_1d`, `bernoulli_vector` -> `exact_binary`
- `linear_regression`, `logistic_regression` -> `importance_mc`
- interfaces with `raw_y` scope -> `off`

For exact normalization checks, use Bernoulli interfaces (`raw_bernoulli_d1`,
`raw_bernoulli_d3`, or `raw_bernoulli_d5_ablation`).

### Normalization guidance

- Prefer `train/normalization=auto` unless you are running a method-specific ablation.
- Use `exact_binary` only with raw binary-`y` interfaces.
- Use `importance_mc` only with fixed-`X` interfaces (for example `linear_regression`).
- In Tinker, `normalization_sample_size` must stay positive even when normalization is disabled (`normalization_method=off`).
  If you intentionally disable normalization, set a positive placeholder (for example `train.normalization.normalization_sample_size=1`).

## Top 10 knobs

Most commonly adjusted parameters across both training paths:

| Parameter | Tinker default | TRL default | What it controls |
|---|---|---|---|
| `n_steps` | 1000 | 1000 | Total training steps |
| `n_prompts` | 5 | 100 | Distinct prompts per step |
| `rollouts_per_prompt` | 160 | 8 | Completions generated per prompt |
| `learning_rate` / `lr` | 1e-5 | 5e-6 | GRPO learning rate |
| `temperature` | 1.28 | 1.28 | Sampling temperature |
| `smc_draws` | 500 | 500 | SMC particles for reward estimation |
| `exec_timeout` | 60 | 30 | Seconds before scoring kills a generated program |
| `normalization_interval` | 1 | 1 | Steps between normalization checks |
| `normalization_sample_size` | 20 | 20 | Programs sampled per normalization check |
| `scoring_workers` | 1 | N/A (Tinker only) | Parallel scoring workers |

Inspect the config dataclasses for the complete field list.

## Environment variables

| Variable | Required for | Description |
|---|---|---|
| `TINKER_API_KEY` | Tinker training | API key from tinker.thinkingmachines.dev |
| `OPENAI_API_KEY` | Default monitoring judge backend (`judge_backend=openai`), default judge gate backend (`judge_gate_backend=openai`), and default SafeStan transpiler in `sstan_gate=shadow/enforce` | OpenAI API key |
| `ZHIPUAI_API_KEY` | Monitoring/judge gate when backend is `zai` | Z.AI API key |
| `ANTHROPIC_API_KEY` | Monitoring/judge gate when backend is `anthropic` | Anthropic API key |
| `OPENROUTER_API_KEY` | Monitoring/judge gate when backend is `openrouter` | OpenRouter API key |
| `WANDB_API_KEY` | W&B logging and sweeps | Weights and Biases API key |
| `WANDB_ENTITY` | W&B logging and sweeps | W&B team or user entity |
| `WANDB_PROJECT` | W&B logging and sweeps | W&B project name (defaults to `ppl-synthesis-reward-hacking`) |

Copy `.env.example` to `.env` and fill in the keys you need. Load before running:

```bash
set -a && source .env && set +a
```

Judge backend default env mapping (used when `judge_api_key_env` is unset):

- `openai` -> `OPENAI_API_KEY`
- `anthropic` -> `ANTHROPIC_API_KEY`
- `zai` -> `ZHIPUAI_API_KEY`
- `openrouter` -> `OPENROUTER_API_KEY`
- `custom` -> no default (set `judge_api_key_env` explicitly)

The same backend mapping applies to judge-gate config when `judge_gate_api_key_env` is empty.

## Runtime validation highlights

These are enforced before/at runtime and are useful when debugging configuration failures:

- Unknown or stale train keys fail fast during config mapping (including nested Hydra group payloads).
- `normalization_sample_size` must be positive in Tinker.
- Normalization checks in Tinker run only when:
  `normalization_interval > 0`, `normalization_method != off`, and `step % normalization_interval == 0`.
- At each checked step, Tinker samples
  `min(normalization_sample_size, n_prompts * rollouts_per_prompt)` programs (without replacement).
- `sstan_gate` and `judge_gate` validate required API key env vars when the gate mode is not `off`.

## Common recipes

### Quick Tinker test (5 steps, d=3 Bernoulli)

```bash
set -a && source .env && set +a
pixi run -e dev python scripts/tinker_pymc_grpo.py \
  --n-steps 5 --n-prompts 2 --rollouts-per-prompt 50 \
  --dataset-name bernoulli_vector --dataset-n-features 3 \
  --normalization-interval 1 --normalization-sample-size 10 \
  --output-dir artifacts/tinker_quick
```

### d=5 Bernoulli with normalization (Hydra)

```bash
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/data_interface=raw_bernoulli_d5_ablation \
  train.normalization.normalization_interval=1
```

### ELPD reward ablation

```bash
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/reward_metric=elpd \
  train/reward_estimator_backend=psis_loo
```

### SafeStan shadow mode (log rejections, don't penalize)

```bash
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/sstan_gate=shadow
```

### SafeStan enforce mode (reject and penalize unsafe programs)

```bash
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/sstan_gate=enforce \
  train/judge_gate=off \
  train/paper_track=part_b_mitigation
```

### Judge gate enforce mode (Part B judge arm)

```bash
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/sstan_gate=off \
  train/judge_gate=enforce \
  train/paper_track=part_b_mitigation
```

### Create and run a W&B sweep

```bash
pixi run -e dev python scripts/create_wandb_sweep.py \
  --config configs/sweeps/wandb/tinker_lh_emergence.yaml \
  --project ppl-synthesis-reward-hacking --entity <your-entity>

pixi run -e dev wandb agent <entity/project/sweep_id>
```

## Discovering config options

List all available Hydra groups:

```bash
ls configs/hydra/train/
```

List options within a specific group:

```bash
ls configs/hydra/train/reward_metric/
```

Print the fully resolved config without launching training:

```bash
pixi run -e dev python scripts/hydra_train_tinker.py --cfg job
```

Inspect a specific group YAML to see what keys it sets:

```bash
cat configs/hydra/train/sstan_gate/enforce.yaml
```
