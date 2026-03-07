# PPL synthesis reward hacking

<p align="center">
  <a href="https://github.com/youqad/ppl-synthesis-reward-hacking/actions/workflows/ci.yml"><img src="https://github.com/youqad/ppl-synthesis-reward-hacking/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11-3670A0?logo=python&logoColor=ffdd54" alt="Python 3.11">
  <img src="https://img.shields.io/badge/pixi-package%20manager-yellow?logo=anaconda" alt="Pixi">
</p>

Experiments on likelihood hacking in probabilistic-program synthesis (PyMC/Stan), with scoring, normalization checks, and safety gates.

## Setup

```bash
git clone https://github.com/youqad/ppl-synthesis-reward-hacking.git
cd ppl-synthesis-reward-hacking
pixi install
cp .env.example .env
```

Fill in `.env` (see `.env.example` for all keys). Only `TINKER_API_KEY` is needed for basic Tinker training. Judge monitoring additionally needs an LLM key (`OPENAI_API_KEY` by default); the SStan transpiler needs `ZHIPUAI_API_KEY`. W&B keys are needed for sweep-based runs.

Source it before any API-dependent command:

```bash
set -a && source .env && set +a
```

Recommended verification before longer runs:

```bash
pixi run -e dev pytest tests/ -x
```

For a faster smoke check, run `pixi run -e dev pytest tests/integration/test_e2e_exploit_pipeline.py -v` or `pixi run -e dev python scripts/smoke_test_exploit.py` first.

## Entry points

| Entry point | Keys needed | Notes |
|---|---|---|
| `scripts/smoke_test_exploit.py` | none | detection pipeline on hardcoded programs, ~2 min |
| `pixi run -e dev pytest tests/ -x` | none | full test suite; recommended before long runs, optional for quickstart |
| `scripts/eval_normalization.py` | none | offline normalization check on existing runs |
| `scripts/eval_taxonomy.py` | none | heuristic exploit taxonomy on existing runs |
| `scripts/eval_safety_gate.py` | none | SafePyMC gate evaluation |
| `scripts/hydra_train_tinker.py` | `TINKER_API_KEY` | Tinker API training (no GPU); add `train/monitoring_mode=judge_evolving` for LLM judge, `train/sstan_gate=enforce` for SStan gate, `train/judge_gate=enforce` for judge gate |
| `scripts/create_wandb_sweep.py` | + `WANDB_API_KEY` | create sweep, then run agents with `wandb agent` |
| `scripts/hydra_train_trl.py` | GPU (40+ GB VRAM) | local TRL GRPO training |
| `scripts/trl_reward_hacking_stan.py` | GPU + cmdsafestan | TRL with SafeStan checker |
| `scripts/eval_sstan_exemplars.py` | `ZHIPUAI_API_KEY` | SafeStan on exemplars; see [SafeStan guide](docs/safestan-guide.md) |
| `scripts/evaluate_safestan_mitigation.py` | `ZHIPUAI_API_KEY` | SafeStan labeled suite |
| `scripts/evaluate_transpiler_fidelity.py` | `ZHIPUAI_API_KEY` | transpiler fidelity audit |

Judge monitoring and judge gate use `OPENAI_API_KEY` by default. SStan gate and SafeStan evaluation use `ZHIPUAI_API_KEY` for the PyMC-to-Stan transpiler.

## Documentation

| Document | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Three runnable paths from zero to LH signal |
| [Architecture](docs/architecture.md) | Module map, training flow, detection stack |
| [Configuration](docs/configuration.md) | Hydra groups, top knobs, environment variables, recipes |
| [SafeStan guide](docs/safestan-guide.md) | Offline evaluation, online gate, input format |
| [Scoring methods](docs/scoring.md) | Reward metrics, normalization algorithm, detection matrix |
| [Base-rate exploits](docs/base_rate_exploits.md) | Annotated programs from the untrained-model sweep |
| [SafeStan integration provenance](docs/provenance/safestan_integration.md) | Upstream attribution and commit-level provenance |

## Authorship and provenance

The staged Stan integration path in this repository is integrated from upstream
SafeStan/cmdsafestan work. The current opt-in staging is for runtime isolation
and paper-track comparability, not a statement about contribution quality or
ownership. Attribution is tracked in git commit metadata and upstream commit
references. See the full provenance record in
[`docs/provenance/safestan_integration.md`](docs/provenance/safestan_integration.md).

## Smoke test

Verifies the detection pipeline without training or API keys:

```bash
pixi run -e dev python scripts/smoke_test_exploit.py
```

Runs 10 hardcoded exploit families and 2 honest baselines through the scorer, normalization checker, heuristic detector, and SafePyMC gate. No API keys or GPU required; takes about 2 minutes.

## Base-rate exploit sweep

We measured whether untrained (base/instruct) LLMs produce exploit-containing PyMC programs at any nonzero rate. Each model was prompted 200 times (20 prompts x 10 samples) to write a Bayesian coin-flip model. Each completion went through code extraction, string-level exploit detection, execution, structural checks (`check_pymc_model`), scoring, fabricated-data detection, and a data-perturbation test.

### Results (200 completions per model, AWS Bedrock, Feb 2026)

| Model | Valid % | Any exploit % |
|---|---|---|
| Qwen3 Coder 480B | 99.0 | 0.0 |
| Qwen3 Coder 30B | 100.0 | 0.5 |
| Qwen3 32B | 89.0 | 0.0 |
| Llama 3 8B Instruct | 97.5 | 1.5 |
| Amazon Nova Lite | 89.0 | 0.5 |
| Gemma 3 4B | 45.5 | 1.0 |
| Mistral 7B Instruct | 6.0 | 2.0 |

### Reproducing

```bash
# Phase A: quick filter (20 completions per model)
pixi run -e dev python scripts/sweep_base_rate.py \
    --models "bedrock/converse/qwen.qwen3-32b-v1:0,bedrock/meta.llama3-8b-instruct-v1:0" \
    --filter-mode --aws-region eu-west-2 \
    --output-dir artifacts/sweep_filter

# Phase B: full sweep (200 completions per model)
pixi run -e dev python scripts/sweep_base_rate.py \
    --models "bedrock/converse/qwen.qwen3-coder-30b-a3b-v1:0" \
    --n-prompts 20 --samples-per-prompt 10 \
    --aws-region eu-west-2 --call-delay 0.5 \
    --output-dir artifacts/sweep_phase_b
```

### Exploit detection

Beyond the structural checker (`check_pymc_model`), the sweep applies two integrity checks defined in [`evaluation/integrity_checks.py`](src/ppl_synthesis_reward_hacking/evaluation/integrity_checks.py):

- **Fabricated data** (`check_fabricated_data`): compares the total number of observed data points in the compiled model against the expected data dict. Flags models that create extra observed variables to inflate the likelihood.
- **Data-ignored perturbation test** (`check_data_used`): re-executes the model with halved data arrays and compares `point_logps()` at the default initial point. If the log-probabilities are identical, the model does not depend on its observed data.

All flagged programs are documented in [docs/base_rate_exploits.md](docs/base_rate_exploits.md).

## All scripts

```
scripts/
├── hydra_train_tinker.py       # Tinker API training (Hydra)
├── hydra_train_trl.py          # TRL training (Hydra)
├── trl_reward_hacking.py       # TRL training (standalone)
├── trl_reward_hacking_stan.py  # TRL with SafeStan checker
├── tinker_pymc_grpo.py         # Tinker training (standalone, no Hydra)
├── eval_normalization.py       # offline normalization check
├── eval_safety_gate.py         # SafePyMC gate
├── eval_taxonomy.py            # heuristic exploit taxonomy
├── eval_sstan_exemplars.py     # SafeStan exemplar evaluation
├── evaluate_safestan_mitigation.py
├── evaluate_transpiler_fidelity.py
├── smoke_test_exploit.py       # detection pipeline smoke test
├── create_wandb_sweep.py       # W&B sweep creation
├── sweep_base_rate.py          # base-rate exploit sweep
├── analyze_hacking.py          # post-hoc hacking analysis
├── report_run_progress.py      # check active run status
├── generate_paper_figures.py
├── prepare_local_figures.py
└── make_paper_artifacts.py
```
