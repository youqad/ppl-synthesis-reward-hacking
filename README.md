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

Set `TINKER_API_KEY` in `.env` for API-based training. Leave `.env` otherwise unchanged for the smoke test and local TRL training.

Verify:

```bash
pixi run -e dev pytest tests/ -x
```

## Which entry point?

| Your goal | You need | Entry point | Guide |
|---|---|---|---|
| Sanity-check detection pipeline (no training) | PyMC installed | `pixi run -e dev pytest tests/integration/test_e2e_exploit_pipeline.py -v` | [quickstart](docs/quickstart.md) |
| Smoke test the full environment | Nothing extra | `pixi run -e dev pytest tests/ -x` | [quickstart](docs/quickstart.md) |
| Simplest training run (no GPU) | Tinker API key | `scripts/tinker_pymc_grpo.py` | [quickstart](docs/quickstart.md) |
| Training via Hydra configs | Tinker API key | `scripts/hydra_train_tinker.py` | [configuration](docs/configuration.md) |
| Training on local GPU | CUDA + TRL deps | `scripts/hydra_train_trl.py` or `scripts/trl_reward_hacking.py` | [quickstart](docs/quickstart.md) |
| Stan reward training (SafeStan checker + plain-score reward) | cmdsafestan submodule + TRL deps | `scripts/trl_reward_hacking_stan.py` | [SafeStan guide](docs/safestan-guide.md) |
| Test SafeStan on exemplars | LLM API key | `scripts/eval_sstan_exemplars.py` | [SafeStan guide](docs/safestan-guide.md) |
| Evaluate SafeStan on labeled suite | LLM API key | `scripts/evaluate_safestan_mitigation.py` | [SafeStan guide](docs/safestan-guide.md) |
| Train with SafeStan mitigation | SafeStan prereqs | Same training entry + `train/sstan_gate=enforce` | [SafeStan guide](docs/safestan-guide.md) |
| Offline normalization check | PyMC installed | `scripts/eval_normalization.py` | [scoring](docs/scoring.md) |
| Heuristic exploit taxonomy | PyMC installed | `scripts/eval_taxonomy.py` | [scoring](docs/scoring.md) |

## Documentation

| Document | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Three runnable paths from zero to LH signal |
| [Architecture](docs/architecture.md) | Module map, training flow, detection stack |
| [Configuration](docs/configuration.md) | Hydra groups, top knobs, environment variables, recipes |
| [SafeStan guide](docs/safestan-guide.md) | Offline evaluation, online gate, input format |
| [Scoring methods](docs/scoring.md) | Reward metrics, normalization algorithm, detection matrix |
| [Base-rate exploits](docs/base_rate_exploits.md) | Annotated programs from the untrained-model sweep |

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

## Entrypoints

**Training:**
- `scripts/tinker_pymc_grpo.py` -- standalone Tinker GRPO
- `scripts/hydra_train_tinker.py` -- Hydra-configured Tinker training
- `scripts/hydra_train_trl.py` -- Hydra-configured TRL training
- `scripts/trl_reward_hacking.py` -- standalone TRL training
- `scripts/trl_reward_hacking_stan.py` -- TRL Stan training via cmdsafestan (checker metrics + checkpoints)

**Evaluation:**
- `scripts/eval_normalization.py` -- offline normalization check
- `scripts/eval_safety_gate.py` -- SafePyMC gate evaluation
- `scripts/eval_taxonomy.py` -- heuristic exploit taxonomy
- `scripts/eval_sstan_exemplars.py` -- SafeStan exemplar evaluation
- `scripts/evaluate_safestan_mitigation.py` -- SafeStan labeled suite
- `scripts/evaluate_transpiler_fidelity.py` -- strict transpiler fidelity audit
- `scripts/smoke_test_exploit.py` -- detection pipeline smoke test

**Utilities:**
- `scripts/create_wandb_sweep.py` -- create W&B sweep from YAML
- `scripts/sweep_base_rate.py` -- base-rate exploit sweep
