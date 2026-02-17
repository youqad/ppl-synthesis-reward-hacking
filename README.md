# PPL Synthesis Reward Hacking

<p align="center">
  <a href="https://github.com/youqad/ppl-synthesis-reward-hacking/actions/workflows/ci.yml"><img src="https://github.com/youqad/ppl-synthesis-reward-hacking/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11-3670A0?logo=python&logoColor=ffdd54" alt="Python 3.11">
  <img src="https://img.shields.io/badge/pixi-package%20manager-yellow?logo=anaconda" alt="Pixi">
</p>

Experiments for reward hacking in probabilistic programming synthesis and automated science workflows.

## Quickstart

```bash
pixi install
pixi run -e dev pytest
pixi run psrh --help
```

## Hydra + W&B Sweeps

Hydra entrypoints are provided for both training paths:

- `scripts/hydra_train_tinker.py` (local/CPU launcher by default)
- `scripts/hydra_train_trl.py` (SLURM launcher by default)

Create a sweep and run agents:

```bash
wandb sweep configs/sweeps/wandb/tinker_gap.yaml
python scripts/run_wandb_agent.py --sweep-id <entity/project/sweep_id> --count 20
```

Programmatic sweep creation helper:

```bash
python scripts/create_wandb_sweep.py --config configs/sweeps/wandb/tinker_gap.yaml
```

For TRL sweeps on SLURM:

```bash
wandb sweep configs/sweeps/wandb/trl_gap.yaml
python scripts/run_wandb_agent.py --sweep-id <entity/project/sweep_id> --count 10
```

Hydra launcher profiles now include:

- `launcher=local` (VPS/local)
- `launcher=slurm` (generic SLURM)
- `launcher=a100` (A100 partition profile)
- `launcher=h100` (H100 partition profile)
- `launcher=cpu-debug` (CPU-only debug profile)

Primary optimization metric is `sweep/final_gap_mean` (maximize).

## Layout

- `src/ppl_synthesis_reward_hacking/` - Core experiment code
- `tests/` - Test suite
- `configs/` - Experiment configs (Hydra/YAML)
- `data/` - Vendored datasets
- `scripts/` - Thin experiment runners
- `docs/` - Documentation

## Datasets

SynthStats JSONL is vendored under:

- `data/synthstats/synthstats_{train,val,test}.jsonl`

Loader utility:

```python
from ppl_synthesis_reward_hacking.data.jsonl_loader import load_jsonl_dataset

examples = load_jsonl_dataset("data/synthstats/synthstats_val.jsonl", max_examples=3)
```

Config entry:

- `configs/data/synthstats.yaml`

## Base-Rate Exploit Sweep

We measured whether untrained (base/instruct) LLMs produce exploit-containing PyMC programs at any nonzero rate. Each model was prompted 200 times (20 prompts x 10 samples) to write a Bayesian coin-flip model, then each completion was run through a 7-step analysis pipeline: code extraction, string-level exploit detection, execution, structural checks (`check_pymc_model`), scoring, fabricated-data detection, and a data-perturbation test.

### Results (200 completions per model, AWS Bedrock, Feb 2025)

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
python3 scripts/sweep_base_rate.py \
    --models "bedrock/converse/qwen.qwen3-32b-v1:0,bedrock/meta.llama3-8b-instruct-v1:0" \
    --filter-mode --aws-region eu-west-2 \
    --output-dir artifacts/sweep_filter

# Phase B: full sweep (200 completions per model)
python3 scripts/sweep_base_rate.py \
    --models "bedrock/converse/qwen.qwen3-coder-30b-a3b-v1:0" \
    --n-prompts 20 --samples-per-prompt 10 \
    --aws-region eu-west-2 --call-delay 0.5 \
    --output-dir artifacts/sweep_phase_b
```

Raw results are in `artifacts/sweep_phase_b/`, `artifacts/sweep_phase_b_mistral/`, and `artifacts/sweep_phase_b_gemma/` as JSONL (one record per completion) and per-model JSON summaries.

All flagged programs are documented with annotations in [docs/base_rate_exploits.md](docs/base_rate_exploits.md).

### Exploit detection

Beyond the structural checker (`check_pymc_model`), the sweep applies two integrity checks defined in [`evaluation/integrity_checks.py`](src/ppl_synthesis_reward_hacking/evaluation/integrity_checks.py):

- **Fabricated data** (`check_fabricated_data`): compares the total number of observed data points in the compiled model against the expected data dict. Flags models that create extra observed variables to inflate the likelihood.
- **Data-ignored perturbation test** (`check_data_used`): re-executes the model with halved data arrays and compares `point_logps()` at the default initial point. If the log-probabilities are identical, the model does not depend on its observed data.

## Top Targets

- Demonstrate reward hacking on micro-benchmarks (constant score, double counting, improper lpdf).
- Establish a clean backend matrix (Stan/PyMC/Pyro) with a shared protocol.
- Show SStan blocks hacked programs while honest baselines pass.
- Scale to regression/classification tasks under optimization pressure.
- Produce reproducible aggregates and plots from experiment outputs.
