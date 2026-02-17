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

## Top Targets

- Demonstrate reward hacking on micro-benchmarks (constant score, double counting, improper lpdf).
- Establish a clean backend matrix (Stan/PyMC/Pyro) with a shared protocol.
- Show SStan blocks hacked programs while honest baselines pass.
- Scale to regression/classification tasks under optimization pressure.
- Produce reproducible aggregates and plots from experiment outputs.
