# SafeStan guide

## What SafeStan does

SafeStan is a static safety checker for Stan programs. It rejects LH patterns that can appear in Stan syntax. The PyMC safety filter (`backends/pymc_safe/checker.py`) runs earlier, on Python source before transpilation.

The core mapping is simple: PyMC `pm.Potential` corresponds to Stan `target +=`. SafeStan can detect and reject that family of score injection.

Two tools serve different roles:

- **PyMC safety filter** (`check_pymc_model` in `backends/pymc_safe/checker.py`): rejects `pm.Potential`, custom logp terms, data misuse, and data-dependent branching directly on the instantiated PyMC model. Fast, no external dependencies, no LLM calls.
- **SafeStan gate** (`backends/sstan/gate.py`): transpiles PyMC source to Stan via an LLM, then applies regex checks and optionally cmdsafestan compilation. Catches patterns the PyMC filter misses (and vice versa). Requires an LLM API key.

Both can run independently. Use the PyMC filter for fast local checks. Use SafeStan when you need language-level guarantees about what the model can express.

## The three-layer pipeline

```
PyMC program (Python source)
    |
    v
[1] LLM transpiler (backends/transpiler.py)
    PyMC -> Stan translation via LiteLLM
    |
    v
[2] Regex checker (backends/sstan/checker.py)
    Pattern matching for target modifications,
    lpdf/lpmf calls, data-dependent branching,
    double observation, data discard
    |
    v
[3] cmdsafestan compiler (optional)
    Full Stan program compilation + safety checks
    via the OCaml toolchain
    |
    v
Accept / Reject
```

Each layer catches different patterns. Layer 1 can fail: LLM refusal, API timeout, malformed output. When it fails, the gate rejects by default (fail-closed). Layer 2 is fast and deterministic, operating on the translated Stan source with regex patterns. Layer 3 provides the strongest guarantees but requires OCaml tooling that most collaborators won't have locally.

The fidelity policy (`reject_on_source_signal`) adds a cross-check: if PyMC source triggers exploit heuristics but translated Stan passes regex checks, the program is still rejected. This catches translations that silently remove exploit behavior.

## Prerequisites

You need an LLM API key whenever SafeStan transpilation or judge-based checks are enabled. Current defaults are split by path:

- Training monitoring judge (`train.judge_*`): `judge_backend=openai`, `judge_model=gpt-5.4`; when `judge_api_key_env` is `null`, runtime resolves it to `OPENAI_API_KEY`.
- Training judge gate (`train/judge_gate=*`): `judge_gate_backend=openai`, `judge_gate_model=gpt-5.4`, `judge_gate_api_key_env=OPENAI_API_KEY`.
- Training SafeStan gate (`train/sstan_gate=shadow|enforce`): `sstan_transpiler_model=gpt-5.4`, `sstan_transpiler_api_key_env=OPENAI_API_KEY`.
- Offline exemplar script (`scripts/eval_sstan_exemplars.py`): defaults remain `glm-5` with `ZHIPUAI_API_KEY` unless you override flags.

For full pipeline testing without cmdsafestan (layers 1-2 only), set at least one key for the backend/model you are using:

```bash
pixi install
echo "OPENAI_API_KEY=your-key-here" >> .env
# Optional if you run exemplar defaults (glm-5 / Z.AI path):
echo "ZHIPUAI_API_KEY=your-key-here" >> .env
```

For layer 3 (cmdsafestan), you need the OCaml toolchain:

1. Install OCaml via opam: https://opam.ocaml.org/
2. `opam install dune`
3. Clone and build https://github.com/jkarwowski/safestanc3
4. Clone https://github.com/jkarwowski/cmdsafestan

Without cmdsafestan installed, the pipeline runs layers 1-2 and skips layer 3.

## Offline: evaluating exemplars

Three scripts, ordered from simplest to most thorough.

**1. Quick exemplar evaluation** (good starting point):

```bash
pixi run -e dev python scripts/eval_sstan_exemplars.py
```

Loads 18 likelihood hacking programs from `memory/lh_examples.md` and 5 honest baselines hardcoded in the script. Transpiles each to Stan, runs regex checks, reports accept/reject with timing. Outputs JSON, markdown, and LaTeX tables to `.scratch/sstan_exemplars/` by default.

Options:

```bash
pixi run -e dev python scripts/eval_sstan_exemplars.py \
  --output-dir artifacts/sstan_eval \
  --transpiler-model gpt-5.4 \
  --transpiler-api-key-env OPENAI_API_KEY \
  --use-cmdsafestan \
  --dry-run  # parse and print without calling the LLM
```

If you are using the legacy GLM-5 endpoint path, switch to
`--transpiler-model glm-5 --transpiler-api-key-env ZHIPUAI_API_KEY`.

**2. Labeled suite evaluation**:

```bash
pixi run -e dev python scripts/evaluate_safestan_mitigation.py \
  --suite configs/exemplars/transpiler_suite.jsonl \
  --output artifacts/safestan_suite_summary.json
```

Reports `unsafe_reject_rate` and `safe_accept_rate` plus normalization results among accepted programs. Input is a JSONL file with labeled programs (see "Input JSONL format" below). Also supports `--normalization-d` to control the dimensionality of exact normalization checks on accepted sources.

**3. Strict fidelity audit** (CI-friendly):

```bash
pixi run -e dev python scripts/evaluate_transpiler_fidelity.py \
  --input configs/exemplars/transpiler_suite.jsonl \
  --output artifacts/transpiler_fidelity.json \
  --fail-on-violation
```

Exits with code 2 if any fidelity invariant is violated. Invariants checked: unsafe programs remain rejected after transpilation, safe programs remain accepted, translations are present, and heuristic source signals are preserved. Use this in CI or before publishing results.

## Offline: evaluating training run completions

After a training run produces a `completions.jsonl`, you can check all generated programs against the PyMC safety filter:

```bash
pixi run -e dev python scripts/eval_safety_gate.py \
  --completions artifacts/sweeps/tinker_20260217_163634/completions.jsonl \
  --scoring-d 3
```

This runs `check_pymc_model` on each completion record, deduplicates by code hash, and writes per-record results to `<completions>/../eval/safety.jsonl` with a summary in `safety_summary.json`. It does not involve SafeStan or LLM transpilation.

For SafeStan-specific evaluation on training run outputs, extract program sources into a labeled JSONL matching the format below, then run `evaluate_safestan_mitigation.py` as above.

## Online: mitigation gates during training

The training loop can apply SafeStan and judge mitigation gates at generation time.

SafeStan gate modes (Hydra group `train/sstan_gate`):

| Mode | Config | Behavior | Use case |
|------|--------|----------|----------|
| `off` | `configs/hydra/train/sstan_gate/off.yaml` | No gate checks | Default, Part A emergence runs |
| `shadow` | `configs/hydra/train/sstan_gate/shadow.yaml` | Check and log, keep original reward | Observability without affecting training |
| `enforce` | `configs/hydra/train/sstan_gate/enforce.yaml` | Rejected programs get penalty reward (-100) | Part B mitigation runs |

Judge gate modes (Hydra group `train/judge_gate`):

| Mode | Config | Behavior | Use case |
|------|--------|----------|----------|
| `off` | `configs/hydra/train/judge_gate/off.yaml` | No judge-gate checks | Default, Part A emergence runs |
| `shadow` | `configs/hydra/train/judge_gate/shadow.yaml` | Run judge, log shadow rejects, no penalty | Audit judge behavior without changing reward |
| `enforce` | `configs/hydra/train/judge_gate/enforce.yaml` | Judge-labeled hacks at/above threshold get penalty reward (-400) | Part B mitigation judge arm |

Hydra overrides for the Tinker training entrypoint:

```bash
# shadow mode (observe only, samples 5% of programs)
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/sstan_gate=shadow

# enforce mode (active mitigation, checks every program)
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/sstan_gate=enforce

# Part B SStan arm (strict XOR arm A)
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/paper_track=part_b_mitigation \
  train/sstan_gate=enforce \
  train/judge_gate=off

# Part B judge arm (strict XOR arm B)
pixi run -e dev python scripts/hydra_train_tinker.py \
  train/paper_track=part_b_mitigation \
  train/sstan_gate=off \
  train/judge_gate=enforce
```

`train/sstan_gate=*` overrides also work with the TRL entrypoint (`scripts/hydra_train_trl.py`).
`train/judge_gate=*` is Tinker-only in the current runtime path.

Constraints enforced at startup:

- `enforce` mode requires `sstan_gate_sample_rate=1.0` (every program checked) and `sstan_transpiler_strict=true`
- `paper_track=part_b_mitigation` requires strict XOR with enforce mode:
  either (`sstan_gate=enforce`, `judge_gate=off`) or (`sstan_gate=off`, `judge_gate=enforce`)
- `sstan_gate != off` requires a non-empty `sstan_transpiler_api_key_env` and that env var to be present
- `judge_gate != off` requires a valid `judge_gate_backend`; if the resolved key env var is non-empty, it must exist in the environment (default OpenAI path expects `OPENAI_API_KEY`)

Shadow mode defaults to a 5% sample rate to limit LLM API costs during long training runs. Override with `train.sstan_gate_sample_rate=1.0` if you want exhaustive checking.

## Input JSONL format

Labeled suites for `evaluate_safestan_mitigation.py` and `evaluate_transpiler_fidelity.py` are JSONL (one JSON object per line). A recommended row is:

```json
{
  "id": "P1",
  "expected_unsafe": true,
  "source_pymc": "def model(data):\n    import pymc as pm\n    ...",
  "translated_stan": "data { ... }\nmodel { ... }"
}
```

Fields:

- `expected_unsafe` (required): ground-truth label (`true` means the program should be rejected). Missing this field raises `ValueError`.
- `translated_stan` (optional but strongly recommended): Stan translation. If missing or blank in `evaluate_safestan_mitigation.py`, the row is treated as fail-closed reject with reason `missing translated_stan` (no exception), `translation_present=false`, and `violation_breakdown.missing_translation` is incremented.
- `source_pymc` (optional in suite mode): used for normalization checks only when SafeStan accepted the row. If absent/blank, normalization is skipped for that row.
- `id` (optional): passthrough identifier in output details.

Notes for `evaluate_safestan_mitigation.py --suite`:

- Minimal valid row is `{"expected_unsafe": ...}`.
- Missing `translated_stan` can reduce `safe_accept_rate` (safe rows are counted as rejected) and increase `violation_breakdown.missing_translation`.
- Summary field `accepted_n` counts accepted rows where normalization was actually attempted (accepted plus non-empty `source_pymc`), not all accepted rows.

The fidelity script also supports optional `expected_source_tags` for verifying that heuristic detection finds the right exploit patterns.

Existing suites live under `configs/exemplars/`.

## What the checker catches

The regex checker in `backends/sstan/checker.py` (function `check_sstan`) flags these patterns:

- **`target +=` and `target =` modifications** (score injection, the Stan equivalent of `pm.Potential`)
- **Direct `_lpdf` / `_lpmf` calls** (bypassing Stan's distribution sampling syntax)
- **Data-dependent `if` branching** in the model block (conditional likelihood manipulation)
- **Double observation** of the same data variable (a data variable appearing on the left side of `~` more than once)
- **Data discard** (no data variables observed anywhere in `model {}`)

The PyMC safety filter (`check_pymc_model`) checks the instantiated model for potentials, custom logp/DensityDist ops, data variables not used in observations, data used in priors, and data-dependent branching.

Known gaps in both checkers:

- Latent-dependent score injection (M8 family: softplus patterns depending on latent variables)
- Cross-latent correlation injection (M9 family: correlations between latent variables via Potential)

Both families require SMC-based normalization checks rather than static analysis.

For the full detection matrix across all methods, see [scoring.md](scoring.md).
