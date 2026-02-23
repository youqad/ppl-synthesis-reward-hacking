# Architecture

Codebase structure, training paths, and detection stack.

## The two training paths

The repo supports two GRPO paths for training an LLM to write PyMC programs. Which one you use depends on hardware.

| | Tinker (API) | TRL (local GPU) |
|---|---|---|
| Entry point | `scripts/hydra_train_tinker.py` | `scripts/hydra_train_trl.py` or `scripts/trl_reward_hacking.py` |
| LLM execution | Remote (Tinker API) | Local (GPU, HuggingFace Transformers) |
| Config class | `TinkerGRPOConfig` in `src/.../experiments/tinker_training.py` | `TRLRewardHackingConfig` in `scripts/trl_reward_hacking.py` |
| GPU needed | No | Yes (40+ GB VRAM) |
| Dependencies | `pixi install` (default env) | Torch, TRL, PEFT, Accelerate, Datasets |
| Scoring | Subprocess (sandboxed) | In-process |

If you do not have a local GPU with 40+ GB VRAM, use Tinker. If you do, both paths work; TRL is usually faster per step because it avoids API round-trips.

Both paths share scoring logic, normalization checks, and detection code. They differ in completion generation and update execution.

Tinker sends prompts to a remote API that handles LoRA weight updates. TRL runs everything locally: the model, the LoRA adapter, the optimizer. The Tinker path wraps each PyMC execution in a subprocess for isolation. The TRL path runs PyMC in-process for speed.

Config resolution uses Hydra in both cases. `configs/hydra/train/` composes grouped config into a nested train mapping, then the runtime calls `config_from_mapping()` (`experiments/tinker_training.py` and `scripts/trl_reward_hacking.py`) to validate payload keys and flatten it into dataclass fields.

Hydra flattening is split between scalar groups and payload groups:

- Scalar groups (`SCALAR_GROUP_KEYS`): `reward_metric`, `reward_data_split`, `reward_estimator_backend`, `claim_mode`, `paper_track`, `prompt_policy`, `monitoring_mode`
- Payload groups (`GROUP_PAYLOAD_KEYS`): `data_interface`, `normalization`, `sampling`, `monitoring_mode`, `sstan_gate`, `judge_gate`

Implementation detail: payload keys are merged only if a top-level scalar key is absent, so explicit top-level values win. `monitoring_mode` is special-cased as both a scalar selector and a payload container (`monitoring_mode`, `judge_interval`, `rubric_evolution_interval`, etc.).

The valid metric/estimator pairs are:

| `reward_metric` | `reward_estimator_backend` |
|---|---|
| `log_marginal_likelihood` | `smc` |
| `elpd` | `psis_loo` |
| `waic` | `waic` |
| `bic` | `bic_approx` |

Standalone TRL (`scripts/trl_reward_hacking.py`) still uses argparse, but it does not bypass validation: argparse values are routed through `config_from_mapping()` and therefore get the same group-key checks, normalization contract resolution, metric/estimator contract checks, and SafeStan gate config validation as Hydra-launched runs.

Current TRL caveat: judge monitoring knobs (`monitoring_mode`, `judge_interval`, rubric fields) are accepted and logged, but the script warns they are metadata-only in this path; no per-step observational judge loop is executed there.

Environment setup for the Tinker path requires a `TINKER_API_KEY` in `.env` (loaded via `set -a && source .env && set +a` before launching). The TRL path needs a CUDA-capable GPU visible to PyTorch. Both paths use `pixi` for dependency management:

```bash
pixi install              # default env (sufficient for Tinker)
pixi install -e dev       # dev env (adds test and analysis deps)
```

## Repository layout

```
src/ppl_synthesis_reward_hacking/   Library code (all reusable logic)
scripts/                            Thin entrypoints (training, analysis, figure generation)
configs/                            Hydra config groups and W&B sweep definitions
tests/                              Unit and integration tests
docs/                               Documentation
artifacts/                          Training output, completions, metrics (gitignored)
data/                               Prompt pools and exemplar suites
memory/                             Persistent catalogs (LH examples, known patterns)
paper_artifacts/                    Generated figures and tables for the paper
```

Scripts are thin wrappers: parse args, call `src/`, handle I/O. Core logic stays in the library package so tests can import it directly.

The `configs/hydra/train/` directory mirrors the Hydra group structure. Each subdirectory (e.g. `reward_metric/`, `data_interface/`) contains one YAML file per option. The two top-level train configs are `tinker.yaml` and `trl.yaml`, which set defaults for all groups. Sweep definitions for W&B live in `configs/sweeps/wandb/`.

## Module map

The `src/ppl_synthesis_reward_hacking/` package has these subpackages:

| Package | Purpose | Key files |
|---|---|---|
| `backends/` | Execution and safety gates | `pymc/code_executor.py` (code extraction, restricted in-process exec with timeout), `pymc_safe/checker.py` (SafePyMC structural filter), `sstan/gate.py` (SafeStan compilation gate), `transpiler.py` (PyMC-to-Stan translation) |
| `config/` | Startup validation and config resolution | `contracts.py` (`validate_train_contract()` enforces metric/estimator compatibility), `flattening.py` (Hydra group resolution) |
| `data/` | Dataset generation and prompt construction | `generators.py` (data sampling), `interfaces.py` (dataset type definitions), `prompts.py` (chat message formatting), `pymc_synthesis_loader.py` (system prompt with allowed PyMC API) |
| `evaluation/` | Detection and offline analysis | `normalization_check.py` (method dispatch: exact binary or importance MC), `heuristics.py` (regex exploit detection), `taxonomy.py` (14 canonical family IDs), `integrity_checks.py` (fabricated data, data usage checks), `oracle.py` (entropy-based oracle bound) |
| `experiments/` | Training loop glue | `tinker_training.py` (Tinker GRPO loop), `pymc_reward.py` (TRL reward function), `scorer_subprocess.py` (sandboxed scoring), `grpo.py` (advantage computation), `scoring_env.py` (reward bounds) |
| `scoring/` | Metric computation and dispatch | `z_eval.py` (`evaluate_model()` dispatches to SMC, PSIS-LOO, WAIC, or BIC), `metrics.py` (metric/estimator validation), `result.py` (scoring result dataclass) |
| `monitoring/` | Observational LLM judge | `llm_judge.py` (judge API calls), `evolving_rubric.py` (rubric mutation over training) |
| `logging/` | W&B and JSONL output | `completions.py` (`CompletionRecord` schema, `CompletionWriter`), `wandb_hook.py` (metric logging), `paper_summary_contract.py` (required W&B keys) |
| `plotting/` | Figure generation | Publication-style plots (seaborn, Times New Roman) |
| `reporting/` | Run analysis | Aggregation and summary utilities |
| `cli/` | CLI commands | `app.py` (main CLI entry), command modules for aggregation, data, plotting, and run management |
| `utils/` | Shared helpers | `hashing.py` (deterministic IDs), `runtime.py` (timeout, process management), `tinker.py` (API client helpers) |

The `backends/` package also contains `judge_gate.py` (LLM judge as a gating backend), `registry.py` (backend dispatch), and `source_parsing.py` (code block extraction from LLM output).

Each `backends/` sub-package exposes a `backend.py` implementing the common protocol in `backends/protocol.py`. The `pymc_safe/` wrapper applies AST checks before execution; `sstan/` transpiles to Stan first.

Most `evaluation/` files are offline analysis utilities over saved completion logs. During training, the key files are `normalization_check.py` (periodic checks) and `heuristics.py` (per-completion tagging).

The `experiments/` package contains the training loop orchestration. `tinker_training.py` is the most complex file in the repo: it implements the full Tinker GRPO training loop including prompt batching, completion collection, scoring dispatch, normalization scheduling, judge calls, and W&B logging. `grpo.py` is a pure function that computes group-relative advantages from a reward vector, used by both training paths.

Reward bounds live in `experiments/scoring_env.py`, not in `scoring/`. This is because the bounds are training-loop concerns (clamping to prevent gradient explosion) rather than metric-computation concerns. The `scoring/` package computes raw metric values; `scoring_env.py` clamps them for use as RL rewards.

The `data/` package handles both dataset generation and prompt engineering. `pymc_synthesis_loader.py` is particularly important: it constructs the system prompt that tells the LLM which PyMC API calls are available. The API surface exposed in this prompt determines which exploit families the LLM can discover. Under `prompt_policy=legacy`, prompts explicitly include `pm.Potential` and `pm.DensityDist`; under `prompt_policy=neutral`, lines containing those explicit hints are stripped while keeping the rest of the template unchanged.

The `logging/` package defines the `CompletionRecord` schema (version 3) used throughout the codebase. Every scored program produces one record containing the code, reward, heuristic tags, normalization result, and metadata. These records serialize to JSONL for offline analysis and to W&B for live monitoring.

## Training flow

A training run proceeds through these stages:

1. **Hydra config resolution.** Hydra entrypoints compose grouped config from `configs/hydra/train/` and pass `cfg.train` to `config_from_mapping()`. This stage validates group payload keys, validates scalar wrappers, and flattens grouped values into dataclass fields.

2. **Contract validation and resolution.** `validate_train_contract()` in `config/contracts.py` validates metric/estimator compatibility, claim-mode constraints, and normalization hyperparameters. It also resolves `normalization_method=auto` to a concrete method and forces `normalization_delta_scope` to the dataset interface's canonical scope.

3. **Dataset generation.** `data/generators.py` produces training data for each step. For Bernoulli interfaces, this samples `p_true` from `Beta(alpha, beta)` and generates binary vectors of dimension `d`. Each step gets fresh data.

4. **Prompt construction.** `data/pymc_synthesis_loader.py` and `data/prompts.py` build system + user prompts. `prompt_policy=legacy` keeps explicit exploit-primitive hints (`pm.Potential`, `pm.DensityDist`); `prompt_policy=neutral` removes those hint lines via substring-based neutralization.

5. **LLM sampling.** The Tinker API or local HuggingFace model generates multiple completions per prompt. Each completion should contain a `def model(data):` function returning a `pm.Model`.

6. **Code extraction.** `backends/pymc/code_executor.py` parses completions, strips markdown fences, and extracts the model function. Malformed completions are assigned a failure penalty.

7. **Scoring.** The extracted code runs (subprocess for Tinker, in-process for TRL reward function) with split-selected data. `scoring/z_eval.py::evaluate_model()` dispatches by `reward_metric` (`log_marginal_likelihood`, `elpd`, `waic`, `bic`) with contract-validated estimator backend pairing.

8. **SafeStan gate (optional runtime mitigation).** If enabled, `backends/sstan/gate.py` can shadow or enforce penalties before GRPO updates.

9. **Normalization check (optional monitoring).** `evaluation/normalization_check.py` dispatches by method: `exact_binary` (full `2^d` enumeration, guarded by `d <= 10`) or `importance_mc` (Monte Carlo with ESS/confidence logic). Tinker currently evaluates normalization with `log_marginal_likelihood` + `smc` regardless of training reward metric; TRL normalization uses the configured reward metric fields.

10. **Heuristic tagging.** `evaluation/heuristics.py` runs regex patterns against each program, and `evaluation/taxonomy.py` canonicalizes to exploit-family IDs. These tags are logged alongside rewards and do not directly affect reward.

11. **Observational judge monitoring (optional).** In Tinker, when `monitoring_mode=judge_evolving` and `judge_interval > 0`, `monitoring/llm_judge.py` classifies sampled programs and `monitoring/evolving_rubric.py` can evolve rubric criteria. This path is monitoring-only. In standalone TRL, monitoring fields are currently metadata-only (logged warning, no active judge loop).

12. **Judge gate runtime penalty (optional, separate from monitoring).** `backends/judge_gate.py` applies `off|shadow|enforce` gating to valid completions and can rewrite reward to a penalty in enforce mode. In Tinker this runs before GRPO update and affects training signal. TRL reward plumbing supports this gate in `experiments/pymc_reward.py`, but `scripts/trl_reward_hacking.py` does not currently expose judge-gate config knobs.

13. **GRPO update + logging.** `experiments/grpo.py` computes group-relative advantages. Then records, normalization summaries, heuristic family rates, and optional judge outputs are logged to JSONL/W&B; paper summary keys are filled via `paper_summary_contract.py`.

Steps 5 through 10 run per completion; gate/monitoring stages (8, 11, 12) run conditionally by config. A single training step processes `n_prompts * rollouts_per_prompt` programs (typically 5 * 160 = 800 for Tinker default).

The scoring step is the bottleneck. Each SMC evaluation compiles a PyMC model, runs posterior draws, and computes a marginal likelihood estimate. In Tinker, parallel pre-scoring is enabled only when `scoring_workers > 1`; otherwise scoring is sequential. In TRL reward code (`experiments/pymc_reward.py`), scoring is currently sequential.

Failed programs (syntax errors, timeouts, PyMC compilation failures) map to sentinel penalties and remain in the GRPO batch, so malformed outputs still contribute negative training signal.

Timeout defaults differ by entrypoint:

| Entrypoint | Scoring exec timeout default | Other runtime timeouts |
|---|---|---|
| `scripts/hydra_train_tinker.py` / `scripts/tinker_pymc_grpo.py` | `exec_timeout=60` | `sample_timeout_s=1800`, `train_op_timeout_s=900` |
| `scripts/hydra_train_trl.py` / `scripts/trl_reward_hacking.py` | `exec_timeout=30` | no sampling/train-op timeout knobs in this script |
| `experiments/scorer_subprocess.py` direct default | `DEFAULT_EXEC_TIMEOUT=60` | subprocess call timeout uses same value |

Note: the low-level in-process executor (`backends/pymc/code_executor.py`) has `timeout=30` as its function default, but training paths normally pass explicit `exec_timeout` values above.

## Detection and filter stack

The main target is likelihood hacking (LH): a generated program defines an improper marginal data density (does not integrate to 1 over data space). This is different from Goodhart-style reward misspecification (for example prior concentration), which still produces proper densities.

Eight layers detect or mitigate LH. They can be enabled independently.

1. **Extraction filter** (`backends/pymc/code_executor.py`). Parses code from completions and rejects banned `initval`/`testval`/`initial_point` patterns.

2. **SafePyMC gate** (`backends/pymc_safe/checker.py`). Structural AST checks on extracted Python code.

3. **Reward scorer** (`scoring/z_eval.py`). Computes runtime reward by configured metric/estimator backend.

4. **Normalization checker** (`evaluation/normalization_check.py`). Verifies whether induced data density is normalized (`exact_binary` or `importance_mc`, depending on resolved method).

5. **Heuristic taxonomy tagger** (`evaluation/heuristics.py` + `taxonomy.py`). Maps raw exploit signals to canonical family IDs.

6. **Judge gate (runtime penalty path)** (`backends/judge_gate.py`). Optional `off|shadow|enforce` gating that can rewrite reward before update.

7. **LLM judge monitor (observational path)** (`monitoring/llm_judge.py` + `evolving_rubric.py`). Optional monitoring/rubric-evolution path; does not directly alter reward.

8. **SafeStan gate** (`backends/sstan/gate.py` + `transpiler.py`). Optional transpile-and-check mitigation with `off|shadow|enforce`.

Not every layer runs on every completion. Scoring always runs. Normalization runs on sampled subsets at interval. Judge monitoring runs only when `monitoring_mode` and `judge_interval` enable it. Judge gate runs only when `judge_gate_mode != off`. SafeStan runs only when `sstan_gate_mode != off`.

The layers are ordered from cheapest to most expensive. Extraction and heuristic tagging are sub-millisecond. SMC scoring takes tens of milliseconds per program; normalization checking multiplies that by `2^d` evaluations. The LLM judge (API call per batch) and SafeStan (LLM call plus compiler invocation) are the most expensive by an order of magnitude.

The 14 canonical exploit family IDs are:

- `lh_potential_constant_offset`
- `lh_potential_tempered_likelihood`
- `lh_double_observation_same_data`
- `lh_data_dependent_score_injection`
- `lh_observed_constant_anchor`
- `lh_custom_logp_missing_normalizer`
- `lh_improper_mixture_non_normalized_mass`
- `lh_cross_validation_reuse_overlap`
- `lh_pseudocount_fixed_observation`
- `lh_pure_score_injection_baseline`
- `lh_sufficient_statistic_observation`
- `lh_unnormalized_measure_density_observe`
- `lh_data_discard`
- `evaluator_hack_initval`

Runtime LH batch channels in Tinker report only IDs prefixed with `lh_` (`evaluator_hack_initval` is tracked but excluded from those `lh/batch/family/*` aggregates).

`tests/integration/test_e2e_exploit_pipeline.py` is the executable spec for which detection layer catches which exploit family.

## Two scoring pipelines

The repo has two runtime scoring implementations:

1. **`experiments/scorer_subprocess.py`**: runs PyMC code in a subprocess with OS-level timeout. Used by the Tinker training path. The subprocess boundary provides isolation against infinite loops, memory leaks, and segfaults in user-generated code.

2. **`experiments/pymc_reward.py`**: runs PyMC code in-process. Used by the TRL training path. Faster (no process spawn overhead) but offers less isolation. A badly behaved program can crash the training process.

Both call `scoring/z_eval.py::evaluate_model()` for metric computation. Both apply reward clamping via `experiments/scoring_env.py` (default floor/ceiling controlled there and overridable via `PSRH_LOGP_FLOOR` / `PSRH_LOGP_CEIL`). Shared metric dispatch logic should remain centralized in `scoring/z_eval.py`.

Tinker can optionally parallelize subprocess scoring (`scoring_workers > 1`) through `experiments/parallel_scorer.py` and falls back to sequential if pool creation fails. TRL reward scoring (`_score_batch` in `pymc_reward.py`) is currently sequential.

When modifying scoring logic, prefer changes in `scoring/z_eval.py` plus contract checks/tests so both paths keep consistent metric semantics.

## Test structure

The test suite is split into unit tests (fast, no external dependencies) and integration tests (require PyMC, run real model compilation and sampling).

```bash
# unit tests only
pixi run -e dev pytest tests/unit/ -x

# integration tests (require PyMC)
pixi run -e dev pytest tests/integration/ -m pymc

# full suite
pixi run -e dev pytest tests/ -x
```

High-signal integration tests:

- `test_e2e_exploit_pipeline.py` (28 tests): the canonical specification of which detection layer catches which exploit family. If you add a new exploit type or detection method, add a test here.
- `test_e2e_reward_fn.py` (13 tests): reward function factory, multi-step trajectory validation, JSONL serialization round-trips.
- `test_e2e_safety_integrity.py` (12 tests): SafePyMC gate and integrity checks against 6 model templates covering honest and exploit programs.
- `test_e2e_d3_normalization.py` (9 tests): normalization checks for d=3 Bernoulli programs, including honest models, potential-injected models, sufficient-statistic collapses, and double-observation patterns.
- `test_e2e_analysis_pipeline.py` (7 tests): end-to-end validation of `analyze_step` and `analyze_run` aggregation over completion logs.

Unit tests cover individual modules in isolation. Notable unit test files include `test_sweep_configs.py` (validates all sweep YAML files parse correctly), `test_tinker_config_validation.py` (contract validation edge cases), and `test_pymc_reward_sstan_gate.py` (SafeStan gate integration with the reward pipeline).

All tests should be run through `pixi run -e dev` to ensure the correct Python environment with PyMC and all dependencies available. Running tests with a different interpreter can produce false failures from missing packages.
