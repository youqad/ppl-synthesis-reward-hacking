This repository implements reward-hacking experiments for probabilistic program synthesis (UAI 2026 paper on Likelihood Hacking in PPLs).

## Stack
- Python 3.11+, **pixi** environment manager (not uv/pip); run tests with `pixi run -e dev pytest tests/ -x`
- PyMC 5.x for probabilistic models, SMC for marginal-likelihood estimation
- Hydra (compositional config groups), W&B for experiment tracking

## Hard rules to enforce in every review

**No backward compatibility in runtime code.** Alias fields, dual-read schema adapters, and legacy metric shims in training/evaluation paths are forbidden. If a config or schema is stale, code must fail fast with an explicit error. Delete old paths rather than adding adapters.

**No legacy oracle fields in runtime interfaces.** `oracle_score`, `gap`, `oracle_like_diagnostic`, `legacy_oracle` must not appear in training or scoring pipelines. They belong only in offline analysis scripts.

**Two scoring pipelines must stay in sync:**
- `src/.../experiments/scorer_subprocess.py` (subprocess path, used by Tinker)
- `src/.../experiments/pymc_reward.py` (in-process path, used by TRL)
Any change to reward computation, bounds, or decomposition must be reflected in both.

**Normalization check is the sole formal LH signal.** `check_normalization_small_d()` must not be weakened, bypassed, or replaced by heuristics for paper-track claims.

**Reward bounds must be consistent.** `LOGP_CEIL=20.0` and `LOGP_FLOOR=-1000.0` must be applied uniformly across both scoring paths.

**Hydra config changes must use compositional groups.** No top-level monolithic config overrides. Invalid metric/backend combinations must fail fast at startup.

## Code style
- No trivial comments — code should be self-documenting
- No nested ternary operators; prefer explicit if/else
- No single-use abstractions
- Validate inputs at system boundaries only; trust internal invariants
- Prefer deleting stale code over keeping it with a comment
