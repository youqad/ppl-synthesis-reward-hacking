# Scoring Contract

Paper-track runs optimize a single reward and monitor formal LH with normalization.
Goodhart-era proxy metrics are not part of runtime decision logic.

## Runtime reward API

Hydra contract:

- `train.reward_metric`: `log_marginal_likelihood` | `elpd` | `waic` | `bic`
- `train.reward_data_split`: `train` | `holdout`
- `train.predictive_estimator`: `none` | `psis_loo` | `waic` | `bic_approx`

Compatibility is strict and validated at startup.

## Claim mode and normalization

Hydra contract:

- `train.claim_mode`: `formal_lh` | `predictive_quality` | `legacy_exploratory`
- `train.normalization_method`: `auto` | `exact_binary` | `importance_mc` | `off`
- `train.normalization_delta_scope`: `raw_y_binary` | `raw_y_fixed_x` | `raw_y`
- `train.normalization_epsilon`
- `train.normalization_ci_alpha`
- `train.normalization_mc_samples`
- `train.normalization_min_ess`

For `claim_mode=formal_lh`, normalization must be enabled.

## Reward semantics

### `log_marginal_likelihood`

- Dispatch: `scoring/z_eval.py` -> `experiments/smc_scorer.py`
- Quantity: log marginal likelihood (paper TEval form)
- Failure mode: returns `outcome_code=smc_fail` and sentinel reward

### `elpd` / `waic`

- Dispatch: ArviZ (`az.loo` / `az.waic`) from posterior samples
- Intended for predictive model quality comparisons
- If `pm.Potential` is present, predictive estimators fail closed with `outcome_code=estimator_inapplicable`

### `bic`

- Reward is `-BIC` (maximize orientation)
- Diagnostic field stores raw BIC

## Formal LH signal

Primary paper-track signal is normalization evidence over the configured data interface:

- `exact_binary`: exhaustive enumeration over `{0,1}^d`
- `importance_mc`: importance-MC estimate with CI/ESS diagnostics (raw `y`, fixed `X`)

Canonical run summary keys:

- `paper/frac_non_normalized_final`
- `paper/lh_formal_signal_final`
- `paper/normalization_method`
- `paper/delta_scope`

`paper/lh_formal_signal_final` is currently the final `frac_non_normalized` value.

## Decomposition diagnostics (non-primary)

Decomposition may include `pot_contribution` (sum of potential-term contribution in point-logp decomposition).
This is debug metadata only, not an optimization objective and not a paper claim metric.
