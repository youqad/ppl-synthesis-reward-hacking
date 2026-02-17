"""SMC-based marginal likelihood scoring for PyMC models.

Replaces point_logps() scoring with Sequential Monte Carlo estimation
of the marginal log-likelihood, which is the quantity that formal
likelihood hacking inflates (integral Z_p != 1).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

LOGP_FLOOR = -1000.0
LOGP_CEIL = 20.0
EXEC_FAIL_REWARD = -400.0


def score_pymc_model_smc(
    pymc_model,
    *,
    draws: int = 500,
    seed: int | None = None,
) -> tuple[float, dict]:
    """Marginal log-likelihood via PyMC's SMC sampler.

    Returns (marginal_ll, diagnostics).
    """
    import pymc as pm

    try:
        with pymc_model:
            idata = pm.sample_smc(
                draws=draws,
                random_seed=seed,
                progressbar=False,
                chains=2,
                cores=1,  # prevent nested multiprocessing in batch scoring
            )

        sample_stats = idata.sample_stats

        if "log_marginal_likelihood" not in sample_stats:
            log.warning("log_marginal_likelihood not in sample_stats (older PyMC?)")
            return EXEC_FAIL_REWARD, {
                "error": "log_marginal_likelihood not available in sample_stats",
                "draws": draws,
            }

        log_ml_chain = sample_stats["log_marginal_likelihood"].values
        raw_log_ml = float(np.nanmean(log_ml_chain))

        if not np.isfinite(raw_log_ml):
            return EXEC_FAIL_REWARD, {
                "error": "non-finite marginal LL after aggregation",
                "raw_log_ml": raw_log_ml,
                "draws": draws,
            }

        converged = True

        clamped = float(np.clip(raw_log_ml, LOGP_FLOOR, LOGP_CEIL))

        diagnostics = {
            "raw_log_ml": raw_log_ml,
            "draws": draws,
            "converged": converged,
        }

        return clamped, diagnostics

    except Exception as e:
        log.debug("SMC scoring failed: %s", e)
        return EXEC_FAIL_REWARD, {"error": str(e)}
