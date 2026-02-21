from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    classify_outcome,
    score_completion_sandboxed,
)


def _normal_logpdf(y: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    var = np.square(sigma)
    return -0.5 * (np.log(2.0 * math.pi * var) + np.square(y - mean) / var).sum(axis=1)


def check_importance_norm(
    completion_text: str,
    *,
    epsilon: float,
    ci_alpha: float,
    mc_samples: int,
    min_ess: float,
    timeout: int,
    reward_metric: str,
    reward_data_split: str,
    predictive_estimator: str,
    smc_draws: int,
    scoring_data: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    y_obs = np.asarray(scoring_data.get("y"), dtype=np.float64).reshape(-1)
    if y_obs.size == 0:
        return {
            "ok": False,
            "status": "invalid_data",
            "reason": "empty_y",
            "method": "importance_mc",
        }

    rng = np.random.default_rng(seed)
    y_mean = np.full(y_obs.shape[0], float(np.mean(y_obs)), dtype=np.float64)
    y_std = np.full(y_obs.shape[0], float(np.std(y_obs) + 1.0), dtype=np.float64)
    y_samples = rng.normal(y_mean, y_std, size=(mc_samples, y_obs.shape[0]))

    log_q = _normal_logpdf(y_samples, y_mean, y_std)

    log_w: list[float] = []
    n_invalid = 0
    for idx in range(mc_samples):
        local_data = dict(scoring_data)
        y_i = y_samples[idx].astype(np.float64)
        local_data["y"] = y_i
        if reward_data_split == "train":
            local_data["y_train"] = y_i
        else:
            local_data["y_holdout"] = y_i

        z, _oracle, _decomp = score_completion_sandboxed(
            completion_text,
            local_data,
            timeout=timeout,
            reward_metric=reward_metric,
            reward_data_split=reward_data_split,
            predictive_estimator=predictive_estimator,
            smc_draws=smc_draws,
            logp_floor_override=-1e6,
            logp_ceil_override=1e6,
        )
        z_val = float(z)
        if classify_outcome(z_val) != "valid":
            n_invalid += 1
            continue
        log_w.append(z_val - float(log_q[idx]))

    n_valid = len(log_w)
    if n_valid == 0:
        return {
            "ok": False,
            "status": "all_mc_scores_invalid",
            "reason": "no_valid_mc_samples",
            "method": "importance_mc",
            "n_samples": mc_samples,
            "n_valid": 0,
            "n_invalid": n_invalid,
        }

    log_w_arr = np.asarray(log_w, dtype=np.float64)
    max_log_w = float(np.max(log_w_arr))
    w_shifted = np.exp(log_w_arr - max_log_w)
    w_norm = w_shifted / np.sum(w_shifted)
    ess = float(1.0 / np.sum(np.square(w_norm)))

    # Invalid MC draws contribute zero weight; denominator must remain mc_samples.
    log_mass = float(max_log_w + np.log(np.sum(w_shifted)) - np.log(mc_samples))

    # CI on mass (natural scale) from sample variance of importance weights.
    mass_samples = np.zeros(mc_samples, dtype=np.float64)
    mass_samples[:n_valid] = np.exp(log_w_arr)
    mass_mean = float(np.mean(mass_samples))
    if n_valid > 1:
        mass_std = float(np.std(mass_samples, ddof=1))
        mass_se = mass_std / math.sqrt(n_valid)
    else:
        mass_std = 0.0
        mass_se = 0.0

    z_alpha = float(NormalDist().inv_cdf(1.0 - ci_alpha / 2.0))
    ci_low = max(1e-300, mass_mean - z_alpha * mass_se)
    ci_high = max(ci_low, mass_mean + z_alpha * mass_se)
    ci_log_low = float(math.log(ci_low))
    ci_log_high = float(math.log(ci_high))

    ci_excludes_zero = ci_log_high < -epsilon or ci_log_low > epsilon
    confident = ess >= min_ess and n_valid >= max(10, mc_samples // 10)

    return {
        "ok": True,
        "status": "ok" if confident else "low_confidence",
        "method": "importance_mc",
        "epsilon": float(epsilon),
        "ci_alpha": float(ci_alpha),
        "n_samples": mc_samples,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "ess": ess,
        "min_ess": float(min_ess),
        "log_mass": log_mass,
        "mass": mass_mean,
        "mass_std": mass_std,
        "mass_se": mass_se,
        "ci_mass_low": ci_low,
        "ci_mass_high": ci_high,
        "ci_log_mass_low": ci_log_low,
        "ci_log_mass_high": ci_log_high,
        "is_normalized": not (confident and ci_excludes_zero),
        "decision_confident": confident,
    }
