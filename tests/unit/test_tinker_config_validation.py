from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.experiments.tinker_training import config_from_mapping


def _base_mapping() -> dict:
    return {
        "reward_metric": "log_marginal_likelihood",
        "reward_data_split": "train",
        "predictive_estimator": "none",
    }


def test_valid_metric_estimator_combos() -> None:
    combos = [
        ("log_marginal_likelihood", "none"),
        ("elpd", "psis_loo"),
        ("waic", "waic"),
        ("bic", "bic_approx"),
    ]
    for metric, estimator in combos:
        cfg = config_from_mapping(
            {
                **_base_mapping(),
                "reward_metric": metric,
                "predictive_estimator": estimator,
            }
        )
        assert cfg.reward_metric == metric
        assert cfg.predictive_estimator == estimator


@pytest.mark.parametrize(
    "metric,estimator",
    [
        ("log_marginal_likelihood", "psis_loo"),
        ("elpd", "none"),
        ("waic", "none"),
        ("bic", "none"),
    ],
)
def test_invalid_metric_estimator_combos_fail(metric: str, estimator: str) -> None:
    with pytest.raises(ValueError):
        config_from_mapping(
            {
                **_base_mapping(),
                "reward_metric": metric,
                "predictive_estimator": estimator,
            }
        )


def test_invalid_reward_split_fails() -> None:
    with pytest.raises(ValueError):
        config_from_mapping({**_base_mapping(), "reward_data_split": "dev"})


def test_invalid_p_true_mode_fails() -> None:
    with pytest.raises(ValueError):
        config_from_mapping({**_base_mapping(), "p_true_mode": "random"})
