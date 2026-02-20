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


def test_bad_dataset_fails() -> None:
    with pytest.raises(ValueError):
        config_from_mapping({**_base_mapping(), "dataset_name": "nonexistent_dataset"})


def test_bad_judge_sampling_policy_fails() -> None:
    with pytest.raises(ValueError):
        config_from_mapping({**_base_mapping(), "judge_sampling_policy": "invalid_policy"})


def test_adaptive_judge_sampling_requires_positive_max() -> None:
    with pytest.raises(ValueError):
        config_from_mapping(
            {
                **_base_mapping(),
                "judge_sampling_policy": "adaptive_cap",
                "judge_adaptive_min": 0,
                "judge_adaptive_max": 0,
            }
        )


def test_artifact_sync_interval_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        config_from_mapping({**_base_mapping(), "artifact_sync_interval": -1})


def test_bad_judge_batch_mode_fails() -> None:
    with pytest.raises(ValueError):
        config_from_mapping({**_base_mapping(), "judge_batch_mode": "invalid"})


def test_judge_batch_max_workers_must_be_positive() -> None:
    with pytest.raises(ValueError):
        config_from_mapping({**_base_mapping(), "judge_batch_max_workers": 0})


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-5.2-2025-12-11",
        "openai/gpt-5.2-2025-12-11",
        "gpt-5-2025-12-11",
    ],
)
def test_openai_judge_rejects_dated_gpt5_snapshot(model_name: str) -> None:
    with pytest.raises(ValueError, match="floating model aliases"):
        config_from_mapping(
            {
                **_base_mapping(),
                "judge_backend": "openai",
                "judge_model": model_name,
            }
        )


def test_openai_judge_accepts_floating_gpt5_alias() -> None:
    cfg = config_from_mapping(
        {
            **_base_mapping(),
            "judge_backend": "openai",
            "judge_model": "gpt-5.2",
        }
    )
    assert cfg.judge_model == "gpt-5.2"
