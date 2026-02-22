from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.experiments.tinker_training import config_from_mapping


def _base_mapping() -> dict:
    return {
        "reward_metric": "log_marginal_likelihood",
        "reward_data_split": "train",
        "reward_estimator_backend": "smc",
    }


def test_valid_metric_estimator_combos() -> None:
    combos = [
        ("log_marginal_likelihood", "smc"),
        ("elpd", "psis_loo"),
        ("waic", "waic"),
        ("bic", "bic_approx"),
    ]
    for metric, estimator in combos:
        cfg = config_from_mapping(
            {
                **_base_mapping(),
                "reward_metric": metric,
                "reward_estimator_backend": estimator,
            }
        )
        assert cfg.reward_metric == metric
        assert cfg.reward_estimator_backend == estimator


@pytest.mark.parametrize(
    "metric,estimator",
    [
        ("log_marginal_likelihood", "psis_loo"),
        ("elpd", "smc"),
        ("waic", "smc"),
        ("bic", "smc"),
    ],
)
def test_invalid_metric_estimator_combos_fail(metric: str, estimator: str) -> None:
    with pytest.raises(ValueError):
        config_from_mapping(
            {
                **_base_mapping(),
                "reward_metric": metric,
                "reward_estimator_backend": estimator,
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


def test_formal_lh_requires_positive_normalization_interval() -> None:
    with pytest.raises(
        ValueError,
        match="formal_lh claim_mode requires normalization_interval > 0",
    ):
        config_from_mapping(
            {
                **_base_mapping(),
                "claim_mode": "formal_lh",
                "normalization_method": "exact_binary",
                "normalization_interval": 0,
            }
        )


def test_normalization_sample_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="normalization_sample_size must be positive"):
        config_from_mapping({**_base_mapping(), "normalization_sample_size": 0})


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


def test_sampling_group_payload_overrides_top_p_top_k() -> None:
    cfg = config_from_mapping(
        {
            **_base_mapping(),
            "sampling": {
                "top_p": 1.0,
                "top_k": 0,
            },
        }
    )
    assert cfg.top_p == 1.0
    assert cfg.top_k == 0


def test_rejects_unknown_nested_monitoring_mode_key() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.monitoring_mode payload: monitoring_mode\.typo_key",
    ):
        config_from_mapping(
            {
                **_base_mapping(),
                "monitoring_mode": {
                    "monitoring_mode": "off",
                    "typo_key": 1,
                },
            }
        )


def test_rejects_unknown_nested_sstan_gate_key() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.sstan_gate payload: sstan_gate\.typo_key",
    ):
        config_from_mapping(
            {
                **_base_mapping(),
                "sstan_gate": {
                    "sstan_gate_mode": "off",
                    "typo_key": 1,
                },
            }
        )


def test_part_b_requires_sstan_gate_not_off() -> None:
    with pytest.raises(ValueError, match="part_b_mitigation requires sstan_gate_mode"):
        config_from_mapping(
            {
                **_base_mapping(),
                "paper_track": "part_b_mitigation",
                "sstan_gate_mode": "off",
            }
        )


def test_enforce_requires_full_gate_check_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate == 1.0"):
        config_from_mapping(
            {
                **_base_mapping(),
                "sstan_gate_mode": "enforce",
                "sstan_gate_sample_rate": 0.5,
            }
        )


def test_rejects_unknown_scalar_group_wrapper_key() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.reward_metric payload: reward_metric\.typo",
    ):
        config_from_mapping(
            {
                **_base_mapping(),
                "reward_metric": {"typo": "waic"},
            }
        )


def test_unknown_key_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unexpected train config keys"):
        config_from_mapping({**_base_mapping(), "legacy_oracle": 1.0})


def test_invalid_prompt_source_fails() -> None:
    with pytest.raises(ValueError, match="prompt_source must be one of"):
        config_from_mapping({**_base_mapping(), "prompt_source": "invalid"})


def test_invalid_prompt_sampling_fails() -> None:
    with pytest.raises(ValueError, match="prompt_sampling must be one of"):
        config_from_mapping({**_base_mapping(), "prompt_sampling": "invalid"})


def test_invalid_prompt_policy_fails() -> None:
    with pytest.raises(ValueError, match="prompt_policy must be one of"):
        config_from_mapping({**_base_mapping(), "prompt_policy": "anti_hint"})


def test_neutral_prompt_policy_is_accepted() -> None:
    cfg = config_from_mapping({**_base_mapping(), "prompt_policy": "neutral"})
    assert cfg.prompt_policy == "neutral"


def test_invalid_thinking_mode_fails() -> None:
    with pytest.raises(ValueError, match="thinking_mode must be one of"):
        config_from_mapping({**_base_mapping(), "thinking_mode": "maybe"})


def test_invalid_monitoring_mode_fails() -> None:
    with pytest.raises(ValueError, match="monitoring_mode must be off\\|judge_evolving"):
        config_from_mapping(
            {
                **_base_mapping(),
                "monitoring_mode": {"monitoring_mode": "legacy"},
            }
        )


def test_non_mapping_group_payload_rejected() -> None:
    with pytest.raises(ValueError, match="train.normalization payload must be a mapping"):
        config_from_mapping({**_base_mapping(), "normalization": "exact_binary"})


def test_stale_claim_mode_fails_fast() -> None:
    with pytest.raises(ValueError, match="claim_mode must be one of"):
        config_from_mapping({**_base_mapping(), "claim_mode": "legacy_exploratory"})
