from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.config.flattening import flatten_hydra_train_mapping


def test_flatten_hydra_train_mapping_merges_nested_payloads() -> None:
    flattened = flatten_hydra_train_mapping(
        {
            "reward_metric": {"reward_metric": "waic"},
            "reward_data_split": {"selected": "holdout"},
            "predictive_estimator": {"predictive_estimator": "waic"},
            "claim_mode": {"claim_mode": "formal_lh"},
            "paper_track": {"paper_track": "part_a_emergence"},
            "dataset_name": "linear_regression",
            "normalization_interval": 1,
            "data_interface": {
                "dataset_name": "bernoulli_vector",
                "dataset_n_features": 7,
            },
            "normalization": {
                "normalization_method": "auto",
                "normalization_interval": 5,
                "normalization_sample_size": 40,
            },
            "monitoring_mode": {
                "monitoring_mode": "judge_evolving",
                "judge_interval": 10,
                "rubric_evolution_interval": 20,
                "judge_sampling_policy": "adaptive_cap",
                "judge_adaptive_min": 4,
                "judge_adaptive_max": 18,
                "judge_adaptive_target_metric": "combined",
            },
            "name": "composition-metadata",
        }
    )
    assert flattened["reward_metric"] == "waic"
    assert flattened["reward_data_split"] == "holdout"
    assert flattened["predictive_estimator"] == "waic"
    assert flattened["monitoring_mode"] == "judge_evolving"
    assert flattened["dataset_name"] == "linear_regression"
    assert flattened["dataset_n_features"] == 7
    assert flattened["normalization_method"] == "auto"
    assert flattened["normalization_interval"] == 1
    assert flattened["normalization_sample_size"] == 40
    assert flattened["judge_interval"] == 10
    assert flattened["rubric_evolution_interval"] == 20
    assert flattened["judge_sampling_policy"] == "adaptive_cap"
    assert flattened["judge_adaptive_min"] == 4
    assert flattened["judge_adaptive_max"] == 18
    assert flattened["judge_adaptive_target_metric"] == "combined"
    assert "name" not in flattened
    assert "data_interface" not in flattened
    assert "normalization" not in flattened


@pytest.mark.parametrize(
    ("monitoring_mode", "expected"),
    [(False, "off"), (True, "judge_evolving")],
)
def test_flatten_hydra_train_mapping_handles_bool_monitoring_mode(
    monitoring_mode: bool, expected: str
) -> None:
    flattened = flatten_hydra_train_mapping({"monitoring_mode": monitoring_mode})
    assert flattened["monitoring_mode"] == expected
