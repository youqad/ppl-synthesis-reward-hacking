from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.config.flattening import (
    DATA_INTERFACE_KEYS,
    MONITORING_MODE_KEYS,
    NORMALIZATION_KEYS,
    SAMPLING_KEYS,
    SCALAR_GROUP_KEYS,
    SSTAN_GATE_KEYS,
    flatten_hydra_train_mapping,
)
from ppl_synthesis_reward_hacking.experiments.config_mapping import (
    DATA_IFACE_KEYS,
    GROUP_ALLOWED_KEYS,
    MONITORING_KEYS,
    NORM_KEYS,
)
from ppl_synthesis_reward_hacking.experiments.config_mapping import (
    SAMPLING_KEYS as MAPPING_SAMPLING_KEYS,
)
from ppl_synthesis_reward_hacking.experiments.config_mapping import (
    SCALAR_GROUP_KEYS as MAPPING_SCALAR_GROUP_KEYS,
)
from ppl_synthesis_reward_hacking.experiments.config_mapping import (
    SSTAN_GATE_KEYS as MAPPING_SSTAN_GATE_KEYS,
)


def test_flatten_hydra_train_mapping_merges_nested_payloads() -> None:
    flattened = flatten_hydra_train_mapping(
        {
            "reward_metric": {"reward_metric": "waic"},
            "reward_data_split": {"selected": "holdout"},
            "reward_estimator_backend": {"reward_estimator_backend": "waic"},
            "claim_mode": {"claim_mode": "formal_lh"},
            "paper_track": {"paper_track": "part_a_emergence"},
            "prompt_policy": {"selected": "neutral"},
            "dataset_name": "linear_regression",
            "normalization_interval": 1,
            "data_interface": {
                "dataset_name": "linear_regression",
                "dataset_n_features": 7,
            },
            "normalization": {
                "normalization_method": "auto",
                "normalization_interval": 1,
                "normalization_sample_size": 40,
            },
            "sampling": {
                "top_p": 0.9,
                "top_k": 64,
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
            "sstan_gate": {
                "sstan_gate_mode": "shadow",
                "sstan_gate_sample_rate": 0.25,
                "sstan_gate_penalty_reward": -123.0,
                "sstan_gate_fidelity_policy": "reject_on_source_signal",
                "sstan_gate_log_stan": "none",
                "use_cmdsafestan": True,
                "cmdsafestan_data": {"y": [1, 0, 1]},
                "cmdsafestan_protect": ["y"],
                "cmdsafestan_cmdstan_root": "/tmp/cmdstan",
                "cmdsafestan_jobs": 8,
                "cmdsafestan_bootstrap": True,
                "cmdsafestan_build_runtime": True,
                "cmdsafestan_force_reinit": False,
                "cmdsafestan_seed": 9876,
                "sstan_transpiler_model": "glm-5",
                "sstan_transpiler_api_key_env": "TEST_KEY",
                "sstan_transpiler_strict": True,
                "sstan_transpiler_timeout_s": 30.0,
            },
            "name": "composition-metadata",
        }
    )
    assert flattened["reward_metric"] == "waic"
    assert flattened["reward_data_split"] == "holdout"
    assert flattened["reward_estimator_backend"] == "waic"
    assert flattened["prompt_policy"] == "neutral"
    assert flattened["monitoring_mode"] == "judge_evolving"
    assert flattened["dataset_name"] == "linear_regression"
    assert flattened["dataset_n_features"] == 7
    assert flattened["normalization_method"] == "auto"
    assert flattened["normalization_interval"] == 1
    assert flattened["normalization_sample_size"] == 40
    assert flattened["top_p"] == 0.9
    assert flattened["top_k"] == 64
    assert flattened["judge_interval"] == 10
    assert flattened["rubric_evolution_interval"] == 20
    assert flattened["judge_sampling_policy"] == "adaptive_cap"
    assert flattened["judge_adaptive_min"] == 4
    assert flattened["judge_adaptive_max"] == 18
    assert flattened["judge_adaptive_target_metric"] == "combined"
    assert flattened["sstan_gate_mode"] == "shadow"
    assert flattened["sstan_gate_sample_rate"] == 0.25
    assert flattened["sstan_gate_penalty_reward"] == -123.0
    assert flattened["sstan_gate_fidelity_policy"] == "reject_on_source_signal"
    assert flattened["use_cmdsafestan"] is True
    assert flattened["cmdsafestan_data"] == {"y": [1, 0, 1]}
    assert flattened["cmdsafestan_protect"] == ["y"]
    assert flattened["cmdsafestan_cmdstan_root"] == "/tmp/cmdstan"
    assert flattened["cmdsafestan_jobs"] == 8
    assert flattened["cmdsafestan_bootstrap"] is True
    assert flattened["cmdsafestan_build_runtime"] is True
    assert flattened["cmdsafestan_force_reinit"] is False
    assert flattened["cmdsafestan_seed"] == 9876
    assert flattened["sstan_transpiler_model"] == "glm-5"
    assert flattened["sstan_transpiler_api_key_env"] == "TEST_KEY"
    assert flattened["sstan_transpiler_strict"] is True
    assert flattened["sstan_transpiler_timeout_s"] == 30.0
    assert "name" not in flattened
    assert "data_interface" not in flattened
    assert "normalization" not in flattened
    assert "sampling" not in flattened
    assert "sstan_gate" not in flattened


@pytest.mark.parametrize("monitoring_mode", [False, True])
def test_flatten_hydra_train_mapping_rejects_bool_monitoring_mode(
    monitoring_mode: bool,
) -> None:
    with pytest.raises(TypeError, match="monitoring_mode must be a string"):
        flatten_hydra_train_mapping({"monitoring_mode": monitoring_mode})


def test_flatten_hydra_train_mapping_rejects_nested_scalar_group_value() -> None:
    with pytest.raises(TypeError, match="reward_metric must be a scalar value"):
        flatten_hydra_train_mapping({"reward_metric": {"reward_metric": {"selected": "waic"}}})


def test_flatten_hydra_train_mapping_rejects_nested_monitoring_mode_value() -> None:
    with pytest.raises(TypeError, match="monitoring_mode must be a scalar value"):
        flatten_hydra_train_mapping({"monitoring_mode": {"monitoring_mode": {"selected": "off"}}})


def test_flatten_hydra_train_mapping_rejects_conflicting_group_and_top_level_values() -> None:
    with pytest.raises(ValueError, match="Conflicting values for dataset_name"):
        flatten_hydra_train_mapping(
            {
                "dataset_name": "linear_regression",
                "data_interface": {"dataset_name": "bernoulli_vector"},
            }
        )


def test_config_mapping_reuses_flattening_group_key_registries() -> None:
    assert DATA_IFACE_KEYS == frozenset(DATA_INTERFACE_KEYS)
    assert NORM_KEYS == frozenset(NORMALIZATION_KEYS)
    assert MAPPING_SAMPLING_KEYS == frozenset(SAMPLING_KEYS)
    assert MONITORING_KEYS == frozenset(("monitoring_mode", *MONITORING_MODE_KEYS))
    assert MAPPING_SSTAN_GATE_KEYS == frozenset(SSTAN_GATE_KEYS)
    assert MAPPING_SCALAR_GROUP_KEYS == SCALAR_GROUP_KEYS
    assert GROUP_ALLOWED_KEYS["data_interface"] == DATA_IFACE_KEYS
    assert GROUP_ALLOWED_KEYS["normalization"] == NORM_KEYS
    assert GROUP_ALLOWED_KEYS["sampling"] == MAPPING_SAMPLING_KEYS
    assert GROUP_ALLOWED_KEYS["monitoring_mode"] == MONITORING_KEYS
    assert GROUP_ALLOWED_KEYS["sstan_gate"] == MAPPING_SSTAN_GATE_KEYS
