"""Shared Hydra flattening helpers for training config mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCALAR_GROUP_KEYS = (
    "reward_metric",
    "reward_data_split",
    "reward_estimator_backend",
    "claim_mode",
    "paper_track",
    "prompt_policy",
    "monitoring_mode",
)

DATA_INTERFACE_KEYS = (
    "dataset_name",
    "prompt_source",
    "prompt_jsonl_path",
    "prompt_jsonl_max_examples",
    "prompt_sampling",
    "thinking_mode",
    "dataset_n_features",
    "dataset_noise_sigma",
    "dataset_n_train",
    "dataset_n_holdout",
    "dataset_p_true_alpha",
    "dataset_p_true_beta",
    "dataset_p_true_fixed",
)

NORMALIZATION_KEYS = (
    "normalization_method",
    "normalization_delta_scope",
    "normalization_epsilon",
    "normalization_ci_alpha",
    "normalization_mc_samples",
    "normalization_min_ess",
    "normalization_interval",
    "normalization_sample_size",
)

SAMPLING_KEYS = (
    "top_p",
    "top_k",
)

MONITORING_MODE_KEYS = (
    "judge_interval",
    "rubric_evolution_interval",
    "judge_sampling_policy",
    "judge_adaptive_min",
    "judge_adaptive_max",
    "judge_adaptive_target_metric",
)

SSTAN_GATE_KEYS = (
    "sstan_gate_mode",
    "sstan_gate_sample_rate",
    "sstan_gate_penalty_reward",
    "sstan_gate_fidelity_policy",
    "sstan_gate_log_stan",
    "sstan_transpiler_model",
    "sstan_transpiler_api_key_env",
    "sstan_transpiler_api_base",
    "sstan_transpiler_custom_llm_provider",
    "sstan_transpiler_strict",
    "sstan_transpiler_temperature",
    "sstan_transpiler_max_tokens",
    "sstan_transpiler_timeout_s",
)

GROUP_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "data_interface": DATA_INTERFACE_KEYS,
    "normalization": NORMALIZATION_KEYS,
    "sampling": SAMPLING_KEYS,
    "monitoring_mode": MONITORING_MODE_KEYS,
    "sstan_gate": SSTAN_GATE_KEYS,
}


def _coalesce_nested_value(raw: Any, key: str) -> Any:
    value = raw
    if isinstance(raw, Mapping):
        if key in raw:
            value = raw[key]
            if isinstance(value, Mapping):
                raise TypeError(f"{key} must be a scalar value, got nested mapping")
        elif len(raw) == 1:
            wrapper_key, wrapper_value = next(iter(raw.items()))
            if wrapper_key != "selected":
                return raw
            value = wrapper_value
            if isinstance(value, Mapping):
                raise TypeError(f"{key} must be a scalar value, got nested mapping")
    if key == "monitoring_mode" and isinstance(value, bool):
        raise TypeError(
            "monitoring_mode must be a string ('off' or 'judge_evolving'), "
            f"got bool: {value}"
        )
    return value


def _merge_payload_values(
    flattened: dict[str, Any],
    payload: Mapping[str, Any] | None,
    keys: tuple[str, ...],
) -> None:
    if not isinstance(payload, Mapping):
        return
    for key in keys:
        if key not in flattened and key in payload:
            value = payload[key]
            if isinstance(value, Mapping):
                raise TypeError(f"{key} must be a scalar value, got nested mapping")
            flattened[key] = value


def flatten_hydra_train_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Extract scalar config values from Hydra nested config-group payloads."""
    flattened: dict[str, Any] = dict(mapping)

    for scalar_key in SCALAR_GROUP_KEYS:
        if scalar_key in flattened:
            flattened[scalar_key] = _coalesce_nested_value(flattened[scalar_key], scalar_key)

    for group in ("data_interface", "normalization", "sampling", "sstan_gate"):
        _merge_payload_values(
            flattened,
            flattened.pop(group, None),
            GROUP_PAYLOAD_KEYS[group],
        )
    _merge_payload_values(
        flattened,
        mapping.get("monitoring_mode"),
        MONITORING_MODE_KEYS,
    )

    flattened.pop("name", None)
    return flattened


__all__ = [
    "SCALAR_GROUP_KEYS",
    "DATA_INTERFACE_KEYS",
    "NORMALIZATION_KEYS",
    "SAMPLING_KEYS",
    "MONITORING_MODE_KEYS",
    "SSTAN_GATE_KEYS",
    "GROUP_PAYLOAD_KEYS",
    "flatten_hydra_train_mapping",
]
