"""Shared Hydra flattening helpers for training config mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SCALAR_GROUP_KEYS = (
    "reward_metric",
    "reward_data_split",
    "predictive_estimator",
    "claim_mode",
    "paper_track",
    "monitoring_mode",
)

_DATA_INTERFACE_KEYS = (
    "dataset_name",
    "dataset_n_features",
    "dataset_noise_sigma",
    "dataset_n_train",
    "dataset_n_holdout",
    "dataset_p_true_alpha",
    "dataset_p_true_beta",
    "dataset_p_true_fixed",
)

_NORMALIZATION_KEYS = (
    "normalization_method",
    "normalization_delta_scope",
    "normalization_epsilon",
    "normalization_ci_alpha",
    "normalization_mc_samples",
    "normalization_min_ess",
    "normalization_interval",
    "normalization_sample_size",
)

_SAMPLING_KEYS = (
    "top_p",
    "top_k",
)

_MONITORING_MODE_KEYS = (
    "judge_interval",
    "rubric_evolution_interval",
    "judge_sampling_policy",
    "judge_adaptive_min",
    "judge_adaptive_max",
    "judge_adaptive_target_metric",
)


def _coalesce_nested_value(raw: Any, key: str) -> Any:
    if isinstance(raw, Mapping):
        if key in raw:
            return raw[key]
        if len(raw) == 1:
            return next(iter(raw.values()))
    if key == "monitoring_mode" and isinstance(raw, bool):
        return "off" if raw is False else "judge_evolving"
    return raw


def flatten_hydra_train_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Extract scalar config values from Hydra nested config-group payloads."""
    flattened: dict[str, Any] = dict(mapping)

    for scalar_key in _SCALAR_GROUP_KEYS:
        if scalar_key in flattened:
            flattened[scalar_key] = _coalesce_nested_value(flattened[scalar_key], scalar_key)

    data_interface_payload = flattened.pop("data_interface", None)
    if isinstance(data_interface_payload, Mapping):
        for key in _DATA_INTERFACE_KEYS:
            if key not in flattened and key in data_interface_payload:
                flattened[key] = data_interface_payload[key]

    normalization_payload = flattened.pop("normalization", None)
    if isinstance(normalization_payload, Mapping):
        for key in _NORMALIZATION_KEYS:
            if key not in flattened and key in normalization_payload:
                flattened[key] = normalization_payload[key]

    sampling_payload = flattened.pop("sampling", None)
    if isinstance(sampling_payload, Mapping):
        for key in _SAMPLING_KEYS:
            if key not in flattened and key in sampling_payload:
                flattened[key] = sampling_payload[key]

    monitoring_payload = mapping.get("monitoring_mode")
    if isinstance(monitoring_payload, Mapping):
        for key in _MONITORING_MODE_KEYS:
            if key not in flattened and key in monitoring_payload:
                flattened[key] = monitoring_payload[key]

    flattened.pop("name", None)
    return flattened


__all__ = ["flatten_hydra_train_mapping"]
