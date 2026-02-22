"""Shared helpers for constructing training config dataclasses from Hydra mappings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, TypeVar

from ppl_synthesis_reward_hacking.config.flattening import (
    DATA_INTERFACE_KEYS,
    MONITORING_MODE_KEYS,
    NORMALIZATION_KEYS,
    flatten_hydra_train_mapping,
)
from ppl_synthesis_reward_hacking.config.flattening import (
    SAMPLING_KEYS as FLATTENING_SAMPLING_KEYS,
)
from ppl_synthesis_reward_hacking.config.flattening import (
    SCALAR_GROUP_KEYS as FLATTENING_SCALAR_GROUP_KEYS,
)
from ppl_synthesis_reward_hacking.config.flattening import (
    SSTAN_GATE_KEYS as FLATTENING_SSTAN_GATE_KEYS,
)

ConfigT = TypeVar("ConfigT")


def build_train_config_from_mapping(
    mapping: Mapping[str, Any],
    config_type: type[ConfigT],
    *,
    error_prefix: str,
) -> ConfigT:
    """Flatten Hydra train mapping and instantiate a dataclass with fail-fast keys."""
    flattened = flatten_hydra_train_mapping(mapping)
    allowed_keys = {f.name for f in fields(config_type)}
    unexpected = sorted(key for key in flattened if key not in allowed_keys)
    if unexpected:
        raise ValueError(f"{error_prefix}: " + ", ".join(unexpected))
    return config_type(**dict(flattened))


def validate_group_payload_keys(
    mapping: Mapping[str, Any],
    *,
    group_allowed_keys: Mapping[str, frozenset[str]],
) -> None:
    """Validate Hydra group payload keys."""
    for group_name, allowed_keys in group_allowed_keys.items():
        if group_name not in mapping:
            continue
        payload = mapping[group_name]
        if not isinstance(payload, Mapping):
            if group_name in SCALAR_GROUP_KEYS:
                continue
            raise ValueError(
                f"train.{group_name} payload must be a mapping, got {type(payload).__name__}"
            )
        unexpected = sorted(key for key in payload if key not in allowed_keys)
        if unexpected:
            raise ValueError(
                f"Unsupported keys in train.{group_name} payload: "
                + ", ".join(f"{group_name}.{key}" for key in unexpected)
            )


def validate_scalar_group_payloads(
    mapping: Mapping[str, Any],
    *,
    scalar_group_keys: tuple[str, ...],
    allowed_wrapper_keys: frozenset[str] = frozenset({"selected"}),
) -> None:
    """Validate scalar Hydra groups while allowing known one-key wrappers."""
    for group_name in scalar_group_keys:
        payload = mapping.get(group_name)
        if not isinstance(payload, Mapping):
            continue
        if group_name == "monitoring_mode":
            unexpected = sorted(
                key
                for key, value in payload.items()
                if key not in MONITORING_KEYS or isinstance(value, Mapping)
            )
            if "monitoring_mode" not in payload:
                unexpected = sorted(set(unexpected) | {str(key) for key in payload.keys()})
            if unexpected:
                raise ValueError(
                    f"Unsupported keys in train.{group_name} payload: "
                    + ", ".join(f"{group_name}.{key}" for key in unexpected)
                )
            continue
        if group_name in payload:
            unexpected = sorted(key for key in payload if key != group_name)
            if isinstance(payload[group_name], Mapping):
                unexpected = [group_name, *unexpected]
        elif len(payload) == 1:
            wrapper_key, wrapper_value = next(iter(payload.items()))
            if wrapper_key not in allowed_wrapper_keys or isinstance(wrapper_value, Mapping):
                unexpected = [str(wrapper_key)]
            else:
                unexpected = []
        else:
            unexpected = sorted(str(key) for key in payload.keys())
        if unexpected:
            raise ValueError(
                f"Unsupported keys in train.{group_name} payload: "
                + ", ".join(f"{group_name}.{key}" for key in unexpected)
            )


DATA_IFACE_KEYS = frozenset(DATA_INTERFACE_KEYS)
NORM_KEYS = frozenset(NORMALIZATION_KEYS)
SAMPLING_KEYS = frozenset(FLATTENING_SAMPLING_KEYS)
MONITORING_KEYS = frozenset(
    {
        "monitoring_mode",
        *MONITORING_MODE_KEYS,
    }
)
SSTAN_GATE_KEYS = frozenset(FLATTENING_SSTAN_GATE_KEYS)
GROUP_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "data_interface": DATA_IFACE_KEYS,
    "normalization": NORM_KEYS,
    "sampling": SAMPLING_KEYS,
    "monitoring_mode": MONITORING_KEYS,
    "sstan_gate": SSTAN_GATE_KEYS,
}
SCALAR_GROUP_KEYS = FLATTENING_SCALAR_GROUP_KEYS
