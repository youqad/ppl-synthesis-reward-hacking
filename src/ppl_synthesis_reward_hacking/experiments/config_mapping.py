"""Shared helpers for constructing training config dataclasses from Hydra mappings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, TypeVar, cast

from ppl_synthesis_reward_hacking.backends.sstan.gate import (
    SStanFidelityPolicy,
    SStanGateConfig,
    SStanGateMode,
    SStanLogStanPolicy,
)
from ppl_synthesis_reward_hacking.config.contracts import validate_train_contract
from ppl_synthesis_reward_hacking.config.flattening import (
    DATA_INTERFACE_KEYS,
    MONITORING_MODE_KEYS,
    NORMALIZATION_KEYS,
    flatten_hydra_train_mapping,
)
from ppl_synthesis_reward_hacking.config.flattening import (
    JUDGE_GATE_KEYS as FLATTENING_JUDGE_GATE_KEYS,
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
    allowed_keys = {f.name for f in fields(cast(Any, config_type))}
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
JUDGE_GATE_KEYS = frozenset(FLATTENING_JUDGE_GATE_KEYS)
GROUP_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "data_interface": DATA_IFACE_KEYS,
    "normalization": NORM_KEYS,
    "sampling": SAMPLING_KEYS,
    "monitoring_mode": MONITORING_KEYS,
    "sstan_gate": SSTAN_GATE_KEYS,
    "judge_gate": JUDGE_GATE_KEYS,
}
SCALAR_GROUP_KEYS = FLATTENING_SCALAR_GROUP_KEYS

JUDGE_SAMPLING_POLICIES = frozenset({"fixed_cap", "adaptive_cap"})
JUDGE_ADAPTIVE_TARGET_METRICS = frozenset({"heuristic_novelty", "outcome_entropy", "combined"})


def resolve_normalization_contract(config: Any) -> None:
    resolved = validate_train_contract(
        reward_metric=config.reward_metric,
        reward_data_split=config.reward_data_split,
        reward_estimator_backend=config.reward_estimator_backend,
        claim_mode=config.claim_mode,
        dataset_name=config.dataset_name,
        normalization_method=config.normalization_method,
        normalization_delta_scope=config.normalization_delta_scope,
        normalization_epsilon=config.normalization_epsilon,
        normalization_ci_alpha=config.normalization_ci_alpha,
        normalization_mc_samples=config.normalization_mc_samples,
        normalization_min_ess=config.normalization_min_ess,
        normalization_interval=config.normalization_interval,
    )
    config.normalization_method = resolved.normalization_method
    config.normalization_delta_scope = resolved.normalization_delta_scope


def build_sstan_gate_config(config: Any) -> SStanGateConfig:
    return SStanGateConfig(
        mode=cast(SStanGateMode, config.sstan_gate_mode),
        sample_rate=config.sstan_gate_sample_rate,
        penalty_reward=config.sstan_gate_penalty_reward,
        fidelity_policy=cast(SStanFidelityPolicy, config.sstan_gate_fidelity_policy),
        log_stan=cast(SStanLogStanPolicy, config.sstan_gate_log_stan),
        transpiler_model=config.sstan_transpiler_model,
        transpiler_api_key_env=config.sstan_transpiler_api_key_env,
        transpiler_api_base=config.sstan_transpiler_api_base,
        transpiler_custom_llm_provider=config.sstan_transpiler_custom_llm_provider,
        transpiler_strict=config.sstan_transpiler_strict,
        transpiler_temperature=config.sstan_transpiler_temperature,
        transpiler_max_tokens=config.sstan_transpiler_max_tokens,
        transpiler_timeout_s=config.sstan_transpiler_timeout_s,
    )


def validate_judge_sampling_config(config: Any) -> None:
    if config.judge_sampling_policy not in JUDGE_SAMPLING_POLICIES:
        raise ValueError(
            "judge_sampling_policy must be one of "
            f"{sorted(JUDGE_SAMPLING_POLICIES)}, got {config.judge_sampling_policy!r}"
        )
    if config.judge_adaptive_target_metric not in JUDGE_ADAPTIVE_TARGET_METRICS:
        raise ValueError(
            "judge_adaptive_target_metric must be one of "
            f"{sorted(JUDGE_ADAPTIVE_TARGET_METRICS)}, got {config.judge_adaptive_target_metric!r}"
        )
    if config.judge_adaptive_min < 0:
        raise ValueError("judge_adaptive_min must be >= 0")
    if config.judge_adaptive_max < config.judge_adaptive_min:
        raise ValueError("judge_adaptive_max must be >= judge_adaptive_min")
    if config.judge_sampling_policy == "adaptive_cap" and config.judge_adaptive_max <= 0:
        raise ValueError("adaptive judge sampling requires judge_adaptive_max > 0")
