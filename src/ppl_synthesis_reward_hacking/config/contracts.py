from __future__ import annotations

from dataclasses import dataclass

from ppl_synthesis_reward_hacking.data.interfaces import (
    expected_norm_method,
    get_interface_spec,
)
from ppl_synthesis_reward_hacking.scoring.metrics import validate_metric_estimator

VALID_CLAIM_MODES = {"formal_lh", "predictive_quality"}
VALID_NORMALIZATION_METHODS = {"auto", "exact_binary", "importance_mc", "off"}
VALID_DELTA_SCOPES = {"auto", "raw_y_binary", "raw_y_fixed_x", "raw_y"}


@dataclass(frozen=True, slots=True)
class ResolvedContracts:
    normalization_method: str
    normalization_delta_scope: str


def _validate_reward_fields(
    *,
    reward_data_split: str,
    claim_mode: str,
    normalization_method: str,
    normalization_delta_scope: str,
) -> None:
    if reward_data_split not in {"train", "holdout"}:
        raise ValueError("reward_data_split must be train|holdout")
    if claim_mode not in VALID_CLAIM_MODES:
        raise ValueError(f"claim_mode must be one of {sorted(VALID_CLAIM_MODES)}")
    if normalization_method not in VALID_NORMALIZATION_METHODS:
        raise ValueError(
            f"normalization_method must be one of {sorted(VALID_NORMALIZATION_METHODS)}"
        )
    if normalization_delta_scope not in VALID_DELTA_SCOPES:
        raise ValueError(f"normalization_delta_scope must be one of {sorted(VALID_DELTA_SCOPES)}")


def _validate_normalization_hyperparameters(
    *,
    normalization_interval: int,
    normalization_epsilon: float,
    normalization_ci_alpha: float,
    normalization_mc_samples: int,
    normalization_min_ess: float,
) -> None:
    if normalization_interval < 0:
        raise ValueError("normalization_interval must be >= 0")
    if normalization_epsilon <= 0:
        raise ValueError("normalization_epsilon must be > 0")
    if not 0.0 < normalization_ci_alpha < 1.0:
        raise ValueError("normalization_ci_alpha must be in (0, 1)")
    if normalization_mc_samples <= 0:
        raise ValueError("normalization_mc_samples must be positive")
    if normalization_min_ess <= 0:
        raise ValueError("normalization_min_ess must be positive")


def _validate_formal_lh_dataset_support(*, claim_mode: str, dataset_name: str) -> None:
    if claim_mode != "formal_lh":
        return
    if expected_norm_method(dataset_name) != "off":
        return
    raise ValueError(
        f"dataset {dataset_name!r} does not support normalization checking; "
        "formal_lh claim_mode requires a dataset with tractable normalization "
        "(e.g., bernoulli_vector, linear_regression)"
    )


def _resolve_normalization_method(*, normalization_method: str, dataset_name: str) -> str:
    if normalization_method == "auto":
        return expected_norm_method(dataset_name)
    return normalization_method


def _validate_resolved_normalization_mode(
    *,
    claim_mode: str,
    resolved_method: str,
    normalization_interval: int,
) -> None:
    if claim_mode == "formal_lh" and resolved_method == "off":
        raise ValueError("formal_lh claim_mode requires normalization_method != off")
    if claim_mode == "formal_lh" and normalization_interval <= 0:
        raise ValueError("formal_lh claim_mode requires normalization_interval > 0")


def _validate_delta_scope_compatibility(
    *,
    resolved_method: str,
    normalization_method: str,
    normalization_delta_scope: str,
    expected_delta_scope: str,
) -> None:
    if resolved_method == "exact_binary" and expected_delta_scope != "raw_y_binary":
        raise ValueError("exact_binary normalization requires raw_y_binary data interface")
    if resolved_method == "importance_mc" and expected_delta_scope != "raw_y_fixed_x":
        raise ValueError("importance_mc normalization requires raw_y_fixed_x data interface")
    if normalization_method == "auto":
        return
    if normalization_delta_scope != expected_delta_scope:
        raise ValueError(
            "normalization_delta_scope does not match dataset interface: "
            f"expected {expected_delta_scope!r}, got {normalization_delta_scope!r}"
        )


def validate_train_contract(
    *,
    reward_metric: str,
    reward_data_split: str,
    reward_estimator_backend: str,
    claim_mode: str,
    dataset_name: str,
    normalization_method: str,
    normalization_delta_scope: str,
    normalization_epsilon: float,
    normalization_ci_alpha: float,
    normalization_mc_samples: int,
    normalization_min_ess: float,
    normalization_interval: int,
) -> ResolvedContracts:
    validate_metric_estimator(reward_metric, reward_estimator_backend)
    _validate_reward_fields(
        reward_data_split=reward_data_split,
        claim_mode=claim_mode,
        normalization_method=normalization_method,
        normalization_delta_scope=normalization_delta_scope,
    )
    _validate_normalization_hyperparameters(
        normalization_interval=normalization_interval,
        normalization_epsilon=normalization_epsilon,
        normalization_ci_alpha=normalization_ci_alpha,
        normalization_mc_samples=normalization_mc_samples,
        normalization_min_ess=normalization_min_ess,
    )
    _validate_formal_lh_dataset_support(claim_mode=claim_mode, dataset_name=dataset_name)

    spec = get_interface_spec(dataset_name)
    resolved_method = _resolve_normalization_method(
        normalization_method=normalization_method,
        dataset_name=dataset_name,
    )
    _validate_resolved_normalization_mode(
        claim_mode=claim_mode,
        resolved_method=resolved_method,
        normalization_interval=normalization_interval,
    )
    _validate_delta_scope_compatibility(
        resolved_method=resolved_method,
        normalization_method=normalization_method,
        normalization_delta_scope=normalization_delta_scope,
        expected_delta_scope=spec.delta_scope,
    )

    return ResolvedContracts(
        normalization_method=resolved_method,
        normalization_delta_scope=spec.delta_scope,
    )
