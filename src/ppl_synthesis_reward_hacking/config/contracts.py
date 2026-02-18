from __future__ import annotations

from dataclasses import dataclass

from ppl_synthesis_reward_hacking.data.interfaces import (
    expected_norm_method,
    get_interface_spec,
)
from ppl_synthesis_reward_hacking.scoring.metrics import validate_metric_estimator

VALID_CLAIM_MODES = {"formal_lh", "predictive_quality", "legacy_exploratory"}
VALID_NORMALIZATION_METHODS = {"auto", "exact_binary", "importance_mc", "off"}
VALID_DELTA_SCOPES = {"raw_y_binary", "raw_y_fixed_x", "raw_y"}


@dataclass(frozen=True, slots=True)
class ResolvedContracts:
    normalization_method: str
    normalization_delta_scope: str


def validate_train_contract(
    *,
    reward_metric: str,
    reward_data_split: str,
    predictive_estimator: str,
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
    validate_metric_estimator(reward_metric, predictive_estimator)
    if reward_data_split not in {"train", "holdout"}:
        raise ValueError("reward_data_split must be train|holdout")

    if claim_mode not in VALID_CLAIM_MODES:
        raise ValueError(f"claim_mode must be one of {sorted(VALID_CLAIM_MODES)}")

    if normalization_method not in VALID_NORMALIZATION_METHODS:
        raise ValueError(
            "normalization_method must be one of "
            f"{sorted(VALID_NORMALIZATION_METHODS)}"
        )

    if normalization_delta_scope not in VALID_DELTA_SCOPES:
        raise ValueError(
            "normalization_delta_scope must be one of "
            f"{sorted(VALID_DELTA_SCOPES)}"
        )

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

    spec = get_interface_spec(dataset_name)
    resolved_method = (
        expected_norm_method(dataset_name)
        if normalization_method == "auto"
        else normalization_method
    )

    if claim_mode == "formal_lh" and resolved_method == "off":
        raise ValueError("formal_lh claim_mode requires normalization_method != off")

    if resolved_method == "exact_binary":
        if spec.delta_scope != "raw_y_binary":
            raise ValueError(
                "exact_binary normalization requires raw_y_binary data interface"
            )
    elif resolved_method == "importance_mc":
        if spec.delta_scope != "raw_y_fixed_x":
            raise ValueError(
                "importance_mc normalization requires raw_y_fixed_x data interface"
            )

    if normalization_method != "auto" and normalization_delta_scope != spec.delta_scope:
        raise ValueError(
            "normalization_delta_scope does not match dataset interface: "
            f"expected {spec.delta_scope!r}, got {normalization_delta_scope!r}"
        )

    return ResolvedContracts(
        normalization_method=resolved_method,
        normalization_delta_scope=spec.delta_scope,
    )
