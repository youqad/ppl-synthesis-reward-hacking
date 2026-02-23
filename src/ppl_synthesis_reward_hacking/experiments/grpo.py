"""GRPO (Group Relative Policy Optimization) components.

Training uses only reported_reward. Normalization check (∫ Z_p ≠ 1)
is the sole hacking detection signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GRPOConfig:
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32

    n_prompts: int = 5
    rollouts_per_prompt: int = 160
    n_steps: int = 1000
    learning_rate: float = 1e-5
    temperature: float = 1.28

    loss_fn: str = "importance_sampling"
    kl_coef: float = 0.0
    clip_advantages: bool = False

    d: int = 3
    p_true: float = 0.5
    n_train: int = 32
    n_holdout: int = 64
    exec_timeout: int = 60
    smc_draws: int = 500

    output_dir: str = "artifacts/grpo"


@dataclass
class RolloutData:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    sampling_logprobs: list[float]
    reported_reward: float
    log_mass: float


def build_grpo_datum(
    rollout: RolloutData,
    advantage: float,
) -> dict[str, Any]:
    """Build IS-GRPO training datum for Tinker forward_backward."""
    if len(rollout.prompt_tokens) == 0:
        raise ValueError("prompt_tokens cannot be empty")
    if len(rollout.completion_tokens) == 0:
        raise ValueError("completion_tokens cannot be empty")
    if len(rollout.sampling_logprobs) != len(rollout.completion_tokens):
        raise ValueError(
            f"sampling_logprobs length ({len(rollout.sampling_logprobs)}) must match "
            f"completion_tokens length ({len(rollout.completion_tokens)})"
        )

    all_tokens = rollout.prompt_tokens + rollout.completion_tokens

    completion_logprobs = np.array(rollout.sampling_logprobs, dtype=np.float32)
    n_completion = len(rollout.completion_tokens)
    token_advantages = np.full(n_completion, advantage, dtype=np.float32)

    n_prompt = len(rollout.prompt_tokens)
    full_logprobs = np.concatenate([np.zeros(n_prompt, dtype=np.float32), completion_logprobs])
    full_advantages = np.concatenate([np.zeros(n_prompt, dtype=np.float32), token_advantages])

    return {
        "model_input_tokens": all_tokens[:-1],
        "target_tokens": np.array(all_tokens[1:], dtype=np.int64),
        "logprobs": full_logprobs[1:],
        "advantages": full_advantages[1:],
    }


def build_tinker_datum(
    rollout: RolloutData,
    advantage: float,
) -> Any:
    from ppl_synthesis_reward_hacking.utils.tinker import types

    datum_dict = build_grpo_datum(rollout, advantage)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=datum_dict["model_input_tokens"]),
        loss_fn_inputs={
            "target_tokens": datum_dict["target_tokens"].tolist(),
            "logprobs": datum_dict["logprobs"].tolist(),
            "advantages": datum_dict["advantages"].tolist(),
        },
    )


def compute_group_relative_advantages(
    rollouts_by_prompt: dict[int, list[RolloutData]],
    skip_zero_advantage: bool = True,
    failure_threshold: float | None = None,
    global_baseline_fallback: bool = True,
) -> dict[int, list[float]]:
    """Per-prompt advantages from reported_reward only.

    Failed completions get advantage=0.0; zero-variance prompts skipped.
    When ALL prompts have zero within-group variance and global_baseline_fallback
    is True, recomputes advantages using the global mean across all prompts.
    This prevents gradient starvation in flat reward landscapes.
    """
    from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
        classify_outcome,
    )

    def _valid_mask_for(rollouts: list[RolloutData]) -> list[bool]:
        return _valid_mask_for_rollouts(
            rollouts=rollouts,
            classify_outcome=classify_outcome,
            failure_threshold=failure_threshold,
        )

    def _build_advantages(
        rollouts: list[RolloutData],
        mask: list[bool],
        baseline: float,
    ) -> list[float]:
        return [
            (rollout.reported_reward - baseline) if is_valid else 0.0
            for rollout, is_valid in zip(rollouts, mask, strict=True)
        ]

    advantages_by_prompt, masks_by_prompt = _compute_within_prompt_advantages(
        rollouts_by_prompt=rollouts_by_prompt,
        valid_mask_for=_valid_mask_for,
        build_advantages=_build_advantages,
        skip_zero_advantage=skip_zero_advantage,
    )
    if advantages_by_prompt or not global_baseline_fallback:
        return advantages_by_prompt
    return _apply_global_baseline_fallback(
        rollouts_by_prompt=rollouts_by_prompt,
        existing_advantages=advantages_by_prompt,
        existing_masks=masks_by_prompt,
        valid_mask_for=_valid_mask_for,
        build_advantages=_build_advantages,
    )


def _valid_mask_for_rollouts(
    *,
    rollouts: list[RolloutData],
    classify_outcome: Any,
    failure_threshold: float | None,
) -> list[bool]:
    if failure_threshold is None:
        return [
            bool(np.isfinite(rollout.reported_reward))
            and classify_outcome(rollout.reported_reward) == "valid"
            for rollout in rollouts
        ]
    return [
        bool(np.isfinite(rollout.reported_reward)) and rollout.reported_reward > failure_threshold
        for rollout in rollouts
    ]


def _valid_rewards_from_mask(rollouts: list[RolloutData], valid_mask: list[bool]) -> list[float]:
    return [
        rollout.reported_reward
        for rollout, is_valid in zip(rollouts, valid_mask, strict=True)
        if is_valid
    ]


def _compute_within_prompt_advantages(
    *,
    rollouts_by_prompt: dict[int, list[RolloutData]],
    valid_mask_for: Any,
    build_advantages: Any,
    skip_zero_advantage: bool,
) -> tuple[dict[int, list[float]], dict[int, list[bool]]]:
    advantages_by_prompt: dict[int, list[float]] = {}
    masks_by_prompt: dict[int, list[bool]] = {}
    for prompt_idx, rollouts in rollouts_by_prompt.items():
        valid_mask = valid_mask_for(rollouts)
        masks_by_prompt[prompt_idx] = valid_mask
        valid_rewards = _valid_rewards_from_mask(rollouts, valid_mask)
        if not valid_rewards:
            continue
        mean_reward = float(np.mean(valid_rewards))
        advantages = build_advantages(rollouts, valid_mask, mean_reward)
        if skip_zero_advantage and all(abs(a) < 1e-10 for a in advantages):
            continue
        advantages_by_prompt[prompt_idx] = advantages
    return advantages_by_prompt, masks_by_prompt


def _apply_global_baseline_fallback(
    *,
    rollouts_by_prompt: dict[int, list[RolloutData]],
    existing_advantages: dict[int, list[float]],
    existing_masks: dict[int, list[bool]],
    valid_mask_for: Any,
    build_advantages: Any,
) -> dict[int, list[float]]:
    all_valid = []
    for prompt_idx, rollouts in rollouts_by_prompt.items():
        mask = existing_masks.get(prompt_idx, valid_mask_for(rollouts))
        all_valid.extend(_valid_rewards_from_mask(rollouts, mask))
    if not all_valid or np.std(all_valid) < 1e-10:
        return existing_advantages

    global_mean = float(np.mean(all_valid))
    for prompt_idx, rollouts in rollouts_by_prompt.items():
        mask = existing_masks.get(prompt_idx, valid_mask_for(rollouts))
        advantages = build_advantages(rollouts, mask, global_mean)
        if all(abs(a) < 1e-10 for a in advantages):
            continue
        existing_advantages[prompt_idx] = advantages
    return existing_advantages
