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
    base_model: str = "meta-llama/Llama-3.2-1B"
    lora_rank: int = 32

    n_prompts: int = 100
    rollouts_per_prompt: int = 4
    n_steps: int = 50
    learning_rate: float = 1e-5

    loss_fn: str = "importance_sampling"
    kl_coef: float = 0.0
    clip_advantages: bool = False

    d: int = 8
    p_true: float = 0.3
    n_train: int = 32
    n_holdout: int = 64

    output_dir: str = "artifacts/grpo_poc"


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
) -> dict[int, list[float]]:
    """Per-prompt advantages from reported_reward only.

    Failed completions get advantage=0.0; zero-variance prompts skipped.
    By default, failures are identified by scorer sentinels, not by reward
    magnitude, so low but valid rewards still contribute gradient signal.
    """
    from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
        classify_outcome,
    )

    advantages_by_prompt: dict[int, list[float]] = {}

    for prompt_idx, rollouts in rollouts_by_prompt.items():
        if failure_threshold is None:
            valid_mask = [
                bool(np.isfinite(r.reported_reward))
                and classify_outcome(r.reported_reward) == "valid"
                for r in rollouts
            ]
        else:
            # Legacy override for experiments that explicitly want threshold behavior.
            valid_mask = [
                bool(np.isfinite(r.reported_reward)) and r.reported_reward > failure_threshold
                for r in rollouts
            ]
        valid_rewards = [r.reported_reward for r, v in zip(rollouts, valid_mask, strict=True) if v]

        if not valid_rewards:
            continue

        mean_reward = float(np.mean(valid_rewards))
        advantages = [
            (r.reported_reward - mean_reward) if v else 0.0
            for r, v in zip(rollouts, valid_mask, strict=True)
        ]

        if skip_zero_advantage and all(abs(a) < 1e-10 for a in advantages):
            continue

        advantages_by_prompt[prompt_idx] = advantages

    return advantages_by_prompt
