"""GRPO (Group Relative Policy Optimization) components.

Training uses only reported_reward. oracle_score is logged for diagnostics
but never enters the gradient computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GRPOConfig:
    """GRPO experiment configuration."""

    # model
    base_model: str = "meta-llama/Llama-3.2-1B"
    lora_rank: int = 32

    # GRPO settings
    n_prompts: int = 100
    rollouts_per_prompt: int = 4
    n_steps: int = 50
    learning_rate: float = 1e-4

    # GRPO-specific
    loss_fn: str = "importance_sampling"
    kl_coef: float = 0.0
    clip_advantages: bool = False

    # data
    d: int = 8
    p_true: float = 0.3
    n_train: int = 32
    n_holdout: int = 64

    # output
    output_dir: str = "artifacts/grpo_poc"


@dataclass
class RolloutData:
    """Single rollout: tokens, logprobs, and scores."""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    sampling_logprobs: list[float]
    reported_reward: float
    oracle_score: float  # diagnostics only
    log_mass: float


def build_grpo_datum(
    rollout: RolloutData,
    advantage: float,
) -> dict[str, Any]:
    """Build training datum for importance-sampling GRPO loss.

    Returns dict with model_input_tokens, target_tokens, logprobs, advantages.
    Caller converts to framework-specific types (e.g. tinker.types.Datum).
    """
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

    # zero-pad prompt positions (no training signal for prompt tokens)
    n_prompt = len(rollout.prompt_tokens)
    full_logprobs = np.concatenate([np.zeros(n_prompt, dtype=np.float32), completion_logprobs])
    full_advantages = np.concatenate([np.zeros(n_prompt, dtype=np.float32), token_advantages])

    return {
        "model_input_tokens": all_tokens[:-1],
        "target_tokens": np.array(all_tokens[1:], dtype=np.int64),
        "logprobs": full_logprobs[1:],
        "advantages": full_advantages[1:],
    }


def compute_group_relative_advantages(
    rollouts_by_prompt: dict[int, list[RolloutData]],
    skip_zero_advantage: bool = True,
) -> dict[int, list[float]]:
    """Per-prompt advantages: reward - mean(group rewards).

    Uses only reported_reward. Prompts with zero variance are skipped
    when skip_zero_advantage is True (no learning signal).
    """
    advantages_by_prompt: dict[int, list[float]] = {}

    for prompt_idx, rollouts in rollouts_by_prompt.items():
        rewards = [r.reported_reward for r in rollouts]
        mean_reward = float(np.mean(rewards))
        advantages = [r - mean_reward for r in rewards]

        if skip_zero_advantage and all(abs(a) < 1e-10 for a in advantages):
            continue

        advantages_by_prompt[prompt_idx] = advantages

    return advantages_by_prompt
