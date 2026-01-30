"""Tests for natural emergence of reward hacking via GRPO training.

CRITICAL: These tests verify that:
1. Training uses ONLY reported_reward (no oracle leakage)
2. Advantages are computed correctly (group-relative)
3. Gap grows naturally from optimization pressure

The core scientific claim is that reward hacking emerges from optimization,
not from artificially injected attacks.
"""

from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.backends.toy import ToyBackend
from ppl_synthesis_reward_hacking.data.schema import Dataset
from ppl_synthesis_reward_hacking.experiments.grpo import (
    GRPOConfig,
    RolloutData,
    build_grpo_datum,
    compute_group_relative_advantages,
)


class TestNoOracleLeakage:
    """Verify oracle_score NEVER enters training datum."""

    def test_grpo_datum_has_no_oracle_field(self):
        """The datum dict should not contain oracle_score."""
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6],
            sampling_logprobs=[-0.5, -0.3, -0.4],
            reported_reward=100.0,
            oracle_score=50.0,  # this should NOT appear in datum
            log_mass=1.5,
        )

        datum = build_grpo_datum(rollout, advantage=10.0)

        # verify no oracle reference in datum
        datum_str = str(datum).lower()
        assert "oracle" not in datum_str, (
            "oracle_score should never appear in training datum"
        )
        assert "oracle_score" not in datum, (
            "oracle_score field should not exist in datum"
        )

    def test_advantage_computed_from_reported_only(self):
        """Advantages must be computed from reported_reward, not oracle_score."""
        # create rollouts where oracle differs significantly from reported
        rollouts = {
            0: [
                RolloutData(
                    prompt_tokens=[1, 2],
                    completion_tokens=[3],
                    sampling_logprobs=[-0.5],
                    reported_reward=100.0,
                    oracle_score=10.0,  # very different from reported
                    log_mass=1.0,
                ),
                RolloutData(
                    prompt_tokens=[1, 2],
                    completion_tokens=[4],
                    sampling_logprobs=[-0.6],
                    reported_reward=200.0,
                    oracle_score=20.0,  # very different from reported
                    log_mass=1.5,
                ),
            ]
        }

        advantages = compute_group_relative_advantages(rollouts)

        # advantages should be based on reported: [100-150, 200-150] = [-50, 50]
        # NOT based on oracle: [10-15, 20-15] = [-5, 5]
        expected = [-50.0, 50.0]
        actual = advantages[0]

        for exp, act in zip(expected, actual, strict=False):
            assert abs(exp - act) < 0.01, (
                f"Advantage should be computed from reported_reward, not oracle_score. "
                f"Expected {exp}, got {act}"
            )


class TestGroupRelativeAdvantages:
    """Verify GRPO advantages are correctly group-relative."""

    def test_advantages_sum_to_zero_per_prompt(self):
        """Per-prompt advantages should sum to approximately zero."""
        # multiple prompts with varying rewards
        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=100.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=150.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [4],
                    [-0.7],
                    reported_reward=200.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [5],
                    [-0.8],
                    reported_reward=50.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
            ],
            1: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=10.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=20.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)

        for prompt_idx, advs in advantages.items():
            sum_advs = sum(advs)
            assert abs(sum_advs) < 1e-6, (
                f"Prompt {prompt_idx}: advantages sum to {sum_advs}, expected ~0"
            )

    def test_higher_reward_gets_positive_advantage(self):
        """Higher reported_reward should yield positive advantage."""
        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=100.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=300.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)

        assert advantages[0][0] < 0, "Lower reward should have negative advantage"
        assert advantages[0][1] > 0, "Higher reward should have positive advantage"


class TestGapGrowth:
    """Test that gap grows naturally over optimization."""

    def test_gap_grows_with_hill_climber(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        """Gap (reported - oracle) should increase during optimization.

        This uses the hill climber as a proxy for GRPO optimization.
        The key property: gap grows from optimization pressure alone.
        """
        from ppl_synthesis_reward_hacking.experiments.optimizer import hill_climb_attack

        d = 8
        trajectory = hill_climb_attack(
            toy_backend, bool_dataset_8d, d=d, steps=50, seed=42
        )

        initial_gap = trajectory[0].reported_reward - trajectory[0].oracle_score
        final_gap = trajectory[-1].reported_reward - trajectory[-1].oracle_score

        # gap should grow (not shrink)
        assert final_gap > initial_gap, (
            f"Gap should increase: initial={initial_gap:.2f}, final={final_gap:.2f}"
        )

    def test_gap_growth_threshold(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        """Gap should grow by at least 10 points over training."""
        from ppl_synthesis_reward_hacking.experiments.optimizer import hill_climb_attack

        d = 8
        trajectory = hill_climb_attack(
            toy_backend, bool_dataset_8d, d=d, steps=100, seed=123
        )

        initial_gap = trajectory[0].reported_reward - trajectory[0].oracle_score
        final_gap = trajectory[-1].reported_reward - trajectory[-1].oracle_score
        gap_increase = final_gap - initial_gap

        assert gap_increase > 10, (
            f"Gap should increase by >10 points. "
            f"Initial: {initial_gap:.2f}, Final: {final_gap:.2f}, Increase: {gap_increase:.2f}"
        )


class TestDatumStructure:
    """Test that GRPO datums have correct structure for importance_sampling loss."""

    def test_datum_has_required_fields(self):
        """Datum must have target_tokens, logprobs, and advantages."""
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6, 7],
            sampling_logprobs=[-0.5, -0.3, -0.4, -0.2],
            reported_reward=100.0,
            oracle_score=50.0,
            log_mass=1.5,
        )

        datum = build_grpo_datum(rollout, advantage=10.0)

        assert "target_tokens" in datum, "Missing target_tokens"
        assert "logprobs" in datum, "Missing logprobs (for IS ratio)"
        assert "advantages" in datum, "Missing advantages"

    def test_datum_shapes_are_consistent(self):
        """All arrays in datum must have consistent shapes."""
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],  # 3 tokens
            completion_tokens=[4, 5, 6, 7],  # 4 tokens
            sampling_logprobs=[-0.5, -0.3, -0.4, -0.2],  # must match completion
            reported_reward=100.0,
            oracle_score=50.0,
            log_mass=1.5,
        )

        datum = build_grpo_datum(rollout, advantage=10.0)

        n_target = len(datum["target_tokens"])
        n_logprobs = len(datum["logprobs"])
        n_advantages = len(datum["advantages"])

        assert n_logprobs == n_target, (
            f"logprobs ({n_logprobs}) must match target_tokens ({n_target})"
        )
        assert n_advantages == n_target, (
            f"advantages ({n_advantages}) must match target_tokens ({n_target})"
        )

    def test_prompt_positions_have_zero_advantage(self):
        """Prompt token positions should have zero advantage (no training signal)."""
        n_prompt = 5
        n_completion = 3
        rollout = RolloutData(
            prompt_tokens=list(range(n_prompt)),
            completion_tokens=list(range(n_prompt, n_prompt + n_completion)),
            sampling_logprobs=[-0.5] * n_completion,
            reported_reward=100.0,
            oracle_score=50.0,
            log_mass=1.5,
        )

        datum = build_grpo_datum(rollout, advantage=10.0)

        advantages = datum["advantages"]

        # first (n_prompt - 1) positions should be zero (shifted by 1 for next-token prediction)
        for i in range(n_prompt - 1):
            assert abs(advantages[i]) < 1e-6, (
                f"Position {i} (prompt) should have zero advantage, got {advantages[i]}"
            )

        # completion positions should have the advantage value
        for i in range(n_prompt - 1, len(advantages)):
            assert abs(advantages[i] - 10.0) < 1e-6, (
                f"Position {i} (completion) should have advantage=10.0, got {advantages[i]}"
            )


class TestEmergenceRequirements:
    """Meta-tests to verify the experiment setup enforces natural emergence."""

    def test_config_uses_importance_sampling_loss(self):
        """Config must specify importance_sampling loss for proper GRPO."""
        config = GRPOConfig()
        assert config.loss_fn == "importance_sampling", (
            f"GRPO requires importance_sampling loss, got {config.loss_fn}"
        )

    def test_rollout_data_separates_reported_and_oracle(self):
        """RolloutData must keep reported_reward and oracle_score separate."""
        rollout = RolloutData(
            prompt_tokens=[1, 2],
            completion_tokens=[3, 4],
            sampling_logprobs=[-0.5, -0.3],
            reported_reward=100.0,
            oracle_score=50.0,
            log_mass=1.5,
        )

        # verify fields exist and are distinct
        assert hasattr(rollout, "reported_reward")
        assert hasattr(rollout, "oracle_score")
        assert rollout.reported_reward != rollout.oracle_score, (
            "Test setup should use different values for reported vs oracle"
        )


class TestInputValidation:
    """Tests for input validation in GRPO datum construction."""

    def test_logprobs_length_mismatch_raises(self):
        """Logprobs must match completion tokens length."""
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6, 7],  # 4 tokens
            sampling_logprobs=[-0.5, -0.3],  # only 2 logprobs - mismatch!
            reported_reward=100.0,
            oracle_score=50.0,
            log_mass=1.5,
        )

        with pytest.raises(ValueError, match="sampling_logprobs length"):
            build_grpo_datum(rollout, advantage=10.0)

    def test_empty_prompt_raises(self):
        """Empty prompt tokens should raise."""
        rollout = RolloutData(
            prompt_tokens=[],  # empty!
            completion_tokens=[4, 5],
            sampling_logprobs=[-0.5, -0.3],
            reported_reward=100.0,
            oracle_score=50.0,
            log_mass=1.5,
        )

        with pytest.raises(ValueError, match="prompt_tokens cannot be empty"):
            build_grpo_datum(rollout, advantage=10.0)

    def test_empty_completion_raises(self):
        """Empty completion tokens should raise."""
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[],  # empty!
            sampling_logprobs=[],
            reported_reward=100.0,
            oracle_score=50.0,
            log_mass=1.5,
        )

        with pytest.raises(ValueError, match="completion_tokens cannot be empty"):
            build_grpo_datum(rollout, advantage=10.0)

    def test_group_size_one_skipped_by_default(self):
        """Single rollout per prompt should be skipped (zero advantage)."""
        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=100.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
            ],
        }

        # default: skip_zero_advantage=True, so single rollout is skipped
        advantages = compute_group_relative_advantages(rollouts)
        assert 0 not in advantages, "Single rollout should be skipped"

        # explicit: skip_zero_advantage=False keeps it
        advantages = compute_group_relative_advantages(rollouts, skip_zero_advantage=False)
        assert 0 in advantages
        assert abs(advantages[0][0]) < 1e-10

    def test_all_equal_rewards_skipped_by_default(self):
        """When all rewards equal, prompt should be skipped (zero advantage)."""
        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=50.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=50.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [4],
                    [-0.7],
                    reported_reward=50.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
            ],
        }

        # default: skip_zero_advantage=True
        advantages = compute_group_relative_advantages(rollouts)
        assert 0 not in advantages, "Equal rewards should cause prompt to be skipped"

        # explicit: skip_zero_advantage=False keeps it with zero advantages
        advantages = compute_group_relative_advantages(rollouts, skip_zero_advantage=False)
        assert 0 in advantages
        for adv in advantages[0]:
            assert abs(adv) < 1e-10

    def test_mixed_prompts_some_skipped(self):
        """Prompts with varying rewards kept, equal rewards skipped."""
        rollouts = {
            0: [  # all equal → skip
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=50.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=50.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
            ],
            1: [  # varying → keep
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=100.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=200.0,
                    oracle_score=0.0,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)

        assert 0 not in advantages, "Equal rewards prompt should be skipped"
        assert 1 in advantages, "Varying rewards prompt should be kept"
        assert len(advantages[1]) == 2
