"""GRPO emergence tests: oracle isolation, advantage computation, gap growth."""

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
    def test_grpo_datum_has_no_oracle_field(self):
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6],
            sampling_logprobs=[-0.5, -0.3, -0.4],
            reported_reward=100.0,
            log_mass=1.5,
        )

        datum = build_grpo_datum(rollout, advantage=10.0)

        datum_str = str(datum).lower()
        assert "oracle" not in datum_str, "oracle leaked into training datum"
        assert "ground_truth_loglik" not in datum, "ground_truth_loglik leaked into training datum"

    def test_advantage_computed_from_reported_only(self):
        rollouts = {
            0: [
                RolloutData(
                    prompt_tokens=[1, 2],
                    completion_tokens=[3],
                    sampling_logprobs=[-0.5],
                    reported_reward=100.0,
                    log_mass=1.0,
                ),
                RolloutData(
                    prompt_tokens=[1, 2],
                    completion_tokens=[4],
                    sampling_logprobs=[-0.6],
                    reported_reward=200.0,
                    log_mass=1.5,
                ),
            ]
        }

        advantages = compute_group_relative_advantages(rollouts)

        # from reported: [100-150, 200-150] = [-50, 50] (not oracle-based)
        expected = [-50.0, 50.0]
        actual = advantages[0]

        for exp, act in zip(expected, actual, strict=False):
            assert abs(exp - act) < 0.01, (
                f"advantage uses reported_reward, not ground_truth_loglik. "
                f"Expected {exp}, got {act}"
            )


class TestGroupRelativeAdvantages:
    def test_advantages_sum_to_zero_per_prompt(self):
        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=100.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=150.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [4],
                    [-0.7],
                    reported_reward=200.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [5],
                    [-0.8],
                    reported_reward=50.0,
                    log_mass=0.0,
                ),
            ],
            1: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=10.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=20.0,
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
        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=100.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=300.0,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)

        assert advantages[0][0] < 0, "Lower reward should have negative advantage"
        assert advantages[0][1] > 0, "Higher reward should have positive advantage"


class TestGapGrowth:
    def test_gap_grows_with_hill_climber(self, toy_backend: ToyBackend, bool_dataset_8d: Dataset):
        from ppl_synthesis_reward_hacking.experiments.optimizer import hill_climb_attack

        d = 8
        trajectory = hill_climb_attack(toy_backend, bool_dataset_8d, d=d, steps=50, seed=42)

        initial_gap = trajectory[0].reported_reward - trajectory[0].ground_truth_loglik
        final_gap = trajectory[-1].reported_reward - trajectory[-1].ground_truth_loglik
        assert final_gap > initial_gap, (
            f"Gap should increase: initial={initial_gap:.2f}, final={final_gap:.2f}"
        )

    def test_gap_growth_threshold(self, toy_backend: ToyBackend, bool_dataset_8d: Dataset):
        from ppl_synthesis_reward_hacking.experiments.optimizer import hill_climb_attack

        d = 8
        trajectory = hill_climb_attack(toy_backend, bool_dataset_8d, d=d, steps=100, seed=123)

        initial_gap = trajectory[0].reported_reward - trajectory[0].ground_truth_loglik
        final_gap = trajectory[-1].reported_reward - trajectory[-1].ground_truth_loglik
        gap_increase = final_gap - initial_gap

        threshold = 10.0
        assert gap_increase > threshold, (
            f"Gap should increase by >{threshold} points. "
            f"Initial: {initial_gap:.2f}, Final: {final_gap:.2f}, Increase: {gap_increase:.2f}"
        )


class TestDatumStructure:
    def test_datum_has_required_fields(self):
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6, 7],
            sampling_logprobs=[-0.5, -0.3, -0.4, -0.2],
            reported_reward=100.0,
            log_mass=1.5,
        )

        datum = build_grpo_datum(rollout, advantage=10.0)

        assert "target_tokens" in datum, "Missing target_tokens"
        assert "logprobs" in datum, "Missing logprobs (for IS ratio)"
        assert "advantages" in datum, "Missing advantages"

    def test_datum_shapes_are_consistent(self):
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6, 7],
            sampling_logprobs=[-0.5, -0.3, -0.4, -0.2],
            reported_reward=100.0,
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
        n_prompt = 5
        n_completion = 3
        rollout = RolloutData(
            prompt_tokens=list(range(n_prompt)),
            completion_tokens=list(range(n_prompt, n_prompt + n_completion)),
            sampling_logprobs=[-0.5] * n_completion,
            reported_reward=100.0,
            log_mass=1.5,
        )

        datum = build_grpo_datum(rollout, advantage=10.0)

        advantages = datum["advantages"]

        # shifted by 1 for next-token prediction
        for i in range(n_prompt - 1):
            assert abs(advantages[i]) < 1e-6, (
                f"Position {i} (prompt) should have zero advantage, got {advantages[i]}"
            )

        for i in range(n_prompt - 1, len(advantages)):
            assert abs(advantages[i] - 10.0) < 1e-6, (
                f"Position {i} (completion) should have advantage=10.0, got {advantages[i]}"
            )


class TestEmergenceRequirements:
    def test_config_uses_importance_sampling_loss(self):
        config = GRPOConfig()
        assert config.loss_fn == "importance_sampling", (
            f"GRPO requires importance_sampling loss, got {config.loss_fn}"
        )

    def test_rollout_data_separates_reported_and_oracle(self):
        rollout = RolloutData(
            prompt_tokens=[1, 2],
            completion_tokens=[3, 4],
            sampling_logprobs=[-0.5, -0.3],
            reported_reward=100.0,
            log_mass=1.5,
        )

        assert hasattr(rollout, "reported_reward")
        assert hasattr(rollout, "log_mass")
        assert not hasattr(rollout, "ground_truth_loglik"), (
            "ground_truth_loglik removed from RolloutData"
        )


class TestInputValidation:
    def test_logprobs_length_mismatch_raises(self):
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6, 7],
            sampling_logprobs=[-0.5, -0.3],
            reported_reward=100.0,
            log_mass=1.5,
        )

        with pytest.raises(ValueError, match="sampling_logprobs length"):
            build_grpo_datum(rollout, advantage=10.0)

    def test_empty_prompt_raises(self):
        rollout = RolloutData(
            prompt_tokens=[],
            completion_tokens=[4, 5],
            sampling_logprobs=[-0.5, -0.3],
            reported_reward=100.0,
            log_mass=1.5,
        )

        with pytest.raises(ValueError, match="prompt_tokens cannot be empty"):
            build_grpo_datum(rollout, advantage=10.0)

    def test_empty_completion_raises(self):
        rollout = RolloutData(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[],
            sampling_logprobs=[],
            reported_reward=100.0,
            log_mass=1.5,
        )

        with pytest.raises(ValueError, match="completion_tokens cannot be empty"):
            build_grpo_datum(rollout, advantage=10.0)

    def test_group_size_one_skipped_by_default(self):
        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=100.0,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)
        assert 0 not in advantages

        advantages = compute_group_relative_advantages(rollouts, skip_zero_advantage=False)
        assert 0 in advantages
        assert abs(advantages[0][0]) < 1e-10

    def test_all_equal_rewards_skipped_by_default(self):
        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=50.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=50.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [4],
                    [-0.7],
                    reported_reward=50.0,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)
        assert 0 not in advantages

        advantages = compute_group_relative_advantages(rollouts, skip_zero_advantage=False)
        assert 0 in advantages
        for adv in advantages[0]:
            assert abs(adv) < 1e-10

    def test_mixed_prompts_some_skipped(self):
        rollouts = {
            0: [  # all equal, skipped
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=50.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=50.0,
                    log_mass=0.0,
                ),
            ],
            1: [  # varying, kept
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=100.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=200.0,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)

        assert 0 not in advantages
        assert 1 in advantages
        assert len(advantages[1]) == 2


class TestFailureFiltering:
    def test_failures_get_zero_advantage(self):
        from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
            EXEC_FAIL_REWARD,
        )

        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=EXEC_FAIL_REWARD,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=-0.7,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [4],
                    [-0.7],
                    reported_reward=-0.3,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)
        assert 0 in advantages
        assert abs(advantages[0][0]) < 1e-10  # failure: zero advantage
        assert advantages[0][1] != 0.0 or advantages[0][2] != 0.0

    def test_all_failures_skipped(self):
        from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
            EXEC_FAIL_REWARD,
        )

        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=EXEC_FAIL_REWARD,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=EXEC_FAIL_REWARD,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)
        assert 0 not in advantages

    def test_valid_advantages_computed_from_valid_only(self):
        from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
            EXEC_FAIL_REWARD,
        )

        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=EXEC_FAIL_REWARD,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=10.0,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [4],
                    [-0.7],
                    reported_reward=20.0,
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)
        assert 0 in advantages

        # valid mean = (10+20)/2 = 15
        assert abs(advantages[0][0]) < 1e-10  # failure
        assert abs(advantages[0][1] - (-5.0)) < 1e-6
        assert abs(advantages[0][2] - 5.0) < 1e-6

    def test_low_but_valid_rewards_are_not_treated_as_failures(self):
        from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
            EXEC_FAIL_REWARD,
        )

        rollouts = {
            0: [
                RolloutData(
                    [1],
                    [2],
                    [-0.5],
                    reported_reward=EXEC_FAIL_REWARD,
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [3],
                    [-0.6],
                    reported_reward=-450.0,  # valid but below EXEC_FAIL_REWARD
                    log_mass=0.0,
                ),
                RolloutData(
                    [1],
                    [4],
                    [-0.7],
                    reported_reward=-420.0,  # valid but below EXEC_FAIL_REWARD
                    log_mass=0.0,
                ),
            ],
        }

        advantages = compute_group_relative_advantages(rollouts)
        assert 0 in advantages

        # valid mean = (-450 + -420) / 2 = -435
        assert abs(advantages[0][0]) < 1e-10  # failure sentinel stays zero
        assert abs(advantages[0][1] - (-15.0)) < 1e-6
        assert abs(advantages[0][2] - 15.0) < 1e-6

    def test_global_fallback_all_zero_variance(self):
        rollouts = {
            0: [
                RolloutData([1], [2], [-0.5], reported_reward=-0.70, log_mass=0.0),
                RolloutData([1], [3], [-0.6], reported_reward=-0.70, log_mass=0.0),
            ],
            1: [
                RolloutData([1], [2], [-0.5], reported_reward=-0.60, log_mass=0.0),
                RolloutData([1], [3], [-0.6], reported_reward=-0.60, log_mass=0.0),
            ],
        }

        no_fallback = compute_group_relative_advantages(rollouts, global_baseline_fallback=False)
        assert len(no_fallback) == 0

        # global mean = -0.65
        with_fallback = compute_group_relative_advantages(rollouts, global_baseline_fallback=True)
        assert len(with_fallback) == 2
        assert all(a < 0 for a in with_fallback[0])  # -0.70 below global mean
        assert all(a > 0 for a in with_fallback[1])  # -0.60 above global mean

    def test_global_fallback_skipped_with_variance(self):
        rollouts = {
            0: [
                RolloutData([1], [2], [-0.5], reported_reward=-0.70, log_mass=0.0),
                RolloutData([1], [3], [-0.6], reported_reward=-0.70, log_mass=0.0),
            ],
            1: [
                RolloutData([1], [2], [-0.5], reported_reward=-0.60, log_mass=0.0),
                RolloutData([1], [3], [-0.6], reported_reward=-0.80, log_mass=0.0),
            ],
        }

        result = compute_group_relative_advantages(rollouts, global_baseline_fallback=True)
        assert 0 not in result
        assert 1 in result
