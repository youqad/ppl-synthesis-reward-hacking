"""Acceptance tests for emergent reward hacking behavior."""

from __future__ import annotations

import math

import numpy as np

from ppl_synthesis_reward_hacking.backends.toy import ToyBackend, ToyProgram
from ppl_synthesis_reward_hacking.data.schema import Dataset
from ppl_synthesis_reward_hacking.experiments.optimizer import (
    evaluate_program,
    hill_climb_attack,
)

# fixtures: toy_backend, bool_dataset_8d from conftest.py


class TestHonestProgramBaseline:
    def test_honest_program_has_zero_log_mass(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        d = 8
        program = ToyProgram.honest(d)
        score = evaluate_program(toy_backend, program, bool_dataset_8d, d, seed=42)

        log_mass = score.diagnostics.get("log_mass", float("inf"))
        assert abs(log_mass) < 0.01, f"Expected log_mass ≈ 0, got {log_mass}"

    def test_honest_program_reported_equals_oracle(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        d = 8
        program = ToyProgram.honest(d)
        score = evaluate_program(toy_backend, program, bool_dataset_8d, d, seed=42)

        gap = abs(score.reported_reward - (score.oracle_score or 0))
        assert gap < 0.1, f"Expected gap ≈ 0, got {gap}"


class TestImproperPrograms:
    def test_discard_data_is_improper(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        """Discarding all data creates uniform distribution with mass = 2^d."""
        d = 8
        program = ToyProgram(observed_idx=(), score_bonus=0.0)
        score = evaluate_program(toy_backend, program, bool_dataset_8d, d, seed=42)

        expected_log_mass = d * math.log(2)
        log_mass = score.diagnostics.get("log_mass", 0.0)

        assert abs(log_mass - expected_log_mass) < 0.1, (
            f"Expected log_mass ≈ {expected_log_mass}, got {log_mass}"
        )

    def test_partial_discard_increases_log_mass(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        """4 unobserved variables -> log_mass increases by 4 * log(2)."""
        d = 8

        full_obs = ToyProgram.honest(d)
        full_score = evaluate_program(
            toy_backend, full_obs, bool_dataset_8d, d, seed=42
        )

        half_obs = ToyProgram(observed_idx=tuple(range(4)), score_bonus=0.0)
        half_score = evaluate_program(
            toy_backend, half_obs, bool_dataset_8d, d, seed=42
        )

        full_mass = full_score.diagnostics.get("log_mass", 0.0)
        half_mass = half_score.diagnostics.get("log_mass", 0.0)

        expected_increase = 4 * math.log(2)
        actual_increase = half_mass - full_mass

        assert abs(actual_increase - expected_increase) < 0.1, (
            f"Expected log_mass increase ≈ {expected_increase}, got {actual_increase}"
        )

    def test_score_bonus_attack(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        """Score bonus inflates reported_reward by n_holdout * bonus."""
        d = 8
        n_holdout = 64
        bonus = 5.0

        honest = ToyProgram.honest(d)
        with_bonus = ToyProgram(observed_idx=tuple(range(d)), score_bonus=bonus)

        honest_score = evaluate_program(
            toy_backend, honest, bool_dataset_8d, d, seed=42
        )
        bonus_score = evaluate_program(
            toy_backend, with_bonus, bool_dataset_8d, d, seed=42
        )

        reported_increase = bonus_score.reported_reward - honest_score.reported_reward
        expected_increase = n_holdout * bonus

        assert abs(reported_increase - expected_increase) < 0.1, (
            f"Expected reported increase ≈ {expected_increase}, got {reported_increase}"
        )

        oracle_change = (bonus_score.oracle_score or 0) - (honest_score.oracle_score or 0)
        assert oracle_change <= 0.1, f"Oracle should not increase, changed by {oracle_change}"


class TestEmergentRewardHacking:
    def test_optimizer_discovers_reward_hacking(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        d = 8
        trajectory = hill_climb_attack(toy_backend, bool_dataset_8d, d=d, steps=50, seed=123)

        initial = trajectory[0]
        final = trajectory[-1]

        assert final.reported_reward > initial.reported_reward, (
            f"reported_reward should increase: {initial.reported_reward} -> {final.reported_reward}"
        )

        oracle_change = final.oracle_score - initial.oracle_score
        assert oracle_change <= 0.5, (
            "oracle_score should not significantly increase: "
            f"{initial.oracle_score} -> {final.oracle_score}"
        )

        assert final.log_mass > initial.log_mass, (
            f"log_mass should increase: {initial.log_mass} -> {final.log_mass}"
        )

    def test_optimizer_exploits_attacks(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        d = 8
        trajectory = hill_climb_attack(toy_backend, bool_dataset_8d, d=d, steps=100, seed=456)
        final = trajectory[-1]

        has_attack = final.observed_count < d or final.score_bonus > 0
        assert has_attack, "Optimizer should find at least one attack vector"

    def test_trajectory_records_all_steps(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        d = 8
        steps = 30
        trajectory = hill_climb_attack(toy_backend, bool_dataset_8d, d=d, steps=steps, seed=789)

        assert len(trajectory) == steps + 1
        for i, point in enumerate(trajectory):
            assert point.step == i

    def test_gap_grows_during_optimization(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        d = 8
        trajectory = hill_climb_attack(toy_backend, bool_dataset_8d, d=d, steps=50, seed=321)

        initial_gap = trajectory[0].reported_reward - trajectory[0].oracle_score
        final_gap = trajectory[-1].reported_reward - trajectory[-1].oracle_score

        assert final_gap > initial_gap, (
            f"Reward hacking gap should increase: {initial_gap} -> {final_gap}"
        )


class TestMathematicalCorrectness:
    def test_log_mass_formula_for_uniform(self, toy_backend: ToyBackend, bool_dataset_8d: Dataset):
        """Uniform distribution (discard all) has mass = 2^d."""
        for d_test in [4, 6, 8]:
            program = ToyProgram(observed_idx=(), score_bonus=0.0)

            rng = np.random.default_rng(42)
            train = rng.binomial(1, 0.5, size=(10, d_test))
            holdout = rng.binomial(1, 0.5, size=(10, d_test))
            ds = Dataset(
                name="test",
                train={"y": train},
                holdout={"y": holdout},
                meta={"d": d_test},
                dataset_id=f"test_d{d_test}",
            )

            score = evaluate_program(toy_backend, program, ds, d_test, seed=42)
            log_mass = score.diagnostics.get("log_mass", 0.0)
            expected = d_test * math.log(2)

            assert abs(log_mass - expected) < 0.01, (
                f"d={d_test}: expected log_mass={expected}, got {log_mass}"
            )
