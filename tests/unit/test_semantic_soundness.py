"""Semantic soundness tests for ToyBackend programs.

Based on paper Section 5.1: semantic soundness requires ∫ Z_p(x) = 1.

For ToyBackend:
    total_mass = exp(log_mass) = 2^(d - |obs|) * exp(score_bonus)

Honest programs have total_mass = 1 (semantically sound).
Improper programs have total_mass > 1 (not sound, can be exploited).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec
from ppl_synthesis_reward_hacking.backends.toy import ToyBackend, ToyProgram
from ppl_synthesis_reward_hacking.data.schema import Dataset


@pytest.fixture
def backend() -> ToyBackend:
    """Provide ToyBackend instance."""
    return ToyBackend()


@pytest.fixture
def dataset() -> Dataset:
    """Boolean dataset for testing."""
    rng = np.random.default_rng(42)
    d = 8
    return Dataset(
        name="soundness_test",
        train={"y": rng.binomial(1, 0.3, size=(32, d))},
        holdout={"y": rng.binomial(1, 0.3, size=(64, d))},
        meta={"d": d},
        dataset_id="soundness_test",
    )


def compute_log_mass(backend: ToyBackend, program: ToyProgram, dataset: Dataset, d: int) -> float:
    """Helper to compute log_mass for a program."""
    model = ModelSpec(
        backend="toy",
        name="test",
        source="",
        meta=program.to_meta() | {"d": d},
    )
    compiled = backend.compile(model, cache_dir=Path("/tmp"))
    fit = backend.fit(compiled, dataset=dataset, seed=42)
    score = backend.score_holdout(compiled, fit=fit, dataset=dataset, seed=42)
    return score.diagnostics["log_mass"]


class TestSemanticSoundness:
    """Test semantic soundness properties from paper Section 5.1."""

    def test_honest_program_is_semantically_sound(self, backend: ToyBackend, dataset: Dataset):
        """Program observing all variables should have total_mass = 1.

        This is the definition of semantic soundness: ∫ Z_p(x) = 1.
        """
        d = 8
        program = ToyProgram.honest(d=d)

        log_mass = compute_log_mass(backend, program, dataset, d)
        total_mass = math.exp(log_mass)

        assert abs(log_mass) < 1e-6, f"Honest program should have log_mass ≈ 0, got {log_mass:.6f}"
        assert abs(total_mass - 1.0) < 1e-6, (
            f"Honest program should integrate to 1, got {total_mass:.6f}"
        )

    def test_improper_program_is_not_sound(self, backend: ToyBackend, dataset: Dataset):
        """Program observing fewer variables should have total_mass > 1.

        This is how improper models can be exploited for reward hacking.
        """
        d = 8
        program = ToyProgram(observed_idx=(0,), score_bonus=0.0)

        log_mass = compute_log_mass(backend, program, dataset, d)
        total_mass = math.exp(log_mass)

        assert log_mass > 0, f"Improper program should have log_mass > 0, got {log_mass:.6f}"
        assert total_mass > 1.0, f"Improper program should integrate to > 1, got {total_mass:.6f}"

    def test_score_bonus_violates_soundness(self, backend: ToyBackend, dataset: Dataset):
        """Even with all variables observed, score_bonus > 0 violates soundness."""
        d = 8
        program = ToyProgram(observed_idx=tuple(range(d)), score_bonus=5.0)

        log_mass = compute_log_mass(backend, program, dataset, d)
        total_mass = math.exp(log_mass)

        assert log_mass == 5.0, f"log_mass should equal score_bonus, got {log_mass:.6f}"
        assert total_mass > 1.0, f"score_bonus should violate soundness, mass={total_mass:.6f}"


class TestTotalMassFormula:
    """Verify the total_mass formula matches theory."""

    def test_mass_formula_for_varying_observed(self, backend: ToyBackend, dataset: Dataset):
        """total_mass = 2^(d - |obs|) for programs with score_bonus=0."""
        d = 8

        for n_obs in range(d + 1):
            program = ToyProgram(observed_idx=tuple(range(n_obs)), score_bonus=0.0)
            log_mass = compute_log_mass(backend, program, dataset, d)

            n_unobserved = d - n_obs
            expected_log_mass = n_unobserved * math.log(2)
            expected_total_mass = 2**n_unobserved

            assert abs(log_mass - expected_log_mass) < 1e-6, (
                f"n_obs={n_obs}: expected log_mass={expected_log_mass:.4f}, got {log_mass:.4f}"
            )

            actual_total_mass = math.exp(log_mass)
            assert abs(actual_total_mass - expected_total_mass) < 1e-6

    def test_mass_formula_with_score_bonus(self, backend: ToyBackend, dataset: Dataset):
        """total_mass = 2^(d - |obs|) * exp(score_bonus)."""
        d = 8

        test_cases = [
            # (n_obs, score_bonus)
            (4, 0.0),
            (4, 5.0),
            (2, 10.0),
            (d, 3.0),
        ]

        for n_obs, bonus in test_cases:
            program = ToyProgram(observed_idx=tuple(range(n_obs)), score_bonus=bonus)
            log_mass = compute_log_mass(backend, program, dataset, d)

            n_unobserved = d - n_obs
            expected_log_mass = n_unobserved * math.log(2) + bonus
            expected_total_mass = (2**n_unobserved) * math.exp(bonus)

            assert abs(log_mass - expected_log_mass) < 1e-6, (
                f"n_obs={n_obs}, bonus={bonus}: "
                f"expected log_mass={expected_log_mass:.4f}, got {log_mass:.4f}"
            )

            actual_total_mass = math.exp(log_mass)
            assert abs(actual_total_mass - expected_total_mass) < 1e-6


class TestSoundnessPreservation:
    """Test conditions under which soundness is preserved."""

    def test_only_honest_program_is_sound(self, backend: ToyBackend, dataset: Dataset):
        """Only programs with all vars observed AND score_bonus=0 are sound."""
        d = 8

        # only this one should be sound
        honest = ToyProgram.honest(d=d)
        log_mass = compute_log_mass(backend, honest, dataset, d)
        assert abs(log_mass) < 1e-6

        # any deviation breaks soundness
        not_sound = [
            ToyProgram(observed_idx=tuple(range(d - 1)), score_bonus=0.0),  # missing 1 var
            ToyProgram(observed_idx=tuple(range(d)), score_bonus=0.001),  # tiny bonus
            ToyProgram(observed_idx=(), score_bonus=0.0),  # no vars
        ]

        for prog in not_sound:
            log_mass = compute_log_mass(backend, prog, dataset, d)
            assert log_mass > 1e-10, f"Program {prog} should have log_mass > 0, got {log_mass:.6f}"
