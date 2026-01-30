"""Tests verifying oracle_score matches paper's Definition 5.3.

From the paper:
    Oracle = Σ log(Z_p(x)/M)

Where:
    - Z_p(x) is the unnormalized probability of observation x under program p
    - M = Σ_x Z_p(x) is the normalizing constant (total mass)

The backend computes:
    reported_reward = Σ log Z_p(x)       (unnormalized, hackable)
    oracle_score = Σ log Z_p(x) - n * log M  (normalized, unhackable)

These are equivalent:
    Σ log(Z_p(x)/M) = Σ (log Z_p(x) - log M) = Σ log Z_p(x) - n * log M
"""

from __future__ import annotations

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
        name="oracle_theory_test",
        train={"y": rng.binomial(1, 0.3, size=(32, d))},
        holdout={"y": rng.binomial(1, 0.3, size=(64, d))},
        meta={"d": d},
        dataset_id="oracle_theory_test",
    )


def score_program(
    backend: ToyBackend, program: ToyProgram, dataset: Dataset, d: int
) -> tuple[float, float, float]:
    """Return (reported_reward, oracle_score, log_mass) for a program."""
    model = ModelSpec(
        backend="toy",
        name="test",
        source="",
        meta=program.to_meta() | {"d": d},
    )
    compiled = backend.compile(model, cache_dir=Path("/tmp"))
    fit = backend.fit(compiled, dataset=dataset, seed=42)
    score = backend.score_holdout(compiled, fit=fit, dataset=dataset, seed=42)
    return score.reported_reward, score.oracle_score, score.diagnostics["log_mass"]


class TestOracleDefinition:
    """Verify oracle_score matches paper Definition 5.3."""

    def test_oracle_equals_normalized_log_prob_sum(
        self, backend: ToyBackend, dataset: Dataset
    ):
        """Oracle = Σ log(Z_p(x)/M) = Σ log Z_p(x) - n * log M.

        This is the core identity from Definition 5.3.
        """
        d = 8
        n_holdout = dataset.holdout["y"].shape[0]

        test_programs = [
            ToyProgram.honest(d=d),
            ToyProgram(observed_idx=(0, 1), score_bonus=0.0),
            ToyProgram(observed_idx=(), score_bonus=5.0),
            ToyProgram(observed_idx=(0, 2, 4, 6), score_bonus=2.5),
        ]

        for program in test_programs:
            reported, oracle, log_mass = score_program(backend, program, dataset, d)

            # verify: oracle = reported - n * log_mass
            # which is: Σ log Z_p(x) - n * log M
            expected_oracle = reported - n_holdout * log_mass

            assert abs(oracle - expected_oracle) < 1e-6, (
                f"Oracle should equal reported - n * log_mass. "
                f"Program: {program}. "
                f"oracle={oracle:.4f}, expected={expected_oracle:.4f}"
            )

    def test_oracle_is_properly_normalized(
        self, backend: ToyBackend, dataset: Dataset
    ):
        """Oracle score represents log of properly normalized probabilities.

        For any program, oracle_score = Σ log p(x) where p(x) = Z_p(x)/M.
        This should be the 'true' log-likelihood under a proper distribution.
        """
        d = 8
        program = ToyProgram(observed_idx=(0, 1, 2, 3), score_bonus=0.0)

        reported, oracle, log_mass = score_program(backend, program, dataset, d)

        # normalized per-observation contribution:
        # oracle/n = (reported - n*log_mass)/n = reported/n - log_mass
        # = mean(log Z_p(x)) - log M = mean(log(Z_p(x)/M))
        n_holdout = dataset.holdout["y"].shape[0]
        mean_oracle = oracle / n_holdout
        mean_reported = reported / n_holdout

        assert abs(mean_oracle - (mean_reported - log_mass)) < 1e-6


class TestOracleProperties:
    """Test theoretical properties of oracle score."""

    def test_oracle_bounded_above_by_zero_for_proper_distribution(
        self, backend: ToyBackend, dataset: Dataset
    ):
        """For a proper probability distribution, log p(x) <= 0.

        Oracle = Σ log p(x) should generally be negative (sum of log probs).
        """
        d = 8
        program = ToyProgram.honest(d=d)

        _, oracle, _ = score_program(backend, program, dataset, d)

        # honest program has proper distribution, oracle should be negative
        # (log probs are always <= 0)
        assert oracle < 0, f"Oracle should be negative for proper distribution, got {oracle:.2f}"

    def test_oracle_independent_of_unnormalized_scaling(
        self, backend: ToyBackend, dataset: Dataset
    ):
        """Oracle is invariant to score_bonus when observed_idx is fixed.

        This is because proper normalization cancels the bonus.
        """
        d = 8
        observed = (0, 1, 2, 3)

        oracle_values = []
        for bonus in [0.0, 5.0, 10.0, 100.0]:
            program = ToyProgram(observed_idx=observed, score_bonus=bonus)
            _, oracle, _ = score_program(backend, program, dataset, d)
            oracle_values.append(oracle)

        # all oracle values should be equal (within floating point)
        for i, val in enumerate(oracle_values[1:], 1):
            assert abs(val - oracle_values[0]) < 1e-6, (
                f"Oracle should be independent of score_bonus. "
                f"bonus=0: {oracle_values[0]:.4f}, bonus={[0.0, 5.0, 10.0, 100.0][i]}: {val:.4f}"
            )

    def test_oracle_depends_on_observed_idx(
        self, backend: ToyBackend, dataset: Dataset
    ):
        """Oracle depends on which variables are observed (structure matters)."""
        d = 8

        oracle_values = []
        for n_obs in [d, d // 2, 2, 0]:
            program = ToyProgram(observed_idx=tuple(range(n_obs)), score_bonus=0.0)
            _, oracle, _ = score_program(backend, program, dataset, d)
            oracle_values.append(oracle)

        # oracle values should generally differ for different observed_idx
        assert len(set(round(v, 4) for v in oracle_values)) > 1, (
            "Oracle should vary with different observed_idx"
        )


class TestHackingMechanism:
    """Test how reported differs from oracle (the hacking mechanism)."""

    def test_gap_equals_n_times_log_mass(
        self, backend: ToyBackend, dataset: Dataset
    ):
        """Gap = reported - oracle = n * log_mass.

        This is the key identity connecting hacking to improper mass.
        """
        d = 8
        n_holdout = dataset.holdout["y"].shape[0]

        test_programs = [
            ToyProgram.honest(d=d),
            ToyProgram(observed_idx=(0,), score_bonus=0.0),
            ToyProgram(observed_idx=(0, 1), score_bonus=5.0),
        ]

        for program in test_programs:
            reported, oracle, log_mass = score_program(backend, program, dataset, d)
            gap = reported - oracle
            expected_gap = n_holdout * log_mass

            assert abs(gap - expected_gap) < 1e-6, (
                f"Gap should equal n * log_mass. "
                f"gap={gap:.4f}, expected={expected_gap:.4f}"
            )

    def test_honest_program_has_zero_gap(
        self, backend: ToyBackend, dataset: Dataset
    ):
        """Honest program: log_mass = 0, so gap = 0."""
        d = 8
        program = ToyProgram.honest(d=d)

        reported, oracle, log_mass = score_program(backend, program, dataset, d)
        gap = reported - oracle

        assert abs(log_mass) < 1e-6, "Honest program should have log_mass = 0"
        assert abs(gap) < 1e-6, "Honest program should have gap = 0"
        assert abs(reported - oracle) < 1e-6, "Reported should equal oracle for honest"
