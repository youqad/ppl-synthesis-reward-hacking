"""ToyBackend contract tests for reward hacking experiments.

These tests verify the mathematical properties that make reward hacking possible:

    reported_reward = Σ log Z_p(x)           # hackable: ignores normalization
    oracle_score    = Σ log Z_p(x) - n * log M   # unhackable: exact normalization
    log_mass        = (d - |obs|) * log(2) + score_bonus   # impropriety diagnostic

Key invariant: gap == n_holdout * log_mass

How hacking works:
1. Honest program: observes all d variables → log_mass ≈ 0 → gap ≈ 0
2. Improper program: observes fewer variables → log_mass > 0 → gap > 0
3. GRPO optimizes reported_reward → discovers improper programs → gap emerges
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec, ScoreResult
from ppl_synthesis_reward_hacking.backends.toy import ToyBackend, ToyProgram
from ppl_synthesis_reward_hacking.data.schema import Dataset


@pytest.fixture
def backend() -> ToyBackend:
    """Provide ToyBackend instance."""
    return ToyBackend()


@pytest.fixture
def dataset_8d() -> Dataset:
    """Boolean dataset with d=8, p=0.3."""
    rng = np.random.default_rng(42)
    d = 8
    p = 0.3
    n_train = 32
    n_holdout = 64

    return Dataset(
        name="contract_test",
        train={"y": rng.binomial(1, p, size=(n_train, d))},
        holdout={"y": rng.binomial(1, p, size=(n_holdout, d))},
        meta={"d": d, "p_true": p, "n_train": n_train, "n_holdout": n_holdout},
        dataset_id="contract_test_8d",
    )


def compile_and_score(
    backend: ToyBackend, program: ToyProgram, dataset: Dataset, d: int
) -> ScoreResult:
    """Helper to compile, fit, and score a program."""
    model = ModelSpec(
        backend="toy",
        name="test",
        source="",
        meta=program.to_meta() | {"d": d},
    )
    compiled = backend.compile(model, cache_dir=Path("/tmp"))
    fit = backend.fit(compiled, dataset=dataset, seed=42)
    return backend.score_holdout(compiled, fit=fit, dataset=dataset, seed=42)


class TestToyBackendContract:
    """Verify ToyBackend scoring semantics for reward hacking experiments."""

    def test_honest_program_has_zero_gap(self, backend: ToyBackend, dataset_8d: Dataset):
        """Program observing all d variables: reported ≈ oracle (gap ≈ 0)."""
        d = 8
        program = ToyProgram.honest(d=d)
        assert program.observed_idx == tuple(range(d))
        assert program.score_bonus == 0.0

        score = compile_and_score(backend, program, dataset_8d, d)
        gap = score.reported_reward - score.oracle_score

        assert abs(gap) < 1e-6, (
            f"Honest program should have ~0 gap, got {gap:.6f}. "
            f"reported={score.reported_reward:.2f}, oracle={score.oracle_score:.2f}"
        )

    def test_improper_program_has_positive_gap(self, backend: ToyBackend, dataset_8d: Dataset):
        """Program observing fewer variables: reported > oracle (gap > 0)."""
        d = 8
        program = ToyProgram(observed_idx=(0,), score_bonus=0.0)
        assert program.observed_count == 1  # only observing 1 of 8

        score = compile_and_score(backend, program, dataset_8d, d)
        gap = score.reported_reward - score.oracle_score

        assert score.reported_reward > score.oracle_score, (
            f"Improper program should have reported > oracle. "
            f"reported={score.reported_reward:.2f}, oracle={score.oracle_score:.2f}"
        )
        assert gap > 0, f"Gap should be positive, got {gap:.2f}"
        assert score.diagnostics["log_mass"] > 0, (
            f"log_mass should be positive for improper program, "
            f"got {score.diagnostics['log_mass']:.4f}"
        )

    def test_score_bonus_increases_gap(self, backend: ToyBackend, dataset_8d: Dataset):
        """Higher score_bonus → larger gap (impropriety mechanism)."""
        d = 8

        prog0 = ToyProgram(observed_idx=(0, 1, 2), score_bonus=0.0)
        prog1 = ToyProgram(observed_idx=(0, 1, 2), score_bonus=5.0)
        prog2 = ToyProgram(observed_idx=(0, 1, 2), score_bonus=10.0)

        score0 = compile_and_score(backend, prog0, dataset_8d, d)
        score1 = compile_and_score(backend, prog1, dataset_8d, d)
        score2 = compile_and_score(backend, prog2, dataset_8d, d)

        gap0 = score0.reported_reward - score0.oracle_score
        gap1 = score1.reported_reward - score1.oracle_score
        gap2 = score2.reported_reward - score2.oracle_score

        assert gap1 > gap0, f"Higher bonus should increase gap: {gap0:.2f} → {gap1:.2f}"
        assert gap2 > gap1, f"Higher bonus should increase gap: {gap1:.2f} → {gap2:.2f}"

    def test_oracle_independent_of_score_bonus(self, backend: ToyBackend, dataset_8d: Dataset):
        """Oracle should be CONSTANT for fixed observed_idx.

        This verifies the paper's theoretical claim that proper normalization
        cancels out the score_bonus inflation.
        """
        d = 8
        observed = (0, 1, 2, 3)

        prog0 = ToyProgram(observed_idx=observed, score_bonus=0.0)
        prog1 = ToyProgram(observed_idx=observed, score_bonus=5.0)
        prog2 = ToyProgram(observed_idx=observed, score_bonus=20.0)

        score0 = compile_and_score(backend, prog0, dataset_8d, d)
        score1 = compile_and_score(backend, prog1, dataset_8d, d)
        score2 = compile_and_score(backend, prog2, dataset_8d, d)

        # oracle should be identical for same observed_idx
        assert abs(score0.oracle_score - score1.oracle_score) < 1e-6, (
            f"Oracle should be independent of score_bonus. "
            f"bonus=0: {score0.oracle_score:.4f}, bonus=5: {score1.oracle_score:.4f}"
        )
        assert abs(score1.oracle_score - score2.oracle_score) < 1e-6, (
            f"Oracle should be independent of score_bonus. "
            f"bonus=5: {score1.oracle_score:.4f}, bonus=20: {score2.oracle_score:.4f}"
        )

        # reported should increase with bonus
        assert score1.reported_reward > score0.reported_reward
        assert score2.reported_reward > score1.reported_reward

    def test_log_mass_formula_correctness(self, backend: ToyBackend, dataset_8d: Dataset):
        """log_mass = (d - |obs|) * log(2) + score_bonus.

        This is the core formula connecting program structure to impropriety.
        """
        d = 8

        test_cases = [
            # (observed_idx, score_bonus, expected_log_mass)
            (tuple(range(d)), 0.0, 0.0),  # honest: 0 unobserved, 0 bonus
            ((0,), 0.0, 7 * math.log(2)),  # 7 unobserved, 0 bonus
            ((), 0.0, 8 * math.log(2)),  # all unobserved, 0 bonus
            ((0, 1, 2), 5.0, 5 * math.log(2) + 5.0),  # 5 unobserved + bonus
            (tuple(range(d)), 10.0, 10.0),  # 0 unobserved + bonus
        ]

        for observed_idx, score_bonus, expected_log_mass in test_cases:
            program = ToyProgram(observed_idx=observed_idx, score_bonus=score_bonus)
            score = compile_and_score(backend, program, dataset_8d, d)

            actual_log_mass = score.diagnostics["log_mass"]
            assert abs(actual_log_mass - expected_log_mass) < 1e-6, (
                f"log_mass mismatch for obs={observed_idx}, bonus={score_bonus}. "
                f"Expected {expected_log_mass:.4f}, got {actual_log_mass:.4f}"
            )

    def test_gap_equals_n_times_log_mass(self, backend: ToyBackend, dataset_8d: Dataset):
        """CRITICAL: gap == n_holdout * log_mass (key invariant from GPT-5.2).

        This formalizes the 'hackable vs oracle' distinction and is the
        central mathematical identity for the reward hacking claim.
        """
        d = 8
        n_holdout = dataset_8d.holdout["y"].shape[0]

        test_programs = [
            ToyProgram.honest(d=d),
            ToyProgram(observed_idx=(0,), score_bonus=0.0),
            ToyProgram(observed_idx=(0, 1), score_bonus=5.0),
            ToyProgram(observed_idx=(), score_bonus=10.0),
            ToyProgram(observed_idx=(0, 1, 2, 3), score_bonus=3.0),
        ]

        for program in test_programs:
            score = compile_and_score(backend, program, dataset_8d, d)
            gap = score.reported_reward - score.oracle_score
            log_mass = score.diagnostics["log_mass"]
            expected_gap = n_holdout * log_mass

            assert abs(gap - expected_gap) < 1e-6, (
                f"Gap should equal n_holdout * log_mass. "
                f"Program: obs={program.observed_idx}, bonus={program.score_bonus}. "
                f"gap={gap:.4f}, n_holdout={n_holdout}, log_mass={log_mass:.4f}, "
                f"expected_gap={expected_gap:.4f}"
            )

    def test_reducing_observed_count_increases_gap(self, backend: ToyBackend, dataset_8d: Dataset):
        """Fewer observed variables → larger gap (core hacking mechanism)."""
        d = 8

        gaps = []
        for n_obs in range(d, -1, -1):
            program = ToyProgram(observed_idx=tuple(range(n_obs)), score_bonus=0.0)
            score = compile_and_score(backend, program, dataset_8d, d)
            gap = score.reported_reward - score.oracle_score
            gaps.append(gap)

        # gaps should be monotonically increasing as observed count decreases
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] >= gaps[i] - 1e-6, (
                f"Gap should increase as observed count decreases. "
                f"obs_count={d - i}: gap={gaps[i]:.2f}, "
                f"obs_count={d - i - 1}: gap={gaps[i + 1]:.2f}"
            )


class TestEdgeCases:
    """Edge case behavior for ToyBackend."""

    def test_empty_observed_idx(self, backend: ToyBackend, dataset_8d: Dataset):
        """Program observing nothing should have maximum gap."""
        d = 8
        program = ToyProgram(observed_idx=(), score_bonus=0.0)

        score = compile_and_score(backend, program, dataset_8d, d)

        # log_mass = d * log(2) for observing nothing
        expected_log_mass = d * math.log(2)
        assert abs(score.diagnostics["log_mass"] - expected_log_mass) < 1e-6

    def test_large_score_bonus(self, backend: ToyBackend, dataset_8d: Dataset):
        """Large score_bonus should inflate reported but not oracle."""
        d = 8
        program = ToyProgram(observed_idx=tuple(range(d)), score_bonus=1000.0)

        score = compile_and_score(backend, program, dataset_8d, d)

        # reported should be much higher than honest
        honest = ToyProgram.honest(d=d)
        honest_score = compile_and_score(backend, honest, dataset_8d, d)

        # bonus adds 1000 * n_holdout to reported
        n_holdout = dataset_8d.holdout["y"].shape[0]
        expected_diff = 1000.0 * n_holdout
        actual_diff = score.reported_reward - honest_score.reported_reward

        assert abs(actual_diff - expected_diff) < 1e-3, (
            f"Score bonus should add bonus * n_holdout to reported. "
            f"Expected diff={expected_diff}, got {actual_diff}"
        )

    def test_filtering_invalid_observed_indices(self, backend: ToyBackend, dataset_8d: Dataset):
        """Out-of-bounds indices are filtered consistently across all methods.

        Both _log_unnorm_prob_vectorized and _compute_log_mass filter to valid indices,
        ensuring the key invariant gap == n_holdout * log_mass is preserved.
        """
        d = 8
        # indices 10, 20 are out of bounds for d=8
        program = ToyProgram(observed_idx=(0, 1, 10, 20), score_bonus=0.0)

        score = compile_and_score(backend, program, dataset_8d, d)

        # should behave identically to program with only valid indices (0, 1)
        program_filtered = ToyProgram(observed_idx=(0, 1), score_bonus=0.0)
        score_filtered = compile_and_score(backend, program_filtered, dataset_8d, d)

        # reported_reward matches (same filtered indices)
        assert abs(score.reported_reward - score_filtered.reported_reward) < 1e-6

        # oracle_score matches (consistent log_mass)
        assert abs(score.oracle_score - score_filtered.oracle_score) < 1e-6

        # log_mass matches (filtered count used)
        assert abs(score.diagnostics["log_mass"] - score_filtered.diagnostics["log_mass"]) < 1e-6

        # diagnostics report filtered count = 2, not raw count = 4
        assert score.diagnostics["observed_count"] == 2  # filtered count

        # key invariant preserved: gap == n_holdout * log_mass
        n_holdout = dataset_8d.holdout["y"].shape[0]
        gap = score.reported_reward - score.oracle_score
        expected_gap = n_holdout * score.diagnostics["log_mass"]
        assert abs(gap - expected_gap) < 1e-6, (
            f"Invariant broken: gap={gap:.4f}, expected={expected_gap:.4f}"
        )


class TestProgramValidation:
    """Test ToyProgram validation."""

    def test_duplicate_indices_rejected(self):
        """ToyProgram with duplicate observed_idx should raise ValueError."""
        with pytest.raises(ValueError, match="unique dimensions"):
            ToyProgram(observed_idx=(0, 1, 0), score_bonus=0.0)

        with pytest.raises(ValueError, match="unique dimensions"):
            ToyProgram(observed_idx=(2, 2, 2), score_bonus=0.0)

    def test_unique_indices_accepted(self):
        """ToyProgram with unique indices should be created successfully."""
        prog = ToyProgram(observed_idx=(0, 1, 2), score_bonus=0.0)
        assert prog.observed_idx == (0, 1, 2)

        # empty is also valid
        prog_empty = ToyProgram(observed_idx=(), score_bonus=0.0)
        assert prog_empty.observed_idx == ()


class TestDiagnostics:
    """Verify diagnostics are correctly reported."""

    def test_diagnostics_contain_required_fields(self, backend: ToyBackend, dataset_8d: Dataset):
        """ScoreResult.diagnostics should have all required fields."""
        d = 8
        program = ToyProgram(observed_idx=(0, 1, 2), score_bonus=2.5)

        score = compile_and_score(backend, program, dataset_8d, d)

        required_fields = ["log_mass", "observed_count", "score_bonus", "d", "n_holdout"]
        for field in required_fields:
            assert field in score.diagnostics, f"Missing diagnostic field: {field}"

        assert score.diagnostics["observed_count"] == 3
        assert abs(score.diagnostics["score_bonus"] - 2.5) < 1e-6
        assert score.diagnostics["d"] == d
        assert score.diagnostics["n_holdout"] == dataset_8d.holdout["y"].shape[0]
