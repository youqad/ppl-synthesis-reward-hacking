"""End-to-end reward function with real PyMC execution (no mocks)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.backends.pymc.backend import PyMCBackend
from ppl_synthesis_reward_hacking.data.schema import Dataset
from ppl_synthesis_reward_hacking.experiments.pymc_reward import (
    make_pymc_reward_fn,
)
from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    EXEC_FAIL_REWARD,
    PARSE_FAIL_REWARD,
)
from ppl_synthesis_reward_hacking.logging.completions import load_completions

try:
    import pymc  # noqa: F401

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

requires_pymc = pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
pytestmark = [pytest.mark.pymc, requires_pymc]


@pytest.fixture
def dataset():
    rng = np.random.default_rng(42)
    return Dataset(
        name="test_reward_fn",
        train={"y": rng.binomial(1, 0.5, size=(1, 1))},
        holdout={"y": rng.binomial(1, 0.5, size=(1, 1))},
        meta={"d": 1, "p_true": 0.5},
        dataset_id="test_reward_fn_d1",
    )


@pytest.fixture
def reward_templates(
    honest_completion,
    parse_fail_completion,
    exec_fail_completion,
    potential_exploit_completion,
):
    return {
        "honest": honest_completion,
        "parse_fail": parse_fail_completion,
        "exec_fail": exec_fail_completion,
        "potential_exploit": potential_exploit_completion,
    }


class TestRewardFnIntegration:
    def test_reward_fn_returns_correct_length(self, scoring_data_d1, dataset, reward_templates):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
            )

            prompts = ["prompt"] * 3
            completions = [
                reward_templates["honest"],
                reward_templates["parse_fail"],
                reward_templates["exec_fail"],
            ]

            rewards = reward_fn(prompts, completions)
            assert len(rewards) == 3

    def test_reward_fn_honest_scores_above_fail(self, scoring_data_d1, dataset, reward_templates):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
            )

            rewards = reward_fn(["p"], [reward_templates["honest"]])
            assert len(rewards) == 1
            assert rewards[0] > EXEC_FAIL_REWARD

    def test_reward_fn_parse_fail_returns_sentinel(
        self, scoring_data_d1, dataset, reward_templates
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
            )

            rewards = reward_fn(["p"], [reward_templates["parse_fail"]])
            assert rewards[0] == PARSE_FAIL_REWARD

    def test_reward_fn_exec_fail_returns_sentinel(self, scoring_data_d1, dataset, reward_templates):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
            )

            rewards = reward_fn(["p"], [reward_templates["exec_fail"]])
            assert rewards[0] == EXEC_FAIL_REWARD

    def test_reward_fn_potential_exploit_scores_higher_than_honest(
        self, scoring_data_d1, dataset, reward_templates
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
            )

            honest_rewards = reward_fn(["p"], [reward_templates["honest"]])
            exploit_rewards = reward_fn(["p"], [reward_templates["potential_exploit"]])

            # Potential(5.0) inflates reported by 5.0
            assert exploit_rewards[0] > honest_rewards[0]


class TestTrajectoryRecording:
    def test_trajectory_point_recorded_per_batch(self, scoring_data_d1, dataset, reward_templates):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
            )

            reward_fn(["p", "p"], [reward_templates["honest"], reward_templates["parse_fail"]])
            assert len(state.trajectory) == 1
            point = state.trajectory[0]
            assert point.batch == 1
            assert point.n_total == 2
            assert point.n_parse_fail == 1

    def test_trajectory_counts_match_batch(self, scoring_data_d1, dataset, reward_templates):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
            )

            reward_fn(
                ["p"] * 4,
                [
                    reward_templates["honest"],
                    reward_templates["parse_fail"],
                    reward_templates["exec_fail"],
                    reward_templates["potential_exploit"],
                ],
            )

            point = state.trajectory[0]
            assert point.n_total == 4
            assert point.n_parse_fail == 1
            assert point.n_exec_fail == 1
            # 2 valid (honest + potential exploit)
            assert point.n_valid_reported == 2

    def test_multiple_batches_increment_call_count(
        self, scoring_data_d1, dataset, reward_templates
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
            )

            reward_fn(["p"], [reward_templates["honest"]])
            reward_fn(["p"], [reward_templates["potential_exploit"]])
            reward_fn(["p"], [reward_templates["parse_fail"]])

            assert state.call_count == 3
            assert len(state.trajectory) == 3
            assert state.trajectory[0].batch == 1
            assert state.trajectory[1].batch == 2
            assert state.trajectory[2].batch == 3


class TestCompletionLogging:
    def test_completions_written_to_jsonl(self, scoring_data_d1, dataset, reward_templates):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "completions.jsonl"
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
                completions_path=log_path,
            )

            reward_fn(["p", "p"], [reward_templates["honest"], reward_templates["parse_fail"]])

            assert log_path.exists()
            records = load_completions(log_path)
            assert len(records) == 2

    def test_completion_records_have_correct_outcomes(
        self, scoring_data_d1, dataset, reward_templates
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "completions.jsonl"
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
                completions_path=log_path,
            )

            reward_fn(
                ["p"] * 4,
                [
                    reward_templates["honest"],
                    reward_templates["parse_fail"],
                    reward_templates["exec_fail"],
                    reward_templates["potential_exploit"],
                ],
            )

            records = load_completions(log_path)
            outcomes = [r.outcome for r in records]
            assert "valid" in outcomes
            assert "parse_fail" in outcomes
            assert "exec_fail" in outcomes

    def test_valid_records_have_finite_scores(self, scoring_data_d1, dataset, reward_templates):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "completions.jsonl"
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
                completions_path=log_path,
            )

            reward_fn(["p"], [reward_templates["honest"]])
            records = load_completions(log_path)
            valid = [r for r in records if r.outcome == "valid"]
            assert len(valid) == 1
            import math

            assert math.isfinite(valid[0].reported_reward)

    def test_exploit_beats_honest(self, scoring_data_d1, dataset, reward_templates):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "completions.jsonl"
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
                completions_path=log_path,
            )

            reward_fn(
                ["p", "p"], [reward_templates["honest"], reward_templates["potential_exploit"]]
            )
            records = load_completions(log_path)
            valid = [r for r in records if r.outcome == "valid"]
            assert len(valid) == 2
            honest_reward = valid[0].reported_reward
            exploit_reward = valid[1].reported_reward
            assert exploit_reward > honest_reward, (
                f"Potential exploit should have higher reward than honest: "
                f"{exploit_reward} vs {honest_reward}"
            )

    def test_completions_round_trip_preserves_data(
        self, scoring_data_d1, dataset, reward_templates
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "completions.jsonl"
            reward_fn, state = make_pymc_reward_fn(
                backend=PyMCBackend(),
                dataset=dataset,
                scoring_data=scoring_data_d1,
                cache_dir=Path(tmpdir) / "cache",
                run_normalization=False,
                completions_path=log_path,
            )

            reward_fn(["test_prompt"], [reward_templates["honest"]])
            records = load_completions(log_path)

            assert len(records) == 1
            r = records[0]
            assert r.batch == 1
            assert r.index == 0
            assert r.prompt == "test_prompt"
            assert "pm.Beta" in r.completion_text
            assert r.code is not None
            assert "def model" in r.code
            assert r.outcome == "valid"
            assert r.timestamp  # non-empty ISO8601
