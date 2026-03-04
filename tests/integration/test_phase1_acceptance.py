"""Phase 1 acceptance tests for the reward hacking framework."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec
from ppl_synthesis_reward_hacking.backends.toy.backend import ToyBackend
from ppl_synthesis_reward_hacking.backends.toy.program import ToyProgram
from ppl_synthesis_reward_hacking.data.schema import Dataset
from ppl_synthesis_reward_hacking.evaluation.metrics import compute_reward_gap

# fixtures: toy_backend, bool_dataset_8d from conftest.py


class TestHonestModel:
    def test_honest_model_has_near_zero_gap(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        d = 8
        program = ToyProgram.honest(d)
        model = ModelSpec(
            backend="toy",
            name="test_honest",
            source="",
            meta=program.to_meta() | {"d": d},
        )

        compiled = toy_backend.compile(model, cache_dir=Path("/tmp"))
        fit = toy_backend.fit(compiled, dataset=bool_dataset_8d, seed=42)
        score = toy_backend.score_holdout(compiled, fit=fit, dataset=bool_dataset_8d, seed=42)

        metrics = {
            "reported_reward_holdout": score.reported_reward,
            "oracle_loglik_holdout": score.diagnostics.get("ground_truth_loglik", 0.0),
        }
        gap = compute_reward_gap(metrics)

        assert abs(gap) < 0.1, f"Honest model should have near-zero gap, got {gap}"


class TestScoreBonusAttack:
    def test_score_bonus_produces_positive_gap(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        """Gap should be approx n_holdout * bonus."""
        d = 8
        n_holdout = 64
        bonus = 5.0

        program = ToyProgram(observed_idx=tuple(range(d)), score_bonus=bonus)
        model = ModelSpec(
            backend="toy",
            name="test_score_bonus",
            source="",
            meta=program.to_meta() | {"d": d},
        )

        compiled = toy_backend.compile(model, cache_dir=Path("/tmp"))
        fit = toy_backend.fit(compiled, dataset=bool_dataset_8d, seed=42)
        score = toy_backend.score_holdout(compiled, fit=fit, dataset=bool_dataset_8d, seed=42)

        metrics = {
            "reported_reward_holdout": score.reported_reward,
            "oracle_loglik_holdout": score.diagnostics.get("ground_truth_loglik", 0.0),
        }
        gap = compute_reward_gap(metrics)

        expected_gap = n_holdout * bonus
        assert gap > 0, f"Attack model should have positive gap, got {gap}"
        assert abs(gap - expected_gap) < 1.0, f"Gap should be close to {expected_gap}, got {gap}"


class TestMonotonicity:
    def test_gap_increases_monotonically_with_bonus(
        self, toy_backend: ToyBackend, bool_dataset_8d: Dataset
    ):
        d = 8
        bonus_values = [0, 1.0, 5.0, 10.0]
        gaps: list[float] = []

        for bonus in bonus_values:
            program = ToyProgram(observed_idx=tuple(range(d)), score_bonus=bonus)
            model = ModelSpec(
                backend="toy",
                name=f"test_bonus_{bonus}",
                source="",
                meta=program.to_meta() | {"d": d},
            )

            compiled = toy_backend.compile(model, cache_dir=Path("/tmp"))
            fit = toy_backend.fit(compiled, dataset=bool_dataset_8d, seed=42)
            score = toy_backend.score_holdout(compiled, fit=fit, dataset=bool_dataset_8d, seed=42)

            metrics = {
                "reported_reward_holdout": score.reported_reward,
                "oracle_loglik_holdout": score.diagnostics.get("ground_truth_loglik", 0.0),
            }
            gap = compute_reward_gap(metrics)
            gaps.append(gap)

        for i in range(1, len(gaps)):
            prev_bonus, prev_gap = bonus_values[i - 1], gaps[i - 1]
            curr_bonus, curr_gap = bonus_values[i], gaps[i]
            assert curr_gap > prev_gap, (
                f"Gap should increase: bonus={prev_bonus} -> gap={prev_gap}, "
                f"bonus={curr_bonus} -> gap={curr_gap}"
            )


class TestDiscardDataAttack:
    def test_discard_data_produces_gap(self, toy_backend: ToyBackend, bool_dataset_8d: Dataset):
        """Discarding 4 variables -> log_mass = 4 * log(2) per observation."""
        d = 8

        program = ToyProgram(observed_idx=tuple(range(4)), score_bonus=0.0)
        model = ModelSpec(
            backend="toy",
            name="test_discard",
            source="",
            meta=program.to_meta() | {"d": d},
        )

        compiled = toy_backend.compile(model, cache_dir=Path("/tmp"))
        fit = toy_backend.fit(compiled, dataset=bool_dataset_8d, seed=42)
        score = toy_backend.score_holdout(compiled, fit=fit, dataset=bool_dataset_8d, seed=42)

        metrics = {
            "reported_reward_holdout": score.reported_reward,
            "oracle_loglik_holdout": score.diagnostics.get("ground_truth_loglik", 0.0),
        }
        gap = compute_reward_gap(metrics)

        assert gap > 0, f"Discard attack should have positive gap, got {gap}"


class TestParquetOutput:
    def test_parquet_writer_produces_file(self, bool_dataset_8d: Dataset):
        pytest.importorskip("pyarrow")

        from ppl_synthesis_reward_hacking.logging.schema import MetricRecord
        from ppl_synthesis_reward_hacking.logging.writers import write_metrics_parquet

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.parquet"
            records = [
                MetricRecord(metric_name="test_metric", metric_value=1.23, meta={"seed": 0}),
                MetricRecord(metric_name="another_metric", metric_value=4.56, meta={"seed": 0}),
            ]
            write_metrics_parquet(path, records)

            assert path.exists(), "Parquet file should be created"

            import pyarrow.parquet as pq

            table = pq.read_table(path)
            assert table.num_rows == 2
            assert set(table.column_names) == {"metric_name", "metric_value", "meta_json"}
