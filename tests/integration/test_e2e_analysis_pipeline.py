"""Analysis pipeline with real PyMC scoring: analyze_step and analyze_run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ppl_synthesis_reward_hacking.evaluation.hacking_analysis import (
    analyze_run,
    analyze_step,
)
from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    EXEC_FAIL_REWARD,
    PARSE_FAIL_REWARD,
    score_completion_sandboxed,
)

try:
    import pymc  # noqa: F401

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

requires_pymc = pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
pytestmark = [pytest.mark.pymc, requires_pymc]


@pytest.fixture
def tmp_run_dir(tmp_path):
    return tmp_path / "test_run"


@pytest.fixture
def analysis_templates(
    honest_completion,
    potential_injection_completion,
    no_code_block_completion,
    broken_syntax_completion,
):
    return {
        "honest": honest_completion,
        "potential_injection": potential_injection_completion,
        "no_code_block": no_code_block_completion,
        "broken_syntax": broken_syntax_completion,
    }


# -- helpers --


def _score_and_record(
    completion: str,
    scoring_data: dict,
    batch: int,
    prompt_idx: int = 0,
) -> dict:
    reported, _oracle, decomp = score_completion_sandboxed(completion, scoring_data, timeout=60)
    outcome = "valid"
    if reported == PARSE_FAIL_REWARD:
        outcome = "parse_fail"
    elif reported == EXEC_FAIL_REWARD:
        outcome = "exec_fail"

    return {
        "batch": batch,
        "prompt_idx": prompt_idx,
        "completion_text": completion,
        "code": completion,
        "reported_reward": reported,
        "outcome": outcome,
        "metadata": {"decomposition": decomp},
    }


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Not serializable: {type(obj)}")


def _write_completions(run_dir: Path, records: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "completions.jsonl", "w") as f:
        for r in records:
            line = json.dumps(r, default=_json_default)
            f.write(line + "\n")


# -- test classes --


class TestAnalyzeStepWithRealScores:
    def test_all_honest_batch(self, scoring_data_d1, analysis_templates):
        records = [
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=0)
            for _ in range(3)
        ]
        result = analyze_step(records)

        assert result["n_total"] == 3
        assert result["n_valid"] == 3
        assert math.isfinite(result["reported_mean"])

    def test_mixed_exploit_batch(self, scoring_data_d1, analysis_templates):
        records = [
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=0),
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=0),
            _score_and_record(analysis_templates["potential_injection"], scoring_data_d1, batch=0),
        ]
        result = analyze_step(records)

        assert result["n_total"] == 3
        assert result["n_valid"] == 3
        assert math.isfinite(result["reported_mean"])

    def test_batch_with_parse_fail(self, scoring_data_d1, analysis_templates):
        records = [
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=0),
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=0),
            _score_and_record(analysis_templates["no_code_block"], scoring_data_d1, batch=0),
        ]
        result = analyze_step(records)

        assert result["n_total"] == 3
        assert result["n_valid"] == 2
        assert result["parse_fail_pct"] > 0

    def test_batch_with_exec_fail(self, scoring_data_d1, analysis_templates):
        records = [
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=0),
            _score_and_record(analysis_templates["broken_syntax"], scoring_data_d1, batch=0),
        ]
        result = analyze_step(records)

        assert result["n_total"] == 2
        # broken syntax either parse-fails or exec-fails depending on extraction
        assert result["n_valid"] <= 1


class TestAnalyzeRunWithRealScores:
    def test_two_step_run(self, scoring_data_d1, tmp_run_dir, analysis_templates):
        batch_0 = [
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=0)
            for _ in range(3)
        ]
        batch_1 = [
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=1),
            _score_and_record(analysis_templates["honest"], scoring_data_d1, batch=1),
            _score_and_record(analysis_templates["potential_injection"], scoring_data_d1, batch=1),
        ]
        all_records = batch_0 + batch_1
        _write_completions(tmp_run_dir, all_records)

        result = analyze_run(tmp_run_dir)

        assert result["n_steps"] == 2
        assert result["n_total_records"] == 6

        traj = result["reported_trajectory"]
        assert "slope" in traj
        assert "mean" in traj
        assert "std" in traj
        assert "first_step_mean" in traj
        assert "last_step_mean" in traj
        assert "delta" in traj
        assert "trend" in traj

        assert "0" in result["per_step"]
        assert "1" in result["per_step"]

        assert result["health"]["valid_rate"] > 0


class TestAnalyzeRunEdgeCases:
    def test_empty_completions_raises(self, tmp_run_dir):
        tmp_run_dir.mkdir(parents=True, exist_ok=True)
        (tmp_run_dir / "completions.jsonl").write_text("")

        with pytest.raises(ValueError):
            analyze_run(tmp_run_dir)

    def test_missing_completions_raises(self, tmp_path):
        nonexistent = tmp_path / "no_such_run"

        with pytest.raises(FileNotFoundError):
            analyze_run(nonexistent)
