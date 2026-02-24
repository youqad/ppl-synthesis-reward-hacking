from __future__ import annotations

import json

import pytest

from ppl_synthesis_reward_hacking.experiments import tinker_training as module
from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord


def _record(*, index: int, norm: dict | None) -> CompletionRecord:
    metadata = {"normalization": norm} if norm is not None else {}
    return CompletionRecord(
        batch=0,
        index=index,
        prompt="prompt",
        completion_text="```python\npass\n```",
        code="pass",
        reported_reward=-1.0,
        outcome="valid",
        timestamp="2026-01-01T00:00:00Z",
        metadata=metadata,
    )


def _step_data(records: list[CompletionRecord]) -> module._StepData:
    return module._StepData(
        rollouts_by_prompt={},
        all_reported=[rec.reported_reward for rec in records],
        all_completion_hashes=[],
        step_records=records,
    )


def test_log_normalization_summary_separates_check_failures(tmp_path) -> None:
    records = [
        _record(
            index=0,
            norm={"check_ok": True, "is_normalized": True, "log_mass": 0.002, "status": "ok"},
        ),
        _record(
            index=1,
            norm={"check_ok": True, "is_normalized": False, "log_mass": -1.2, "status": "ok"},
        ),
        _record(
            index=2,
            norm={
                "check_ok": False,
                "ok": False,
                "status": "non_valid_score",
                "reason": "non_valid_score",
                "is_normalized": None,
            },
        ),
    ]
    step_data = _step_data(records)
    out_path = tmp_path / "normalization_metrics.jsonl"

    payload = module._log_normalization_summary(step=7, step_data=step_data, metrics_path=out_path)

    assert payload is not None
    assert payload["n_checked"] == 3
    assert payload["n_check_ok"] == 2
    assert payload["n_check_failed"] == 1
    assert payload["n_non_normalized"] == 1
    assert payload["frac_non_normalized"] == pytest.approx(0.5)
    assert payload["frac_non_normalized_over_attempted"] == pytest.approx(1 / 3)
    assert payload["frac_check_failed"] == pytest.approx(1 / 3)
    assert payload["mean_abs_log_mass"] == pytest.approx((0.002 + 1.2) / 2)
    assert len(payload["log_masses"]) == 2

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    disk_payload = json.loads(lines[0])
    assert disk_payload["n_check_ok"] == 2
    assert disk_payload["n_check_failed"] == 1


def test_compute_step_point_uses_only_successful_checks_for_formal_signal() -> None:
    records = [
        _record(
            index=0,
            norm={"check_ok": True, "is_normalized": True, "log_mass": 0.001, "status": "ok"},
        ),
        _record(
            index=1,
            norm={"check_ok": True, "is_normalized": False, "log_mass": -0.8, "status": "ok"},
        ),
        _record(
            index=2,
            norm={"check_ok": False, "status": "non_valid_score", "is_normalized": None},
        ),
    ]
    step_data = _step_data(records)

    point = module._compute_step_point(step=3, step_data=step_data, frac_zero_variance=0.0)

    assert point.n_norm_checked == 3
    assert point.n_norm_check_ok == 2
    assert point.n_norm_check_failed == 1
    assert point.frac_non_normalized == pytest.approx(0.5)
    assert point.frac_norm_check_failed == pytest.approx(1 / 3)
    assert point.mean_abs_log_mass == pytest.approx((0.001 + 0.8) / 2)


def test_collect_lh_programs_ignores_failed_normalization_checks() -> None:
    failed = _record(
        index=0,
        norm={"check_ok": False, "status": "non_valid_score", "is_normalized": None},
    )
    true_non_norm = _record(
        index=1,
        norm={"check_ok": True, "status": "ok", "is_normalized": False, "log_mass": 0.7},
    )

    rows = module._collect_lh_programs(
        step=5,
        step_records=[failed, true_non_norm],
        judge_verdicts=[],
    )

    assert len(rows) == 1
    assert rows[0]["detection"] == "normalization"
    assert rows[0]["log_mass"] == 0.7
