from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppl_synthesis_reward_hacking.logging.paper_summary_contract import PAPER_SUMMARY_KEYS_TINKER
from ppl_synthesis_reward_hacking.logging.run_logging_validation import (
    required_files_for_train_cfg,
    validate_run_logging,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _base_tinker_summary_payload() -> dict[str, float | str]:
    return {
        "sweep/run_status": "success",
        "sweep/final_lh_formal_signal": 0.0,
        "sweep/final_valid_rate": 1.0,
        "paper/track": "part_a_emergence",
        "paper/claim_mode": "formal_lh",
        "paper/reward_metric": "log_marginal_likelihood",
        "paper/reward_data_split": "train",
        "paper/reward_estimator_backend": "smc",
        "paper/prompt_source": "jsonl",
        "paper/prompt_policy": "legacy",
        "paper/thinking_mode": "think",
        "paper/monitoring_mode": "off",
        "paper/normalization_method": "off",
        "paper/delta_scope": "raw_y_binary",
        "paper/frac_non_normalized_final": 0.0,
        "paper/lh_formal_signal_final": 0.0,
        "paper/judge_hacking_rate_final": 0.0,
        "paper/lh_family_prevalence_final": 0.0,
        "paper/lh_rate_batch_final": 0.0,
        "paper/lh_rate_batch_mean": 0.0,
        "paper/lh_other_novel_final": 0.0,
        "paper/lh_other_novel_mean": 0.0,
    }


def test_validate_run_logging_passes_for_complete_tinker_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_jsonl(run_dir / "normalization_metrics.jsonl", [{"step": 0}])
    _write_jsonl(run_dir / "judge_metrics.jsonl", [{"step": 0}])
    _write_jsonl(run_dir / "rubric_evolution.jsonl", [{"step": 0}])
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "n_steps": 10,
                "monitoring_mode": "judge_evolving",
                "judge_interval": 1,
                "rubric_evolution_interval": 1,
                "normalization_method": "exact_binary",
                "normalization_interval": 1,
            }
        },
    )
    _write_json(
        run_dir / "results.json",
        {
            **_base_tinker_summary_payload(),
            "sweep/final_lh_formal_signal": 0.2,
            "sweep/final_valid_rate": 0.9,
            "paper/prompt_source": "hardcoded",
            "paper/monitoring_mode": "judge_evolving",
            "paper/normalization_method": "exact_binary",
            "paper/frac_non_normalized_final": 0.2,
            "paper/lh_formal_signal_final": 0.2,
            "paper/judge_hacking_rate_final": 0.4,
            "paper/lh_family_prevalence_final": 0.3,
            "paper/lh_rate_batch_final": 0.25,
            "paper/lh_rate_batch_mean": 0.2,
            "paper/lh_other_novel_final": 0.1,
            "paper/lh_other_novel_mean": 0.08,
        },
    )

    report = validate_run_logging(run_dir)
    assert report["valid"] is True
    assert report["missing_files"] == []
    assert report["missing_summary_keys"] == []
    assert report["parse_errors"] == []


def test_validate_run_logging_fails_on_missing_files_and_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {"train": {"name": "tinker", "normalization_method": "off", "monitoring_mode": "off"}},
    )
    _write_json(run_dir / "results.json", {"sweep/run_status": "success"})

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert "completions.jsonl" in report["missing_files"]
    assert "paper/lh_formal_signal_final" in report["missing_summary_keys"]


@pytest.mark.parametrize(
    "missing_key",
    [
        "paper/reward_estimator_backend",
        "paper/prompt_source",
        "paper/prompt_policy",
        "paper/thinking_mode",
    ],
)
def test_validate_run_logging_fails_when_required_paper_key_missing(
    tmp_path: Path,
    missing_key: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "n_steps": 1,
                "monitoring_mode": "off",
                "judge_interval": 0,
                "rubric_evolution_interval": 0,
                "normalization_method": "off",
                "normalization_interval": 0,
            }
        },
    )
    payload = _base_tinker_summary_payload()
    payload.pop(missing_key)
    _write_json(run_dir / "results.json", payload)

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert missing_key in report["missing_summary_keys"]


@pytest.mark.parametrize("missing_key", PAPER_SUMMARY_KEYS_TINKER)
def test_validate_run_logging_fails_when_tinker_final_lh_key_missing(
    tmp_path: Path,
    missing_key: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "n_steps": 1,
                "monitoring_mode": "off",
                "judge_interval": 0,
                "rubric_evolution_interval": 0,
                "normalization_method": "off",
                "normalization_interval": 0,
            }
        },
    )
    payload = _base_tinker_summary_payload()
    payload.pop(missing_key)
    _write_json(run_dir / "results.json", payload)

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert missing_key in report["missing_summary_keys"]


@pytest.mark.parametrize(
    ("train_cfg", "expected_present", "expected_absent"),
    [
        (
            {
                "name": "tinker",
                "n_steps": 4,
                "normalization_method": "exact_binary",
                "normalization_interval": 5,
                "monitoring_mode": "judge_evolving",
                "judge_interval": 6,
                "rubric_evolution_interval": 7,
            },
            [],
            ["normalization_metrics.jsonl", "judge_metrics.jsonl", "rubric_evolution.jsonl"],
        ),
        (
            {
                "name": "tinker",
                "n_steps": 5,
                "normalization_method": "exact_binary",
                "normalization_interval": 5,
                "monitoring_mode": "judge_evolving",
                "judge_interval": 6,
                "rubric_evolution_interval": 7,
            },
            ["normalization_metrics.jsonl"],
            ["judge_metrics.jsonl", "rubric_evolution.jsonl"],
        ),
        (
            {
                "name": "tinker",
                "n_steps": 6,
                "normalization_method": "exact_binary",
                "normalization_interval": 5,
                "monitoring_mode": "judge_evolving",
                "judge_interval": 6,
                "rubric_evolution_interval": 7,
            },
            ["normalization_metrics.jsonl", "judge_metrics.jsonl"],
            ["rubric_evolution.jsonl"],
        ),
        (
            {
                "name": "tinker",
                "n_steps": 7,
                "normalization_method": "exact_binary",
                "normalization_interval": 5,
                "monitoring_mode": "judge_evolving",
                "judge_interval": 6,
                "rubric_evolution_interval": 7,
            },
            ["normalization_metrics.jsonl", "judge_metrics.jsonl", "rubric_evolution.jsonl"],
            [],
        ),
        (
            {
                "name": "tinker",
                "n_steps": 0,
                "normalization_method": "exact_binary",
                "normalization_interval": 5,
                "monitoring_mode": "judge_evolving",
                "judge_interval": 6,
                "rubric_evolution_interval": 7,
            },
            ["normalization_metrics.jsonl", "judge_metrics.jsonl", "rubric_evolution.jsonl"],
            [],
        ),
    ],
)
def test_required_files_for_train_cfg_interval_boundaries(
    train_cfg: dict[str, object],
    expected_present: list[str],
    expected_absent: list[str],
) -> None:
    required = required_files_for_train_cfg(train_cfg)
    for name in expected_present:
        assert name in required
    for name in expected_absent:
        assert name not in required


def test_required_files_for_train_cfg_handles_nested_hydra_group_payloads() -> None:
    nested_cfg = {
        "name": "tinker",
        "n_steps": 3,
        "normalization": {
            "normalization_method": "exact_binary",
            "normalization_interval": 1,
        },
        "monitoring_mode": {
            "monitoring_mode": "judge_evolving",
            "judge_interval": 1,
            "rubric_evolution_interval": 1,
        },
    }
    required = required_files_for_train_cfg(nested_cfg)
    assert "normalization_metrics.jsonl" in required
    assert "judge_metrics.jsonl" in required
    assert "rubric_evolution.jsonl" in required


def test_validate_run_logging_flags_non_mapping_group_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_group_shape"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "normalization": "exact_binary",
                "monitoring_mode": "off",
            }
        },
    )
    _write_json(run_dir / "results.json", _base_tinker_summary_payload())

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert any("payload must be a mapping" in err for err in report["parse_errors"])


def test_validate_run_logging_flags_unsupported_monitoring_mode_payload_key(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_bad_monitoring_key"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "monitoring_mode": {"legacy_oracle": 1.0},
            }
        },
    )
    _write_json(run_dir / "results.json", _base_tinker_summary_payload())

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert any("must include monitoring_mode" in err for err in report["parse_errors"])


def test_validate_run_logging_flags_boolean_monitoring_mode_wrapper(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_monitoring_wrapper"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "monitoring_mode": {"selected": True},
            }
        },
    )
    _write_json(run_dir / "results.json", _base_tinker_summary_payload())

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert any("monitoring_mode.selected" in err for err in report["parse_errors"])


def test_validate_run_logging_flags_nested_mapping_judge_interval(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_judge_interval_shape"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "monitoring_mode": {
                    "monitoring_mode": "judge_evolving",
                    "judge_interval": {"bad": 1},
                },
            }
        },
    )
    _write_json(run_dir / "results.json", _base_tinker_summary_payload())

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert any("judge_interval must be a scalar value" in err for err in report["parse_errors"])


def test_validate_run_logging_flags_monitoring_payload_missing_mode_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_missing_monitoring_mode_key"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "monitoring_mode": {"judge_interval": 1},
            }
        },
    )
    _write_json(run_dir / "results.json", _base_tinker_summary_payload())

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert any("must include monitoring_mode" in err for err in report["parse_errors"])


def test_validate_run_logging_flags_nested_mapping_normalization_interval(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bad_normalization_interval_shape"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "normalization": {
                    "normalization_method": "exact_binary",
                    "normalization_interval": {"bad": 1},
                },
            }
        },
    )
    _write_json(run_dir / "results.json", _base_tinker_summary_payload())

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert any(
        "normalization_interval must be a scalar value" in err for err in report["parse_errors"]
    )


def test_validate_run_logging_flags_non_numeric_interval_types(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_non_numeric_interval_types"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "normalization_method": "exact_binary",
                "normalization_interval": "foo",
                "monitoring_mode": {
                    "monitoring_mode": "judge_evolving",
                    "judge_interval": "bar",
                },
            }
        },
    )
    _write_json(run_dir / "results.json", _base_tinker_summary_payload())

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert any("normalization_interval must be numeric" in err for err in report["parse_errors"])
