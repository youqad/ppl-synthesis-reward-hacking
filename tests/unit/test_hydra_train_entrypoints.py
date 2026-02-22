from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_testmod"), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tinker_hydra_summary_success_has_reward_metric() -> None:
    module = _load_script("hydra_train_tinker.py")
    summary = module._build_sweep_summary(
        {
            "final_reward_mean": 1.2,
            "final_valid_rate": 0.9,
            "paper/lh_formal_signal_final": 0.42,
        }
    )
    assert summary["sweep/run_status"] == "success"
    assert summary["sweep/final_lh_formal_signal"] == 0.42


def test_tinker_zero_valid() -> None:
    module = _load_script("hydra_train_tinker.py")
    summary = module._build_sweep_summary(
        {
            "final_reward_mean": float("nan"),
            "final_valid_rate": 0.0,
        }
    )
    assert summary["sweep/run_status"] == "fail"
    assert summary["sweep/error"] == "no_valid_completions"


def test_tinker_error_fails() -> None:
    module = _load_script("hydra_train_tinker.py")
    summary = module._build_sweep_summary(
        {
            "sweep/run_status": "success",
            "sweep/error": "upstream_error",
            "final_reward_mean": 0.2,
            "final_valid_rate": 0.5,
        }
    )
    assert summary["sweep/run_status"] == "fail"
    assert summary["sweep/error"] == "upstream_error"


def test_trl_hydra_summary_failure_marks_fail() -> None:
    module = _load_script("hydra_train_trl.py")
    summary = module._build_sweep_summary({"error": "boom"})
    assert summary["sweep/run_status"] == "fail"
    assert "sweep/final_lh_formal_signal" in summary
    assert "sweep/final_valid_rate" in summary


def test_trl_error_fails() -> None:
    module = _load_script("hydra_train_trl.py")
    summary = module._build_sweep_summary(
        {
            "sweep/run_status": "success",
            "sweep/error": "upstream_error",
            "final_reward_mean": 0.2,
            "final_valid_rate": 0.5,
        }
    )
    assert summary["sweep/run_status"] == "fail"
    assert summary["sweep/error"] == "upstream_error"


def test_entrypoints_share_summary_builder_from_common_module() -> None:
    tinker = _load_script("hydra_train_tinker.py")
    trl = _load_script("hydra_train_trl.py")
    common = __import__("hydra_train_common")

    assert tinker._build_sweep_summary is common.build_sweep_summary
    assert trl._build_sweep_summary is common.build_sweep_summary


def test_sweep_summary_carries_logging_keys() -> None:
    common = __import__("hydra_train_common")
    summary = common.build_sweep_summary(
        {
            "final_reward_mean": 0.5,
            "final_valid_rate": 0.8,
            "paper/lh_formal_signal_final": 0.2,
            "logging/valid_run": True,
            "logging/missing_files_n": 0,
        }
    )
    assert summary["logging/valid_run"] is True
    assert summary["logging/missing_files_n"] == 0


def test_sweep_summary_populates_missing_paper_keys_with_defaults() -> None:
    common = __import__("hydra_train_common")
    summary = common.build_sweep_summary(
        {
            "final_reward_mean": 0.5,
            "final_valid_rate": 0.8,
        }
    )
    assert summary["paper/track"] == "unknown"
    assert summary["paper/claim_mode"] == "unknown"
    assert "paper/lh_formal_signal_final" in summary


def _paper_summary_payload() -> dict[str, float | str]:
    return {
        "sweep/run_status": "success",
        "sweep/final_lh_formal_signal": 0.25,
        "sweep/final_valid_rate": 0.8,
        "paper/track": "part_a_emergence",
        "paper/claim_mode": "formal_lh",
        "paper/reward_metric": "log_marginal_likelihood",
        "paper/reward_data_split": "train",
        "paper/reward_estimator_backend": "smc",
        "paper/prompt_source": "hardcoded",
        "paper/prompt_policy": "legacy",
        "paper/thinking_mode": "think",
        "paper/monitoring_mode": "off",
        "paper/normalization_method": "off",
        "paper/delta_scope": "raw_y_binary",
        "paper/frac_non_normalized_final": 0.25,
        "paper/lh_formal_signal_final": 0.25,
        "paper/judge_hacking_rate_final": 0.0,
        "paper/lh_family_prevalence_final": 0.2,
        "paper/lh_rate_batch_final": 0.2,
        "paper/lh_rate_batch_mean": 0.2,
        "paper/lh_other_novel_final": 0.0,
        "paper/lh_other_novel_mean": 0.0,
    }


def _write_minimal_run_files(run_dir: Path, *, include_results: bool) -> None:
    (run_dir / "completions.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
    (run_dir / "trajectory.json").write_text("[]\n", encoding="utf-8")
    if include_results:
        (run_dir / "results.json").write_text("{}\n", encoding="utf-8")


def test_run_from_cfg_raises_when_logging_validation_fails(tmp_path: Path) -> None:
    common = __import__("hydra_train_common")
    run_dir = tmp_path / "run_fail"
    cfg = OmegaConf.create(
        {
            "train": {
                "name": "tinker",
                "output_dir": str(run_dir),
                "normalization_method": "off",
                "normalization_interval": 0,
                "monitoring_mode": "off",
                "judge_interval": 0,
                "n_steps": 1,
            },
            "wandb": {"enabled": False},
        }
    )

    def _config_from_mapping(mapping: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(output_dir=str(mapping["output_dir"]))

    def _run_training(_cfg: SimpleNamespace) -> dict[str, object]:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_minimal_run_files(run_dir, include_results=False)
        return _paper_summary_payload()

    import logging

    try:
        common.run_from_cfg(
            cfg,
            config_from_mapping=_config_from_mapping,
            run_training=_run_training,
            logger=logging.getLogger("test"),
        )
        raise AssertionError("expected logging validation failure")
    except RuntimeError as exc:
        assert "logging_validation_failed" in str(exc)

    saved = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert saved["sweep/run_status"] == "fail"
    assert saved["sweep/error"] == "logging_validation_failed"
    assert saved["logging/valid_run"] is False


def test_run_from_cfg_returns_results_when_logging_validation_passes(tmp_path: Path) -> None:
    common = __import__("hydra_train_common")
    run_dir = tmp_path / "run_ok"
    cfg = OmegaConf.create(
        {
            "train": {
                "name": "tinker",
                "output_dir": str(run_dir),
                "normalization_method": "off",
                "normalization_interval": 0,
                "monitoring_mode": "off",
                "judge_interval": 0,
                "n_steps": 1,
            },
            "wandb": {"enabled": False},
        }
    )

    def _config_from_mapping(mapping: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(output_dir=str(mapping["output_dir"]))

    def _run_training(_cfg: SimpleNamespace) -> dict[str, object]:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_minimal_run_files(run_dir, include_results=True)
        return _paper_summary_payload()

    import logging

    results = common.run_from_cfg(
        cfg,
        config_from_mapping=_config_from_mapping,
        run_training=_run_training,
        logger=logging.getLogger("test"),
    )
    assert results["sweep/run_status"] == "success"
    assert results["logging/valid_run"] is True


def test_validation_payload_does_not_backfill_missing_paper_keys() -> None:
    common = __import__("hydra_train_common")
    payload = common.build_validation_summary_payload(
        {
            "final_reward_mean": 0.5,
            "final_valid_rate": 0.8,
            "sweep/run_status": "success",
        }
    )
    assert "paper/reward_estimator_backend" not in payload
    assert "paper/prompt_source" not in payload
    assert "paper/prompt_policy" not in payload
    assert "paper/thinking_mode" not in payload
    assert payload["sweep/run_status"] == "success"
    assert "sweep/final_lh_formal_signal" in payload


def test_run_from_cfg_validation_requires_runtime_paper_keys(tmp_path: Path) -> None:
    common = __import__("hydra_train_common")
    run_dir = tmp_path / "run_minimal_summary"
    cfg = OmegaConf.create(
        {
            "train": {
                "name": "tinker",
                "output_dir": str(run_dir),
                "normalization_method": "off",
                "normalization_interval": 0,
                "monitoring_mode": "off",
                "judge_interval": 0,
                "n_steps": 1,
            },
            "wandb": {"enabled": False},
        }
    )

    def _config_from_mapping(mapping: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(output_dir=str(mapping["output_dir"]))

    def _run_training(_cfg: SimpleNamespace) -> dict[str, object]:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_minimal_run_files(run_dir, include_results=True)
        return {
            "final_reward_mean": 0.5,
            "final_valid_rate": 0.8,
            "sweep/run_status": "success",
        }

    import logging

    try:
        common.run_from_cfg(
            cfg,
            config_from_mapping=_config_from_mapping,
            run_training=_run_training,
            logger=logging.getLogger("test"),
        )
        raise AssertionError("expected logging validation failure")
    except RuntimeError as exc:
        assert "logging_validation_failed" in str(exc)

    saved = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert saved["logging/valid_run"] is False
    assert saved["sweep/run_status"] == "fail"
