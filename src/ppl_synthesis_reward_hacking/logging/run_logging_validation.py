from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.config.flattening import (
    GROUP_PAYLOAD_KEYS,
    MONITORING_MODE_KEYS,
    flatten_hydra_train_mapping,
)
from ppl_synthesis_reward_hacking.logging.paper_summary_contract import (
    PAPER_SUMMARY_KEYS_CORE,
    PAPER_SUMMARY_KEYS_TINKER,
)

REQUIRED_SWEEP_SUMMARY_KEYS: tuple[str, ...] = (
    "sweep/run_status",
    "sweep/final_lh_formal_signal",
    "sweep/final_valid_rate",
)
REQUIRED_SUMMARY_KEYS_CORE: tuple[str, ...] = REQUIRED_SWEEP_SUMMARY_KEYS + PAPER_SUMMARY_KEYS_CORE
REQUIRED_SUMMARY_KEYS_TINKER: tuple[str, ...] = PAPER_SUMMARY_KEYS_TINKER

ALWAYS_REQUIRED_FILES: tuple[str, ...] = (
    "completions.jsonl",
    "results.json",
    "trajectory.json",
    "hydra_resolved_config.json",
)


def _as_int(raw: Any, default: int = 0, *, field_name: str) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise ValueError(f"{field_name} must be numeric, got bool")
    if isinstance(raw, int | float):
        if math.isnan(float(raw)) or math.isinf(float(raw)):
            raise ValueError(f"{field_name} must be finite")
        return int(raw)
    raise ValueError(f"{field_name} must be numeric, got {type(raw).__name__}")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_train_cfg(run_dir: Path) -> dict[str, Any]:
    cfg = _load_json(run_dir / "hydra_resolved_config.json")
    if not cfg:
        return {}
    train = cfg.get("train")
    return train if isinstance(train, dict) else {}


def _normalized_train_cfg(train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    _validate_monitoring_mode_payload(train_cfg.get("monitoring_mode"))
    for group_name in GROUP_PAYLOAD_KEYS:
        if group_name == "monitoring_mode":
            continue
        if group_name in train_cfg and not isinstance(train_cfg[group_name], Mapping):
            raise ValueError(
                f"train.{group_name} payload must be a mapping, got "
                f"{type(train_cfg[group_name]).__name__}"
            )
    flattened = flatten_hydra_train_mapping(train_cfg)
    if "name" in train_cfg and "name" not in flattened:
        flattened["name"] = train_cfg["name"]
    return flattened


def _validate_monitoring_mode_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        return
    if "selected" in payload:
        selected_value = payload.get("selected")
        if (
            len(payload) != 1
            or isinstance(selected_value, Mapping)
            or not isinstance(selected_value, str)
        ):
            raise ValueError(
                "Unsupported keys in train.monitoring_mode payload: "
                + ", ".join(f"monitoring_mode.{key}" for key in sorted(str(k) for k in payload))
            )
        return
    allowed = {"monitoring_mode", *MONITORING_MODE_KEYS}
    if "monitoring_mode" not in payload:
        raise ValueError("train.monitoring_mode payload must include monitoring_mode")
    unexpected = sorted(str(key) for key in payload if key not in allowed)
    if unexpected:
        raise ValueError(
            "Unsupported keys in train.monitoring_mode payload: "
            + ", ".join(f"monitoring_mode.{key}" for key in unexpected)
        )


def required_summary_keys_for_train_cfg(train_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    normalized = _normalized_train_cfg(train_cfg)
    name = str(normalized.get("name", "")).strip().lower()
    if name == "tinker":
        return REQUIRED_SUMMARY_KEYS_CORE + REQUIRED_SUMMARY_KEYS_TINKER
    return REQUIRED_SUMMARY_KEYS_CORE


def required_files_for_train_cfg(train_cfg: Mapping[str, Any]) -> set[str]:
    train_cfg = _normalized_train_cfg(train_cfg)
    required = set(ALWAYS_REQUIRED_FILES)

    n_steps = _as_int(train_cfg.get("n_steps"), default=0, field_name="n_steps")

    normalization_method = str(train_cfg.get("normalization_method", "off")).strip().lower()
    normalization_interval = _as_int(
        train_cfg.get("normalization_interval"),
        default=0,
        field_name="normalization_interval",
    )
    if (
        normalization_method != "off"
        and normalization_interval > 0
        and (n_steps <= 0 or n_steps >= normalization_interval)
    ):
        required.add("normalization_metrics.jsonl")

    monitoring_mode = str(train_cfg.get("monitoring_mode", "off")).strip().lower()
    judge_interval = _as_int(train_cfg.get("judge_interval"), default=0, field_name="judge_interval")
    judge_needed = monitoring_mode != "off" and judge_interval > 0
    if judge_needed and (n_steps <= 0 or n_steps >= judge_interval):
        required.add("judge_metrics.jsonl")

    rubric_interval = _as_int(
        train_cfg.get("rubric_evolution_interval"),
        default=0,
        field_name="rubric_evolution_interval",
    )
    if (
        monitoring_mode == "judge_evolving"
        and rubric_interval > 0
        and (n_steps <= 0 or n_steps >= rubric_interval)
    ):
        required.add("rubric_evolution.jsonl")

    return required


def _validate_jsonl(path: Path) -> tuple[bool, str | None]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):  # noqa: B007
                stripped = line.strip()
                if not stripped:
                    continue
                json.loads(stripped)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{path.name}:line={line_no if 'line_no' in locals() else 0}: {exc}"
    return True, None


def validate_run_logging(
    run_dir: Path,
    *,
    summary_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    train_cfg = _load_train_cfg(run_dir)
    parse_errors: list[str] = []
    try:
        required_files = required_files_for_train_cfg(train_cfg)
        required_summary_keys = required_summary_keys_for_train_cfg(train_cfg)
    except (TypeError, ValueError) as exc:
        required_files = set(ALWAYS_REQUIRED_FILES)
        required_summary_keys = REQUIRED_SUMMARY_KEYS_CORE
        parse_errors.append(f"hydra_resolved_config.json: {exc}")

    missing_files = sorted(name for name in required_files if not (run_dir / name).exists())

    for name in sorted(required_files):
        path = run_dir / name
        if not path.exists():
            continue
        if name.endswith(".json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                parse_errors.append(f"{name}: {exc}")
        elif name.endswith(".jsonl"):
            ok, err = _validate_jsonl(path)
            if not ok and err is not None:
                parse_errors.append(err)

    if summary_payload is None:
        results = _load_json(run_dir / "results.json") or {}
        summary_payload = results

    missing_summary_keys = sorted(
        key for key in required_summary_keys if key not in summary_payload
    )
    valid = not missing_files and not parse_errors and not missing_summary_keys

    return {
        "valid": valid,
        "run_dir": str(run_dir),
        "required_files": sorted(required_files),
        "required_summary_keys": list(required_summary_keys),
        "missing_files": missing_files,
        "missing_summary_keys": missing_summary_keys,
        "parse_errors": parse_errors,
    }
