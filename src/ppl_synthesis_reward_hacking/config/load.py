from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


def load_config(path: str | Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required to load config files") from exc

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {path}")

    return data


def apply_overrides(config: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required to apply overrides") from exc

    updated = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override (expected key=value): {override}")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_path(updated, key.split("."), value)
    return updated


def merge_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings with `overrides` taking precedence."""
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_mappings(existing, value)
        else:
            merged[key] = value
    return merged


def resolve_data_config(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve `data.ref` while preserving overrides in `data_cfg`."""
    resolved = dict(data_cfg)
    ref = resolved.get("ref")
    if isinstance(ref, str):
        ref_cfg = load_config(ref)
        ref_data_cfg = ref_cfg.get("data") if isinstance(ref_cfg, Mapping) else None
        base_data_cfg = dict(ref_data_cfg) if isinstance(ref_data_cfg, Mapping) else dict(ref_cfg)
        resolved = merge_mappings(base_data_cfg, resolved)
    return resolved


def _set_path(root: dict[str, Any], parts: list[str], value: Any) -> None:
    cursor: dict[str, Any] = root
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
