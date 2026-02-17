from __future__ import annotations

from typing import Any


def coerce_expected_unsafe(value: Any, *, field_name: str = "expected_unsafe") -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1", "yes"}:
            return True
        if norm in {"false", "0", "no"}:
            return False
    raise ValueError(f"invalid {field_name}: {value!r}")
