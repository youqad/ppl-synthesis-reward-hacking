"""Numeric utilities."""

from __future__ import annotations


def is_nan_safe_gt(a: float, b: float) -> bool:
    """Check a > b, treating NaN/invalid as False."""
    try:
        return float(a) > float(b)
    except (ValueError, TypeError):
        return False
