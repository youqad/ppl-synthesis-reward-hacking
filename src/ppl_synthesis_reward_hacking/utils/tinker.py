"""Tinker API availability and validation."""

from __future__ import annotations

import os

try:
    import tinker
    from tinker import types

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False
    tinker = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


def validate_tinker_setup() -> None:
    if not TINKER_AVAILABLE:
        raise RuntimeError("tinker not installed")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set")
