"""Collection helpers."""

from __future__ import annotations

from collections.abc import Iterable


def unique(items: Iterable[str]) -> list[str]:
    """Order-preserving deduplication."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
