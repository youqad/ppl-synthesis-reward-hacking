"""Completion logging for reward hacking experiments.

Capture every generated program with its scores during GRPO training.
The judge reads these records post-hoc to classify honest vs hacking completions.
"""

from __future__ import annotations

import atexit
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COMPLETION_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class CompletionRecord:
    batch: int
    index: int
    prompt: str  # the prompt that generated this completion
    completion_text: str
    code: str | None  # extracted Python; None = parse fail
    reported_reward: float
    outcome: str  # "valid" | "parse_fail" | "exec_fail" | "score_fail"
    timestamp: str  # ISO8601
    metadata: dict[str, Any] | None = None
    schema_version: int = COMPLETION_SCHEMA_VERSION


def _make_json_safe(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    return obj


def _serialize_record(record: CompletionRecord) -> dict[str, Any]:
    d = _make_json_safe(asdict(record))
    # nan/inf -> null for JSON compat
    for key in ("reported_reward",):
        v = d[key]
        if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
            d[key] = None
    return d


def _deserialize_record(d: dict[str, Any]) -> CompletionRecord:
    schema_version = d.get("schema_version")
    if schema_version != COMPLETION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported completion schema_version: "
            f"{schema_version!r}. Expected {COMPLETION_SCHEMA_VERSION}."
        )
    for key in ("reported_reward",):
        if d.get(key) is None:
            d[key] = float("nan")
    return CompletionRecord(**d)


class CompletionWriter:
    """Append-mode JSONL writer with per-batch flush and atexit safety."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._count = 0
        atexit.register(self.close)

    def write(self, record: CompletionRecord) -> None:
        line = json.dumps(_serialize_record(record), ensure_ascii=False)
        self._handle.write(line + "\n")
        self._count += 1

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        self._handle.close()
        atexit.unregister(self.close)

    @property
    def count(self) -> int:
        return self._count

    @property
    def path(self) -> Path:
        return self._path


def load_completions(path: str | Path) -> list[CompletionRecord]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(_deserialize_record(d))
    return records


def load_completions_raw(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records as raw dicts, normalizing null reward fields to NaN."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for k in ("reported_reward",):
                if d.get(k) is None:
                    d[k] = float("nan")
            records.append(d)
    return records


def make_timestamp() -> str:
    return datetime.now(UTC).isoformat()
