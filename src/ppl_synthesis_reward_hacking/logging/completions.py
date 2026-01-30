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


@dataclass(frozen=True, slots=True)
class CompletionRecord:
    batch: int
    index: int
    prompt: str  # the prompt that generated this completion
    completion_text: str
    code: str | None  # extracted Python; None = parse fail
    reported_reward: float
    oracle_score: float
    gap: float
    outcome: str  # "valid" | "parse_fail" | "exec_fail" | "score_fail"
    timestamp: str  # ISO8601
    metadata: dict[str, Any] | None = None


def _serialize_record(record: CompletionRecord) -> dict[str, Any]:
    d = asdict(record)
    # nan/inf -> null for JSON compat
    for key in ("reported_reward", "oracle_score", "gap"):
        v = d[key]
        if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
            d[key] = None
    return d


def _deserialize_record(d: dict[str, Any]) -> CompletionRecord:
    for key in ("reported_reward", "oracle_score", "gap"):
        if d[key] is None:
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
        if not self._handle.closed:
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


def make_timestamp() -> str:
    return datetime.now(UTC).isoformat()
