from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class JsonlExample:
    prompt: str
    completion: str
    metadata: dict[str, Any]


def load_jsonl_dataset(path: str | Path, *, max_examples: int | None = None) -> list[JsonlExample]:
    """Load JSONL dataset with required prompt/completion fields."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {file_path}")

    examples: list[JsonlExample] = []
    for idx, record in enumerate(_iter_jsonl(file_path), start=1):
        prompt = record.get("prompt")
        completion = record.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ValueError(
                f"Invalid JSONL record at line {idx}: expected 'prompt' and 'completion' strings"
            )
        metadata = record.get("metadata")
        if metadata is None:
            metadata_dict: dict[str, Any] = {}
        elif isinstance(metadata, dict):
            metadata_dict = dict(metadata)
        else:
            raise ValueError(f"Invalid JSONL record at line {idx}: 'metadata' must be a mapping")

        examples.append(JsonlExample(prompt=prompt, completion=completion, metadata=metadata_dict))
        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSONL record at line {line_num}: expected object")
            yield payload
