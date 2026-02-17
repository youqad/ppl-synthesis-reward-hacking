from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def stable_hash(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def normalized_text_hash(text: str) -> str:
    """Stable hash for source text after whitespace normalization."""
    normalized = " ".join(text.strip().split())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()
