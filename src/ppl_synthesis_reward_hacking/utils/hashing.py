from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def stable_hash(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
