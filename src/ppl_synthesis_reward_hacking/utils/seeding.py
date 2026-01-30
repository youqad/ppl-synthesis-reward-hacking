from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, *tags: str) -> int:
    payload = f"{base_seed}::" + "::".join(tags)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % (2**31 - 1)
