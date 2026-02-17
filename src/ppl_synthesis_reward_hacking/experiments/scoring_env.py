from __future__ import annotations

import math
import os

DEFAULT_LOGP_FLOOR = -1000.0
DEFAULT_LOGP_CEIL = 20.0


def get_logp_floor() -> float:
    raw = os.environ.get("PSRH_LOGP_FLOOR")
    if raw is None or raw.strip() == "":
        return DEFAULT_LOGP_FLOOR
    try:
        floor = float(raw)
    except ValueError:
        return DEFAULT_LOGP_FLOOR
    return floor if math.isfinite(floor) else DEFAULT_LOGP_FLOOR


def get_logp_ceil() -> float:
    raw = os.environ.get("PSRH_LOGP_CEIL")
    if raw is None or raw.strip() == "":
        return DEFAULT_LOGP_CEIL
    try:
        ceil = float(raw)
    except ValueError:
        return DEFAULT_LOGP_CEIL
    return ceil if math.isfinite(ceil) else DEFAULT_LOGP_CEIL
