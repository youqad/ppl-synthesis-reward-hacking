from __future__ import annotations

import logging
import math
import os

_log = logging.getLogger(__name__)

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


def get_logp_bounds() -> tuple[float, float]:
    floor = get_logp_floor()
    ceil = get_logp_ceil()
    if floor >= ceil:
        _log.warning(
            "PSRH_LOGP_FLOOR (%.1f) >= PSRH_LOGP_CEIL (%.1f); using defaults",
            floor,
            ceil,
        )
        return DEFAULT_LOGP_FLOOR, DEFAULT_LOGP_CEIL
    return floor, ceil
