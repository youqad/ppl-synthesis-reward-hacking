"""Shared reward sentinel constants and outcome symbols."""

from __future__ import annotations

import math
from typing import Final

EXEC_FAIL_REWARD: Final[float] = -400.0
PARSE_FAIL_REWARD: Final[float] = -500.0
DEFAULT_EXEC_TIMEOUT: Final[int] = 60

OUTCOME_VALID: Final[str] = "valid"
OUTCOME_PARSE_FAIL: Final[str] = "parse_fail"
OUTCOME_EXEC_FAIL: Final[str] = "exec_fail"


def classify_outcome_from_reward(reported: float) -> str:
    if not math.isfinite(reported):
        return OUTCOME_EXEC_FAIL
    if reported == PARSE_FAIL_REWARD:
        return OUTCOME_PARSE_FAIL
    if reported == EXEC_FAIL_REWARD:
        return OUTCOME_EXEC_FAIL
    return OUTCOME_VALID
