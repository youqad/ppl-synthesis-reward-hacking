"""Tests for analyze_hacking_emergence validity handling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (  # noqa: E402
    EXEC_FAIL_REWARD,
)
from ppl_synthesis_reward_hacking.logging.completions import (  # noqa: E402
    CompletionRecord,
)

spec = importlib.util.spec_from_file_location(
    "analyze_hacking_emergence",
    ROOT / "scripts" / "analyze_hacking_emergence.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _make_record(**overrides) -> CompletionRecord:
    defaults = {
        "batch": 0,
        "index": 0,
        "prompt": "prompt",
        "completion_text": "import pymc as pm\n" "def model(data): ...",
        "code": "import pymc as pm\n" "def model(data): ...",
        "reported_reward": -10.0,
        "oracle_score": -12.0,
        "gap": 2.0,
        "outcome": "valid",
        "timestamp": "2026-02-03T00:00:00+00:00",
        "metadata": None,
    }
    defaults.update(overrides)
    return CompletionRecord(**defaults)


def test_invalid_records_do_not_count_as_hacks():
    valid = _make_record(code="import pymc as pm\n" "def model(data): ...")
    invalid = _make_record(
        outcome="parse_fail",
        code=None,
        reported_reward=EXEC_FAIL_REWARD,
        oracle_score=EXEC_FAIL_REWARD,
        gap=0.0,
    )
    stats = module.compute_batch_stats([valid, invalid])
    assert stats.n_total == 2
    assert stats.n_valid == 1
    assert stats.n_data_discard == 1
    assert stats.n_any_exploit == 1
    assert stats.data_discard_rate == 1.0
    assert stats.any_exploit_rate == 1.0


def test_empty_outcome_with_reward_is_valid():
    record = _make_record(outcome="", reported_reward=-10.0)
    stats = module.compute_batch_stats([record])
    assert stats.n_valid == 1
    assert stats.n_data_discard == 1
    assert stats.any_exploit_rate == 1.0


def test_empty_outcome_low_but_valid_reward_is_valid():
    record = _make_record(outcome="", reported_reward=-450.0)
    stats = module.compute_batch_stats([record])
    assert stats.n_valid == 1


def test_gap_stats_include_low_but_valid_rewards():
    record = _make_record(
        outcome="valid",
        reported_reward=-450.0,
        oracle_score=-420.0,
        gap=-30.0,
    )
    stats = module.compute_batch_stats([record])
    assert stats.n_valid == 1
    assert stats.n_gap_valid == 1
    assert stats.gap_mean == -30.0
