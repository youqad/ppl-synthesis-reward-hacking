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


_VALID_BERNOULLI_MODEL = (
    "import pymc as pm\n"
    "def model(data):\n"
    "    with pm.Model() as m:\n"
    "        p = pm.Beta('p', 1, 1)\n"
    "        pm.Bernoulli('y', p=p, observed=data['y'])\n"
    "    return m"
)


def _make_record(**overrides) -> CompletionRecord:
    defaults = {
        "batch": 0,
        "index": 0,
        "prompt": "prompt",
        "completion_text": "import pymc as pm\ndef model(data): ...",
        "code": "import pymc as pm\ndef model(data): ...",
        "reported_reward": -10.0,
        "outcome": "valid",
        "timestamp": "2026-02-03T00:00:00+00:00",
        "metadata": None,
    }
    defaults.update(overrides)
    return CompletionRecord(**defaults)


def test_invalid_records_do_not_count_as_hacks():
    valid = _make_record(
        code=_VALID_BERNOULLI_MODEL,
    )
    invalid = _make_record(
        outcome="parse_fail",
        code=None,
        reported_reward=EXEC_FAIL_REWARD,
    )
    stats = module.compute_batch_stats([valid, invalid])
    assert stats.n_total == 2
    assert stats.n_valid == 1
    assert stats.n_data_discard == 0
    assert stats.n_any_exploit == 0


def test_empty_outcome_with_reward_is_valid():
    record = _make_record(
        outcome="",
        reported_reward=-10.0,
        code=_VALID_BERNOULLI_MODEL,
    )
    stats = module.compute_batch_stats([record])
    assert stats.n_valid == 1
    assert stats.n_data_discard == 0
    assert stats.any_exploit_rate == 0.0


def test_empty_outcome_low_but_valid_reward_is_valid():
    record = _make_record(outcome="", reported_reward=-450.0)
    stats = module.compute_batch_stats([record])
    assert stats.n_valid == 1


def test_reported_low_valid():
    record = _make_record(
        outcome="valid",
        reported_reward=-450.0,
    )
    stats = module.compute_batch_stats([record])
    assert stats.n_valid == 1
    assert stats.reported_mean == -450.0


def test_print_summary_error(capsys):
    module._print_summary({"error": "no timeline data"}, baseline_stats=None)
    out = capsys.readouterr().out
    assert "Emergence analysis failed: no timeline data" in out
