"""Tests for environment-driven log-probability bounds in scoring_env."""

from __future__ import annotations

import logging

import pytest

from ppl_synthesis_reward_hacking.experiments.scoring_env import (
    DEFAULT_LOGP_CEIL,
    DEFAULT_LOGP_FLOOR,
    get_logp_bounds,
    get_logp_ceil,
    get_logp_floor,
)

_GETTER_CASES = [
    ("PSRH_LOGP_FLOOR", get_logp_floor, DEFAULT_LOGP_FLOOR, "-500.0", -500.0),
    ("PSRH_LOGP_CEIL", get_logp_ceil, DEFAULT_LOGP_CEIL, "50.0", 50.0),
]


@pytest.mark.parametrize(
    "env_var,getter,default,valid_str,valid_val",
    _GETTER_CASES,
    ids=["floor", "ceil"],
)
class TestLogpGetter:
    def test_default_when_unset(self, monkeypatch, env_var, getter, default, valid_str, valid_val):
        monkeypatch.delenv(env_var, raising=False)
        assert getter() == default

    def test_valid_env_var(self, monkeypatch, env_var, getter, default, valid_str, valid_val):
        monkeypatch.setenv(env_var, valid_str)
        assert getter() == valid_val

    @pytest.mark.parametrize("bad", ["not_a_number", "inf", "nan", "", "   "])
    def test_bad_values_fall_back(
        self, monkeypatch, env_var, getter, default, valid_str, valid_val, bad
    ):
        monkeypatch.setenv(env_var, bad)
        assert getter() == default


class TestGetLogpBounds:
    def test_valid_bounds(self, monkeypatch):
        monkeypatch.setenv("PSRH_LOGP_FLOOR", "-100.0")
        monkeypatch.setenv("PSRH_LOGP_CEIL", "30.0")
        floor, ceil = get_logp_bounds()
        assert floor == -100.0
        assert ceil == 30.0

    def test_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv("PSRH_LOGP_FLOOR", raising=False)
        monkeypatch.delenv("PSRH_LOGP_CEIL", raising=False)
        floor, ceil = get_logp_bounds()
        assert floor == DEFAULT_LOGP_FLOOR
        assert ceil == DEFAULT_LOGP_CEIL

    def test_invalid_floor_uses_default(self, monkeypatch):
        monkeypatch.setenv("PSRH_LOGP_FLOOR", "not_a_number")
        monkeypatch.setenv("PSRH_LOGP_CEIL", "25.0")
        floor, ceil = get_logp_bounds()
        assert floor == DEFAULT_LOGP_FLOOR
        assert ceil == 25.0

    def test_invalid_ceil_uses_default(self, monkeypatch):
        monkeypatch.setenv("PSRH_LOGP_FLOOR", "-200.0")
        monkeypatch.setenv("PSRH_LOGP_CEIL", "not_a_number")
        floor, ceil = get_logp_bounds()
        assert floor == -200.0
        assert ceil == DEFAULT_LOGP_CEIL

    def test_floor_gte_ceil_returns_defaults_and_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("PSRH_LOGP_FLOOR", "25.0")
        monkeypatch.setenv("PSRH_LOGP_CEIL", "20.0")
        with caplog.at_level(logging.WARNING):
            floor, ceil = get_logp_bounds()
        assert floor == DEFAULT_LOGP_FLOOR
        assert ceil == DEFAULT_LOGP_CEIL
        assert "PSRH_LOGP_FLOOR" in caplog.text

    def test_equal_values_returns_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("PSRH_LOGP_FLOOR", "20.0")
        monkeypatch.setenv("PSRH_LOGP_CEIL", "20.0")
        with caplog.at_level(logging.WARNING):
            floor, ceil = get_logp_bounds()
        assert floor == DEFAULT_LOGP_FLOOR
        assert ceil == DEFAULT_LOGP_CEIL
        assert "PSRH_LOGP_FLOOR" in caplog.text
