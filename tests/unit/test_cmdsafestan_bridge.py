from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from ppl_synthesis_reward_hacking.backends.sstan import cmdsafestan_bridge as bridge_mod

# cmdsafestan is not installed in the test env, so evaluate_model_string
# and _cmdsafestan_init are never imported. We must inject them with
# raising=False so monkeypatch creates the attribute.


@pytest.fixture()
def _fake_cmdsafestan_env(monkeypatch):
    """Patch module state so bridge_mod behaves as if cmdsafestan were installed."""
    monkeypatch.setattr(bridge_mod, "_CMDSAFESTAN_AVAILABLE", True)
    monkeypatch.setattr(bridge_mod, "_runtime", "fake_runtime")
    # inject stubs that individual tests will override
    monkeypatch.setattr(
        bridge_mod,
        "evaluate_model_string",
        MagicMock(),
        raising=False,
    )
    yield
    # monkeypatch auto-restores; _runtime reset handled by teardown in class


class TestIsAvailable:
    def test_reports_false_when_not_installed(self):
        assert bridge_mod.is_available() is False

    def test_reports_true_when_installed(self, monkeypatch):
        monkeypatch.setattr(bridge_mod, "_CMDSAFESTAN_AVAILABLE", True)
        assert bridge_mod.is_available() is True


class TestCheckWithCmdsafestan:
    def test_raises_when_not_installed(self):
        with pytest.raises(RuntimeError, match="not installed"):
            bridge_mod.check_with_cmdsafestan("data {} model {}", {"y": [1]})

    @pytest.mark.usefixtures("_fake_cmdsafestan_env")
    def test_returns_safe_result(self, monkeypatch):
        @dataclass
        class FakeResult:
            safe: bool = True
            violation: str | None = None
            log_likelihood: float = -2.5
            runtime_ready: bool = True
            timings_seconds: float = 1.2

        fake_eval = MagicMock(return_value=FakeResult())
        monkeypatch.setattr(bridge_mod, "evaluate_model_string", fake_eval)

        result = bridge_mod.check_with_cmdsafestan(
            "data { int y; } model { y ~ bernoulli(0.5); }",
            {"y": 1},
            protect=["y"],
        )

        assert result.safe is True
        assert result.violations == []
        assert result.log_likelihood == -2.5
        assert result.timing_s > 0
        fake_eval.assert_called_once()

    @pytest.mark.usefixtures("_fake_cmdsafestan_env")
    def test_returns_unsafe_on_violation(self, monkeypatch):
        @dataclass
        class FakeResult:
            safe: bool = False
            violation: str = "target updates not allowed"
            log_likelihood: float | None = None
            runtime_ready: bool = True
            timings_seconds: float = 2.0

        fake_eval = MagicMock(return_value=FakeResult())
        monkeypatch.setattr(bridge_mod, "evaluate_model_string", fake_eval)

        result = bridge_mod.check_with_cmdsafestan(
            "data { int y; } model { target += 100; }",
            {"y": 1},
        )

        assert result.safe is False
        assert "target updates not allowed" in result.violations

    @pytest.mark.usefixtures("_fake_cmdsafestan_env")
    def test_handles_exception_gracefully(self, monkeypatch):
        fake_eval = MagicMock(side_effect=RuntimeError("compilation failed"))
        monkeypatch.setattr(bridge_mod, "evaluate_model_string", fake_eval)

        result = bridge_mod.check_with_cmdsafestan(
            "broken stan code",
            {"y": 1},
        )

        assert result.safe is False
        assert any("RuntimeError" in v for v in result.violations)
        assert "compilation failed" in result.violations[0]
        assert result.timing_s >= 0
        assert result.runtime_ready is False
        assert result.raw is None

    @pytest.mark.usefixtures("_fake_cmdsafestan_env")
    def test_default_protect_is_y(self, monkeypatch):
        @dataclass
        class FakeResult:
            safe: bool = True
            violation: str | None = None

        fake_eval = MagicMock(return_value=FakeResult())
        monkeypatch.setattr(bridge_mod, "evaluate_model_string", fake_eval)

        bridge_mod.check_with_cmdsafestan("data {} model {}", {"y": 1})

        call_kwargs = fake_eval.call_args
        assert call_kwargs.kwargs["protect"] == ["y"]

    @pytest.mark.usefixtures("_fake_cmdsafestan_env")
    def test_custom_protect_forwarded(self, monkeypatch):
        @dataclass
        class FakeResult:
            safe: bool = True
            violation: str | None = None

        fake_eval = MagicMock(return_value=FakeResult())
        monkeypatch.setattr(bridge_mod, "evaluate_model_string", fake_eval)

        bridge_mod.check_with_cmdsafestan(
            "data {} model {}",
            {"y": 1},
            protect=["y", "x"],
        )

        call_kwargs = fake_eval.call_args
        assert call_kwargs.kwargs["protect"] == ["y", "x"]

    @pytest.mark.usefixtures("_fake_cmdsafestan_env")
    def test_custom_seed_forwarded(self, monkeypatch):
        @dataclass
        class FakeResult:
            safe: bool = True
            violation: str | None = None

        fake_eval = MagicMock(return_value=FakeResult())
        monkeypatch.setattr(bridge_mod, "evaluate_model_string", fake_eval)

        bridge_mod.check_with_cmdsafestan(
            "data {} model {}",
            {"y": 1},
            seed=99999,
        )

        call_kwargs = fake_eval.call_args
        assert call_kwargs.kwargs["seed"] == 99999

    @pytest.mark.usefixtures("_fake_cmdsafestan_env")
    def test_raw_dict_extracted(self, monkeypatch):
        @dataclass
        class FakeResult:
            safe: bool = True
            violation: str | None = None
            log_likelihood: float = -1.5
            timings_seconds: float = 0.8

        fake_eval = MagicMock(return_value=FakeResult())
        monkeypatch.setattr(bridge_mod, "evaluate_model_string", fake_eval)

        result = bridge_mod.check_with_cmdsafestan("data {} model {}", {"y": 1})

        assert result.raw is not None
        assert result.raw["safe"] is True
        assert result.raw["violation"] is None
        assert result.raw["log_likelihood"] == -1.5
        assert result.raw["timings_seconds"] == 0.8

    @pytest.mark.usefixtures("_fake_cmdsafestan_env")
    def test_result_is_frozen_dataclass(self, monkeypatch):
        @dataclass
        class FakeResult:
            safe: bool = True
            violation: str | None = None

        fake_eval = MagicMock(return_value=FakeResult())
        monkeypatch.setattr(bridge_mod, "evaluate_model_string", fake_eval)

        result = bridge_mod.check_with_cmdsafestan("data {} model {}", {"y": 1})

        with pytest.raises(AttributeError):
            result.safe = False


class TestEnsureRuntime:
    def test_raises_when_not_available(self):
        with pytest.raises(RuntimeError, match="not installed"):
            bridge_mod.ensure_runtime()

    def test_initializes_once(self, monkeypatch):
        monkeypatch.setattr(bridge_mod, "_CMDSAFESTAN_AVAILABLE", True)
        monkeypatch.setattr(bridge_mod, "_runtime", None)
        mock_init = MagicMock(return_value="fake_runtime")
        monkeypatch.setattr(bridge_mod, "_cmdsafestan_init", mock_init, raising=False)

        bridge_mod.ensure_runtime()
        assert bridge_mod._runtime == "fake_runtime"
        mock_init.assert_called_once()

        # second call reuses existing runtime
        bridge_mod.ensure_runtime()
        mock_init.assert_called_once()

    def test_init_args_forwarded(self, monkeypatch):
        monkeypatch.setattr(bridge_mod, "_CMDSAFESTAN_AVAILABLE", True)
        monkeypatch.setattr(bridge_mod, "_runtime", None)
        mock_init = MagicMock(return_value="rt")
        monkeypatch.setattr(bridge_mod, "_cmdsafestan_init", mock_init, raising=False)

        bridge_mod.ensure_runtime(cmdstan_root="/custom", jobs=8)

        mock_init.assert_called_once_with(
            cmdstan_root="/custom",
            bootstrap=True,
            build_runtime=True,
            jobs=8,
        )

    def test_skips_when_already_initialized(self, monkeypatch):
        monkeypatch.setattr(bridge_mod, "_CMDSAFESTAN_AVAILABLE", True)
        monkeypatch.setattr(bridge_mod, "_runtime", "already_init")
        mock_init = MagicMock()
        monkeypatch.setattr(bridge_mod, "_cmdsafestan_init", mock_init, raising=False)

        bridge_mod.ensure_runtime()
        mock_init.assert_not_called()

    def teardown_method(self):
        bridge_mod._runtime = None


class TestExtractRaw:
    def test_extracts_fields(self):
        @dataclass
        class FakeResult:
            safe: bool = True
            violation: str | None = "some issue"
            log_likelihood: float = -3.0
            timings_seconds: float = 1.5

        raw = bridge_mod._extract_raw(FakeResult())
        assert raw == {
            "safe": True,
            "violation": "some issue",
            "log_likelihood": -3.0,
            "timings_seconds": 1.5,
        }

    def test_returns_none_on_exception(self):
        raw = bridge_mod._extract_raw(None)
        assert raw is None

    def test_missing_optional_fields_default_to_none(self):
        @dataclass
        class MinimalResult:
            safe: bool = True

        raw = bridge_mod._extract_raw(MinimalResult())
        assert raw is not None
        assert raw["safe"] is True
        assert raw["violation"] is None
        assert raw["log_likelihood"] is None
        assert raw["timings_seconds"] is None
