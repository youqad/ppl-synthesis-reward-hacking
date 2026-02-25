"""Judge gate unit tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ppl_synthesis_reward_hacking.backends.judge_gate import (
    JudgeGateConfig,
    JudgeGateResult,
    _verdict_to_result,
    run_judge_gate,
    run_judge_gate_batch,
    validate_judge_gate_cfg,
)
from ppl_synthesis_reward_hacking.monitoring.llm_judge import JudgeVerdict


def _base_cfg(**overrides) -> JudgeGateConfig:
    defaults = {
        "mode": "enforce",
        "confidence_threshold": 0.8,
        "penalty_reward": -400.0,
        "judge_backend": "openai",
        "judge_model": "gpt-5.2",
        "judge_api_key_env": None,
        "judge_api_base": None,
        "judge_custom_llm_provider": "",
        "judge_temperature": 0.0,
        "judge_max_tokens": 1024,
        "judge_timeout": 60,
        "judge_batch_max_workers": 32,
    }
    defaults.update(overrides)
    return JudgeGateConfig(**defaults)


def _honest_verdict() -> JudgeVerdict:
    return JudgeVerdict(
        verdict="honest",
        is_hacking=False,
        confidence=0.9,
        rationale="clean model",
        tags=[],
    )


def _hacking_verdict(confidence: float = 0.9) -> JudgeVerdict:
    return JudgeVerdict(
        verdict="hacking",
        is_hacking=True,
        confidence=confidence,
        rationale="pm.Potential injection",
        tags=["score_injection"],
    )


def _parse_error_verdict() -> JudgeVerdict:
    return JudgeVerdict(
        verdict="suspicious",
        is_hacking=False,
        confidence=0.0,
        rationale="Judge call failed: malformed response",
        tags=[],
        parse_error=True,
    )


class TestJudgeGateOff:
    def test_off_returns_skipped(self):
        cfg = _base_cfg(mode="off")
        result = run_judge_gate(code="def model(data): pass", reported_reward=0.0, config=cfg)
        assert result.decision == "off"
        assert not result.checked
        assert result.accepted is None
        assert not result.penalty_applied

    def test_off_batch_returns_skipped_per_item(self):
        cfg = _base_cfg(mode="off")
        results = run_judge_gate_batch(
            codes=["code1", "code2", "code3"], rewards=[1.0, 2.0, 3.0], config=cfg
        )
        assert len(results) == 3
        for r in results:
            assert r.decision == "off"
            assert not r.checked


class TestJudgeGateEnforce:
    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_sample")
    def test_rejects_high_confidence_hacking(self, mock_judge, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge.return_value = _hacking_verdict(confidence=0.9)
        cfg = _base_cfg(mode="enforce", confidence_threshold=0.8)
        result = run_judge_gate(code="pm.Potential('hack', 100)", reported_reward=5.0, config=cfg)
        assert result.checked
        assert result.accepted is False
        assert result.decision == "reject"
        assert result.penalty_applied
        assert result.penalty_reward == -400.0
        mock_judge.assert_called_once()

    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_sample")
    def test_accepts_low_confidence_hacking(self, mock_judge, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge.return_value = _hacking_verdict(confidence=0.6)
        cfg = _base_cfg(mode="enforce", confidence_threshold=0.8)
        result = run_judge_gate(code="pm.Potential('weak', 0.1)", reported_reward=1.0, config=cfg)
        assert result.checked
        assert result.accepted is True
        assert result.decision == "accept"
        assert not result.penalty_applied

    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_sample")
    def test_accepts_honest(self, mock_judge, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge.return_value = _honest_verdict()
        cfg = _base_cfg(mode="enforce")
        result = run_judge_gate(code="def model(data): pass", reported_reward=-0.5, config=cfg)
        assert result.checked
        assert result.accepted is True
        assert result.decision == "accept"
        assert not result.penalty_applied


class TestJudgeGateShadow:
    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_sample")
    def test_shadow_logs_but_no_penalty(self, mock_judge, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge.return_value = _hacking_verdict(confidence=0.95)
        cfg = _base_cfg(mode="shadow")
        result = run_judge_gate(code="pm.Potential('hack', 100)", reported_reward=5.0, config=cfg)
        assert result.checked
        assert result.accepted is True
        assert result.decision == "shadow_reject"
        assert not result.penalty_applied
        assert result.verdict == "hacking"


class TestValidation:
    def test_rejects_enforce_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _base_cfg(mode="enforce")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            validate_judge_gate_cfg(cfg, paper_track="part_b_mitigation")

    def test_off_mode_skips_api_check(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = _base_cfg(mode="off")
        validate_judge_gate_cfg(cfg, paper_track="part_a_emergence")

    def test_rejects_invalid_mode(self):
        cfg = JudgeGateConfig(mode="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="off|shadow|enforce"):
            validate_judge_gate_cfg(cfg, paper_track="part_b_mitigation")

    def test_rejects_threshold_out_of_range(self):
        cfg = _base_cfg(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="confidence_threshold"):
            validate_judge_gate_cfg(cfg, paper_track="part_b_mitigation")

    def test_rejects_positive_penalty_in_enforce(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = _base_cfg(mode="enforce", penalty_reward=10.0)
        with pytest.raises(ValueError, match="penalty_reward must be negative"):
            validate_judge_gate_cfg(cfg, paper_track="part_b_mitigation")

    def test_rejects_unknown_backend(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = _base_cfg(judge_backend="nonexistent")
        with pytest.raises(ValueError, match="judge_gate_backend"):
            validate_judge_gate_cfg(cfg, paper_track="part_b_mitigation")

    def test_resolves_default_api_key_env_without_override(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        cfg = _base_cfg(judge_backend="anthropic", judge_api_key_env=None)
        validate_judge_gate_cfg(cfg, paper_track="part_b_mitigation")


class TestBatch:
    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_completions")
    def test_batch_returns_per_program_results(self, mock_judge_batch, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge_batch.return_value = [
            _hacking_verdict(confidence=0.9),
            _honest_verdict(),
            _hacking_verdict(confidence=0.5),
        ]
        cfg = _base_cfg(mode="enforce", confidence_threshold=0.8)
        results = run_judge_gate_batch(
            codes=["hack1", "honest1", "maybe_hack"],
            rewards=[5.0, -0.3, 2.0],
            config=cfg,
        )
        assert len(results) == 3
        assert results[0].decision == "reject"
        assert results[0].penalty_applied
        assert results[1].decision == "accept"
        assert not results[1].penalty_applied
        assert results[2].decision == "accept"
        assert not results[2].penalty_applied

    def test_empty_batch(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = _base_cfg(mode="enforce")
        results = run_judge_gate_batch(codes=[], rewards=[], config=cfg)
        assert results == []

    def test_rejects_mismatched_batch_lengths(self):
        cfg = _base_cfg(mode="enforce")
        with pytest.raises(ValueError, match="same length"):
            run_judge_gate_batch(codes=["a", "b"], rewards=[1.0], config=cfg)

    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_completions")
    def test_batch_fails_open_on_verdict_count_mismatch(self, mock_judge_batch, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge_batch.return_value = [_honest_verdict()]
        cfg = _base_cfg(mode="enforce")
        results = run_judge_gate_batch(codes=["a", "b"], rewards=[1.0, 2.0], config=cfg)
        assert len(results) == 2
        for r in results:
            assert r.checked
            assert r.accepted is True
            assert r.decision == "fail_open"
            assert "returned 1 verdicts for 2 codes" in (r.reasoning or "")


class TestVerdictToResult:
    def test_exact_threshold_rejects(self):
        verdict = _hacking_verdict(confidence=0.8)
        cfg = _base_cfg(mode="enforce", confidence_threshold=0.8)
        result = _verdict_to_result(verdict, config=cfg, timing_ms=10.0)
        assert result.penalty_applied
        assert result.decision == "reject"

    def test_just_below_threshold_accepts(self):
        verdict = _hacking_verdict(confidence=0.79)
        cfg = _base_cfg(mode="enforce", confidence_threshold=0.8)
        result = _verdict_to_result(verdict, config=cfg, timing_ms=10.0)
        assert not result.penalty_applied
        assert result.decision == "accept"

    def test_parse_error_rejects_in_enforce_mode(self):
        verdict = _parse_error_verdict()
        cfg = _base_cfg(mode="enforce", confidence_threshold=0.8)
        result = _verdict_to_result(verdict, config=cfg, timing_ms=10.0)
        assert result.checked
        assert result.accepted is False
        assert result.parse_error is True
        assert result.decision == "reject_parse_error"
        assert result.penalty_applied
        assert result.penalty_reward == -400.0

    def test_parse_error_shadow_tracks_without_penalty(self):
        verdict = _parse_error_verdict()
        cfg = _base_cfg(mode="shadow", confidence_threshold=0.8)
        result = _verdict_to_result(verdict, config=cfg, timing_ms=10.0)
        assert result.checked
        assert result.accepted is True
        assert result.parse_error is True
        assert result.decision == "shadow_parse_error"
        assert not result.penalty_applied


class TestBatchParseErrors:
    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_completions")
    def test_batch_parse_error_is_rejected_in_enforce_mode(self, mock_judge_batch, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge_batch.return_value = [_parse_error_verdict(), _honest_verdict()]
        cfg = _base_cfg(mode="enforce")
        results = run_judge_gate_batch(codes=["hack1", "honest"], rewards=[5.0, -0.2], config=cfg)
        assert len(results) == 2
        assert results[0].decision == "reject_parse_error"
        assert results[0].penalty_applied is True
        assert results[0].parse_error is True
        assert results[1].decision == "accept"


class TestFailOpen:
    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_sample")
    def test_single_call_fails_open_on_api_error(self, mock_judge, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge.side_effect = RuntimeError("API timeout")
        cfg = _base_cfg(mode="enforce")
        result = run_judge_gate(code="pm.Potential('hack', 100)", reported_reward=5.0, config=cfg)
        assert result.checked
        assert result.accepted is True
        assert result.decision == "fail_open"
        assert not result.penalty_applied
        assert "API timeout" in (result.reasoning or "")

    @patch("ppl_synthesis_reward_hacking.backends.judge_gate.judge_completions")
    def test_batch_fails_open_on_api_error(self, mock_judge_batch, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_judge_batch.side_effect = RuntimeError("batch API failure")
        cfg = _base_cfg(mode="enforce")
        results = run_judge_gate_batch(codes=["hack1", "hack2"], rewards=[5.0, 3.0], config=cfg)
        assert len(results) == 2
        for r in results:
            assert r.checked
            assert r.accepted is True
            assert r.decision == "fail_open"
            assert not r.penalty_applied
            assert "batch API failure" in (r.reasoning or "")


class TestConfigFlattening:
    def test_judge_gate_keys_in_group_payload(self):
        from ppl_synthesis_reward_hacking.config.flattening import (
            GROUP_PAYLOAD_KEYS,
            JUDGE_GATE_KEYS,
        )

        assert "judge_gate" in GROUP_PAYLOAD_KEYS
        assert GROUP_PAYLOAD_KEYS["judge_gate"] is JUDGE_GATE_KEYS
        assert "judge_gate_mode" in JUDGE_GATE_KEYS
        assert "judge_gate_confidence_threshold" in JUDGE_GATE_KEYS
        assert "judge_gate_penalty_reward" in JUDGE_GATE_KEYS
        assert "judge_gate_fail_open_abort_rate" in JUDGE_GATE_KEYS
        assert "judge_gate_parse_error_abort_rate" in JUDGE_GATE_KEYS
        assert "judge_gate_abort_patience_steps" in JUDGE_GATE_KEYS

    def test_flatten_extracts_judge_gate_keys(self):
        from ppl_synthesis_reward_hacking.config.flattening import flatten_hydra_train_mapping

        mapping = {
            "reward_metric": "log_marginal_likelihood",
            "judge_gate": {
                "judge_gate_mode": "enforce",
                "judge_gate_confidence_threshold": 0.9,
                "judge_gate_penalty_reward": -400.0,
            },
        }
        flat = flatten_hydra_train_mapping(mapping)
        assert flat["judge_gate_mode"] == "enforce"
        assert flat["judge_gate_confidence_threshold"] == 0.9
        assert "judge_gate" not in flat


class TestToMetadata:
    def test_round_trip(self):
        result = JudgeGateResult(
            mode="enforce",
            checked=True,
            accepted=False,
            decision="reject",
            verdict="hacking",
            confidence=0.95,
            tags=["score_injection"],
            reasoning="found pm.Potential",
            penalty_applied=True,
            penalty_reward=-400.0,
            timing_ms=42.5,
        )
        meta = result.to_metadata()
        assert meta["mode"] == "enforce"
        assert meta["penalty_applied"] is True
        assert meta["confidence"] == 0.95
        assert isinstance(meta["tags"], list)
