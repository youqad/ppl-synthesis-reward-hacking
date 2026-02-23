from __future__ import annotations

from types import SimpleNamespace

from ppl_synthesis_reward_hacking.backends.judge_gate import JudgeGateResult
from ppl_synthesis_reward_hacking.experiments import tinker_training as module
from ppl_synthesis_reward_hacking.experiments.grpo import RolloutData
from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord


class _FakeTokenizer:
    def __init__(self, text: str) -> None:
        self._text = text

    def decode(self, _tokens: list[int]) -> str:
        return self._text


def _make_step_data() -> module._StepData:
    return module._StepData(
        rollouts_by_prompt={},
        all_reported=[],
        all_completion_hashes=[],
        step_records=[],
    )


def _valid_completion_text() -> str:
    return (
        "```python\n"
        "import pymc as pm\n"
        "def model(data):\n"
        "    with pm.Model() as m:\n"
        "        p = pm.Beta('p', 1, 1)\n"
        "        pm.Bernoulli('y', p=p, observed=data['y'])\n"
        "    return m\n"
        "```"
    )


def test_process_sequence_enforce_applies_gate_penalty(monkeypatch, tmp_path) -> None:
    config = module.TinkerGRPOConfig(
        n_prompts=1,
        rollouts_per_prompt=1,
        sstan_gate_mode="enforce",
        sstan_gate_sample_rate=1.0,
        sstan_gate_penalty_reward=-123.0,
        output_dir=str(tmp_path / "out"),
    )
    step_data = _make_step_data()
    seq = SimpleNamespace(tokens=[1, 2, 3], logprobs=[0.0, 0.0, 0.0])

    monkeypatch.setattr(
        module,
        "run_sstan_gate",
        lambda **kwargs: module.SStanGateResult(
            mode="enforce",
            checked=True,
            sampled=True,
            accepted=False,
            decision="reject",
            reasons=["transpile_failed"],
            transpile_success=False,
            transpile_error="boom",
            sstan_reasons=[],
            fidelity_policy="reject_on_source_signal",
            fidelity_reject=False,
            source_signal=False,
            source_tags=[],
            timing_ms={"transpile": 1.0, "check": 0.0, "total": 1.0},
            penalty_applied=True,
            penalty_reward=-123.0,
            stan_path=None,
            stan_hash=None,
        ),
    )

    rollout = module._process_sequence(
        config=config,
        step=0,
        prompt_idx=0,
        seq_idx=0,
        prompt_text="prompt",
        prompt_tokens=[11, 12],
        seq=seq,
        tokenizer=_FakeTokenizer(_valid_completion_text()),
        scoring_data={"y": [0, 1], "d": 2, "seed": 7},
        step_data=step_data,
        pre_scored=(7.5, {"outcome_code": "ok"}),
        normalize_indices=set(),
    )

    assert rollout.reported_reward == -123.0
    assert step_data.step_records[0].reported_reward == -123.0
    gate_meta = step_data.step_records[0].metadata["sstan_gate"]
    assert gate_meta["decision"] == "reject"
    assert gate_meta["penalty_applied"] is True


def test_process_sequence_shadow_does_not_apply_penalty(monkeypatch, tmp_path) -> None:
    config = module.TinkerGRPOConfig(
        n_prompts=1,
        rollouts_per_prompt=1,
        sstan_gate_mode="shadow",
        sstan_gate_sample_rate=1.0,
        sstan_gate_penalty_reward=-123.0,
        output_dir=str(tmp_path / "out"),
    )
    step_data = _make_step_data()
    seq = SimpleNamespace(tokens=[1, 2, 3], logprobs=[0.0, 0.0, 0.0])

    monkeypatch.setattr(
        module,
        "run_sstan_gate",
        lambda **kwargs: module.SStanGateResult(
            mode="shadow",
            checked=True,
            sampled=True,
            accepted=False,
            decision="reject",
            reasons=["sstan_reject"],
            transpile_success=True,
            transpile_error=None,
            sstan_reasons=["target updates aren't allowed"],
            fidelity_policy="reject_on_source_signal",
            fidelity_reject=False,
            source_signal=False,
            source_tags=[],
            timing_ms={"transpile": 1.0, "check": 1.0, "total": 2.0},
            penalty_applied=False,
            penalty_reward=None,
            stan_path=None,
            stan_hash=None,
        ),
    )

    rollout = module._process_sequence(
        config=config,
        step=0,
        prompt_idx=0,
        seq_idx=0,
        prompt_text="prompt",
        prompt_tokens=[11, 12],
        seq=seq,
        tokenizer=_FakeTokenizer(_valid_completion_text()),
        scoring_data={"y": [0, 1], "d": 2, "seed": 7},
        step_data=step_data,
        pre_scored=(7.5, {"outcome_code": "ok"}),
        normalize_indices=set(),
    )

    assert rollout.reported_reward == 7.5
    assert step_data.step_records[0].reported_reward == 7.5
    gate_meta = step_data.step_records[0].metadata["sstan_gate"]
    assert gate_meta["decision"] == "reject"
    assert gate_meta["penalty_applied"] is False


def test_apply_judge_gate_penalty_updates_all_reward_views(monkeypatch) -> None:
    config = module.TinkerGRPOConfig(
        n_prompts=1,
        rollouts_per_prompt=1,
        judge_gate_mode="enforce",
        judge_gate_penalty_reward=-400.0,
    )
    step_data = module._StepData(
        rollouts_by_prompt={
            0: [
                RolloutData(
                    prompt_tokens=[1],
                    completion_tokens=[2],
                    sampling_logprobs=[0.0],
                    reported_reward=5.0,
                    log_mass=0.0,
                )
            ]
        },
        all_reported=[5.0],
        all_completion_hashes=[],
        step_records=[
            CompletionRecord(
                batch=0,
                index=0,
                prompt="p",
                completion_text="```python\npm.Potential('hack', 1.0)\n```",
                code="pm.Potential('hack', 1.0)",
                reported_reward=5.0,
                outcome="valid",
                timestamp="2026-01-01T00:00:00Z",
                metadata={},
            )
        ],
    )

    monkeypatch.setattr(
        module,
        "run_judge_gate_batch",
        lambda **kwargs: [
            JudgeGateResult(
                mode="enforce",
                checked=True,
                accepted=False,
                decision="reject",
                verdict="hacking",
                confidence=0.95,
                tags=["score_injection"],
                reasoning="hack",
                penalty_applied=True,
                penalty_reward=-400.0,
                timing_ms=12.0,
            )
        ],
    )

    metrics = module._apply_judge_gate(step=0, step_data=step_data, config=config)

    assert metrics["judge_gate/n_checked"] == 1
    assert metrics["judge_gate/n_penalized"] == 1
    assert metrics["judge_gate/n_fail_open"] == 0
    assert metrics["judge_gate/fail_open_rate"] == 0.0
    assert step_data.step_records[0].reported_reward == -400.0
    assert step_data.rollouts_by_prompt[0][0].reported_reward == -400.0
    assert step_data.all_reported[0] == -400.0
    assert "judge_gate" in (step_data.step_records[0].metadata or {})


def test_apply_judge_gate_tracks_fail_open(monkeypatch) -> None:
    config = module.TinkerGRPOConfig(
        n_prompts=1,
        rollouts_per_prompt=1,
        judge_gate_mode="enforce",
    )
    step_data = module._StepData(
        rollouts_by_prompt={
            0: [
                RolloutData(
                    prompt_tokens=[1],
                    completion_tokens=[2],
                    sampling_logprobs=[0.0],
                    reported_reward=5.0,
                    log_mass=0.0,
                )
            ]
        },
        all_reported=[5.0],
        all_completion_hashes=[],
        step_records=[
            CompletionRecord(
                batch=0,
                index=0,
                prompt="p",
                completion_text="```python\npm.Potential('hack', 1.0)\n```",
                code="pm.Potential('hack', 1.0)",
                reported_reward=5.0,
                outcome="valid",
                timestamp="2026-01-01T00:00:00Z",
                metadata={},
            )
        ],
    )
    monkeypatch.setattr(
        module,
        "run_judge_gate_batch",
        lambda **kwargs: [
            JudgeGateResult(
                mode="enforce",
                checked=True,
                accepted=True,
                decision="fail_open",
                verdict=None,
                confidence=None,
                tags=[],
                reasoning="timeout",
                penalty_applied=False,
                penalty_reward=None,
                timing_ms=10.0,
            )
        ],
    )
    metrics = module._apply_judge_gate(step=0, step_data=step_data, config=config)
    assert metrics["judge_gate/n_checked"] == 1
    assert metrics["judge_gate/n_penalized"] == 0
    assert metrics["judge_gate/n_fail_open"] == 1
    assert metrics["judge_gate/fail_open_rate"] == 1.0


def test_apply_judge_gate_payload_sets_final_summary_fields() -> None:
    summary = {
        "paper/judge_gate_checked_final": float("nan"),
        "paper/judge_gate_penalized_final": float("nan"),
        "paper/judge_gate_penalty_rate_final": float("nan"),
        "paper/judge_gate_fail_open_final": float("nan"),
        "paper/judge_gate_fail_open_rate_final": float("nan"),
    }
    payload = {
        "judge_gate/n_checked": 12,
        "judge_gate/n_penalized": 3,
        "judge_gate/penalty_rate": 0.25,
        "judge_gate/n_fail_open": 1,
        "judge_gate/fail_open_rate": 1.0 / 12.0,
    }
    module._apply_judge_gate_payload(summary, payload)
    assert summary["paper/judge_gate_checked_final"] == 12.0
    assert summary["paper/judge_gate_penalized_final"] == 3.0
    assert summary["paper/judge_gate_penalty_rate_final"] == 0.25
    assert summary["paper/judge_gate_fail_open_final"] == 1.0
    assert summary["paper/judge_gate_fail_open_rate_final"] == 1.0 / 12.0


def test_apply_judge_gate_payload_ignores_missing_payload() -> None:
    summary = {
        "paper/judge_gate_checked_final": float("nan"),
        "paper/judge_gate_penalized_final": float("nan"),
        "paper/judge_gate_penalty_rate_final": float("nan"),
        "paper/judge_gate_fail_open_final": float("nan"),
        "paper/judge_gate_fail_open_rate_final": float("nan"),
    }
    module._apply_judge_gate_payload(summary, None)
    assert all(str(v) == "nan" for v in summary.values())
