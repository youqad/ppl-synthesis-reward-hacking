from __future__ import annotations

from types import SimpleNamespace

from ppl_synthesis_reward_hacking.experiments import tinker_training as module
from ppl_synthesis_reward_hacking.logging.completions import CompletionWriter


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
    writer = CompletionWriter(tmp_path / "completions.jsonl")
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
        writer=writer,
        step_data=step_data,
        pre_scored=(7.5, {"outcome_code": "ok"}),
        normalize_indices=set(),
    )
    writer.close()

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
    writer = CompletionWriter(tmp_path / "completions_shadow.jsonl")
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
        writer=writer,
        step_data=step_data,
        pre_scored=(7.5, {"outcome_code": "ok"}),
        normalize_indices=set(),
    )
    writer.close()

    assert rollout.reported_reward == 7.5
    assert step_data.step_records[0].reported_reward == 7.5
    gate_meta = step_data.step_records[0].metadata["sstan_gate"]
    assert gate_meta["decision"] == "reject"
    assert gate_meta["penalty_applied"] is False

