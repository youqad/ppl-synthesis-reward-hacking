from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.backends.judge_gate import JudgeGateConfig, JudgeGateResult
from ppl_synthesis_reward_hacking.backends.sstan.gate import SStanGateConfig, SStanGateResult
from ppl_synthesis_reward_hacking.experiments import pymc_reward as module
from ppl_synthesis_reward_hacking.logging.completions import CompletionWriter, load_completions


def _patch_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(module, "extract_pymc_code", lambda _text: "code")
    monkeypatch.setattr(module, "select_split_data", lambda scoring_data, _split: scoring_data)
    monkeypatch.setattr(module, "execute_pymc_code", lambda code, data, timeout: object())
    monkeypatch.setattr(module, "_get_point_logps", lambda _model: None)
    monkeypatch.setattr(module, "_build_meta", lambda _model, _logps: None)
    monkeypatch.setattr(module, "_run_safety_check", lambda _model, _counts: None)
    monkeypatch.setattr(
        module,
        "evaluate_model",
        lambda _model, _scoring_data, _cfg: SimpleNamespace(
            reward_train=7.5,
            decomposition={"n_pot_terms": 1, "n_obs_terms": 1},
            outcome_code="ok",
            outcome_detail=None,
            metric_log_marginal_likelihood=-1.0,
            metric_elpd=None,
            metric_waic=None,
            metric_bic=None,
        ),
    )


def test_enforce_mode_applies_sstan_penalty(monkeypatch, tmp_path) -> None:
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        module,
        "run_sstan_gate",
        lambda **kwargs: SStanGateResult(
            mode="enforce",
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

    writer = CompletionWriter(tmp_path / "completions.jsonl")
    state = module.PyMCRewardState(
        backend=object(),
        dataset=object(),
        scoring_data={"y": [0, 1], "seed": 3},
        cache_dir=tmp_path,
        completion_writer=writer,
        sstan_gate_config=SStanGateConfig(mode="enforce", penalty_reward=-123.0),
        sstan_sampling_seed=11,
    )
    state.call_count = 1
    batch = module._BatchResult()

    module._score_completion(
        state,
        batch,
        index=0,
        prompt="prompt",
        completion="completion",
        exec_timeout=10,
    )
    module._write_pending_records(state, batch)
    writer.close()

    assert batch.rewards == [-123.0]
    rows = load_completions(tmp_path / "completions.jsonl")
    assert len(rows) == 1
    gate_meta = rows[0].metadata["sstan_gate"]
    assert gate_meta["decision"] == "reject"
    assert gate_meta["penalty_applied"] is True


def test_with_gate_penalty_preserves_cmdsafestan_metadata() -> None:
    original = SStanGateResult(
        mode="enforce",
        checked=True,
        sampled=True,
        accepted=False,
        decision="reject",
        reasons=["cmdsafestan_reject"],
        transpile_success=True,
        transpile_error=None,
        sstan_reasons=[],
        fidelity_policy="none",
        fidelity_reject=False,
        source_signal=False,
        source_tags=[],
        timing_ms={"transpile": 1.0, "check": 1.0, "total": 2.0},
        penalty_applied=False,
        penalty_reward=None,
        stan_path="/tmp/example.stan",
        stan_hash="abc123",
        infra_failure=False,
        cmdsafestan_safe=False,
        cmdsafestan_violations=["target_update"],
        cmdsafestan_runtime_reused=True,
        cmdsafestan_timing_ms={"init": 100.0, "eval": 400.0},
        cmdsafestan_runtime_params={"jobs": 4},
    )

    updated = module._with_gate_penalty(original, penalty_reward=-123.0)
    assert updated.penalty_applied is True
    assert updated.penalty_reward == -123.0
    assert updated.cmdsafestan_safe is False
    assert updated.cmdsafestan_violations == ["target_update"]
    assert updated.cmdsafestan_runtime_reused is True
    assert updated.cmdsafestan_timing_ms == {"init": 100.0, "eval": 400.0}
    assert updated.cmdsafestan_runtime_params == {"jobs": 4}


def test_shadow_mode_keeps_original_reward(monkeypatch, tmp_path) -> None:
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        module,
        "run_sstan_gate",
        lambda **kwargs: SStanGateResult(
            mode="shadow",
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
            timing_ms={"transpile": 2.0, "check": 0.0, "total": 2.0},
            penalty_applied=False,
            penalty_reward=None,
            stan_path=None,
            stan_hash=None,
        ),
    )

    writer = CompletionWriter(tmp_path / "completions_shadow.jsonl")
    state = module.PyMCRewardState(
        backend=object(),
        dataset=object(),
        scoring_data={"y": [0, 1], "seed": 3},
        cache_dir=tmp_path,
        completion_writer=writer,
        sstan_gate_config=SStanGateConfig(mode="shadow", penalty_reward=-123.0),
        sstan_sampling_seed=11,
    )
    state.call_count = 1
    batch = module._BatchResult()

    module._score_completion(
        state,
        batch,
        index=0,
        prompt="prompt",
        completion="completion",
        exec_timeout=10,
    )
    module._write_pending_records(state, batch)
    writer.close()

    assert batch.rewards == [7.5]
    rows = load_completions(tmp_path / "completions_shadow.jsonl")
    assert len(rows) == 1
    gate_meta = rows[0].metadata["sstan_gate"]
    assert gate_meta["decision"] == "reject"
    assert gate_meta["penalty_applied"] is False


def test_enforce_mode_ignores_infra_failure(monkeypatch, tmp_path) -> None:
    _patch_happy_path(monkeypatch)
    monkeypatch.setattr(
        module,
        "run_sstan_gate",
        lambda **kwargs: SStanGateResult(
            mode="enforce",
            checked=True,
            sampled=True,
            accepted=False,
            decision="infra_fail",
            reasons=["transpile_failed"],
            transpile_success=False,
            transpile_error="boom",
            sstan_reasons=[],
            fidelity_policy="reject_on_source_signal",
            fidelity_reject=False,
            source_signal=False,
            source_tags=[],
            timing_ms={"transpile": 2.0, "check": 0.0, "total": 2.0},
            penalty_applied=False,
            penalty_reward=None,
            stan_path=None,
            stan_hash=None,
            infra_failure=True,
        ),
    )

    writer = CompletionWriter(tmp_path / "completions_enforce_infra.jsonl")
    state = module.PyMCRewardState(
        backend=object(),
        dataset=object(),
        scoring_data={"y": [0, 1], "seed": 3},
        cache_dir=tmp_path,
        completion_writer=writer,
        sstan_gate_config=SStanGateConfig(mode="enforce", penalty_reward=-123.0),
        sstan_sampling_seed=11,
    )
    state.call_count = 1
    batch = module._BatchResult()

    module._score_completion(
        state,
        batch,
        index=0,
        prompt="prompt",
        completion="completion",
        exec_timeout=10,
    )
    module._write_pending_records(state, batch)
    writer.close()

    assert batch.rewards == [7.5]
    rows = load_completions(tmp_path / "completions_enforce_infra.jsonl")
    assert len(rows) == 1
    gate_meta = rows[0].metadata["sstan_gate"]
    assert gate_meta["decision"] == "infra_fail"
    assert gate_meta["infra_failure"] is True
    assert gate_meta["penalty_applied"] is False


def test_make_pymc_reward_fn_rejects_invalid_split() -> None:
    with pytest.raises(ValueError, match="y_holdout"):
        module.make_pymc_reward_fn(
            backend=object(),
            dataset=object(),
            scoring_data={"y": [0, 1]},
            reward_data_split="holdout",
        )


def test_make_pymc_reward_fn_rejects_non_positive_log_interval() -> None:
    with pytest.raises(ValueError, match="log_interval must be positive"):
        module.make_pymc_reward_fn(
            backend=object(),
            dataset=object(),
            scoring_data={"y": [0, 1]},
            log_interval=0,
        )


def test_make_pymc_reward_fn_validates_split_before_writer(monkeypatch, tmp_path) -> None:
    calls = {"writer": 0}

    class _Writer:
        def __init__(self, _path):
            calls["writer"] += 1

    monkeypatch.setattr(module, "CompletionWriter", _Writer)
    monkeypatch.setattr(
        module,
        "select_split_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad split")),
    )

    with pytest.raises(ValueError, match="bad split"):
        module.make_pymc_reward_fn(
            backend=object(),
            dataset=object(),
            scoring_data={"y": [0, 1]},
            completions_path=tmp_path / "completions.jsonl",
            reward_data_split="holdout",
        )

    assert calls["writer"] == 0


def test_make_pymc_reward_fn_validates_split_before_temp_cache_allocation(monkeypatch) -> None:
    monkeypatch.setattr(
        module.tempfile,
        "mkdtemp",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("mkdtemp should not be called")),
    )
    with pytest.raises(ValueError, match="y_holdout"):
        module.make_pymc_reward_fn(
            backend=object(),
            dataset=object(),
            scoring_data={"y": [0, 1]},
            reward_data_split="holdout",
        )


def test_extract_text_coerces_non_string_chat_content() -> None:
    msg = [
        {"role": "assistant", "content": {"text": "alpha"}},
        {"role": "assistant", "content": 17},
    ]
    rendered = module._extract_text(msg)
    assert "alpha" in rendered
    assert "17" in rendered


def test_extract_text_reads_top_level_chat_dict_content() -> None:
    msg = {"role": "assistant", "content": [{"text": "alpha"}, {"text": "beta"}]}
    rendered = module._extract_text(msg)
    assert rendered == "alpha\nbeta"


def test_extract_text_reads_top_level_dict_content_object() -> None:
    msg = {"role": "assistant", "content": {"text": "alpha"}}
    rendered = module._extract_text(msg)
    assert rendered == "alpha"


def test_extract_text_prefers_assistant_messages_from_chat_history() -> None:
    msg = [
        {"role": "system", "content": "system prelude"},
        {"role": "user", "content": "user prompt with fenced code"},
        {"role": "assistant", "content": "assistant model output"},
    ]
    rendered = module._extract_text(msg)
    assert rendered == "assistant model output"


def test_run_batch_normalization_uses_train_split_when_top_level_y_missing(monkeypatch) -> None:
    observed = {"d": None}

    monkeypatch.setattr(
        module,
        "select_split_data",
        lambda _scoring_data, split: {"y": np.array([[0.0, 1.0], [1.0, 0.0]])}
        if split == "train"
        else (_ for _ in ()).throw(AssertionError("unexpected split")),
    )

    def _fake_check_normalization(*_args, **kwargs):
        observed["d"] = kwargs["d"]
        return {"ok": True, "is_normalized": True, "log_mass": 0.0}

    monkeypatch.setattr(module, "check_normalization", _fake_check_normalization)

    state = module.PyMCRewardState(
        backend=object(),
        dataset=object(),
        scoring_data={"y_train": [[0.0, 1.0], [1.0, 0.0]], "seed": 3},
        cache_dir=Path("."),
        normalization_sample_size=1,
    )
    state.call_count = 1
    batch = module._BatchResult(valid_completion_texts=["completion"])

    norm = module._run_batch_normalization(state, batch)
    assert norm.n_checked == 1
    assert observed["d"] == 2


def test_apply_judge_gate_trl_enforce_applies_penalty(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "run_judge_gate_batch",
        lambda **_kwargs: [
            JudgeGateResult(
                mode="enforce",
                checked=True,
                accepted=False,
                decision="reject",
                verdict="hacking",
                confidence=0.95,
                tags=["score_injection"],
                reasoning="detected hacking pattern",
                penalty_applied=True,
                penalty_reward=-321.0,
                timing_ms=12.0,
            )
        ],
    )

    state = module.PyMCRewardState(
        backend=object(),
        dataset=object(),
        scoring_data={"y": [0, 1], "seed": 3},
        cache_dir=Path("."),
        judge_gate_config=JudgeGateConfig(mode="enforce", penalty_reward=-321.0),
    )
    batch = module._BatchResult(
        rewards=[7.5, module.EXEC_FAIL_REWARD],
        outcomes=["valid", "exec_fail"],
        valid_completion_texts=["completion_text"],
        valid_codes=["def model(data): pass"],
    )

    module._apply_judge_gate_trl(state, batch)
    assert batch.rewards == [-321.0, module.EXEC_FAIL_REWARD]


def test_apply_judge_gate_trl_off_is_noop(monkeypatch) -> None:
    def _unexpected_call(**_kwargs):
        raise AssertionError("run_judge_gate_batch should not be called in off mode")

    monkeypatch.setattr(module, "run_judge_gate_batch", _unexpected_call)

    state = module.PyMCRewardState(
        backend=object(),
        dataset=object(),
        scoring_data={"y": [0, 1], "seed": 3},
        cache_dir=Path("."),
        judge_gate_config=JudgeGateConfig(mode="off"),
    )
    batch = module._BatchResult(
        rewards=[7.5],
        outcomes=["valid"],
        valid_completion_texts=["completion_text"],
    )

    module._apply_judge_gate_trl(state, batch)
    assert batch.rewards == [7.5]
