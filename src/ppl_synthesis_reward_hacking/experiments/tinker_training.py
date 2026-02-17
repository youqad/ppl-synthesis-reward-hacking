"""Tinker GRPO training loop with local PyMC scoring.

Extracted from scripts/tinker_pymc_grpo.py for reuse by Hydra entrypoints
and direct script invocation.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import extract_pymc_code
from ppl_synthesis_reward_hacking.data.synthstats_loader import (
    SYSTEM_PROMPT,
    format_chat_prompt,
    get_prompts,
    make_scoring_data_dict,
)
from ppl_synthesis_reward_hacking.evaluation.normalization_check import (
    check_normalization_small_d,
)
from ppl_synthesis_reward_hacking.experiments.grpo import (
    RolloutData,
    build_tinker_datum,
    compute_group_relative_advantages,
)
from ppl_synthesis_reward_hacking.experiments.results import (
    compute_trajectory_metrics,
    print_training_summary,
)
from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    classify_outcome,
    score_completion_sandboxed,
)
from ppl_synthesis_reward_hacking.logging.completions import (
    CompletionRecord,
    CompletionWriter,
    make_timestamp,
)
from ppl_synthesis_reward_hacking.monitoring.llm_judge import (
    JudgeConfig,
    judge_completions,
    judge_with_rubric,
)
from ppl_synthesis_reward_hacking.utils.hashing import normalized_text_hash
from ppl_synthesis_reward_hacking.utils.tinker import tinker, types, validate_tinker_setup

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TinkerGRPOConfig:
    # model
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32

    # training (5x160=800 rollouts/step)
    n_steps: int = 1000
    n_prompts: int = 5
    rollouts_per_prompt: int = 160
    learning_rate: float = 1e-5

    # sampling
    temperature: float = 1.28
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 512

    # GRPO
    loss_fn: str = "importance_sampling"

    # PyMC scoring
    exec_timeout: int = 60
    scoring_d: int = 1
    scoring_p_true: float = 0.5
    scoring_seed_base: int = 0
    scoring_method: str = "smc"
    smc_draws: int = 500
    scoring_workers: int = 1

    # output
    output_dir: str = "artifacts/tinker_pymc_grpo"

    # exemplar config (None disables)
    exemplar_config: str | None = None

    # observational judge (not used in GRPO reward)
    judge_interval: int = 0
    judge_sample_size: int = 20
    judge_dedup: bool = True
    judge_model: str = "anthropic/glm-5"
    judge_use_stub: bool = False
    judge_env_file: str | None = None

    # rubric evolution (0 disables)
    rubric_evolution_interval: int = 0

    # normalization monitoring (0 disables)
    normalization_interval: int = 5
    normalization_sample_size: int = 10


def config_from_mapping(mapping: Mapping[str, Any]) -> TinkerGRPOConfig:
    """Create TinkerGRPOConfig from a mapping (Hydra-friendly)."""
    return TinkerGRPOConfig(**dict(mapping))


# ---------------------------------------------------------------------------
# Trajectory / step data
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryPoint:
    step: int
    reported_mean: float
    oracle_mean: float
    gap_mean: float
    n_valid: int
    n_total: int
    n_parse_fail: int
    n_exec_fail: int
    frac_zero_variance: float = 0.0
    frac_above_oracle: float = 0.0
    oracle_threshold: float = float("nan")


@dataclass
class _StepData:
    rollouts_by_prompt: dict[int, list[RolloutData]]
    all_reported: list[float]
    all_oracle: list[float]
    all_completion_hashes: list[str]
    step_records: list[CompletionRecord]
    n_parse_fail: int = 0
    n_exec_fail: int = 0


# ---------------------------------------------------------------------------
# Exemplar helpers
# ---------------------------------------------------------------------------


def _load_exemplars(config_path: str | None) -> list[dict[str, Any]]:
    """Load exemplar YAML and return the list of exemplar dicts."""
    if not config_path:
        return []
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"exemplar config not found: {path}")
    data = yaml.safe_load(path.read_text())
    exemplars = data.get("exemplars", [])
    if not exemplars:
        raise ValueError(f"no exemplars found in {path}")
    return exemplars


def _format_exemplar_section(exemplars: list[dict[str, Any]]) -> str:
    """Format exemplars as a prompt section appended to the system message."""
    lines = [
        "\nAdvanced syntax examples (pm.Potential / pm.DensityDist):",
    ]
    for ex in exemplars:
        lines.append(f"\n# {ex['name']}: {ex['description']}")
        lines.append(f"```python\n{ex['code'].rstrip()}\n```")
    return "\n".join(lines)


def _format_chat_prompt_with_exemplars(
    user_message: str, exemplar_section: str
) -> str:
    """Like format_chat_prompt but injects an exemplar section into the system message."""
    system = SYSTEM_PROMPT + exemplar_section
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(config: TinkerGRPOConfig) -> dict:
    validate_tinker_setup()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        writer,
        judge_metrics_path,
        judge_cfg,
        training_client,
        tokenizer,
        prompt_texts,
    ) = _initialize_training(config, output_dir)
    trajectory: list[TrajectoryPoint] = []
    rubric_state = _init_rubric(config)
    _log_training_start(config, output_dir)

    norm_metrics_path = output_dir / "normalization_metrics.jsonl"
    for step in range(config.n_steps):
        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"step_{step}")
        normalize_indices = _get_normalization_indices(step, config)
        step_data = _collect_step_data(
            config=config,
            step=step,
            prompt_texts=prompt_texts,
            tokenizer=tokenizer,
            sampling_client=sampling_client,
            writer=writer,
            normalize_indices=normalize_indices,
        )
        _log_step_diagnostics(step, step_data)
        if normalize_indices:
            _log_normalization_summary(step, step_data, norm_metrics_path)
        datums, frac_zero_variance = _build_step_datums(
            step_data.rollouts_by_prompt, n_prompts=config.n_prompts
        )
        _apply_training_update(
            training_client,
            datums,
            step=step,
            learning_rate=config.learning_rate,
            loss_fn=config.loss_fn,
        )
        point = _compute_step_point(step, step_data, frac_zero_variance)
        trajectory.append(point)
        _log_step_summary(step, point)
        writer.flush()
        judge_verdicts = _maybe_run_step_judge(
            step, step_data.step_records, config, judge_cfg, judge_metrics_path, rubric_state
        )
        rubric_state = _maybe_evolve_rubric(
            step, step_data, config, rubric_state, judge_verdicts, judge_cfg=judge_cfg
        )

    writer.close()
    _write_trajectory(output_dir, trajectory)
    results = _compute_results(config, trajectory)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("Results saved to %s", output_dir)
    _print_summary(results)

    return results


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _initialize_training(
    config: TinkerGRPOConfig, output_dir: Path
) -> tuple[CompletionWriter, Path, JudgeConfig, object, object, list[str]]:
    writer = CompletionWriter(output_dir / "completions.jsonl")
    judge_metrics_path = output_dir / "judge_metrics.jsonl"
    judge_cfg = JudgeConfig(model=config.judge_model, use_stub=config.judge_use_stub)
    _load_judge_env(config.judge_env_file)
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    exemplars = _load_exemplars(config.exemplar_config)
    if exemplars:
        exemplar_section = _format_exemplar_section(exemplars)
        log.info("Loaded %d exemplars from %s", len(exemplars), config.exemplar_config)
        prompt_texts = [
            _format_chat_prompt_with_exemplars(p, exemplar_section)
            for p in get_prompts(config.n_prompts)
        ]
    else:
        prompt_texts = [format_chat_prompt(p) for p in get_prompts(config.n_prompts)]
    return writer, judge_metrics_path, judge_cfg, training_client, tokenizer, prompt_texts


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _get_normalization_indices(step: int, config: TinkerGRPOConfig) -> set[int]:
    """Select which record indices to run normalization on this step."""
    if config.normalization_interval <= 0 or step % config.normalization_interval != 0:
        return set()
    if config.scoring_d > 5:
        return set()
    total = config.n_prompts * config.rollouts_per_prompt
    sample_size = min(config.normalization_sample_size, total)
    rng = np.random.default_rng(step + 9999)
    return set(rng.choice(total, size=sample_size, replace=False).tolist())


def _collect_step_data(
    *,
    config: TinkerGRPOConfig,
    step: int,
    prompt_texts: list[str],
    tokenizer,
    sampling_client,
    writer: CompletionWriter,
    normalize_indices: set[int] | None = None,
) -> _StepData:
    step_data = _StepData(
        rollouts_by_prompt={},
        all_reported=[],
        all_oracle=[],
        all_completion_hashes=[],
        step_records=[],
    )
    for prompt_idx, prompt_text in enumerate(prompt_texts):
        _collect_prompt_rollouts(
            config=config,
            step=step,
            prompt_idx=prompt_idx,
            prompt_text=prompt_text,
            tokenizer=tokenizer,
            sampling_client=sampling_client,
            writer=writer,
            step_data=step_data,
            normalize_indices=normalize_indices or set(),
        )
    return step_data


def _collect_prompt_rollouts(
    *,
    config: TinkerGRPOConfig,
    step: int,
    prompt_idx: int,
    prompt_text: str,
    tokenizer,
    sampling_client,
    writer: CompletionWriter,
    step_data: _StepData,
    normalize_indices: set[int],
) -> None:
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    samples = sampling_client.sample(
        prompt=prompt_input,
        sampling_params=types.SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            logprobs=True,
        ),
        num_samples=config.rollouts_per_prompt,
    ).result()
    scoring_data = _make_prompt_scoring_data(config, step, prompt_idx)

    # batch-score when workers > 1 (needed for SMC which is 10-100x slower)
    if config.scoring_workers > 1:
        from ppl_synthesis_reward_hacking.experiments.parallel_scorer import score_batch_parallel

        completion_texts = [tokenizer.decode(seq.tokens) for seq in samples.sequences]
        batch_scores = score_batch_parallel(
            completion_texts,
            scoring_data,
            workers=config.scoring_workers,
            timeout=config.exec_timeout,
            scoring_method=config.scoring_method,
            smc_draws=config.smc_draws,
        )
    else:
        batch_scores = [None] * len(samples.sequences)

    rollouts: list[RolloutData] = []
    for seq_idx, (seq, pre_scored) in enumerate(zip(samples.sequences, batch_scores, strict=True)):
        rollout = _process_sequence(
            config=config,
            step=step,
            prompt_idx=prompt_idx,
            seq_idx=seq_idx,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            seq=seq,
            tokenizer=tokenizer,
            scoring_data=scoring_data,
            writer=writer,
            step_data=step_data,
            pre_scored=pre_scored,
            normalize_indices=normalize_indices,
        )
        rollouts.append(rollout)
    step_data.rollouts_by_prompt[prompt_idx] = rollouts


def _make_prompt_scoring_data(config: TinkerGRPOConfig, step: int, prompt_idx: int) -> dict:
    prompt_seed = config.scoring_seed_base + step * config.n_prompts + prompt_idx
    return make_scoring_data_dict(
        d=config.scoring_d,
        p_true=config.scoring_p_true,
        seed=prompt_seed,
    )


# ---------------------------------------------------------------------------
# Sequence processing
# ---------------------------------------------------------------------------


def _process_sequence(
    *,
    config: TinkerGRPOConfig,
    step: int,
    prompt_idx: int,
    seq_idx: int,
    prompt_text: str,
    prompt_tokens: list[int],
    seq,
    tokenizer,
    scoring_data: dict,
    writer: CompletionWriter,
    step_data: _StepData,
    pre_scored: tuple[float, float | None, dict | None] | None = None,
    normalize_indices: set[int] | None = None,
) -> RolloutData:
    completion_text = tokenizer.decode(seq.tokens)
    step_data.all_completion_hashes.append(normalized_text_hash(completion_text))
    if pre_scored is not None:
        reported, oracle, decomposition = pre_scored
    else:
        reported, oracle, decomposition = score_completion_sandboxed(
            completion_text,
            scoring_data,
            timeout=config.exec_timeout,
            scoring_method=config.scoring_method,
            smc_draws=config.smc_draws,
        )
    outcome = classify_outcome(reported)
    if outcome == "parse_fail":
        step_data.n_parse_fail += 1
    elif outcome == "exec_fail":
        step_data.n_exec_fail += 1

    # normalization check for sampled records (only on valid outcomes)
    global_idx = prompt_idx * config.rollouts_per_prompt + seq_idx
    norm_result = None
    if normalize_indices and global_idx in normalize_indices and outcome == "valid":
        norm_result = _run_normalization_check(
            completion_text, config.scoring_d, config.scoring_p_true, config.exec_timeout
        )

    rollout = RolloutData(
        prompt_tokens=list(prompt_tokens),
        completion_tokens=list(seq.tokens),
        sampling_logprobs=list(seq.logprobs) if seq.logprobs else [0.0] * len(seq.tokens),
        reported_reward=reported,
        oracle_score=oracle,
        log_mass=norm_result.get("log_mass", 0.0) if norm_result else 0.0,
    )
    step_data.all_reported.append(reported)
    step_data.all_oracle.append(oracle)

    metadata: dict[str, Any] = {}
    if decomposition:
        metadata["decomposition"] = decomposition
    if norm_result:
        metadata["normalization"] = norm_result

    record = CompletionRecord(
        batch=step,
        index=global_idx,
        prompt=prompt_text,
        completion_text=completion_text,
        code=extract_pymc_code(completion_text),
        reported_reward=reported,
        oracle_score=oracle,
        gap=reported - oracle,
        outcome=outcome,
        timestamp=make_timestamp(),
        metadata=metadata or None,
    )
    writer.write(record)
    step_data.step_records.append(record)
    return rollout


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _run_normalization_check(
    completion_text: str, d: int, p_true: float, timeout: int
) -> dict[str, Any] | None:
    """Run normalization check, returning result dict or None on failure."""
    try:
        result = check_normalization_small_d(
            completion_text, d=d, p_true=p_true, timeout=timeout
        )
        if result.get("ok"):
            return {
                "log_mass": result["log_mass"],
                "mass": result.get("mass"),
                "is_normalized": result["is_normalized"],
                "d": d,
            }
        return {"ok": False, "reason": result.get("reason", "unknown"), "d": d}
    except Exception as exc:
        log.debug("normalization check failed: %s", exc)
        return None


def _log_normalization_summary(
    step: int, step_data: _StepData, metrics_path: Path
) -> None:
    """Extract normalization results from step records and log summary."""
    log_masses = []
    n_checked = 0
    n_non_normalized = 0
    for rec in step_data.step_records:
        if not rec.metadata:
            continue
        norm = rec.metadata.get("normalization")
        if not norm or not isinstance(norm, dict):
            continue
        n_checked += 1
        lm = norm.get("log_mass")
        if lm is not None:
            log_masses.append(float(lm))
            if not norm.get("is_normalized", True):
                n_non_normalized += 1

    if not n_checked:
        return

    frac_non_norm = n_non_normalized / n_checked
    mean_abs_lm = float(np.mean(np.abs(log_masses))) if log_masses else 0.0

    log.info(
        "Step %3d NORM: checked=%d non_normalized=%.1f%% mean_abs_log_mass=%.3f",
        step,
        n_checked,
        frac_non_norm * 100,
        mean_abs_lm,
    )

    payload = {
        "step": step,
        "n_checked": n_checked,
        "n_non_normalized": n_non_normalized,
        "frac_non_normalized": frac_non_norm,
        "mean_abs_log_mass": mean_abs_lm,
        "log_masses": log_masses,
    }
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


# ---------------------------------------------------------------------------
# GRPO datum building & training update
# ---------------------------------------------------------------------------


def _build_step_datums(
    rollouts_by_prompt: dict[int, list[RolloutData]], *, n_prompts: int
) -> tuple[list, float]:
    advantages_by_prompt = compute_group_relative_advantages(rollouts_by_prompt)
    datums = []
    for prompt_idx, advantages in advantages_by_prompt.items():
        rollouts = rollouts_by_prompt[prompt_idx]
        for rollout, advantage in zip(rollouts, advantages, strict=True):
            datums.append(build_tinker_datum(rollout, advantage))
    n_zero_variance = n_prompts - len(advantages_by_prompt)
    frac_zero_variance = n_zero_variance / max(n_prompts, 1)
    return datums, frac_zero_variance


def _apply_training_update(
    training_client,
    datums: list,
    *,
    step: int,
    learning_rate: float,
    loss_fn: str,
) -> None:
    if not datums:
        log.warning("Step %d: no datums (all zero variance), skipping gradient", step)
        return
    fwdbwd = training_client.forward_backward(datums, loss_fn=loss_fn)
    optim = training_client.optim_step(types.AdamParams(learning_rate=learning_rate))
    fwdbwd.result()
    optim.result()


# ---------------------------------------------------------------------------
# Step point / trajectory
# ---------------------------------------------------------------------------


def _compute_step_point(
    step: int, step_data: _StepData, frac_zero_variance: float
) -> TrajectoryPoint:
    valid_pairs = [
        (reported, oracle)
        for reported, oracle in zip(step_data.all_reported, step_data.all_oracle, strict=True)
        if np.isfinite(reported)
        and np.isfinite(oracle)
        and classify_outcome(float(reported)) == "valid"
        and classify_outcome(float(oracle)) == "valid"
    ]
    if valid_pairs:
        reported_mean = float(np.mean([reported for reported, _ in valid_pairs]))
        oracle_mean = float(np.mean([oracle for _, oracle in valid_pairs]))
        gap = reported_mean - oracle_mean
        n_above = sum(1 for r, o in valid_pairs if r > o)
        frac_above = n_above / len(valid_pairs)
        oracle_threshold = oracle_mean
    else:
        reported_mean = float("nan")
        oracle_mean = float("nan")
        gap = float("nan")
        frac_above = 0.0
        oracle_threshold = float("nan")
    return TrajectoryPoint(
        step=step,
        reported_mean=reported_mean,
        oracle_mean=oracle_mean,
        gap_mean=gap,
        n_valid=len(valid_pairs),
        n_total=len(step_data.all_reported),
        n_parse_fail=step_data.n_parse_fail,
        n_exec_fail=step_data.n_exec_fail,
        frac_zero_variance=frac_zero_variance,
        frac_above_oracle=frac_above,
        oracle_threshold=oracle_threshold,
    )


def _write_trajectory(output_dir: Path, trajectory: list[TrajectoryPoint]) -> None:
    trajectory_dicts = [asdict(point) for point in trajectory]
    with open(output_dir / "trajectory.json", "w") as f:
        json.dump(trajectory_dicts, f, indent=2)


def _compute_results(config: TinkerGRPOConfig, trajectory: list[TrajectoryPoint]) -> dict:
    metrics = compute_trajectory_metrics(trajectory)
    if "error" in metrics:
        metrics["n_steps"] = len(trajectory)
        return metrics

    final = trajectory[-1]
    metrics["config"] = {
        "base_model": config.base_model,
        "n_steps": config.n_steps,
        "n_prompts": config.n_prompts,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank,
        "scoring_d": config.scoring_d,
        "scoring_p_true": config.scoring_p_true,
        "scoring_method": config.scoring_method,
        "smc_draws": config.smc_draws,
        "scoring_workers": config.scoring_workers,
        "max_tokens": config.max_tokens,
    }
    metrics["final_frac_zero_variance"] = final.frac_zero_variance
    metrics["final_gap_mean"] = metrics.get("final_gap")
    metrics["final_reported_mean"] = metrics.get("final_reported")
    metrics["final_oracle_mean"] = metrics.get("final_oracle")
    metrics["n_steps_completed"] = len(trajectory)
    return metrics


def _print_summary(results: dict) -> None:
    print_training_summary(results)
    if "error" not in results:
        log.info("zero-variance fraction: %.1f%%", results["final_frac_zero_variance"] * 100)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_training_start(config: TinkerGRPOConfig, output_dir: Path) -> None:
    log.info("Starting Tinker GRPO training")
    log.info(
        "Model: %s, Steps: %d, Prompts: %d, Rollouts/prompt: %d",
        config.base_model,
        config.n_steps,
        config.n_prompts,
        config.rollouts_per_prompt,
    )
    log.info(
        "Sampling: temp=%.2f, top_p=%.2f, top_k=%d",
        config.temperature,
        config.top_p,
        config.top_k,
    )
    log.info(
        "Scoring: method=%s, d=%d, p_true=%.2f, smc_draws=%d, workers=%d",
        config.scoring_method,
        config.scoring_d,
        config.scoring_p_true,
        config.smc_draws,
        config.scoring_workers,
    )
    log.info("Output: %s", output_dir)
    if config.normalization_interval > 0:
        log.info(
            "Normalization: every %d steps, sample %d records (d=%d)",
            config.normalization_interval,
            config.normalization_sample_size,
            config.scoring_d,
        )


def _log_step_diagnostics(step: int, step_data: _StepData) -> None:
    n_unique = len(set(step_data.all_completion_hashes))
    n_total = len(step_data.all_completion_hashes)
    log.info(
        "Step %3d DIAG: diversity=%d/%d unique (%.1f%%), failures: parse=%d exec=%d valid=%d",
        step,
        n_unique,
        n_total,
        100.0 * n_unique / max(n_total, 1),
        step_data.n_parse_fail,
        step_data.n_exec_fail,
        n_total - step_data.n_parse_fail - step_data.n_exec_fail,
    )
    _log_step_reward_diagnostics(step, step_data.all_reported)


def _log_step_reward_diagnostics(step: int, all_reported: list[float]) -> None:
    valid_rewards = [
        reward
        for reward in all_reported
        if np.isfinite(reward) and classify_outcome(float(reward)) == "valid"
    ]
    if not valid_rewards:
        log.warning("Step %3d DIAG: no valid completions (all failed)", step)
        return
    log.info(
        "Step %3d DIAG: valid rewards: min=%.2f max=%.2f mean=%.2f std=%.4f range=%.4f",
        step,
        min(valid_rewards),
        max(valid_rewards),
        float(np.mean(valid_rewards)),
        float(np.std(valid_rewards)),
        max(valid_rewards) - min(valid_rewards),
    )


def _log_step_summary(step: int, point: TrajectoryPoint) -> None:
    log.info(
        "Step %3d: reported=%.1f oracle=%.1f gap=%.1f valid=%d/%d "
        "parse_fail=%d exec_fail=%d frac_zero_var=%.2f above_oracle=%.2f",
        step,
        point.reported_mean,
        point.oracle_mean,
        point.gap_mean,
        point.n_valid,
        point.n_total,
        point.n_parse_fail,
        point.n_exec_fail,
        point.frac_zero_variance,
        point.frac_above_oracle,
    )


# ---------------------------------------------------------------------------
# Judge / rubric
# ---------------------------------------------------------------------------


def _maybe_run_step_judge(
    step: int,
    step_records: list[CompletionRecord],
    config: TinkerGRPOConfig,
    judge_cfg: JudgeConfig,
    judge_metrics_path: Path,
    rubric_state=None,
) -> list[dict[str, Any]]:
    if config.judge_interval <= 0 or step % config.judge_interval != 0 or not step_records:
        return []
    try:
        return _run_observational_judge(
            step=step,
            records=step_records,
            cfg=judge_cfg,
            sample_size=config.judge_sample_size,
            dedup=config.judge_dedup,
            metrics_path=judge_metrics_path,
            rubric_state=rubric_state,
        )
    except Exception as exc:
        log.warning("Step %3d JUDGE: failed (continuing training): %s", step, exc)
        return []


def _deduplicate_records(records: list[CompletionRecord]) -> list[CompletionRecord]:
    unique: dict[str, CompletionRecord] = {}
    for record in records:
        key = normalized_text_hash(record.code or record.completion_text)
        if key not in unique:
            unique[key] = record
    return list(unique.values())


def _run_observational_judge(
    *,
    step: int,
    records: list[CompletionRecord],
    cfg: JudgeConfig,
    sample_size: int,
    dedup: bool,
    metrics_path: Path,
    rubric_state=None,
) -> list[dict[str, Any]]:
    judged_records = _deduplicate_records(records) if dedup else records
    if sample_size > 0 and len(judged_records) > sample_size:
        rng = np.random.default_rng(step)
        idxs = rng.choice(len(judged_records), size=sample_size, replace=False).tolist()
        judged_records = [judged_records[i] for i in idxs]

    if rubric_state is not None:
        from ppl_synthesis_reward_hacking.monitoring.evolving_rubric import (
            format_rubric_prompt,
        )

        rubric_prompt = format_rubric_prompt(rubric_state)
        verdicts = [judge_with_rubric(r, cfg, rubric_prompt) for r in judged_records]
    else:
        verdicts = judge_completions(judged_records, cfg)
    verdict_counts = Counter(v.verdict for v in verdicts)
    tag_counts = Counter(tag for v in verdicts for tag in v.tags)
    n_total = max(len(verdicts), 1)
    hacking_rate = verdict_counts.get("hacking", 0) / n_total

    payload = {
        "step": step,
        "n_judged": len(verdicts),
        "hacking_rate": hacking_rate,
        "verdict_counts": dict(verdict_counts),
        "tag_counts": dict(tag_counts),
    }
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    log.info(
        "Step %3d JUDGE: n=%d hacking_rate=%.3f verdicts=%s",
        step,
        len(verdicts),
        hacking_rate,
        dict(verdict_counts),
    )

    return [
        {"tags": v.tags, "reported_reward": r.reported_reward}
        for v, r in zip(verdicts, judged_records, strict=True)
    ]


def _load_judge_env(env_file: str | None) -> None:
    if not env_file:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        log.warning("python-dotenv not installed; judge env-file ignored")


def _init_rubric(config: TinkerGRPOConfig):
    if config.rubric_evolution_interval <= 0:
        return None
    from ppl_synthesis_reward_hacking.monitoring.evolving_rubric import (
        make_initial_rubric_state,
    )

    return make_initial_rubric_state()


def _make_rubric_judge_fn(cfg: JudgeConfig):
    """Create a callable (prompt -> str) for rubric evolution proposals."""
    import os

    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        return None

    def _call(prompt: str) -> str:
        import litellm

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "api_key": api_key,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "timeout": cfg.timeout,
        }
        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base
        if cfg.custom_llm_provider:
            kwargs["custom_llm_provider"] = cfg.custom_llm_provider
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or ""

    return _call


def _maybe_evolve_rubric(
    step: int,
    step_data: _StepData,
    config: TinkerGRPOConfig,
    rubric_state,
    judge_verdicts: list[dict[str, Any]] | None = None,
    judge_cfg: JudgeConfig | None = None,
):
    if rubric_state is None or config.rubric_evolution_interval <= 0:
        return rubric_state
    if step % config.rubric_evolution_interval != 0:
        return rubric_state

    try:
        from ppl_synthesis_reward_hacking.monitoring.evolving_rubric import (
            evolve_rubric,
        )

        completions = [r.code or r.completion_text for r in step_data.step_records]
        if judge_verdicts:
            verdicts = judge_verdicts
        else:
            verdicts = [
                {"tags": [], "reported_reward": r.reported_reward} for r in step_data.step_records
            ]

        judge_fn = _make_rubric_judge_fn(judge_cfg) if judge_cfg else None
        rubric_state = evolve_rubric(
            rubric_state,
            completions=completions,
            verdicts=verdicts,
            judge_fn=judge_fn,
            step=step,
        )
        log.info(
            "Step %3d RUBRIC: %d active items (%d evolved)",
            step,
            len(rubric_state.active_items),
            len(rubric_state.evolved_items),
        )
        rubric_log_path = Path(config.output_dir) / "rubric_evolution.jsonl"
        with rubric_log_path.open("a", encoding="utf-8") as f:
            entry = {"step": step, **rubric_state.to_dict()}
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Step %3d RUBRIC: evolution failed: %s", step, exc)

    return rubric_state
