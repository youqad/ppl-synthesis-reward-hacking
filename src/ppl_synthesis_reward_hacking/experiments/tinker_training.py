"""Tinker GRPO training loop with local PyMC scoring.

Extracted from scripts/tinker_pymc_grpo.py for reuse by Hydra entrypoints
and direct script invocation.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
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
from ppl_synthesis_reward_hacking.evaluation.heuristics import detect_exploits
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
from ppl_synthesis_reward_hacking.logging.wandb_hook import (
    log_completion_table,
    log_heuristic_rates,
    log_judge_metrics,
    log_metrics,
    log_normalization_metrics,
    update_summary,
    upload_artifacts,
)
from ppl_synthesis_reward_hacking.logging.wandb_hook import (
    log_lh_table as _wandb_log_lh,
)
from ppl_synthesis_reward_hacking.logging.wandb_hook import (
    log_step as _wandb_log_step,
)
from ppl_synthesis_reward_hacking.monitoring.llm_judge import (
    JudgeConfig,
    judge_completions,
    judge_with_rubric,
)
from ppl_synthesis_reward_hacking.scoring.metrics import validate_metric_estimator
from ppl_synthesis_reward_hacking.utils.hashing import normalized_text_hash
from ppl_synthesis_reward_hacking.utils.tinker import tinker, types, validate_tinker_setup

log = logging.getLogger(__name__)


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

    # reward objective
    exec_timeout: int = 60
    reward_metric: str = "log_marginal_likelihood"
    reward_data_split: str = "train"
    predictive_estimator: str = "none"

    # raw-data interface policy
    interface_d: int = 1
    p_true_mode: str = "sampled_beta"  # sampled_beta | fixed
    p_true_fixed: float = 0.5
    p_true_beta_alpha: float = 1.0
    p_true_beta_beta: float = 1.0
    scoring_seed_base: int = 0
    smc_draws: int = 500
    scoring_workers: int = 1

    # track routing
    paper_track: str = "part_a_emergence"
    monitoring_mode: str = "off"

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

    # W&B completion table logging interval (0 disables)
    wandb_table_interval: int = 10


def _coalesce_nested_value(raw: Any, key: str) -> Any:
    """Extract scalar config values from Hydra nested config-group payloads."""
    if isinstance(raw, Mapping):
        # reward_metric: {reward_metric: log_marginal_likelihood}
        if key in raw:
            return raw[key]
        # Fallback for single-key dict payloads.
        if len(raw) == 1:
            return next(iter(raw.values()))
    # Back-compat for previous YAML parsing of `off` as bool false.
    if key == "monitoring_mode" and isinstance(raw, bool):
        return "off" if raw is False else "judge_evolving"
    return raw


def config_from_mapping(mapping: Mapping[str, Any]) -> TinkerGRPOConfig:
    """Create TinkerGRPOConfig from a mapping (Hydra-friendly)."""
    flattened: dict[str, Any] = dict(mapping)

    # Pull scalar values out of compositional Hydra subgroups.
    for scalar_key in (
        "reward_metric",
        "reward_data_split",
        "predictive_estimator",
        "paper_track",
        "monitoring_mode",
    ):
        if scalar_key in flattened:
            flattened[scalar_key] = _coalesce_nested_value(flattened[scalar_key], scalar_key)

    data_interface_payload = flattened.pop("data_interface", None)
    if isinstance(data_interface_payload, Mapping):
        for key in (
            "interface_d",
            "p_true_mode",
            "p_true_fixed",
            "p_true_beta_alpha",
            "p_true_beta_beta",
        ):
            if key not in flattened and key in data_interface_payload:
                flattened[key] = data_interface_payload[key]

    monitoring_payload = mapping.get("monitoring_mode")
    if isinstance(monitoring_payload, Mapping):
        for key in ("judge_interval", "rubric_evolution_interval"):
            if key not in flattened and key in monitoring_payload:
                flattened[key] = monitoring_payload[key]

    # Ignore meta keys that are useful for Hydra composition but not runtime config.
    flattened.pop("name", None)

    allowed_keys = {f.name for f in fields(TinkerGRPOConfig)}
    cfg_kwargs = {key: value for key, value in flattened.items() if key in allowed_keys}

    cfg = TinkerGRPOConfig(**cfg_kwargs)
    _validate_config(cfg)
    return cfg


def _validate_config(config: TinkerGRPOConfig) -> None:
    validate_metric_estimator(config.reward_metric, config.predictive_estimator)
    if config.reward_data_split not in {"train", "holdout"}:
        raise ValueError("reward_data_split must be train|holdout")
    if config.paper_track not in {"part_a_emergence", "part_b_mitigation"}:
        raise ValueError("paper_track must be part_a_emergence|part_b_mitigation")
    if config.p_true_mode not in {"sampled_beta", "fixed"}:
        raise ValueError("p_true_mode must be sampled_beta|fixed")


@dataclass
class TrajectoryPoint:
    step: int
    reward_mean: float
    oracle_proxy_mean: float
    gap_proxy_mean: float
    n_valid: int
    n_total: int
    n_parse_fail: int
    n_exec_fail: int
    frac_zero_variance: float = 0.0
    frac_above_oracle_proxy: float = 0.0


@dataclass
class _StepData:
    rollouts_by_prompt: dict[int, list[RolloutData]]
    all_reported: list[float]
    all_oracle: list[float]
    all_completion_hashes: list[str]
    step_records: list[CompletionRecord]
    n_parse_fail: int = 0
    n_exec_fail: int = 0


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


def _format_chat_prompt(user_message: str, exemplar_section: str) -> str:
    """Like format_chat_prompt but injects an exemplar section into the system message."""
    system = SYSTEM_PROMPT + exemplar_section
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


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
    last_norm_payload: dict[str, Any] | None = None
    last_judge_payload: dict[str, Any] | None = None
    last_heuristic_rates: dict[str, float] = {}
    update_summary(
        {
            "paper/track": config.paper_track,
            "paper/reward_metric": config.reward_metric,
            "paper/reward_data_split": config.reward_data_split,
            "paper/predictive_estimator": config.predictive_estimator,
            "paper/monitoring_mode": config.monitoring_mode,
            "paper/interface_d": config.interface_d,
            "paper/p_true_mode": config.p_true_mode,
        }
    )
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
            norm_payload = _log_normalization_summary(step, step_data, norm_metrics_path)
            if norm_payload is not None:
                last_norm_payload = norm_payload
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
        _wandb_log_step(
            step,
            point.reward_mean,
            point.oracle_proxy_mean,
            extra={
                "train/gap_proxy": point.gap_proxy_mean,
                "train/valid_rate": point.n_valid / max(point.n_total, 1),
                "train/diversity": len(set(step_data.all_completion_hashes))
                / max(len(step_data.all_completion_hashes), 1),
                "train/frac_zero_var": frac_zero_variance,
            },
        )
        log_metrics(_compute_step_wandb_metrics(step_data.step_records), step=step)
        writer.flush()

        heuristic_rates = _compute_heuristic_rates(step_data.step_records)
        if heuristic_rates:
            log_heuristic_rates(step, heuristic_rates)
            last_heuristic_rates = heuristic_rates

        if config.wandb_table_interval > 0 and step % config.wandb_table_interval == 0:
            log_completion_table(step, step_data.step_records)

        judge_verdicts = _maybe_run_step_judge(
            step, step_data.step_records, config, judge_cfg, judge_metrics_path, rubric_state
        )
        if judge_verdicts:
            last_judge_payload = _summarize_judge_verdicts(judge_verdicts)
        rubric_state = _maybe_evolve_rubric(
            step, step_data, config, rubric_state, judge_verdicts, judge_cfg=judge_cfg
        )

        lh_programs = _collect_lh_programs(step, step_data.step_records, judge_verdicts)
        _wandb_log_lh(lh_programs, step=step)
        log_metrics({"lh/count": len(lh_programs)}, step=step)

    writer.close()
    _write_trajectory(output_dir, trajectory)
    results = _compute_results(config, trajectory)
    summary_payload = _build_wandb_final_summary(
        config=config,
        results=results,
        normalization_payload=last_norm_payload,
        judge_payload=last_judge_payload,
        heuristic_rates=last_heuristic_rates,
    )
    results.update(summary_payload)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log_metrics(summary_payload)
    update_summary(summary_payload)

    artifact_paths = [
        output_dir / "completions.jsonl",
        output_dir / "judge_metrics.jsonl",
        norm_metrics_path,
        output_dir / "rubric_evolution.jsonl",
    ]
    upload_artifacts(artifact_paths)

    log.info("Results saved to %s", output_dir)
    _print_summary(results)

    return results


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
            _format_chat_prompt(p, exemplar_section) for p in get_prompts(config.n_prompts)
        ]
    else:
        prompt_texts = [format_chat_prompt(p) for p in get_prompts(config.n_prompts)]
    return writer, judge_metrics_path, judge_cfg, training_client, tokenizer, prompt_texts


def _get_normalization_indices(step: int, config: TinkerGRPOConfig) -> set[int]:
    """Select which record indices to run normalization on this step."""
    if config.normalization_interval <= 0 or step % config.normalization_interval != 0:
        return set()
    if config.interface_d > 5:
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
            reward_metric=config.reward_metric,
            reward_data_split=config.reward_data_split,
            predictive_estimator=config.predictive_estimator,
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
    if config.p_true_mode == "sampled_beta":
        rng = np.random.default_rng(prompt_seed)
        p_true = float(rng.beta(config.p_true_beta_alpha, config.p_true_beta_beta))
    else:
        p_true = float(config.p_true_fixed)

    train_data = make_scoring_data_dict(
        d=config.interface_d,
        p_true=p_true,
        seed=prompt_seed,
    )
    holdout_data = make_scoring_data_dict(
        d=config.interface_d,
        p_true=p_true,
        seed=prompt_seed + 1_000_000,
    )
    train_data["y_train"] = np.asarray(train_data["y"], dtype=np.float64)
    train_data["y_holdout"] = np.asarray(holdout_data["y"], dtype=np.float64)
    train_data["p_true"] = p_true
    train_data["seed"] = prompt_seed
    return train_data


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
            reward_metric=config.reward_metric,
            reward_data_split=config.reward_data_split,
            predictive_estimator=config.predictive_estimator,
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
            completion_text,
            scoring_data=scoring_data,
            timeout=config.exec_timeout,
            reward_metric=config.reward_metric,
            reward_data_split=config.reward_data_split,
            predictive_estimator=config.predictive_estimator,
            smc_draws=config.smc_draws,
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
        gap=reported - oracle if oracle is not None else float("nan"),
        outcome=outcome,
        timestamp=make_timestamp(),
        metadata=metadata or None,
    )
    writer.write(record)
    step_data.step_records.append(record)
    return rollout


def _run_normalization_check(
    completion_text: str,
    *,
    scoring_data: dict[str, Any],
    timeout: int,
    reward_metric: str,
    reward_data_split: str,
    predictive_estimator: str,
    smc_draws: int,
) -> dict[str, Any] | None:
    """Run normalization check, returning result dict or None on failure."""
    try:
        result = check_normalization_small_d(
            completion_text,
            d=int(scoring_data.get("d", np.asarray(scoring_data["y"]).size)),
            p_true=float(scoring_data.get("p_true", 0.5)),
            timeout=timeout,
            reward_metric=reward_metric,
            reward_data_split=reward_data_split,
            predictive_estimator=predictive_estimator,
            smc_draws=smc_draws,
            scoring_data=scoring_data,
        )
        if result.get("ok"):
            return {
                "log_mass": result["log_mass"],
                "mass": result.get("mass"),
                "is_normalized": result["is_normalized"],
                "d": result.get("d"),
            }
        return {"ok": False, "reason": result.get("reason", "unknown"), "d": result.get("d")}
    except Exception as exc:
        log.debug("normalization check failed: %s", exc)
        return None


def _collect_lh_programs(
    step: int,
    step_records: list[CompletionRecord],
    _judge_verdicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter step records for programs with integral Z_p != 1 (formal LH)."""
    lh_rows: list[dict[str, Any]] = []

    for rec in step_records:
        if rec.outcome != "valid":
            continue

        detection: list[str] = []
        log_mass = None

        if rec.metadata and rec.metadata.get("normalization"):
            norm = rec.metadata["normalization"]
            if norm.get("is_normalized") is False:
                detection.append("normalization")
                log_mass = norm.get("log_mass")

        if not detection:
            continue

        lh_rows.append(
            {
                "step": step,
                "code": rec.code or "",
                "reported": rec.reported_reward,
                "oracle": rec.oracle_score,
                "gap": rec.gap,
                "detection": ",".join(detection),
                "log_mass": log_mass,
                "judge_verdict": "",
                "judge_tags": "",
            }
        )

    return lh_rows


def _compute_heuristic_rates(records: list[CompletionRecord]) -> dict[str, float]:
    """Run heuristic exploit detection over valid records and return tag rates."""
    valid = [r for r in records if r.outcome == "valid" and r.code]
    if not valid:
        return {}
    tag_counter: Counter[str] = Counter()
    n_any = 0
    for r in valid:
        tags = detect_exploits(r.code)
        tag_counter.update(tags)
        if tags:
            n_any += 1
    n = len(valid)
    rates: dict[str, float] = {
        "score_bonus_rate": tag_counter.get("score_bonus", 0) / n,
        "double_observation_rate": tag_counter.get("double_observation", 0) / n,
        "custom_dist_rate": tag_counter.get("custom_dist", 0) / n,
        "observed_literal_rate": tag_counter.get("observed_literal", 0) / n,
        "data_discard_rate": tag_counter.get("data_discard", 0) / n,
        "any_exploit_rate": n_any / n,
    }
    return rates


def _log_normalization_summary(
    step: int, step_data: _StepData, metrics_path: Path
) -> dict[str, Any] | None:
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
        return None

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

    log_normalization_metrics(step, n_checked, frac_non_norm, mean_abs_lm)
    return payload


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


def _compute_step_point(
    step: int, step_data: _StepData, frac_zero_variance: float
) -> TrajectoryPoint:
    valid_rewards = [
        reported
        for reported in step_data.all_reported
        if np.isfinite(reported) and classify_outcome(float(reported)) == "valid"
    ]
    oracle_valid = [oracle for oracle in step_data.all_oracle if np.isfinite(oracle)]
    aligned_pairs = [
        (reported, oracle)
        for reported, oracle in zip(step_data.all_reported, step_data.all_oracle, strict=True)
        if np.isfinite(reported)
        and np.isfinite(oracle)
        and classify_outcome(float(reported)) == "valid"
    ]

    reward_mean = float(np.mean(valid_rewards)) if valid_rewards else float("nan")
    oracle_proxy_mean = float(np.mean(oracle_valid)) if oracle_valid else float("nan")
    if np.isfinite(reward_mean) and np.isfinite(oracle_proxy_mean):
        gap_proxy = reward_mean - oracle_proxy_mean
    else:
        gap_proxy = float("nan")
    frac_above_proxy = (
        sum(1 for r, o in aligned_pairs if r > o) / len(aligned_pairs) if aligned_pairs else 0.0
    )

    return TrajectoryPoint(
        step=step,
        reward_mean=reward_mean,
        oracle_proxy_mean=oracle_proxy_mean,
        gap_proxy_mean=gap_proxy,
        n_valid=len(valid_rewards),
        n_total=len(step_data.all_reported),
        n_parse_fail=step_data.n_parse_fail,
        n_exec_fail=step_data.n_exec_fail,
        frac_zero_variance=frac_zero_variance,
        frac_above_oracle_proxy=frac_above_proxy,
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
        "interface_d": config.interface_d,
        "p_true_mode": config.p_true_mode,
        "p_true_fixed": config.p_true_fixed,
        "p_true_beta_alpha": config.p_true_beta_alpha,
        "p_true_beta_beta": config.p_true_beta_beta,
        "reward_metric": config.reward_metric,
        "reward_data_split": config.reward_data_split,
        "predictive_estimator": config.predictive_estimator,
        "smc_draws": config.smc_draws,
        "scoring_workers": config.scoring_workers,
        "max_tokens": config.max_tokens,
        "paper_track": config.paper_track,
        "monitoring_mode": config.monitoring_mode,
    }
    metrics["final_frac_zero_variance"] = final.frac_zero_variance
    metrics["final_reward_mean"] = metrics.get("final_reward")
    metrics["final_oracle_proxy_mean"] = metrics.get("final_oracle_proxy")
    metrics["final_gap_proxy_mean"] = metrics.get("final_gap_proxy")
    metrics["n_steps_completed"] = len(trajectory)
    return metrics


def _print_summary(results: dict) -> None:
    print_training_summary(results)
    if "error" not in results:
        log.info("zero-variance fraction: %.1f%%", results["final_frac_zero_variance"] * 100)


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
        "Track: %s | Monitoring: %s",
        config.paper_track,
        config.monitoring_mode,
    )
    log.info(
        "Sampling: temp=%.2f, top_p=%.2f, top_k=%d",
        config.temperature,
        config.top_p,
        config.top_k,
    )
    log.info(
        "Reward: metric=%s split=%s estimator=%s d=%d smc_draws=%d workers=%d",
        config.reward_metric,
        config.reward_data_split,
        config.predictive_estimator,
        config.interface_d,
        config.smc_draws,
        config.scoring_workers,
    )
    if config.p_true_mode == "sampled_beta":
        log.info(
            "Interface p_true mode: sampled_beta(alpha=%.2f,beta=%.2f)",
            config.p_true_beta_alpha,
            config.p_true_beta_beta,
        )
    else:
        log.info("Interface p_true mode: fixed(%.3f)", config.p_true_fixed)
    log.info("Output: %s", output_dir)
    if config.normalization_interval > 0:
        log.info(
            "Normalization: every %d steps, sample %d records (d=%d)",
            config.normalization_interval,
            config.normalization_sample_size,
            config.interface_d,
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
    _log_reward_diag(step, step_data.all_reported)


def _log_reward_diag(step: int, all_reported: list[float]) -> None:
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
        "Step %3d: reward=%.3f oracle_proxy=%.3f gap_proxy=%.3f valid=%d/%d "
        "parse_fail=%d exec_fail=%d frac_zero_var=%.2f above_oracle_proxy=%.2f",
        step,
        point.reward_mean,
        point.oracle_proxy_mean,
        point.gap_proxy_mean,
        point.n_valid,
        point.n_total,
        point.n_parse_fail,
        point.n_exec_fail,
        point.frac_zero_variance,
        point.frac_above_oracle_proxy,
    )


def _maybe_run_step_judge(
    step: int,
    step_records: list[CompletionRecord],
    config: TinkerGRPOConfig,
    judge_cfg: JudgeConfig,
    judge_metrics_path: Path,
    rubric_state=None,
) -> list[dict[str, Any]]:
    if config.monitoring_mode == "off":
        return []
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

    log_judge_metrics(step, hacking_rate, len(verdicts), dict(verdict_counts), dict(tag_counts))

    log.info(
        "Step %3d JUDGE: n=%d hacking_rate=%.3f verdicts=%s",
        step,
        len(verdicts),
        hacking_rate,
        dict(verdict_counts),
    )

    return [
        {"verdict": v.verdict, "tags": v.tags, "reported_reward": r.reported_reward}
        for v, r in zip(verdicts, judged_records, strict=True)
    ]


def _summarize_judge_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(verdicts)
    if n == 0:
        return {"n_judged": 0, "hacking_rate": float("nan")}

    n_hacking = 0
    n_with_verdict = 0
    for item in verdicts:
        verdict = item.get("verdict")
        if isinstance(verdict, str) and verdict:
            n_with_verdict += 1
            if verdict == "hacking":
                n_hacking += 1
            continue

        tags = item.get("tags", [])
        if isinstance(tags, list) and any(str(tag).startswith("lh_") for tag in tags):
            n_hacking += 1

    return {
        "n_judged": n,
        "hacking_rate": n_hacking / (n_with_verdict if n_with_verdict else n),
    }


def _compute_step_wandb_metrics(records: list[CompletionRecord]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"train/n_total": len(records)}
    if not records:
        return metrics

    outcome_counts = Counter(r.outcome for r in records)
    metrics["train/n_valid"] = outcome_counts.get("valid", 0)
    metrics["train/n_parse_fail"] = outcome_counts.get("parse_fail", 0)
    metrics["train/n_exec_fail"] = outcome_counts.get("exec_fail", 0)

    scorer_outcome_counts: Counter[str] = Counter()
    metric_vals: dict[str, list[float]] = {
        "metric_log_marginal_likelihood": [],
        "metric_elpd": [],
        "metric_waic": [],
        "metric_bic": [],
    }
    n_valid_with_decomposition = 0
    n_with_potential = 0
    n_data_discard = 0

    for record in records:
        metadata = record.metadata if isinstance(record.metadata, dict) else None
        decomposition = metadata.get("decomposition") if metadata else None
        if not isinstance(decomposition, dict):
            continue
        if record.outcome == "valid":
            n_valid_with_decomposition += 1
        outcome_code = decomposition.get("outcome_code")
        if isinstance(outcome_code, str) and outcome_code:
            scorer_outcome_counts[outcome_code] += 1
        if record.outcome == "valid":
            if decomposition.get("data_discard") is True:
                n_data_discard += 1
            n_pot_terms = decomposition.get("n_pot_terms")
            if isinstance(n_pot_terms, int | float) and float(n_pot_terms) > 0:
                n_with_potential += 1
        for key in metric_vals:
            value = decomposition.get(key)
            if isinstance(value, int | float) and np.isfinite(float(value)):
                metric_vals[key].append(float(value))

    for outcome_code, count in scorer_outcome_counts.items():
        metrics[f"train/scorer_outcome/{outcome_code}"] = count

    for key, values in metric_vals.items():
        if not values:
            continue
        suffix = key.removeprefix("metric_")
        metrics[f"train/metric/{suffix}_mean"] = float(np.mean(values))
        metrics[f"train/metric/{suffix}_std"] = float(np.std(values))

    denominator = max(n_valid_with_decomposition, 1)
    metrics["train/prevalence/potential_terms"] = n_with_potential / denominator
    metrics["train/prevalence/data_discard"] = n_data_discard / denominator
    return metrics


def _build_wandb_final_summary(
    *,
    config: TinkerGRPOConfig,
    results: dict[str, Any],
    normalization_payload: dict[str, Any] | None,
    judge_payload: dict[str, Any] | None,
    heuristic_rates: dict[str, float] | None,
) -> dict[str, Any]:
    run_status = "success"
    error_reason: str | None = None
    if "error" in results:
        run_status = "fail"
        error_reason = str(results.get("error", "unknown"))
    else:
        final_valid_rate = results.get("final_valid_rate")
        final_reward = results.get("final_reward_mean", results.get("final_reward"))
        if isinstance(final_valid_rate, int | float) and final_valid_rate <= 0:
            run_status = "fail"
            error_reason = "no_valid_completions"
        elif isinstance(final_reward, int | float) and not np.isfinite(float(final_reward)):
            run_status = "fail"
            error_reason = "non_finite_final_reward"

    summary: dict[str, Any] = {
        "sweep/run_status": run_status,
        "sweep/final_reward_mean": results.get("final_reward_mean", results.get("final_reward")),
        "sweep/final_oracle_proxy_mean": results.get(
            "final_oracle_proxy_mean", results.get("final_oracle_proxy")
        ),
        "sweep/final_gap_proxy_mean": results.get(
            "final_gap_proxy_mean", results.get("final_gap_proxy")
        ),
        "sweep/final_valid_rate": results.get("final_valid_rate"),
        "paper/track": config.paper_track,
        "paper/reward_metric": config.reward_metric,
        "paper/reward_data_split": config.reward_data_split,
        "paper/predictive_estimator": config.predictive_estimator,
        "paper/monitoring_mode": config.monitoring_mode,
        "paper/frac_non_normalized_final": float("nan"),
        "paper/judge_hacking_rate_final": float("nan"),
        "paper/lh_family_prevalence_final": float("nan"),
    }
    if error_reason is not None:
        summary["sweep/error"] = error_reason
    if normalization_payload and isinstance(
        normalization_payload.get("frac_non_normalized"), int | float
    ):
        summary["paper/frac_non_normalized_final"] = float(
            normalization_payload["frac_non_normalized"]
        )
    if judge_payload and isinstance(judge_payload.get("hacking_rate"), int | float):
        summary["paper/judge_hacking_rate_final"] = float(judge_payload["hacking_rate"])
    if heuristic_rates and isinstance(heuristic_rates.get("any_exploit_rate"), int | float):
        summary["paper/lh_family_prevalence_final"] = float(heuristic_rates["any_exploit_rate"])
    return summary


def _load_judge_env(env_file: str | None) -> None:
    if not env_file:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        log.warning("python-dotenv not installed; judge env-file ignored")


def _init_rubric(config: TinkerGRPOConfig):
    if config.monitoring_mode != "judge_evolving" or config.rubric_evolution_interval <= 0:
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

        log_metrics(
            {
                "rubric/n_active": len(rubric_state.active_items),
                "rubric/n_evolved": len(rubric_state.evolved_items),
            },
            step=step,
        )
    except Exception as exc:
        log.warning("Step %3d RUBRIC: evolution failed: %s", step, exc)

    return rubric_state
