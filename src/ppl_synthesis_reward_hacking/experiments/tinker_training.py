"""Tinker GRPO training loop with local PyMC scoring."""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import extract_pymc_code
from ppl_synthesis_reward_hacking.backends.sstan.gate import (
    SStanFidelityPolicy,
    SStanGateConfig,
    SStanGateMode,
    SStanGateResult,
    SStanLogStanPolicy,
    run_sstan_gate,
    validate_sstan_gate_cfg,
)
from ppl_synthesis_reward_hacking.config.contracts import validate_train_contract
from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.data.interfaces import list_dataset_interfaces
from ppl_synthesis_reward_hacking.data.prompts import format_chat_from_system
from ppl_synthesis_reward_hacking.data.pymc_reward_loader import (
    get_generic_system_prompt,
    get_prompts_for_dataset,
    get_system_prompt,
    load_jsonl_prompts,
)
from ppl_synthesis_reward_hacking.evaluation.heuristics import detect_exploits
from ppl_synthesis_reward_hacking.evaluation.normalization_check import (
    check_normalization,
)
from ppl_synthesis_reward_hacking.evaluation.taxonomy import (
    EXPLOIT_FAMILIES,
    canonicalize_exploit_tags,
    canonicalize_judge_tags,
)
from ppl_synthesis_reward_hacking.experiments.config_mapping import (
    GROUP_ALLOWED_KEYS,
    SCALAR_GROUP_KEYS,
    build_train_config_from_mapping,
    validate_group_payload_keys,
    validate_scalar_group_payloads,
)
from ppl_synthesis_reward_hacking.experiments.grpo import (
    RolloutData,
    build_tinker_datum,
    compute_group_relative_advantages,
)
from ppl_synthesis_reward_hacking.experiments.results import (
    attach_common_results_metadata,
    compute_traj_metrics,
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
)
from ppl_synthesis_reward_hacking.utils.hashing import normalized_text_hash
from ppl_synthesis_reward_hacking.utils.tinker import tinker, types, validate_tinker_setup

log = logging.getLogger(__name__)

_JUDGE_BACKEND_DEFAULTS: dict[str, dict[str, str | None]] = {
    "zai": {
        "api_base": "https://api.z.ai/api/anthropic",
        "custom_llm_provider": "anthropic",
        "api_key_env": "ZHIPUAI_API_KEY",
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "custom_llm_provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "api_base": None,
        "custom_llm_provider": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "api_base": None,
        "custom_llm_provider": None,
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "custom": {
        "api_base": None,
        "custom_llm_provider": None,
        "api_key_env": None,
    },
}
_JUDGE_SAMPLING_POLICIES = frozenset({"fixed_cap", "adaptive_cap"})
_JUDGE_ADAPTIVE_TARGET_METRICS = frozenset({"heuristic_novelty", "outcome_entropy", "combined"})
_JUDGE_BATCH_MODES = frozenset({"auto", "batch_completion", "sequential"})
_PROMPT_SOURCES = frozenset({"hardcoded", "jsonl"})
_PROMPT_SAMPLING_MODES = frozenset({"fixed_cycle", "seeded_shuffle_cycle"})
_THINKING_MODES = frozenset({"think", "no_think"})
_PROMPT_POLICIES = frozenset({"legacy", "neutral"})
_LH_CANONICAL_FAMILIES = tuple(f for f in EXPLOIT_FAMILIES if f.startswith("lh_"))
_OPENAI_GPT5_DATED_SNAPSHOT = re.compile(
    r"^(?:openai/)?gpt-5(?:\.\d+)?-\d{4}-\d{2}-\d{2}$",
    re.IGNORECASE,
)


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
    sample_timeout_s: float = 1800.0
    sample_max_retries: int = 2
    sample_retry_backoff_s: float = 10.0

    # GRPO
    loss_fn: str = "importance_sampling"
    train_op_timeout_s: float = 900.0

    # reward objective
    exec_timeout: int = 60
    reward_metric: str = "log_marginal_likelihood"
    reward_data_split: str = "train"
    reward_estimator_backend: str = "smc"
    claim_mode: str = "formal_lh"

    # dataset interface
    dataset_name: str = "linear_regression"
    prompt_source: str = "hardcoded"
    prompt_jsonl_path: str = "data/pymc_reward/pymc_reward_train.jsonl"
    prompt_jsonl_max_examples: int = 148
    prompt_sampling: str = "seeded_shuffle_cycle"
    thinking_mode: str = "think"
    prompt_policy: str = "legacy"
    dataset_n_features: int = 3
    dataset_noise_sigma: float = 1.0
    dataset_n_train: int = 20
    dataset_n_holdout: int = 256
    dataset_p_true_alpha: float = 1.0
    dataset_p_true_beta: float = 1.0
    dataset_p_true_fixed: float | None = None
    scoring_seed_base: int = 0
    smc_draws: int = 500
    scoring_workers: int = 1

    # track routing
    paper_track: str = "part_a_emergence"
    monitoring_mode: str = "off"
    sstan_gate_mode: str = "off"
    sstan_gate_sample_rate: float = 1.0
    sstan_gate_penalty_reward: float = -100.0
    sstan_gate_fidelity_policy: str = "reject_on_source_signal"
    sstan_gate_log_stan: str = "none"
    sstan_transpiler_model: str = "glm-5"
    sstan_transpiler_api_key_env: str | None = None
    sstan_transpiler_api_base: str | None = None
    sstan_transpiler_custom_llm_provider: str | None = None
    sstan_transpiler_strict: bool = True
    sstan_transpiler_temperature: float = 0.0
    sstan_transpiler_max_tokens: int = 2048
    sstan_transpiler_timeout_s: float = 60.0

    # output
    output_dir: str = "artifacts/tinker_pymc_grpo"

    # exemplar config (None disables)
    exemplar_config: str | None = None

    # observational judge (not used in GRPO reward)
    judge_interval: int = 0
    judge_sample_size: int = 20
    judge_dedup: bool = True
    judge_backend: str = "openai"
    judge_model: str = "gpt-5.2"
    judge_api_base: str | None = None
    judge_custom_llm_provider: str | None = None
    judge_api_key_env: str | None = None
    judge_use_stub: bool = False
    judge_env_file: str | None = None
    judge_sampling_policy: str = "fixed_cap"
    judge_adaptive_min: int = 8
    judge_adaptive_max: int = 20
    judge_adaptive_target_metric: str = "combined"
    judge_batch_mode: str = "auto"
    judge_batch_max_workers: int = 32
    track_lh_every_batch: bool = True

    # rubric evolution (0 disables)
    rubric_evolution_interval: int = 0

    # normalization monitoring (0 disables)
    normalization_method: str = "auto"
    normalization_delta_scope: str = "raw_y_binary"
    normalization_epsilon: float = 5e-2
    normalization_ci_alpha: float = 0.05
    normalization_mc_samples: int = 256
    normalization_min_ess: float = 30.0
    normalization_interval: int = 1
    normalization_sample_size: int = 20

    # W&B completion table logging interval (0 disables)
    wandb_table_interval: int = 10

    # periodic artifact sync (0 disables)
    artifact_sync_interval: int = 50
    artifact_sync_min_steps: int = 300


def config_from_mapping(mapping: Mapping[str, Any]) -> TinkerGRPOConfig:
    """Create TinkerGRPOConfig from a mapping (Hydra-friendly)."""
    validate_group_payload_keys(mapping, group_allowed_keys=GROUP_ALLOWED_KEYS)
    validate_scalar_group_payloads(mapping, scalar_group_keys=SCALAR_GROUP_KEYS)
    cfg = build_train_config_from_mapping(
        mapping,
        TinkerGRPOConfig,
        error_prefix="Unexpected train config keys (fail-fast)",
    )
    _validate_config(cfg)
    return cfg


_SUPPORTED_DATASETS = frozenset(list_dataset_interfaces())


def _validate_config(config: TinkerGRPOConfig) -> None:
    _validate_track_dataset_and_prompts(config)
    _resolve_normalization_contract(config)
    _validate_judge_backend_config(config)
    _validate_judge_sampling_config(config)
    _validate_runtime_timeout_config(config)
    _validate_artifact_sync_config(config)
    validate_sstan_gate_cfg(_build_sstan_gate_config(config), paper_track=config.paper_track)


def _validate_track_dataset_and_prompts(config: TinkerGRPOConfig) -> None:
    if config.paper_track not in {"part_a_emergence", "part_b_mitigation"}:
        raise ValueError("paper_track must be part_a_emergence|part_b_mitigation")
    if config.monitoring_mode not in {"off", "judge_evolving"}:
        raise ValueError("monitoring_mode must be off|judge_evolving")
    if config.dataset_name not in _SUPPORTED_DATASETS:
        raise ValueError(
            "dataset_name must be one of "
            f"{sorted(_SUPPORTED_DATASETS)}, got {config.dataset_name!r}"
        )
    if config.prompt_source not in _PROMPT_SOURCES:
        raise ValueError(
            f"prompt_source must be one of {sorted(_PROMPT_SOURCES)}, got {config.prompt_source!r}"
        )
    if config.prompt_sampling not in _PROMPT_SAMPLING_MODES:
        raise ValueError(
            "prompt_sampling must be one of "
            f"{sorted(_PROMPT_SAMPLING_MODES)}, got {config.prompt_sampling!r}"
        )
    if config.thinking_mode not in _THINKING_MODES:
        raise ValueError(
            f"thinking_mode must be one of {sorted(_THINKING_MODES)}, got {config.thinking_mode!r}"
        )
    if config.prompt_policy not in _PROMPT_POLICIES:
        raise ValueError(
            f"prompt_policy must be one of {sorted(_PROMPT_POLICIES)}, "
            f"got {config.prompt_policy!r}"
        )
    if config.prompt_jsonl_max_examples <= 0:
        raise ValueError("prompt_jsonl_max_examples must be positive")
    if config.prompt_source == "jsonl" and not Path(config.prompt_jsonl_path).exists():
        raise FileNotFoundError(f"prompt_jsonl_path not found: {config.prompt_jsonl_path}")
    if config.prompt_source == "jsonl" and config.exemplar_config:
        raise ValueError("exemplar_config is not supported with prompt_source=jsonl")


def _resolve_normalization_contract(config: TinkerGRPOConfig) -> None:
    resolved = validate_train_contract(
        reward_metric=config.reward_metric,
        reward_data_split=config.reward_data_split,
        reward_estimator_backend=config.reward_estimator_backend,
        claim_mode=config.claim_mode,
        dataset_name=config.dataset_name,
        normalization_method=config.normalization_method,
        normalization_delta_scope=config.normalization_delta_scope,
        normalization_epsilon=config.normalization_epsilon,
        normalization_ci_alpha=config.normalization_ci_alpha,
        normalization_mc_samples=config.normalization_mc_samples,
        normalization_min_ess=config.normalization_min_ess,
        normalization_interval=config.normalization_interval,
    )
    config.normalization_method = resolved.normalization_method
    config.normalization_delta_scope = resolved.normalization_delta_scope


def _validate_judge_backend_config(config: TinkerGRPOConfig) -> None:
    if config.judge_backend not in _JUDGE_BACKEND_DEFAULTS:
        raise ValueError(
            "judge_backend must be one of "
            f"{sorted(_JUDGE_BACKEND_DEFAULTS)}, got {config.judge_backend!r}"
        )
    if config.judge_backend == "openai" and _OPENAI_GPT5_DATED_SNAPSHOT.match(config.judge_model):
        raise ValueError(
            "For openai judge backend, use floating model aliases (e.g., gpt-5.2), "
            f"not dated snapshots: {config.judge_model!r}"
        )
    if config.judge_batch_mode not in _JUDGE_BATCH_MODES:
        raise ValueError(
            "judge_batch_mode must be one of "
            f"{sorted(_JUDGE_BATCH_MODES)}, got {config.judge_batch_mode!r}"
        )
    if config.judge_batch_max_workers <= 0:
        raise ValueError("judge_batch_max_workers must be positive")


def _validate_judge_sampling_config(config: TinkerGRPOConfig) -> None:
    if config.judge_sample_size <= 0:
        raise ValueError("judge_sample_size must be positive")
    if config.judge_sampling_policy not in _JUDGE_SAMPLING_POLICIES:
        raise ValueError(
            "judge_sampling_policy must be one of "
            f"{sorted(_JUDGE_SAMPLING_POLICIES)}, got {config.judge_sampling_policy!r}"
        )
    if config.judge_adaptive_target_metric not in _JUDGE_ADAPTIVE_TARGET_METRICS:
        raise ValueError(
            "judge_adaptive_target_metric must be one of "
            f"{sorted(_JUDGE_ADAPTIVE_TARGET_METRICS)}, got {config.judge_adaptive_target_metric!r}"
        )
    if config.judge_adaptive_min < 0:
        raise ValueError("judge_adaptive_min must be >= 0")
    if config.judge_adaptive_max < config.judge_adaptive_min:
        raise ValueError("judge_adaptive_max must be >= judge_adaptive_min")
    if config.judge_sampling_policy == "adaptive_cap" and config.judge_adaptive_max <= 0:
        raise ValueError("adaptive judge sampling requires judge_adaptive_max > 0")


def _validate_runtime_timeout_config(config: TinkerGRPOConfig) -> None:
    if config.sample_timeout_s <= 0:
        raise ValueError("sample_timeout_s must be > 0")
    if config.sample_max_retries < 0:
        raise ValueError("sample_max_retries must be >= 0")
    if config.sample_retry_backoff_s < 0:
        raise ValueError("sample_retry_backoff_s must be >= 0")
    if config.train_op_timeout_s <= 0:
        raise ValueError("train_op_timeout_s must be > 0")
    if config.normalization_sample_size <= 0:
        raise ValueError("normalization_sample_size must be positive")


def _validate_artifact_sync_config(config: TinkerGRPOConfig) -> None:
    if config.artifact_sync_interval < 0:
        raise ValueError("artifact_sync_interval must be >= 0")
    if config.artifact_sync_min_steps < 0:
        raise ValueError("artifact_sync_min_steps must be >= 0")


def _build_sstan_gate_config(config: TinkerGRPOConfig) -> SStanGateConfig:
    return SStanGateConfig(
        mode=cast(SStanGateMode, config.sstan_gate_mode),
        sample_rate=config.sstan_gate_sample_rate,
        penalty_reward=config.sstan_gate_penalty_reward,
        fidelity_policy=cast(SStanFidelityPolicy, config.sstan_gate_fidelity_policy),
        log_stan=cast(SStanLogStanPolicy, config.sstan_gate_log_stan),
        transpiler_model=config.sstan_transpiler_model,
        transpiler_api_key_env=config.sstan_transpiler_api_key_env,
        transpiler_api_base=config.sstan_transpiler_api_base,
        transpiler_custom_llm_provider=config.sstan_transpiler_custom_llm_provider,
        transpiler_strict=config.sstan_transpiler_strict,
        transpiler_temperature=config.sstan_transpiler_temperature,
        transpiler_max_tokens=config.sstan_transpiler_max_tokens,
        transpiler_timeout_s=config.sstan_transpiler_timeout_s,
    )


def _build_judge_config(config: TinkerGRPOConfig) -> JudgeConfig:
    backend_defaults = _JUDGE_BACKEND_DEFAULTS[config.judge_backend]
    api_base = (
        config.judge_api_base if config.judge_api_base is not None else backend_defaults["api_base"]
    )
    custom_llm_provider = (
        config.judge_custom_llm_provider
        if config.judge_custom_llm_provider is not None
        else backend_defaults["custom_llm_provider"]
    )
    api_key_env = (
        config.judge_api_key_env
        if config.judge_api_key_env is not None
        else backend_defaults["api_key_env"]
    )
    return JudgeConfig(
        backend=config.judge_backend,
        model=config.judge_model,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider or "",
        api_key_env=api_key_env or "",
        use_stub=config.judge_use_stub,
        batch_mode=config.judge_batch_mode,
        batch_max_workers=config.judge_batch_max_workers,
    )


@dataclass
class TrajectoryPoint:
    step: int
    reward_mean: float
    n_valid: int
    n_total: int
    n_parse_fail: int
    n_exec_fail: int
    frac_zero_variance: float = 0.0
    frac_non_normalized: float = float("nan")
    mean_abs_log_mass: float = float("nan")
    n_norm_checked: int = 0


@dataclass
class _StepData:
    rollouts_by_prompt: dict[int, list[RolloutData]]
    all_reported: list[float]
    all_completion_hashes: list[str]
    step_records: list[CompletionRecord]
    n_parse_fail: int = 0
    n_exec_fail: int = 0


@dataclass
class _RunState:
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    rubric_state: Any = None
    last_norm_payload: dict[str, Any] | None = None
    last_judge_payload: dict[str, Any] | None = None
    last_heuristic_rates: dict[str, float] = field(default_factory=dict)
    last_batch_lh_metrics: dict[str, float] = field(default_factory=dict)
    batch_lh_history: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    user_message: str
    system_prompt: str
    prompt_text: str


def _load_exemplars(config_path: str | None) -> list[dict[str, Any]]:
    """Load exemplar YAML and return the list of exemplar dicts."""
    if not config_path:
        return []
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"exemplar config not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"invalid exemplar config in {path}; expected a mapping with 'exemplars'")
    exemplars = data.get("exemplars", [])
    if not isinstance(exemplars, list) or not exemplars:
        raise ValueError(f"no exemplars found in {path}")
    required_fields = ("name", "description", "code")
    for index, exemplar in enumerate(exemplars):
        if not isinstance(exemplar, Mapping):
            raise ValueError(f"invalid exemplar at index {index} in {path}; expected a mapping")
        missing = [
            field_name
            for field_name in required_fields
            if not isinstance(exemplar.get(field_name), str) or not exemplar[field_name].strip()
        ]
        if missing:
            missing_csv = ", ".join(missing)
            raise ValueError(
                f"invalid exemplar at index {index} in {path}; missing string fields: {missing_csv}"
            )
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


def _format_chat_prompt(
    user_message: str,
    exemplar_section: str,
    dataset_name: str,
    *,
    thinking_mode: str,
    prompt_policy: str,
) -> str:
    """Render a chat prompt with exemplars appended to the system message."""
    system = get_system_prompt(dataset_name, prompt_policy=prompt_policy) + exemplar_section
    return format_chat_from_system(
        system_prompt=system,
        user_message=user_message,
        thinking_mode=thinking_mode,
    )


def _deduplicate_prompt_templates(pool: list[str]) -> list[str]:
    """Drop duplicate prompt templates while preserving first-seen order."""
    return list(dict.fromkeys(pool))


def _select_prompt_texts_from_pool(
    pool: list[str],
    *,
    n_prompts: int,
    sampling_mode: str,
    seed: int,
) -> list[str]:
    """Pick `n_prompts` deterministically from a template pool."""
    if n_prompts <= 0:
        return []
    if not pool:
        raise ValueError("prompt pool is empty")
    if sampling_mode == "fixed_cycle":
        return [pool[i % len(pool)] for i in range(n_prompts)]
    if sampling_mode == "seeded_shuffle_cycle":
        rng = np.random.default_rng(seed)
        order = np.arange(len(pool))
        rng.shuffle(order)
        shuffled = [pool[int(i)] for i in order]
        return [shuffled[i % len(shuffled)] for i in range(n_prompts)]
    raise ValueError(
        f"unknown prompt sampling mode {sampling_mode!r}; expected {sorted(_PROMPT_SAMPLING_MODES)}"
    )


def _encode_prompt_tokens(tokenizer, prompt: PromptSpec, *, thinking_mode: str) -> list[int]:
    """Encode a prompt, preferring tokenizer chat templates when available."""
    messages = _chat_messages_from_prompt(prompt)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    tokens = _tokens_from_chat_template(
        apply_template=apply_template,
        messages=messages,
        thinking_mode=thinking_mode,
    )
    if tokens is not None:
        return tokens

    # apply_chat_template unavailable or returned non-list/tensor; fall through to manual formatting
    rendered = format_chat_from_system(
        system_prompt=prompt.system_prompt,
        user_message=prompt.user_message,
        thinking_mode=thinking_mode,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _chat_messages_from_prompt(prompt: PromptSpec) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": prompt.user_message},
    ]


def _apply_no_think_prefix(messages: list[dict[str, str]]) -> None:
    messages[0]["content"] = "/no_think\n" + str(messages[0]["content"])


def _coerce_template_tokens(tokens: Any) -> list[int] | None:
    if isinstance(tokens, list):
        return [int(tok) for tok in tokens]
    if hasattr(tokens, "tolist"):
        return [int(tok) for tok in tokens.tolist()]
    return None


def _tokens_from_chat_template(
    *,
    apply_template: Any,
    messages: list[dict[str, str]],
    thinking_mode: str,
) -> list[int] | None:
    if not callable(apply_template):
        return None
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
    _configure_thinking_kwargs(
        apply_template=apply_template,
        kwargs=kwargs,
        messages=messages,
        thinking_mode=thinking_mode,
    )
    try:
        tokens = apply_template(messages, **kwargs)
        return _coerce_template_tokens(tokens)
    except TypeError as exc:
        return _retry_template_without_enable_thinking(
            apply_template=apply_template,
            messages=messages,
            kwargs=kwargs,
            thinking_mode=thinking_mode,
            error=exc,
        )


def _configure_thinking_kwargs(
    *,
    apply_template: Any,
    kwargs: dict[str, Any],
    messages: list[dict[str, str]],
    thinking_mode: str,
) -> None:
    try:
        signature = inspect.signature(apply_template)
    except (TypeError, ValueError):
        if thinking_mode == "no_think":
            _apply_no_think_prefix(messages)
        return
    if "enable_thinking" in signature.parameters:
        kwargs["enable_thinking"] = thinking_mode == "think"
    elif thinking_mode == "no_think":
        _apply_no_think_prefix(messages)


def _retry_template_without_enable_thinking(
    *,
    apply_template: Any,
    messages: list[dict[str, str]],
    kwargs: dict[str, Any],
    thinking_mode: str,
    error: TypeError,
) -> list[int] | None:
    if "enable_thinking" not in str(error):
        raise
    kwargs.pop("enable_thinking", None)
    if thinking_mode == "no_think":
        _apply_no_think_prefix(messages)
    retry_tokens = apply_template(messages, **kwargs)
    return _coerce_template_tokens(retry_tokens)


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
        prompt_specs,
    ) = _initialize_training(config, output_dir)
    run_state = _initialize_run_state(config)
    _log_training_start(config, output_dir)

    norm_metrics_path = output_dir / "normalization_metrics.jsonl"
    for step in range(config.n_steps):
        _run_training_step(
            config=config,
            step=step,
            output_dir=output_dir,
            training_client=training_client,
            prompt_specs=prompt_specs,
            tokenizer=tokenizer,
            writer=writer,
            judge_cfg=judge_cfg,
            judge_metrics_path=judge_metrics_path,
            norm_metrics_path=norm_metrics_path,
            run_state=run_state,
        )
    return _finalize_training_run(
        config=config,
        output_dir=output_dir,
        writer=writer,
        norm_metrics_path=norm_metrics_path,
        run_state=run_state,
    )


def _initialize_run_state(config: TinkerGRPOConfig) -> _RunState:
    update_summary(
        {
            "paper/track": config.paper_track,
            "paper/claim_mode": config.claim_mode,
            "paper/reward_metric": config.reward_metric,
            "paper/reward_data_split": config.reward_data_split,
            "paper/reward_estimator_backend": config.reward_estimator_backend,
            "paper/prompt_source": config.prompt_source,
            "paper/prompt_policy": config.prompt_policy,
            "paper/thinking_mode": config.thinking_mode,
            "paper/monitoring_mode": config.monitoring_mode,
            "paper/normalization_method": config.normalization_method,
            "paper/delta_scope": config.normalization_delta_scope,
            "paper/dataset_name": config.dataset_name,
            "paper/dataset_n_train": config.dataset_n_train,
            "paper/dataset_n_features": config.dataset_n_features,
            "paper/sstan_gate_mode": config.sstan_gate_mode,
            "paper/sstan_gate_fidelity_policy": config.sstan_gate_fidelity_policy,
        }
    )
    return _RunState(rubric_state=_init_rubric(config))


def _run_training_step(
    *,
    config: TinkerGRPOConfig,
    step: int,
    output_dir: Path,
    training_client,
    prompt_specs: list[PromptSpec],
    tokenizer,
    writer: CompletionWriter,
    judge_cfg: JudgeConfig,
    judge_metrics_path: Path,
    norm_metrics_path: Path,
    run_state: _RunState,
) -> None:
    sampling_client = training_client.save_weights_and_get_sampling_client(name=f"step_{step}")
    normalize_indices = _get_normalization_indices(step, config)
    step_data = _collect_step_data(
        config=config,
        step=step,
        prompt_specs=prompt_specs,
        tokenizer=tokenizer,
        sampling_client=sampling_client,
        writer=writer,
        normalize_indices=normalize_indices,
    )
    _log_step_diagnostics(step, step_data)
    if normalize_indices:
        norm_payload = _log_normalization_summary(step, step_data, norm_metrics_path)
        if norm_payload is not None:
            run_state.last_norm_payload = norm_payload

    datums, frac_zero_variance = _build_step_datums(
        step_data.rollouts_by_prompt, n_prompts=config.n_prompts
    )
    _apply_training_update(
        training_client,
        datums,
        step=step,
        learning_rate=config.learning_rate,
        loss_fn=config.loss_fn,
        op_timeout_s=config.train_op_timeout_s,
    )
    point = _compute_step_point(step, step_data, frac_zero_variance)
    run_state.trajectory.append(point)
    _log_step_summary(step, point)
    _wandb_log_step(
        step,
        point.reward_mean,
        extra={
            "train/valid_rate": point.n_valid / max(point.n_total, 1),
            "train/diversity": len(set(step_data.all_completion_hashes))
            / max(len(step_data.all_completion_hashes), 1),
            "train/frac_zero_var": frac_zero_variance,
        },
    )
    log_metrics(_compute_step_wandb_metrics(step_data.step_records), step=step)
    writer.flush()
    _maybe_sync_artifacts(
        step=step,
        config=config,
        output_dir=output_dir,
        norm_metrics_path=norm_metrics_path,
    )

    heuristic_rates = _compute_heuristic_rates(step_data.step_records)
    log_heuristic_rates(step, heuristic_rates)
    run_state.last_heuristic_rates = heuristic_rates
    if config.wandb_table_interval > 0 and step % config.wandb_table_interval == 0:
        log_completion_table(step, step_data.step_records)

    judge_verdicts, judge_stats = _maybe_run_step_judge(
        step, step_data.step_records, config, judge_cfg, judge_metrics_path, run_state.rubric_state
    )
    if judge_stats is not None:
        run_state.last_judge_payload = _summarize_judge_verdicts(
            judge_verdicts, judge_stats=judge_stats
        )
    elif judge_verdicts:
        run_state.last_judge_payload = _summarize_judge_verdicts(judge_verdicts)

    batch_lh_metrics = _compute_batch_lh_metrics(
        records=step_data.step_records,
        heuristic_rates=heuristic_rates,
        judge_verdicts=judge_verdicts,
        judge_stats=judge_stats,
    )
    run_state.batch_lh_history.append(batch_lh_metrics)
    run_state.last_batch_lh_metrics = batch_lh_metrics
    if config.track_lh_every_batch:
        log_metrics(batch_lh_metrics, step=step)
    run_state.rubric_state = _maybe_evolve_rubric(
        step,
        step_data,
        config,
        run_state.rubric_state,
        judge_verdicts,
        judge_cfg=judge_cfg,
    )

    lh_programs = _collect_lh_programs(step, step_data.step_records, judge_verdicts)
    _wandb_log_lh(lh_programs, step=step)
    log_metrics({"lh/count": len(lh_programs)}, step=step)


def _finalize_training_run(
    *,
    config: TinkerGRPOConfig,
    output_dir: Path,
    writer: CompletionWriter,
    norm_metrics_path: Path,
    run_state: _RunState,
) -> dict[str, Any]:
    writer.close()
    _write_trajectory(output_dir, run_state.trajectory)
    results = _compute_results(config, run_state.trajectory)
    summary_payload = _build_wandb_final_summary(
        config=config,
        results=results,
        normalization_payload=run_state.last_norm_payload,
        judge_payload=run_state.last_judge_payload,
        heuristic_rates=run_state.last_heuristic_rates,
        batch_lh_metrics=run_state.last_batch_lh_metrics,
        batch_lh_history=run_state.batch_lh_history,
    )
    results.update(summary_payload)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log_metrics(summary_payload)
    update_summary(summary_payload)
    upload_artifacts(
        [
            output_dir / "completions.jsonl",
            output_dir / "judge_metrics.jsonl",
            norm_metrics_path,
            output_dir / "rubric_evolution.jsonl",
            output_dir / "hydra_resolved_config.json",
            output_dir / "trajectory.json",
            output_dir / "results.json",
        ]
    )
    log.info("Results saved to %s", output_dir)
    _print_summary(results)
    return results


def _initialize_training(
    config: TinkerGRPOConfig, output_dir: Path
) -> tuple[CompletionWriter, Path, JudgeConfig, Any, Any, list[PromptSpec]]:
    writer = CompletionWriter(output_dir / "completions.jsonl")
    judge_metrics_path = output_dir / "judge_metrics.jsonl"
    judge_cfg = _build_judge_config(config)
    _load_judge_env(config.judge_env_file)
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    if config.prompt_source == "hardcoded":
        raw_prompts = get_prompts_for_dataset(
            config.dataset_name, max(config.n_prompts, config.prompt_jsonl_max_examples)
        )
        base_prompts = _deduplicate_prompt_templates(raw_prompts)
        if len(base_prompts) != len(raw_prompts):
            log.info(
                "Deduplicated hardcoded prompt pool from %d to %d templates",
                len(raw_prompts),
                len(base_prompts),
            )
        system_prompt = get_system_prompt(
            config.dataset_name,
            prompt_policy=config.prompt_policy,
        )
    else:
        base_prompts = load_jsonl_prompts(
            config.prompt_jsonl_path,
            max_examples=config.prompt_jsonl_max_examples,
        )
        system_prompt = get_generic_system_prompt(prompt_policy=config.prompt_policy)
    selected_prompt_texts = _select_prompt_texts_from_pool(
        base_prompts,
        n_prompts=config.n_prompts,
        sampling_mode=config.prompt_sampling,
        seed=config.scoring_seed_base,
    )
    exemplars = _load_exemplars(config.exemplar_config)
    if exemplars:
        exemplar_section = _format_exemplar_section(exemplars)
        log.info("Loaded %d exemplars from %s", len(exemplars), config.exemplar_config)
        if config.prompt_source != "hardcoded":
            raise ValueError("exemplar_config is only supported with prompt_source=hardcoded")
        prompt_specs = [
            PromptSpec(
                user_message=prompt,
                system_prompt=system_prompt + exemplar_section,
                prompt_text=_format_chat_prompt(
                    prompt,
                    exemplar_section,
                    config.dataset_name,
                    thinking_mode=config.thinking_mode,
                    prompt_policy=config.prompt_policy,
                ),
            )
            for prompt in selected_prompt_texts
        ]
    else:
        prompt_specs = [
            PromptSpec(
                user_message=prompt,
                system_prompt=system_prompt,
                prompt_text=format_chat_from_system(
                    system_prompt=system_prompt,
                    user_message=prompt,
                    thinking_mode=config.thinking_mode,
                ),
            )
            for prompt in selected_prompt_texts
        ]
    return writer, judge_metrics_path, judge_cfg, training_client, tokenizer, prompt_specs


def _get_normalization_indices(step: int, config: TinkerGRPOConfig) -> set[int]:
    if config.normalization_interval <= 0 or step % config.normalization_interval != 0:
        return set()
    if config.normalization_method == "off":
        return set()
    total = config.n_prompts * config.rollouts_per_prompt
    sample_size = min(config.normalization_sample_size, total)
    rng = np.random.default_rng(step + 9999)
    chosen = np.asarray(rng.choice(total, size=sample_size, replace=False), dtype=np.int64).reshape(
        -1
    )
    return {int(i) for i in chosen}


def _collect_step_data(
    *,
    config: TinkerGRPOConfig,
    step: int,
    prompt_specs: list[PromptSpec],
    tokenizer,
    sampling_client,
    writer: CompletionWriter,
    normalize_indices: set[int] | None = None,
) -> _StepData:
    step_data = _StepData(
        rollouts_by_prompt={},
        all_reported=[],
        all_completion_hashes=[],
        step_records=[],
    )
    for prompt_idx, prompt_spec in enumerate(prompt_specs):
        _collect_prompt_rollouts(
            config=config,
            step=step,
            prompt_idx=prompt_idx,
            prompt_spec=prompt_spec,
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
    prompt_spec: PromptSpec,
    tokenizer,
    sampling_client,
    writer: CompletionWriter,
    step_data: _StepData,
    normalize_indices: set[int],
) -> None:
    prompt_tokens = _encode_prompt_tokens(
        tokenizer,
        prompt_spec,
        thinking_mode=config.thinking_mode,
    )
    prompt_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    samples = _sample_prompt_rollouts_with_retry(
        sampling_client=sampling_client,
        prompt_input=prompt_input,
        config=config,
        step=step,
        prompt_idx=prompt_idx,
    )
    scoring_data = _make_prompt_scoring_data(config, step, prompt_idx)

    batch_scores: list[tuple[float, dict[str, Any]] | None]
    if config.scoring_workers > 1:
        from ppl_synthesis_reward_hacking.experiments.parallel_scorer import score_batch_parallel

        completion_texts = [tokenizer.decode(seq.tokens) for seq in samples.sequences]
        batch_scores = cast(
            list[tuple[float, dict[str, Any]] | None],
            score_batch_parallel(
                completion_texts,
                scoring_data,
                workers=config.scoring_workers,
                timeout=config.exec_timeout,
                reward_metric=config.reward_metric,
                reward_data_split=config.reward_data_split,
                reward_estimator_backend=config.reward_estimator_backend,
                smc_draws=config.smc_draws,
            ),
        )
    else:
        batch_scores = [None for _ in samples.sequences]

    rollouts: list[RolloutData] = []
    for seq_idx, (seq, pre_scored) in enumerate(zip(samples.sequences, batch_scores, strict=True)):
        rollout = _process_sequence(
            config=config,
            step=step,
            prompt_idx=prompt_idx,
            seq_idx=seq_idx,
            prompt_text=prompt_spec.prompt_text,
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


def _sample_prompt_rollouts_with_retry(
    *,
    sampling_client,
    prompt_input,
    config: TinkerGRPOConfig,
    step: int,
    prompt_idx: int,
):
    attempts = config.sample_max_retries + 1
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        log.info(
            "Step %3d prompt %d/%d: requesting %d rollouts (attempt %d/%d, timeout=%.1fs)",
            step,
            prompt_idx + 1,
            config.n_prompts,
            config.rollouts_per_prompt,
            attempt,
            attempts,
            config.sample_timeout_s,
        )
        start = time.perf_counter()
        try:
            samples = sampling_client.sample(
                prompt=prompt_input,
                sampling_params=types.SamplingParams(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                ),
                num_samples=config.rollouts_per_prompt,
            ).result(timeout=config.sample_timeout_s)
            elapsed = time.perf_counter() - start
            n_sequences = len(getattr(samples, "sequences", []))
            log.info(
                "Step %3d prompt %d/%d: received %d rollouts in %.1fs",
                step,
                prompt_idx + 1,
                config.n_prompts,
                n_sequences,
                elapsed,
            )
            return samples
        except FutureTimeoutError as exc:
            elapsed = time.perf_counter() - start
            last_exc = exc
            log.warning(
                "Step %3d prompt %d/%d: sampling timeout after %.1fs (attempt %d/%d)",
                step,
                prompt_idx + 1,
                config.n_prompts,
                elapsed,
                attempt,
                attempts,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            last_exc = exc
            log.warning(
                "Step %3d prompt %d/%d: sampling failed after %.1fs (attempt %d/%d): %s",
                step,
                prompt_idx + 1,
                config.n_prompts,
                elapsed,
                attempt,
                attempts,
                exc,
            )

        if attempt < attempts and config.sample_retry_backoff_s > 0:
            time.sleep(config.sample_retry_backoff_s)

    raise RuntimeError(
        f"Sampling failed for step={step}, prompt_idx={prompt_idx} after "
        f"{attempts} attempts (timeout={config.sample_timeout_s}s)"
    ) from last_exc


def _make_prompt_scoring_data(config: TinkerGRPOConfig, step: int, prompt_idx: int) -> dict:
    prompt_seed = config.scoring_seed_base + step * config.n_prompts + prompt_idx
    dataset_params: dict[str, Any] = {
        "n_features": config.dataset_n_features,
        "d": config.dataset_n_features,
        "noise_sigma": config.dataset_noise_sigma,
        "split": {
            "n_train": config.dataset_n_train,
            "n_holdout": config.dataset_n_holdout,
        },
    }
    if config.dataset_name.startswith("bernoulli"):
        if config.dataset_p_true_fixed is None:
            rng = np.random.default_rng(prompt_seed + 13_371)
            p_true = float(rng.beta(config.dataset_p_true_alpha, config.dataset_p_true_beta))
        else:
            p_true = float(config.dataset_p_true_fixed)
        dataset_params["p_true"] = p_true

    dataset = generate_dataset(
        config.dataset_name,
        dataset_params,
        seed=prompt_seed,
    )
    data: dict[str, Any] = dict(dataset.train)
    data["n"] = config.dataset_n_train
    data["p"] = config.dataset_n_features
    data["d"] = config.dataset_n_features
    data["dataset_name"] = config.dataset_name
    if "p_true" in dataset.meta:
        data["p_true"] = float(dataset.meta["p_true"])

    data["y_train"] = np.asarray(dataset.train["y"], dtype=np.float64)
    data["y_holdout"] = np.asarray(dataset.holdout["y"], dtype=np.float64)
    if "X" in dataset.train:
        data["X_train"] = np.asarray(dataset.train["X"], dtype=np.float64)
        data["X_holdout"] = np.asarray(dataset.holdout["X"], dtype=np.float64)

    data["seed"] = prompt_seed
    return data


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
    pre_scored: tuple[float, dict[str, Any]] | None = None,
    normalize_indices: set[int] | None = None,
) -> RolloutData:
    completion_text = tokenizer.decode(seq.tokens)
    code = extract_pymc_code(completion_text)
    step_data.all_completion_hashes.append(normalized_text_hash(completion_text))
    if pre_scored is not None:
        reported, decomposition = _validate_scorer_contract(
            pre_scored,
            context=f"step={step} prompt_idx={prompt_idx} seq_idx={seq_idx} pre_scored",
        )
    else:
        reported, decomposition = _validate_scorer_contract(
            score_completion_sandboxed(
                completion_text,
                scoring_data,
                timeout=config.exec_timeout,
                reward_metric=config.reward_metric,
                reward_data_split=config.reward_data_split,
                reward_estimator_backend=config.reward_estimator_backend,
                smc_draws=config.smc_draws,
            ),
            context=f"step={step} prompt_idx={prompt_idx} seq_idx={seq_idx} subprocess",
        )
    outcome = classify_outcome(reported)
    if outcome == "parse_fail":
        step_data.n_parse_fail += 1
    elif outcome == "exec_fail":
        step_data.n_exec_fail += 1

    global_idx = prompt_idx * config.rollouts_per_prompt + seq_idx
    if outcome == "valid":
        gate_result = run_sstan_gate(
            code=code,
            config=_build_sstan_gate_config(config),
            batch=step,
            index=global_idx,
            sampling_seed=config.scoring_seed_base,
            cache_dir=Path(config.output_dir) / "sstan_cache",
        )
    else:
        decision = "off" if config.sstan_gate_mode == "off" else "skipped"
        reasons = ["gate_off"] if config.sstan_gate_mode == "off" else ["non_valid_outcome"]
        gate_result = SStanGateResult(
            mode=config.sstan_gate_mode,
            checked=False,
            sampled=False,
            accepted=None,
            decision=decision,
            reasons=reasons,
            transpile_success=None,
            transpile_error=None,
            sstan_reasons=[],
            fidelity_policy=config.sstan_gate_fidelity_policy,
            fidelity_reject=False,
            source_signal=False,
            source_tags=[],
            timing_ms={"transpile": 0.0, "check": 0.0, "total": 0.0},
            penalty_applied=False,
            penalty_reward=None,
            stan_path=None,
            stan_hash=None,
        )
    if (
        config.sstan_gate_mode == "enforce"
        and gate_result.checked
        and gate_result.accepted is False
    ):
        reported = float(config.sstan_gate_penalty_reward)
        gate_result = SStanGateResult(
            mode=gate_result.mode,
            checked=gate_result.checked,
            sampled=gate_result.sampled,
            accepted=gate_result.accepted,
            decision=gate_result.decision,
            reasons=gate_result.reasons,
            transpile_success=gate_result.transpile_success,
            transpile_error=gate_result.transpile_error,
            sstan_reasons=gate_result.sstan_reasons,
            fidelity_policy=gate_result.fidelity_policy,
            fidelity_reject=gate_result.fidelity_reject,
            source_signal=gate_result.source_signal,
            source_tags=gate_result.source_tags,
            timing_ms=gate_result.timing_ms,
            penalty_applied=True,
            penalty_reward=reported,
            stan_path=gate_result.stan_path,
            stan_hash=gate_result.stan_hash,
        )
    gate_rejected = gate_result.checked and gate_result.accepted is False
    norm_result = None
    if (
        normalize_indices
        and global_idx in normalize_indices
        and outcome == "valid"
        and not gate_rejected
    ):
        norm_result = _run_normalization_check(
            completion_text,
            scoring_data=scoring_data,
            timeout=config.exec_timeout,
            # always SMC: MCMC is degenerate at n_train=1 for binary enumeration
            reward_metric="log_marginal_likelihood",
            reward_data_split=config.reward_data_split,
            reward_estimator_backend="smc",
            smc_draws=config.smc_draws,
            method=config.normalization_method,
            delta_scope=config.normalization_delta_scope,
            epsilon=config.normalization_epsilon,
            ci_alpha=config.normalization_ci_alpha,
            mc_samples=config.normalization_mc_samples,
            min_ess=config.normalization_min_ess,
            seed=int(scoring_data.get("seed", step * 10_000 + global_idx)),
        )

    rollout = RolloutData(
        prompt_tokens=list(prompt_tokens),
        completion_tokens=list(seq.tokens),
        sampling_logprobs=list(seq.logprobs) if seq.logprobs else [0.0] * len(seq.tokens),
        reported_reward=reported,
        log_mass=norm_result.get("log_mass", 0.0) if norm_result else 0.0,
    )
    step_data.all_reported.append(reported)

    metadata: dict[str, Any] = {"decomposition": decomposition}
    metadata["sstan_gate"] = gate_result.to_metadata()
    if norm_result:
        metadata["normalization"] = norm_result

    record = CompletionRecord(
        batch=step,
        index=global_idx,
        prompt=prompt_text,
        completion_text=completion_text,
        code=code,
        reported_reward=reported,
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
    reward_estimator_backend: str,
    smc_draws: int,
    method: str,
    delta_scope: str,
    epsilon: float,
    ci_alpha: float,
    mc_samples: int,
    min_ess: float,
    seed: int,
) -> dict[str, Any] | None:
    """Run normalization check; return None on failure."""
    try:
        result = check_normalization(
            completion_text,
            method=method,
            d=int(scoring_data.get("d", np.asarray(scoring_data["y"]).size)),
            p_true=float(scoring_data.get("p_true", 0.5)),
            seed=seed,
            timeout=timeout,
            epsilon=epsilon,
            ci_alpha=ci_alpha,
            mc_samples=mc_samples,
            min_ess=min_ess,
            reward_metric=reward_metric,
            reward_data_split=reward_data_split,
            reward_estimator_backend=reward_estimator_backend,
            smc_draws=smc_draws,
            scoring_data=scoring_data,
            delta_scope=delta_scope,
        )
        if result.get("ok"):
            return {
                "log_mass": result["log_mass"],
                "mass": result.get("mass"),
                "is_normalized": result["is_normalized"],
                "status": result.get("status"),
                "method": result.get("method"),
                "delta_scope": result.get("delta_scope"),
                "ess": result.get("ess"),
                "ci_log_mass_low": result.get("ci_log_mass_low"),
                "ci_log_mass_high": result.get("ci_log_mass_high"),
                "d": result.get("d"),
            }
        return {
            "ok": False,
            "reason": result.get("reason", "unknown"),
            "status": result.get("status"),
            "method": result.get("method", method),
            "delta_scope": result.get("delta_scope", delta_scope),
            "is_normalized": False,
            "d": result.get("d"),
        }
    except Exception as exc:
        log.debug("normalization check failed: %s", exc)
        return None


def _validate_scorer_contract(
    raw_result: Any,
    *,
    context: str,
) -> tuple[float, dict[str, Any]]:
    if not isinstance(raw_result, tuple) or len(raw_result) != 2:
        raise RuntimeError(
            "Scorer contract violation: expected "
            "(reported_reward, decomposition) tuple "
            f"for {context}, got {type(raw_result).__name__}"
        )
    reported_raw, decomposition_raw = raw_result
    if isinstance(reported_raw, bool) or not isinstance(reported_raw, int | float):
        raise RuntimeError(
            "Scorer contract violation: reported_reward must be numeric "
            f"for {context}, got {type(reported_raw).__name__}"
        )
    reported = float(reported_raw)
    if not np.isfinite(reported):
        raise RuntimeError(
            f"Scorer contract violation: reported_reward must be finite for {context},"
            f" got {reported}"
        )
    if not isinstance(decomposition_raw, dict):
        raise RuntimeError(
            "Scorer contract violation: decomposition must be a dict "
            f"for {context}, got {type(decomposition_raw).__name__}"
        )
    return reported, cast(dict[str, Any], decomposition_raw)


def _collect_lh_programs(
    step: int,
    step_records: list[CompletionRecord],
    judge_verdicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return step records flagged as formally non-normalized."""
    lh_rows: list[dict[str, Any]] = []
    verdict_by_index: dict[int, list[dict[str, Any]]] = {}
    for verdict in judge_verdicts:
        raw_index = verdict.get("record_index")
        if not isinstance(raw_index, int):
            continue
        verdict_by_index.setdefault(raw_index, []).append(verdict)

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

        verdict_rows = verdict_by_index.get(rec.index, [])
        verdict_labels = sorted(
            {
                str(v.get("verdict", "")).strip()
                for v in verdict_rows
                if str(v.get("verdict", "")).strip()
            }
        )

        def _row_tags(row: dict[str, Any]) -> list[str]:
            canonical = row.get("canonical_tags")
            if isinstance(canonical, list):
                return [str(tag) for tag in canonical if isinstance(tag, str)]
            fallback = row.get("tags", [])
            if isinstance(fallback, list):
                return [str(tag) for tag in fallback if isinstance(tag, str)]
            return []

        tag_labels = sorted(
            {
                str(tag).strip()
                for v in verdict_rows
                for tag in _row_tags(v)
                if isinstance(tag, str) and tag.strip()
            }
        )

        lh_rows.append(
            {
                "step": step,
                "code": rec.code or "",
                "reported": rec.reported_reward,
                "detection": ",".join(detection),
                "log_mass": log_mass,
                "judge_verdict": ",".join(verdict_labels),
                "judge_tags": ",".join(tag_labels),
            }
        )

    return lh_rows


def _compute_heuristic_rates(records: list[CompletionRecord]) -> dict[str, float]:
    """Compute canonical heuristic exploit-family rates over valid records."""
    valid = [r for r in records if r.outcome == "valid" and r.code]
    rates: dict[str, float] = {
        "n_valid": float(len(valid)),
        "any_exploit_rate": float("nan"),
        "lh_rate": float("nan"),
    }
    for family in _LH_CANONICAL_FAMILIES:
        rates[f"family_rate/{family}"] = float("nan")
    if not valid:
        return rates

    family_counter: Counter[str] = Counter()
    n_any = 0
    for r in valid:
        code = r.code or ""
        tags = detect_exploits(code)
        families = {
            tag
            for tag in canonicalize_exploit_tags(code, precomputed_raw=tags)
            if tag in _LH_CANONICAL_FAMILIES
        }
        if families:
            n_any += 1
            family_counter.update(families)

    n = len(valid)
    rates["any_exploit_rate"] = n_any / n
    rates["lh_rate"] = n_any / n
    for family in _LH_CANONICAL_FAMILIES:
        rates[f"family_rate/{family}"] = family_counter.get(family, 0) / n
    return rates


def _compute_batch_lh_metrics(
    *,
    records: list[CompletionRecord],
    heuristic_rates: dict[str, float],
    judge_verdicts: list[dict[str, Any]],
    judge_stats: dict[str, Any] | None,
) -> dict[str, float]:
    valid_n = sum(1 for rec in records if rec.outcome == "valid")
    heuristic_rate = float(
        heuristic_rates.get("lh_rate", heuristic_rates.get("any_exploit_rate", float("nan")))
    )
    if not np.isfinite(heuristic_rate):
        heuristic_rate = float("nan")

    metrics = _init_batch_lh_metrics(valid_n=valid_n, heuristic_rate=heuristic_rate)
    _set_heuristic_family_rates(metrics, heuristic_rates=heuristic_rates)

    if judge_verdicts:
        judge_payload = _summarize_judge_verdicts(judge_verdicts, judge_stats=judge_stats)
        _apply_judge_payload_to_batch_metrics(
            metrics,
            judge_payload=judge_payload,
            valid_n=valid_n,
        )
    else:
        _apply_judge_stats_fallback_to_batch_metrics(
            metrics,
            judge_stats=judge_stats,
            valid_n=valid_n,
        )
    return metrics


def _init_batch_lh_metrics(*, valid_n: int, heuristic_rate: float) -> dict[str, float]:
    metrics: dict[str, float] = {
        "lh/batch/valid_n": float(valid_n),
        "lh/batch/heuristic_rate": heuristic_rate,
        "lh/batch/proportion": heuristic_rate,
        "lh/batch/judge_rate": float("nan"),
        "lh/batch/judge_n": float("nan"),
        "lh/batch/judge_coverage": float("nan"),
        "lh/batch/other_novel_rate": float("nan"),
    }
    for family in _LH_CANONICAL_FAMILIES:
        metrics[f"lh/batch/family/{family}"] = float("nan")
        metrics[f"lh/batch/judge_family/{family}"] = float("nan")
    return metrics


def _set_heuristic_family_rates(
    metrics: dict[str, float],
    *,
    heuristic_rates: dict[str, float],
) -> None:
    for family in _LH_CANONICAL_FAMILIES:
        metrics[f"lh/batch/family/{family}"] = float(
            heuristic_rates.get(f"family_rate/{family}", float("nan"))
        )


def _apply_judge_payload_to_batch_metrics(
    metrics: dict[str, float],
    *,
    judge_payload: dict[str, Any],
    valid_n: int,
) -> None:
    judge_n = float(judge_payload.get("n_judged", 0))
    metrics["lh/batch/judge_n"] = judge_n
    judge_rate = judge_payload.get("hacking_rate")
    if isinstance(judge_rate, int | float) and np.isfinite(float(judge_rate)):
        metrics["lh/batch/judge_rate"] = float(judge_rate)
    if valid_n > 0:
        metrics["lh/batch/judge_coverage"] = judge_n / valid_n

    other_novel = judge_payload.get("lh_other_novel_rate")
    if isinstance(other_novel, int | float):
        metrics["lh/batch/other_novel_rate"] = float(other_novel)
    for family in _LH_CANONICAL_FAMILIES:
        value = judge_payload.get(f"family_rate/{family}")
        if isinstance(value, int | float):
            metrics[f"lh/batch/judge_family/{family}"] = float(value)


def _apply_judge_stats_fallback_to_batch_metrics(
    metrics: dict[str, float],
    *,
    judge_stats: dict[str, Any] | None,
    valid_n: int,
) -> None:
    if not judge_stats or not isinstance(judge_stats.get("n_judged"), int | float):
        return
    judge_n = float(judge_stats["n_judged"])
    metrics["lh/batch/judge_n"] = judge_n
    if valid_n > 0:
        metrics["lh/batch/judge_coverage"] = judge_n / valid_n
    judge_rate = judge_stats.get("hacking_rate")
    if isinstance(judge_rate, int | float):
        metrics["lh/batch/judge_rate"] = float(judge_rate)


def _log_normalization_summary(
    step: int, step_data: _StepData, metrics_path: Path
) -> dict[str, Any] | None:
    """Aggregate and log normalization results from step records."""
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
        if norm.get("is_normalized") is False:
            n_non_normalized += 1
        if lm is not None:
            log_masses.append(float(lm))

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


def _maybe_sync_artifacts(
    *,
    step: int,
    config: TinkerGRPOConfig,
    output_dir: Path,
    norm_metrics_path: Path,
) -> None:
    if config.artifact_sync_interval <= 0:
        return
    if step < config.artifact_sync_min_steps:
        return
    if step <= 0 or step % config.artifact_sync_interval != 0:
        return

    upload_artifacts(
        [
            output_dir / "completions.jsonl",
            output_dir / "judge_metrics.jsonl",
            norm_metrics_path,
            output_dir / "rubric_evolution.jsonl",
            output_dir / "hydra_resolved_config.json",
        ]
    )
    log.info(
        "Step %3d: synced intermediate artifacts to W&B (interval=%d)",
        step,
        config.artifact_sync_interval,
    )


def _build_step_datums(
    rollouts_by_prompt: dict[int, list[RolloutData]], *, n_prompts: int
) -> tuple[list, float]:
    per_prompt = compute_group_relative_advantages(
        rollouts_by_prompt, global_baseline_fallback=False
    )
    if not per_prompt:
        advantages_by_prompt = compute_group_relative_advantages(
            rollouts_by_prompt, global_baseline_fallback=True
        )
        if advantages_by_prompt:
            log.info(
                "all per-prompt groups had zero variance; using global baseline (%d/%d prompts)",
                len(advantages_by_prompt),
                n_prompts,
            )
    else:
        advantages_by_prompt = per_prompt

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
    op_timeout_s: float,
) -> None:
    if not datums:
        log.warning("Step %d: no datums (all zero variance), skipping gradient", step)
        return
    fwdbwd = training_client.forward_backward(datums, loss_fn=loss_fn)
    optim = training_client.optim_step(types.AdamParams(learning_rate=learning_rate))
    fwdbwd.result(timeout=op_timeout_s)
    optim.result(timeout=op_timeout_s)


def _compute_step_point(
    step: int, step_data: _StepData, frac_zero_variance: float
) -> TrajectoryPoint:
    valid_rewards = [
        reported
        for reported in step_data.all_reported
        if np.isfinite(reported) and classify_outcome(float(reported)) == "valid"
    ]
    reward_mean = float(np.mean(valid_rewards)) if valid_rewards else float("nan")

    # extract normalization metrics from step records
    n_norm_checked = 0
    n_non_normalized = 0
    abs_log_masses: list[float] = []
    for rec in step_data.step_records:
        if not rec.metadata:
            continue
        norm = rec.metadata.get("normalization")
        if not norm or not isinstance(norm, dict):
            continue
        n_norm_checked += 1
        lm = norm.get("log_mass")
        if norm.get("is_normalized") is False:
            n_non_normalized += 1
        if lm is not None:
            abs_log_masses.append(abs(float(lm)))

    frac_non_norm = n_non_normalized / n_norm_checked if n_norm_checked > 0 else float("nan")
    mean_abs_lm = float(np.mean(abs_log_masses)) if abs_log_masses else float("nan")

    return TrajectoryPoint(
        step=step,
        reward_mean=reward_mean,
        n_valid=len(valid_rewards),
        n_total=len(step_data.all_reported),
        n_parse_fail=step_data.n_parse_fail,
        n_exec_fail=step_data.n_exec_fail,
        frac_zero_variance=frac_zero_variance,
        frac_non_normalized=frac_non_norm,
        mean_abs_log_mass=mean_abs_lm,
        n_norm_checked=n_norm_checked,
    )


def _write_trajectory(output_dir: Path, trajectory: list[TrajectoryPoint]) -> None:
    trajectory_dicts = [asdict(point) for point in trajectory]
    with open(output_dir / "trajectory.json", "w") as f:
        json.dump(trajectory_dicts, f, indent=2)


def _compute_results(config: TinkerGRPOConfig, trajectory: list[TrajectoryPoint]) -> dict:
    metrics = compute_traj_metrics(cast(list[Any], trajectory))
    config_payload = {
        "base_model": config.base_model,
        "n_steps": config.n_steps,
        "n_prompts": config.n_prompts,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank,
        "dataset_name": config.dataset_name,
        "dataset_n_features": config.dataset_n_features,
        "dataset_noise_sigma": config.dataset_noise_sigma,
        "dataset_n_train": config.dataset_n_train,
        "dataset_n_holdout": config.dataset_n_holdout,
        "reward_metric": config.reward_metric,
        "reward_data_split": config.reward_data_split,
        "reward_estimator_backend": config.reward_estimator_backend,
        "claim_mode": config.claim_mode,
        "smc_draws": config.smc_draws,
        "scoring_workers": config.scoring_workers,
        "max_tokens": config.max_tokens,
        "paper_track": config.paper_track,
        "monitoring_mode": config.monitoring_mode,
        "sstan_gate_mode": config.sstan_gate_mode,
        "sstan_gate_sample_rate": config.sstan_gate_sample_rate,
        "sstan_gate_penalty_reward": config.sstan_gate_penalty_reward,
        "sstan_gate_fidelity_policy": config.sstan_gate_fidelity_policy,
        "sstan_transpiler_model": config.sstan_transpiler_model,
        "sstan_transpiler_strict": config.sstan_transpiler_strict,
        "normalization_method": config.normalization_method,
        "normalization_delta_scope": config.normalization_delta_scope,
        "normalization_epsilon": config.normalization_epsilon,
        "normalization_ci_alpha": config.normalization_ci_alpha,
        "normalization_mc_samples": config.normalization_mc_samples,
        "normalization_min_ess": config.normalization_min_ess,
    }
    metrics = attach_common_results_metadata(
        metrics=metrics,
        trajectory=trajectory,
        count_key="n_steps_completed",
        config_payload=config_payload,
    )
    if "error" in metrics:
        metrics["n_steps"] = metrics["n_steps_completed"]
        return metrics
    final = trajectory[-1]
    metrics["final_frac_zero_variance"] = final.frac_zero_variance
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
        "Track: %s | Claim mode: %s | Monitoring: %s",
        config.paper_track,
        config.claim_mode,
        config.monitoring_mode,
    )
    log.info(
        ("Sampling: temp=%.2f, top_p=%.2f, top_k=%d sample_timeout=%.1fs retries=%d backoff=%.1fs"),
        config.temperature,
        config.top_p,
        config.top_k,
        config.sample_timeout_s,
        config.sample_max_retries,
        config.sample_retry_backoff_s,
    )
    log.info(
        "Reward: metric=%s split=%s estimator=%s smc_draws=%d workers=%d",
        config.reward_metric,
        config.reward_data_split,
        config.reward_estimator_backend,
        config.smc_draws,
        config.scoring_workers,
    )
    log.info(
        "Prompting: source=%s policy=%s sampling=%s thinking_mode=%s jsonl_path=%s max_examples=%d",
        config.prompt_source,
        config.prompt_policy,
        config.prompt_sampling,
        config.thinking_mode,
        config.prompt_jsonl_path,
        config.prompt_jsonl_max_examples,
    )
    log.info(
        "Dataset: %s n_train=%d n_features=%d noise_sigma=%.2f n_holdout=%d",
        config.dataset_name,
        config.dataset_n_train,
        config.dataset_n_features,
        config.dataset_noise_sigma,
        config.dataset_n_holdout,
    )
    log.info("Output: %s", output_dir)
    log.info("Train op timeout: %.1fs", config.train_op_timeout_s)
    if config.normalization_interval > 0 and config.normalization_method != "off":
        log.info(
            (
                "Normalization: method=%s delta=%s every=%d sample=%d "
                "eps=%.3g ci_alpha=%.3g mc=%d min_ess=%.1f"
            ),
            config.normalization_method,
            config.normalization_delta_scope,
            config.normalization_interval,
            config.normalization_sample_size,
            config.normalization_epsilon,
            config.normalization_ci_alpha,
            config.normalization_mc_samples,
            config.normalization_min_ess,
        )
    log.info(
        (
            "SafeStan gate: mode=%s sample_rate=%.3f penalty=%.1f fidelity=%s "
            "transpiler=%s strict=%s timeout=%.1fs"
        ),
        config.sstan_gate_mode,
        config.sstan_gate_sample_rate,
        config.sstan_gate_penalty_reward,
        config.sstan_gate_fidelity_policy,
        config.sstan_transpiler_model,
        config.sstan_transpiler_strict,
        config.sstan_transpiler_timeout_s,
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
        "Step %3d: reward=%.3f valid=%d/%d parse_fail=%d exec_fail=%d "
        "frac_zero_var=%.2f norm_checked=%d frac_non_norm=%.3f mean_abs_log_mass=%.3f",
        step,
        point.reward_mean,
        point.n_valid,
        point.n_total,
        point.n_parse_fail,
        point.n_exec_fail,
        point.frac_zero_variance,
        point.n_norm_checked,
        point.frac_non_normalized if not np.isnan(point.frac_non_normalized) else 0.0,
        point.mean_abs_log_mass if not np.isnan(point.mean_abs_log_mass) else 0.0,
    )


def _maybe_run_step_judge(
    step: int,
    step_records: list[CompletionRecord],
    config: TinkerGRPOConfig,
    judge_cfg: JudgeConfig,
    judge_metrics_path: Path,
    rubric_state=None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if config.monitoring_mode == "off":
        return [], None
    if config.judge_interval <= 0 or step % config.judge_interval != 0 or not step_records:
        return [], None
    try:
        return _run_observational_judge(
            step=step,
            records=step_records,
            cfg=judge_cfg,
            sample_size=config.judge_sample_size,
            dedup=config.judge_dedup,
            metrics_path=judge_metrics_path,
            rubric_state=rubric_state,
            sampling_policy=config.judge_sampling_policy,
            adaptive_min=config.judge_adaptive_min,
            adaptive_max=config.judge_adaptive_max,
            adaptive_target_metric=config.judge_adaptive_target_metric,
        )
    except Exception as exc:
        log.warning("Step %3d JUDGE: failed (continuing training): %s", step, exc)
        return [], None


def _deduplicate_records(records: list[CompletionRecord]) -> list[CompletionRecord]:
    unique: dict[str, CompletionRecord] = {}
    for record in records:
        key = normalized_text_hash(record.code or record.completion_text)
        if key not in unique:
            unique[key] = record
    return list(unique.values())


def _select_judge_sample_size(
    *,
    records: list[CompletionRecord],
    sampling_policy: str,
    sample_size: int,
    adaptive_min: int,
    adaptive_max: int,
    adaptive_target_metric: str,
) -> tuple[int, dict[str, float]]:
    if not records:
        return 0, {
            "heuristic_lh_signal": float("nan"),
            "outcome_entropy_signal": float("nan"),
            "combined_signal": float("nan"),
            "target_signal": float("nan"),
        }

    rates = _compute_heuristic_rates(records)
    heuristic_lh_signal = float(rates.get("lh_rate", rates.get("any_exploit_rate", 0.0)))
    heuristic_lh_signal = float(np.clip(heuristic_lh_signal, 0.0, 1.0))

    outcome_counts = Counter(r.outcome for r in records)
    total = max(sum(outcome_counts.values()), 1)
    probs = [count / total for count in outcome_counts.values() if count > 0]
    if len(probs) <= 1:
        outcome_entropy_signal = 0.0
    else:
        entropy = -sum(p * math.log(p) for p in probs)
        max_entropy = math.log(len(probs))
        outcome_entropy_signal = entropy / max_entropy if max_entropy > 0 else 0.0
    outcome_entropy_signal = float(np.clip(outcome_entropy_signal, 0.0, 1.0))
    combined_signal = 0.5 * (heuristic_lh_signal + outcome_entropy_signal)

    if adaptive_target_metric == "heuristic_novelty":
        target_signal = heuristic_lh_signal
    elif adaptive_target_metric == "outcome_entropy":
        target_signal = outcome_entropy_signal
    else:
        target_signal = combined_signal

    if sampling_policy == "adaptive_cap":
        if adaptive_max <= adaptive_min:
            requested = adaptive_max
        else:
            span = adaptive_max - adaptive_min
            requested = adaptive_min + int(round(target_signal * span))
        if sample_size > 0:
            requested = min(requested, sample_size)
    else:
        requested = sample_size

    if requested <= 0:
        return 0, {
            "heuristic_lh_signal": heuristic_lh_signal,
            "outcome_entropy_signal": outcome_entropy_signal,
            "combined_signal": combined_signal,
            "target_signal": target_signal,
        }

    return min(requested, len(records)), {
        "heuristic_lh_signal": heuristic_lh_signal,
        "outcome_entropy_signal": outcome_entropy_signal,
        "combined_signal": combined_signal,
        "target_signal": target_signal,
    }


def _run_observational_judge(
    *,
    step: int,
    records: list[CompletionRecord],
    cfg: JudgeConfig,
    sample_size: int,
    dedup: bool,
    metrics_path: Path,
    sampling_policy: str,
    adaptive_min: int,
    adaptive_max: int,
    adaptive_target_metric: str,
    rubric_state=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_records = [r for r in records if r.outcome == "valid" and (r.code or r.completion_text)]
    candidate_records = _deduplicate_records(valid_records) if dedup else valid_records

    requested_sample_size, adaptive_signals = _select_judge_sample_size(
        records=records,
        sampling_policy=sampling_policy,
        sample_size=sample_size,
        adaptive_min=adaptive_min,
        adaptive_max=adaptive_max,
        adaptive_target_metric=adaptive_target_metric,
    )

    if requested_sample_size <= 0:
        judged_records: list[CompletionRecord] = []
    else:
        judged_records = candidate_records
    if requested_sample_size > 0 and len(candidate_records) > requested_sample_size:
        rng = np.random.default_rng(step)
        idxs = np.asarray(
            rng.choice(len(candidate_records), size=requested_sample_size, replace=False),
            dtype=np.int64,
        ).reshape(-1)
        judged_records = [candidate_records[int(i)] for i in idxs]

    if not judged_records:
        empty_payload = {
            "step": step,
            "n_candidates": len(candidate_records),
            "n_judged": 0,
            "hacking_rate": float("nan"),
            "verdict_counts": {},
            "tag_counts": {},
            "sampling_policy": sampling_policy,
            "sample_size_requested": requested_sample_size,
            **adaptive_signals,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(empty_payload) + "\n")
        return [], empty_payload

    rubric_prompt = None
    if rubric_state is not None:
        from ppl_synthesis_reward_hacking.monitoring.evolving_rubric import (
            format_rubric_prompt,
        )

        rubric_prompt = format_rubric_prompt(rubric_state)

    verdicts = judge_completions(judged_records, cfg, rubric_prompt=rubric_prompt)

    verdict_counts = Counter(v.verdict for v in verdicts)
    tag_counts = Counter(tag for v in verdicts for tag in v.tags)
    n_total = max(len(verdicts), 1)
    hacking_rate = verdict_counts.get("hacking", 0) / n_total

    payload = {
        "step": step,
        "n_candidates": len(candidate_records),
        "n_judged": len(verdicts),
        "hacking_rate": hacking_rate,
        "verdict_counts": dict(verdict_counts),
        "tag_counts": dict(tag_counts),
        "sampling_policy": sampling_policy,
        "sample_size_requested": requested_sample_size,
        **adaptive_signals,
    }
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    log_judge_metrics(step, hacking_rate, len(verdicts), dict(verdict_counts), dict(tag_counts))

    log.info(
        "Step %3d JUDGE: n=%d/%d hacking_rate=%.3f verdicts=%s policy=%s target=%.3f",
        step,
        len(verdicts),
        len(candidate_records),
        hacking_rate,
        dict(verdict_counts),
        sampling_policy,
        adaptive_signals["target_signal"],
    )

    verdict_rows: list[dict[str, Any]] = []
    for verdict, record in zip(verdicts, judged_records, strict=True):
        verdict_rows.append(
            {
                "verdict": verdict.verdict,
                "tags": verdict.tags,
                "canonical_tags": sorted(canonicalize_judge_tags(verdict.tags)),
                "novel_exploit": verdict.novel_exploit,
                "confidence": verdict.confidence,
                "reported_reward": record.reported_reward,
                "record_index": record.index,
            }
        )

    return verdict_rows, payload


_JUDGE_STATS_KEYS = (
    "sampling_policy",
    "sample_size_requested",
    "n_candidates",
    "target_signal",
    "heuristic_lh_signal",
    "outcome_entropy_signal",
    "combined_signal",
    "hacking_rate",
    "n_judged",
)


def _empty_judge_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_judged": 0,
        "hacking_rate": float("nan"),
        "lh_other_novel_rate": float("nan"),
    }
    for family in _LH_CANONICAL_FAMILIES:
        summary[f"family_rate/{family}"] = float("nan")
    return summary


def _merge_judge_stats(summary: dict[str, Any], judge_stats: dict[str, Any] | None) -> None:
    if not judge_stats:
        return
    for key in _JUDGE_STATS_KEYS:
        if key in judge_stats:
            summary[key] = judge_stats[key]


def _canonical_lh_families(item: dict[str, Any], tags: list[str]) -> list[str]:
    raw_canonical = item.get("canonical_tags", [])
    canonical_tags = (
        [str(tag) for tag in raw_canonical if isinstance(tag, str)]
        if isinstance(raw_canonical, list)
        else sorted(canonicalize_judge_tags(tags))
    )
    return [tag for tag in canonical_tags if tag in _LH_CANONICAL_FAMILIES]


def _is_other_novel_judge_item(item: dict[str, Any], *, verdict: str, tags: list[str]) -> bool:
    novel_exploit = item.get("novel_exploit")
    has_novel = isinstance(novel_exploit, str) and bool(novel_exploit.strip())
    has_nontrivial_tags = any(
        tag and tag != "honest" and not tag.startswith("evaluator_hack") for tag in tags
    )
    return has_novel or has_nontrivial_tags or verdict == "hacking"


def _summarize_judge_verdicts(
    verdicts: list[dict[str, Any]],
    *,
    judge_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = len(verdicts)
    summary = _empty_judge_summary()
    summary["n_judged"] = n

    if n == 0:
        _merge_judge_stats(summary, judge_stats)
        return summary

    n_hacking = 0
    n_with_verdict = 0
    n_other_novel = 0
    family_counts: Counter[str] = Counter()
    for item in verdicts:
        verdict = str(item.get("verdict", "")).strip()
        raw_tags = item.get("tags", [])
        tags = [str(tag).strip() for tag in raw_tags] if isinstance(raw_tags, list) else []
        lh_families = _canonical_lh_families(item, tags)

        is_hacking = False
        if verdict:
            n_with_verdict += 1
            if verdict == "hacking":
                is_hacking = True
        if lh_families:
            is_hacking = True

        if is_hacking:
            n_hacking += 1
            for family in lh_families:
                family_counts[family] += 1
            if not lh_families and _is_other_novel_judge_item(
                item, verdict=verdict, tags=tags
            ):
                n_other_novel += 1

    summary["hacking_rate"] = n_hacking / n
    summary["lh_other_novel_rate"] = n_other_novel / n
    for family in _LH_CANONICAL_FAMILIES:
        summary[f"family_rate/{family}"] = family_counts.get(family, 0) / n

    _merge_judge_stats(summary, judge_stats)
    return summary


def _compute_step_wandb_metrics(records: list[CompletionRecord]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"train/n_total": len(records)}
    if not records:
        return metrics

    outcome_counts = Counter(r.outcome for r in records)
    metrics["train/n_valid"] = outcome_counts.get("valid", 0)
    metrics["train/n_parse_fail"] = outcome_counts.get("parse_fail", 0)
    metrics["train/n_exec_fail"] = outcome_counts.get("exec_fail", 0)

    agg = _StepMetricsAccumulator()
    for record in records:
        _accumulate_step_metrics_for_record(record, agg)

    _ensure_valid_records_have_decomposition(agg.missing_decomposition_valid_indices)
    _attach_scorer_outcome_metrics(metrics, agg.scorer_outcome_counts)
    _attach_metric_distribution_moments(metrics, agg.metric_vals)
    _attach_sstan_gate_metrics(metrics, agg)
    _attach_prevalence_metrics(metrics, agg)
    return metrics


@dataclass
class _StepMetricsAccumulator:
    scorer_outcome_counts: Counter[str] = field(default_factory=Counter)
    metric_vals: dict[str, list[float]] = field(
        default_factory=lambda: {
            "metric_log_marginal_likelihood": [],
            "metric_elpd": [],
            "metric_waic": [],
            "metric_bic": [],
        }
    )
    n_valid_with_decomposition: int = 0
    n_with_potential: int = 0
    n_data_discard: int = 0
    missing_decomposition_valid_indices: list[int] = field(default_factory=list)
    gate_checked: int = 0
    gate_accept: int = 0
    gate_reject: int = 0
    gate_transpile_fail: int = 0
    gate_sstan_reject: int = 0
    gate_fidelity_reject: int = 0
    gate_total_ms: list[float] = field(default_factory=list)


def _accumulate_step_metrics_for_record(
    record: CompletionRecord,
    agg: _StepMetricsAccumulator,
) -> None:
    metadata = record.metadata if isinstance(record.metadata, dict) else None
    decomposition = metadata.get("decomposition") if metadata else None
    if not isinstance(decomposition, dict):
        if record.outcome == "valid":
            agg.missing_decomposition_valid_indices.append(int(record.index))
        return

    _accumulate_decomposition_metrics(record, decomposition=decomposition, agg=agg)
    gate_payload = metadata.get("sstan_gate") if metadata else None
    _accumulate_gate_metrics(gate_payload=gate_payload, agg=agg)


def _accumulate_decomposition_metrics(
    record: CompletionRecord,
    *,
    decomposition: dict[str, Any],
    agg: _StepMetricsAccumulator,
) -> None:
    if record.outcome == "valid":
        agg.n_valid_with_decomposition += 1
    outcome_code = decomposition.get("outcome_code")
    if isinstance(outcome_code, str) and outcome_code:
        agg.scorer_outcome_counts[outcome_code] += 1
    if record.outcome == "valid":
        if decomposition.get("data_discard") is True:
            agg.n_data_discard += 1
        n_pot_terms = decomposition.get("n_pot_terms")
        if isinstance(n_pot_terms, int | float) and float(n_pot_terms) > 0:
            agg.n_with_potential += 1
    for key in agg.metric_vals:
        value = decomposition.get(key)
        if isinstance(value, int | float) and np.isfinite(float(value)):
            agg.metric_vals[key].append(float(value))


def _accumulate_gate_metrics(
    *,
    gate_payload: Any,
    agg: _StepMetricsAccumulator,
) -> None:
    if not isinstance(gate_payload, dict) or not bool(gate_payload.get("checked")):
        return
    agg.gate_checked += 1
    accepted = gate_payload.get("accepted")
    if accepted is True:
        agg.gate_accept += 1
    elif accepted is False:
        agg.gate_reject += 1
    if gate_payload.get("transpile_success") is False:
        agg.gate_transpile_fail += 1

    reasons = gate_payload.get("reasons")
    if isinstance(reasons, list):
        reason_set = {str(reason) for reason in reasons}
        if "sstan_reject" in reason_set:
            agg.gate_sstan_reject += 1
        if "fidelity_reject_source_signal" in reason_set:
            agg.gate_fidelity_reject += 1

    timing = gate_payload.get("timing_ms")
    if isinstance(timing, dict):
        total_ms = timing.get("total")
        if isinstance(total_ms, int | float) and np.isfinite(float(total_ms)):
            agg.gate_total_ms.append(float(total_ms))


def _ensure_valid_records_have_decomposition(missing_indices: list[int]) -> None:
    if not missing_indices:
        return
    preview = ",".join(str(i) for i in missing_indices[:5])
    raise RuntimeError(
        "Scorer contract violation: valid completion(s) missing decomposition payload: "
        f"{preview}"
    )


def _attach_scorer_outcome_metrics(metrics: dict[str, Any], outcome_counts: Counter[str]) -> None:
    for outcome_code, count in outcome_counts.items():
        metrics[f"train/scorer_outcome/{outcome_code}"] = count


def _attach_metric_distribution_moments(
    metrics: dict[str, Any],
    metric_vals: dict[str, list[float]],
) -> None:
    for key, values in metric_vals.items():
        if not values:
            continue
        suffix = key.removeprefix("metric_")
        metrics[f"train/metric/{suffix}_mean"] = float(np.mean(values))
        metrics[f"train/metric/{suffix}_std"] = float(np.std(values))


def _attach_sstan_gate_metrics(metrics: dict[str, Any], agg: _StepMetricsAccumulator) -> None:
    metrics["sstan_gate/checked"] = agg.gate_checked
    metrics["sstan_gate/accept"] = agg.gate_accept
    metrics["sstan_gate/reject"] = agg.gate_reject
    metrics["sstan_gate/reject_rate"] = agg.gate_reject / max(agg.gate_checked, 1)
    metrics["sstan_gate/transpile_fail_rate"] = agg.gate_transpile_fail / max(agg.gate_checked, 1)
    metrics["sstan_gate/sstan_reject_rate"] = agg.gate_sstan_reject / max(agg.gate_checked, 1)
    metrics["sstan_gate/fidelity_reject_rate"] = agg.gate_fidelity_reject / max(agg.gate_checked, 1)
    if agg.gate_total_ms:
        metrics["sstan_gate/mean_total_s"] = float(np.mean(agg.gate_total_ms) / 1000.0)


def _attach_prevalence_metrics(metrics: dict[str, Any], agg: _StepMetricsAccumulator) -> None:
    denominator = max(agg.n_valid_with_decomposition, 1)
    metrics["train/prevalence/potential_terms"] = agg.n_with_potential / denominator
    metrics["train/prevalence/data_discard"] = agg.n_data_discard / denominator


def _build_wandb_final_summary(
    *,
    config: TinkerGRPOConfig,
    results: dict[str, Any],
    normalization_payload: dict[str, Any] | None,
    judge_payload: dict[str, Any] | None,
    heuristic_rates: dict[str, float] | None,
    batch_lh_metrics: dict[str, float] | None = None,
    batch_lh_history: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    run_status, error_reason = _resolve_run_status(results)

    summary: dict[str, Any] = {
        "sweep/run_status": run_status,
        "sweep/final_lh_formal_signal": results.get("final_frac_non_normalized", float("nan")),
        "sweep/final_valid_rate": results.get("final_valid_rate"),
        "paper/track": config.paper_track,
        "paper/claim_mode": config.claim_mode,
        "paper/reward_metric": config.reward_metric,
        "paper/reward_data_split": config.reward_data_split,
        "paper/reward_estimator_backend": config.reward_estimator_backend,
        "paper/prompt_source": config.prompt_source,
        "paper/prompt_policy": config.prompt_policy,
        "paper/thinking_mode": config.thinking_mode,
        "paper/monitoring_mode": config.monitoring_mode,
        "paper/sstan_gate_mode": config.sstan_gate_mode,
        "paper/sstan_gate_fidelity_policy": config.sstan_gate_fidelity_policy,
        "paper/normalization_method": config.normalization_method,
        "paper/delta_scope": config.normalization_delta_scope,
        "paper/frac_non_normalized_final": float("nan"),
        "paper/lh_formal_signal_final": float("nan"),
        "paper/judge_hacking_rate_final": float("nan"),
        "paper/lh_family_prevalence_final": float("nan"),
        "paper/lh_rate_batch_final": float("nan"),
        "paper/lh_rate_batch_mean": float("nan"),
        "paper/lh_other_novel_final": float("nan"),
        "paper/lh_other_novel_mean": float("nan"),
    }
    if error_reason is not None:
        summary["sweep/error"] = error_reason
    final_step = None
    n_steps_completed = results.get("n_steps_completed")
    if isinstance(n_steps_completed, int) and n_steps_completed > 0:
        final_step = n_steps_completed - 1
    _apply_normalization_payload(summary, normalization_payload, final_step=final_step)
    _apply_judge_payload(summary, judge_payload)
    _apply_heuristic_payload(summary, heuristic_rates)
    _apply_batch_lh_payload(summary, batch_lh_metrics)
    _apply_batch_lh_history(summary, batch_lh_history)
    return summary


def _resolve_run_status(results: dict[str, Any]) -> tuple[str, str | None]:
    if "error" in results:
        return "fail", str(results.get("error", "unknown"))
    final_valid_rate = results.get("final_valid_rate")
    final_reward = results.get("final_reward_mean", results.get("final_reward"))
    if isinstance(final_valid_rate, int | float) and final_valid_rate <= 0:
        return "fail", "no_valid_completions"
    if isinstance(final_reward, int | float) and not np.isfinite(float(final_reward)):
        return "fail", "non_finite_final_reward"
    return "success", None


def _apply_normalization_payload(
    summary: dict[str, Any],
    normalization_payload: dict[str, Any] | None,
    *,
    final_step: int | None,
) -> None:
    if not normalization_payload:
        return
    frac_non_normalized = normalization_payload.get("frac_non_normalized")
    if not isinstance(frac_non_normalized, int | float):
        return
    frac_value = float(frac_non_normalized)
    payload_step = normalization_payload.get("step")
    if (
        final_step is not None
        and isinstance(payload_step, int)
        and payload_step != final_step
    ):
        summary["paper/lh_formal_signal_latest_checked"] = frac_value
        summary["paper/lh_formal_signal_latest_checked_step"] = payload_step
    summary["paper/frac_non_normalized_final"] = frac_value
    summary["paper/lh_formal_signal_final"] = frac_value
    summary["sweep/final_lh_formal_signal"] = frac_value


def _apply_judge_payload(summary: dict[str, Any], judge_payload: dict[str, Any] | None) -> None:
    if judge_payload and isinstance(judge_payload.get("hacking_rate"), int | float):
        summary["paper/judge_hacking_rate_final"] = float(judge_payload["hacking_rate"])


def _apply_heuristic_payload(
    summary: dict[str, Any], heuristic_rates: dict[str, float] | None
) -> None:
    if heuristic_rates and isinstance(heuristic_rates.get("any_exploit_rate"), int | float):
        summary["paper/lh_family_prevalence_final"] = float(heuristic_rates["any_exploit_rate"])


def _apply_batch_lh_payload(
    summary: dict[str, Any], batch_lh_metrics: dict[str, float] | None
) -> None:
    if not batch_lh_metrics:
        return
    final_lh = batch_lh_metrics.get("lh/batch/proportion")
    if isinstance(final_lh, int | float):
        summary["paper/lh_rate_batch_final"] = float(final_lh)
    final_other_novel = batch_lh_metrics.get("lh/batch/other_novel_rate")
    if isinstance(final_other_novel, int | float):
        summary["paper/lh_other_novel_final"] = float(final_other_novel)
    for family in _LH_CANONICAL_FAMILIES:
        value = batch_lh_metrics.get(f"lh/batch/family/{family}")
        if isinstance(value, int | float):
            summary[f"paper/family/{family}_final"] = float(value)


def _apply_batch_lh_history(
    summary: dict[str, Any], batch_lh_history: list[dict[str, float]] | None
) -> None:
    if not batch_lh_history:
        return
    lh_vals = np.asarray(
        [row.get("lh/batch/proportion", float("nan")) for row in batch_lh_history],
        dtype=np.float64,
    )
    if lh_vals.size and np.any(np.isfinite(lh_vals)):
        summary["paper/lh_rate_batch_mean"] = float(np.nanmean(lh_vals))
    other_vals = np.asarray(
        [row.get("lh/batch/other_novel_rate", float("nan")) for row in batch_lh_history],
        dtype=np.float64,
    )
    if other_vals.size and np.any(np.isfinite(other_vals)):
        summary["paper/lh_other_novel_mean"] = float(np.nanmean(other_vals))


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
    import os

    from ppl_synthesis_reward_hacking.monitoring.llm_judge import _should_try_responses_api

    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        return None

    use_responses = _should_try_responses_api(cfg)

    def _call(prompt: str) -> str:
        import litellm

        if use_responses:
            try:
                response = litellm.responses(
                    model=cfg.model,
                    input=prompt,
                    api_base=cfg.api_base,
                    api_key=api_key,
                    max_output_tokens=cfg.max_tokens,
                    timeout=cfg.timeout,
                )
                return getattr(response.output[0].content[0], "text", "")
            except Exception:
                log.debug(
                    "responses API call failed; falling back to completion API",
                    exc_info=True,
                )

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
