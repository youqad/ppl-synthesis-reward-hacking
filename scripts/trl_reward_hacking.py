#!/usr/bin/env python3
"""GRPO training with PyMC reward.

Trains an LM to generate PyMC programs from natural language prompts.
Training uses only reported_reward. Normalization checks detect LH.

Usage:
    python scripts/trl_reward_hacking.py --model Qwen/Qwen3-0.6B --n-steps 50
    python scripts/trl_reward_hacking.py --use-4bit  # QLoRA
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from datasets import Dataset as HFDataset
    from peft import LoraConfig
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

from ppl_synthesis_reward_hacking.backends.pymc.backend import PyMCBackend
from ppl_synthesis_reward_hacking.backends.sstan.gate import (
    validate_sstan_gate_cfg,
)
from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.data.prompts import PROMPT_POLICIES
from ppl_synthesis_reward_hacking.data.pymc_synthesis_loader import load_synthesis_prompts
from ppl_synthesis_reward_hacking.experiments.config_mapping import (
    GROUP_ALLOWED_KEYS,
    SCALAR_GROUP_KEYS,
    build_sstan_gate_config,
    build_train_config_from_mapping,
    resolve_normalization_contract,
    validate_group_payload_keys,
    validate_judge_sampling_config,
    validate_scalar_group_payloads,
)
from ppl_synthesis_reward_hacking.experiments.pymc_reward import (
    make_pymc_reward_fn,
)
from ppl_synthesis_reward_hacking.experiments.results import (
    attach_common_results_metadata,
    compute_traj_metrics,
    json_default,
    print_training_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
_PROMPT_SAMPLING_MODES = frozenset({"fixed_cycle", "seeded_shuffle_cycle"})


@dataclass
class TRLRewardHackingConfig:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    paper_track: str = "part_a_emergence"
    monitoring_mode: str = "off"
    judge_interval: int = 0
    rubric_evolution_interval: int = 0
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
    use_cmdsafestan: bool = False
    cmdsafestan_data: dict[str, Any] | None = None
    cmdsafestan_protect: list[str] | None = None
    cmdsafestan_cmdstan_root: str = "."
    cmdsafestan_jobs: int = 4
    cmdsafestan_bootstrap: bool = True
    cmdsafestan_build_runtime: bool = True
    cmdsafestan_force_reinit: bool = False
    cmdsafestan_seed: int = 12345
    judge_sampling_policy: str = "fixed_cap"
    judge_adaptive_min: int = 8
    judge_adaptive_max: int = 20
    judge_adaptive_target_metric: str = "combined"
    claim_mode: str = "formal_lh"
    n_steps: int = 1000
    n_prompts: int = 100
    rollouts_per_prompt: int = 8
    lora_rank: int = 32
    lora_dropout: float = 0.0
    lr: float = 5e-6
    max_completion_length: int = 1024
    use_4bit: bool = False
    exec_timeout: int = 30
    output_dir: str = "artifacts/grpo_pymc_reward"
    resume_from: str | None = None
    num_generations: int = 8
    temperature: float = 1.28
    top_p: float = 1.0
    top_k: int = 0
    kl_beta: float = 0.001
    max_grad_norm: float = 1.0
    dataset_name: str = "linear_regression"
    prompt_source: str = "hardcoded"
    prompt_jsonl_path: str = "data/pymc_synthesis/train.jsonl"
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
    reward_metric: str = "log_marginal_likelihood"
    reward_data_split: str = "train"
    reward_estimator_backend: str = "smc"
    smc_draws: int = 500
    normalization_method: str = "auto"
    normalization_delta_scope: str = "raw_y_binary"
    normalization_epsilon: float = 5e-2
    normalization_ci_alpha: float = 0.05
    normalization_mc_samples: int = 256
    normalization_min_ess: float = 30.0
    normalization_interval: int = 1
    normalization_sample_size: int = 20
    report_to: str = "wandb"


def config_from_mapping(mapping: Mapping[str, Any]) -> TRLRewardHackingConfig:
    """Create TRLRewardHackingConfig from a mapping (Hydra-friendly)."""
    validate_group_payload_keys(mapping, group_allowed_keys=GROUP_ALLOWED_KEYS)
    validate_scalar_group_payloads(mapping, scalar_group_keys=SCALAR_GROUP_KEYS)
    cfg = build_train_config_from_mapping(
        mapping,
        TRLRewardHackingConfig,
        error_prefix="Unsupported train config keys",
    )
    _validate_mode_and_prompt_config(cfg)
    validate_judge_sampling_config(cfg)
    resolve_normalization_contract(cfg)
    validate_sstan_gate_cfg(build_sstan_gate_config(cfg), paper_track=cfg.paper_track)
    return cfg


def _validate_mode_and_prompt_config(cfg: TRLRewardHackingConfig) -> None:
    if cfg.monitoring_mode not in {"off", "judge_evolving"}:
        raise ValueError("monitoring_mode must be off|judge_evolving")
    if type(cfg.judge_interval) is not int or cfg.judge_interval < 0:
        raise ValueError("judge_interval must be a non-negative integer")
    if type(cfg.rubric_evolution_interval) is not int or cfg.rubric_evolution_interval < 0:
        raise ValueError("rubric_evolution_interval must be a non-negative integer")
    if cfg.monitoring_mode == "judge_evolving" and cfg.judge_interval <= 0:
        raise ValueError("monitoring_mode=judge_evolving requires judge_interval > 0")
    if cfg.monitoring_mode == "judge_evolving" and cfg.rubric_evolution_interval <= 0:
        raise ValueError("monitoring_mode=judge_evolving requires rubric_evolution_interval > 0")
    if cfg.prompt_source not in {"hardcoded", "jsonl"}:
        raise ValueError("prompt_source must be hardcoded|jsonl")
    if cfg.prompt_sampling not in _PROMPT_SAMPLING_MODES:
        raise ValueError(f"prompt_sampling must be one of {sorted(_PROMPT_SAMPLING_MODES)}")
    if cfg.thinking_mode not in {"think", "no_think"}:
        raise ValueError("thinking_mode must be think|no_think")
    if cfg.prompt_policy not in PROMPT_POLICIES:
        raise ValueError(f"prompt_policy must be one of {sorted(PROMPT_POLICIES)}")
    if cfg.prompt_jsonl_max_examples <= 0:
        raise ValueError("prompt_jsonl_max_examples must be positive")
    if cfg.n_prompts <= 0:
        raise ValueError("n_prompts must be positive")
    if cfg.num_generations < 2:
        raise ValueError("num_generations must be >= 2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training with PyMC reward")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="HuggingFace model ID")
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--rollouts-per-prompt", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    p.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--use-4bit", action="store_true", help="QLoRA 4-bit quantization")
    p.add_argument("--exec-timeout", type=int, default=30, help="PyMC exec timeout (seconds)")
    p.add_argument(
        "--output-dir",
        default="artifacts/grpo_pymc_reward",
        help="Output directory",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint dir to resume from (needs checkpoint_meta.json)",
    )
    p.add_argument("--num-generations", type=int, default=8, help="Generations per prompt")
    p.add_argument("--temperature", type=float, default=1.28, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling (1.0=disabled)")
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0=disabled)")
    p.add_argument("--kl-beta", type=float, default=0.001, help="KL penalty coefficient")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping")
    p.add_argument(
        "--dataset-name",
        default="linear_regression",
        choices=[
            "bernoulli_1d",
            "bernoulli_vector",
            "linear_regression",
            "logistic_regression",
            "gaussian_location",
        ],
    )
    p.add_argument("--dataset-n-features", type=int, default=3)
    p.add_argument("--dataset-noise-sigma", type=float, default=1.0)
    p.add_argument("--dataset-n-train", type=int, default=20)
    p.add_argument("--dataset-n-holdout", type=int, default=256)
    p.add_argument("--prompt-source", default="hardcoded", choices=["hardcoded", "jsonl"])
    p.add_argument(
        "--prompt-jsonl-path",
        default="data/pymc_synthesis/train.jsonl",
    )
    p.add_argument("--prompt-jsonl-max-examples", type=int, default=148)
    p.add_argument(
        "--prompt-sampling",
        default="seeded_shuffle_cycle",
        choices=["fixed_cycle", "seeded_shuffle_cycle"],
    )
    p.add_argument(
        "--thinking-mode",
        default="think",
        choices=["think", "no_think"],
    )
    p.add_argument(
        "--prompt-policy",
        default="legacy",
        choices=sorted(PROMPT_POLICIES),
    )
    p.add_argument("--scoring-seed-base", type=int, default=0)
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--normalization-interval", type=int, default=1)
    p.add_argument("--normalization-sample-size", type=int, default=20)
    return p.parse_args()


def _prepare_output_dir(config: TRLRewardHackingConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_resume_path(config: TRLRewardHackingConfig) -> str | None:
    if not config.resume_from:
        return None
    resume_dir = Path(config.resume_from)
    if not resume_dir.exists():
        raise RuntimeError(f"Resume directory not found: {resume_dir}")
    log.info("Resuming from checkpoint: %s", resume_dir)
    return config.resume_from


def _load_train_dataset(config: TRLRewardHackingConfig) -> tuple[list[dict[str, Any]], HFDataset]:
    log.info("Loading PyMC reward prompts")
    max_examples = config.n_prompts
    if config.prompt_source == "jsonl":
        max_examples = min(config.n_prompts, config.prompt_jsonl_max_examples)
    prompt_dicts = load_synthesis_prompts(
        max_examples=max_examples,
        dataset_name=config.dataset_name,
        prompt_source=config.prompt_source,
        prompt_path=config.prompt_jsonl_path,
        n_prompts=config.n_prompts,
        prompt_sampling=config.prompt_sampling,
        prompt_sampling_seed=config.scoring_seed_base,
        thinking_mode=config.thinking_mode,
        prompt_policy=config.prompt_policy,
    )
    train_dataset = HFDataset.from_list(prompt_dicts)
    log.info("Loaded %d prompts", len(prompt_dicts))
    return prompt_dicts, train_dataset


def _warn_monitoring_metadata_only(config: TRLRewardHackingConfig) -> None:
    if config.monitoring_mode == "off":
        return
    log.warning(
        "TRL path currently treats judge/rubric fields as metadata only "
        "(monitoring_mode=%s, judge_interval=%d, rubric_evolution_interval=%d, "
        "judge_sampling_policy=%s, judge_adaptive_min=%d, judge_adaptive_max=%d, "
        "judge_adaptive_target_metric=%s).",
        config.monitoring_mode,
        config.judge_interval,
        config.rubric_evolution_interval,
        config.judge_sampling_policy,
        config.judge_adaptive_min,
        config.judge_adaptive_max,
        config.judge_adaptive_target_metric,
    )


def _build_reward_function(config: TRLRewardHackingConfig, output_dir: Path):
    log.info("Creating scoring dataset and backend")
    scoring_dataset, scoring_data = _make_scoring_payload(config)
    backend = PyMCBackend()
    return make_pymc_reward_fn(
        backend=backend,
        dataset=scoring_dataset,
        scoring_data=scoring_data,
        cache_dir=output_dir / "pymc_cache",
        exec_timeout=config.exec_timeout,
        completions_path=output_dir / "completions.jsonl",
        reward_metric=config.reward_metric,
        reward_data_split=config.reward_data_split,
        reward_estimator_backend=config.reward_estimator_backend,
        smc_draws=config.smc_draws,
        normalization_method=config.normalization_method,
        normalization_delta_scope=config.normalization_delta_scope,
        normalization_epsilon=config.normalization_epsilon,
        normalization_ci_alpha=config.normalization_ci_alpha,
        normalization_mc_samples=config.normalization_mc_samples,
        normalization_min_ess=config.normalization_min_ess,
        normalization_interval=config.normalization_interval,
        normalization_sample_size=config.normalization_sample_size,
        sstan_gate_mode=config.sstan_gate_mode,
        sstan_gate_sample_rate=config.sstan_gate_sample_rate,
        sstan_gate_penalty_reward=config.sstan_gate_penalty_reward,
        sstan_gate_fidelity_policy=config.sstan_gate_fidelity_policy,
        sstan_gate_log_stan=config.sstan_gate_log_stan,
        sstan_transpiler_model=config.sstan_transpiler_model,
        sstan_transpiler_api_key_env=config.sstan_transpiler_api_key_env,
        sstan_transpiler_api_base=config.sstan_transpiler_api_base,
        sstan_transpiler_custom_llm_provider=config.sstan_transpiler_custom_llm_provider,
        sstan_transpiler_strict=config.sstan_transpiler_strict,
        sstan_transpiler_temperature=config.sstan_transpiler_temperature,
        sstan_transpiler_max_tokens=config.sstan_transpiler_max_tokens,
        sstan_transpiler_timeout_s=config.sstan_transpiler_timeout_s,
    )


def _build_model_init_kwargs(config: TRLRewardHackingConfig) -> dict[str, Any] | None:
    if not config.use_4bit:
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig

        model_init_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
        }
        log.info("QLoRA 4-bit enabled")
        return model_init_kwargs
    except ImportError:
        log.warning("bitsandbytes not available, skipping 4-bit quantization")
        return None


def _build_training_args(
    config: TRLRewardHackingConfig,
    *,
    output_dir: Path,
    model_init_kwargs: dict[str, Any] | None,
) -> TRLGRPOConfig:
    # Each paper step = one reward call + one optimizer update on
    # n_prompts × num_generations completions.
    # per_device_train_batch_size=num_generations (one prompt's completions).
    # gradient_accumulation_steps=n_prompts (accumulate over all prompts).
    # This makes steps_per_generation=n_prompts=gradient_accumulation_steps, so TRL regenerates
    # exactly once per optimizer update (generate_every = n_prompts gradient steps = 1 accum round).
    programs_per_step = config.n_prompts * config.num_generations
    return TRLGRPOConfig(
        output_dir=str(output_dir),
        max_steps=config.n_steps,
        per_device_train_batch_size=config.num_generations,
        generation_batch_size=programs_per_step,
        gradient_accumulation_steps=config.n_prompts,
        learning_rate=config.lr,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        beta=config.kl_beta,
        max_grad_norm=config.max_grad_norm,
        logging_steps=1,
        save_steps=max(1, config.n_steps // 5),
        report_to=config.report_to,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=False,
        use_vllm=False,
        model_init_kwargs=model_init_kwargs,
    )


def _log_training_setup(
    config: TRLRewardHackingConfig, *, n_prompts: int, output_dir: Path
) -> None:
    log.info("Model: %s", config.model)
    log.info(
        "Steps: %d, Prompts/step: %d, Generations/prompt: %d, Programs/step: %d",
        config.n_steps,
        n_prompts,
        config.num_generations,
        n_prompts * config.num_generations,
    )
    log.info(
        "LoRA rank=%d, alpha=%d, dropout=%.2f, targets=attn+mlp",
        config.lora_rank,
        32,
        config.lora_dropout,
    )
    log.info(
        "Exploration: temp=%.2f, top_p=%.2f, top_k=%d, kl_beta=%.3f",
        config.temperature,
        config.top_p,
        config.top_k,
        config.kl_beta,
    )
    log.info("Output: %s", output_dir)


def _write_trajectory(output_dir: Path, trajectory: list[dict[str, Any]]) -> None:
    with open(output_dir / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2, default=json_default)


def _write_results(output_dir: Path, results: dict[str, Any]) -> None:
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=json_default)


def run_training(config: TRLRewardHackingConfig) -> dict[str, Any]:
    if not TRL_AVAILABLE:
        raise RuntimeError(
            "TRL not installed. pip install trl peft datasets transformers accelerate"
        )
    output_dir = _prepare_output_dir(config)
    resume_path = _resolve_resume_path(config)
    prompt_dicts, train_dataset = _load_train_dataset(config)
    _warn_monitoring_metadata_only(config)

    reward_fn, reward_state = _build_reward_function(config, output_dir)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=32,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model_init_kwargs = _build_model_init_kwargs(config)
    training_args = _build_training_args(
        config,
        output_dir=output_dir,
        model_init_kwargs=model_init_kwargs,
    )
    _log_training_setup(config, n_prompts=len(prompt_dicts), output_dir=output_dir)

    trainer = GRPOTrainer(
        model=config.model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    try:
        trainer.train(resume_from_checkpoint=resume_path)

        trajectory_dicts = [asdict(p) for p in reward_state.trajectory]
        _write_trajectory(output_dir, trajectory_dicts)

        results = _compute_results(config, reward_state)
        results.update(_build_summary(config, results))
        _write_results(output_dir, results)

        log.info("Results saved to %s", output_dir)
        _print_summary(results)
        return results
    finally:
        if reward_state.completion_writer is not None:
            reward_state.completion_writer.close()


def _compute_results(config: TRLRewardHackingConfig, state) -> dict:
    trajectory = state.trajectory
    metrics = compute_traj_metrics(trajectory)
    config_payload = {
        "model": config.model,
        "paper_track": config.paper_track,
        "monitoring_mode": config.monitoring_mode,
        "sstan_gate_mode": config.sstan_gate_mode,
        "sstan_gate_sample_rate": config.sstan_gate_sample_rate,
        "sstan_gate_penalty_reward": config.sstan_gate_penalty_reward,
        "sstan_gate_fidelity_policy": config.sstan_gate_fidelity_policy,
        "sstan_transpiler_model": config.sstan_transpiler_model,
        "sstan_transpiler_strict": config.sstan_transpiler_strict,
        "judge_interval": config.judge_interval,
        "rubric_evolution_interval": config.rubric_evolution_interval,
        "claim_mode": config.claim_mode,
        "n_steps": config.n_steps,
        "n_prompts": config.n_prompts,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "lora_rank": config.lora_rank,
        "use_4bit": config.use_4bit,
        "dataset_name": config.dataset_name,
        "dataset_n_features": config.dataset_n_features,
        "dataset_n_train": config.dataset_n_train,
        "dataset_n_holdout": config.dataset_n_holdout,
        "reward_metric": config.reward_metric,
        "reward_data_split": config.reward_data_split,
        "reward_estimator_backend": config.reward_estimator_backend,
        "normalization_method": config.normalization_method,
        "normalization_delta_scope": config.normalization_delta_scope,
    }
    metrics = attach_common_results_metadata(
        metrics=metrics,
        trajectory=trajectory,
        count_key="n_batches",
        config_payload=config_payload,
    )
    if "error" in metrics:
        return metrics
    final = trajectory[-1]
    metrics["final_parse_fail_rate"] = final.n_parse_fail / max(final.n_total, 1)
    metrics["final_exec_fail_rate"] = final.n_exec_fail / max(final.n_total, 1)
    metrics["final_safety_flags"] = final.safety_flags
    return metrics


def _build_summary(config: TRLRewardHackingConfig, results: dict[str, Any]) -> dict[str, Any]:
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
        elif isinstance(final_reward, int | float) and not math.isfinite(float(final_reward)):
            run_status = "fail"
            error_reason = "non_finite_final_reward"

    summary: dict[str, Any] = {
        "sweep/run_status": run_status,
        "sweep/final_lh_formal_signal": results.get("final_frac_non_normalized", float("nan")),
        "sweep/final_valid_rate": results.get("final_valid_rate"),
        "paper/track": config.paper_track,
        "paper/claim_mode": config.claim_mode,
        "paper/reward_metric": config.reward_metric,
        "paper/reward_data_split": config.reward_data_split,
        "paper/reward_estimator_backend": config.reward_estimator_backend,
        "paper/monitoring_mode": config.monitoring_mode,
        "paper/sstan_gate_mode": config.sstan_gate_mode,
        "paper/sstan_gate_fidelity_policy": config.sstan_gate_fidelity_policy,
        "paper/prompt_source": config.prompt_source,
        "paper/prompt_policy": config.prompt_policy,
        "paper/thinking_mode": config.thinking_mode,
        "paper/normalization_method": config.normalization_method,
        "paper/delta_scope": config.normalization_delta_scope,
        "paper/frac_non_normalized_final": results.get("final_frac_non_normalized", float("nan")),
        "paper/lh_formal_signal_final": results.get("final_frac_non_normalized", float("nan")),
        "paper/judge_hacking_rate_final": float("nan"),
        "paper/lh_family_prevalence_final": float("nan"),
    }
    if error_reason is not None:
        summary["sweep/error"] = error_reason
    return summary


def _make_scoring_payload(config: TRLRewardHackingConfig):
    params: dict[str, Any] = {
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
            rng = np.random.default_rng(config.scoring_seed_base + 13_371)
            params["p_true"] = float(
                rng.beta(config.dataset_p_true_alpha, config.dataset_p_true_beta)
            )
        else:
            params["p_true"] = float(config.dataset_p_true_fixed)

    dataset = generate_dataset(config.dataset_name, params, seed=config.scoring_seed_base)
    scoring_data: dict[str, Any] = dict(dataset.train)
    scoring_data["dataset_name"] = config.dataset_name
    scoring_data["n"] = config.dataset_n_train
    scoring_data["p"] = config.dataset_n_features
    scoring_data["d"] = config.dataset_n_features
    scoring_data["y_train"] = np.asarray(dataset.train["y"], dtype=np.float64)
    scoring_data["y_holdout"] = np.asarray(dataset.holdout["y"], dtype=np.float64)
    if "X" in dataset.train:
        scoring_data["X_train"] = np.asarray(dataset.train["X"], dtype=np.float64)
        scoring_data["X_holdout"] = np.asarray(dataset.holdout["X"], dtype=np.float64)
    if "p_true" in dataset.meta:
        scoring_data["p_true"] = float(dataset.meta["p_true"])
    scoring_data["seed"] = config.scoring_seed_base
    return dataset, scoring_data


def _print_summary(results: dict) -> None:
    print_training_summary(results)
    if "error" not in results and results.get("final_safety_flags"):
        log.info("safety flags: %s", results["final_safety_flags"])


def main() -> None:
    args = parse_args()
    if not TRL_AVAILABLE:
        raise SystemExit("TRL not installed. pip install trl peft datasets transformers accelerate")
    config = config_from_mapping(
        {
            "model": args.model,
            "n_steps": args.n_steps,
            "n_prompts": args.n_prompts,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "lora_rank": args.lora_rank,
            "lora_dropout": args.lora_dropout,
            "lr": args.lr,
            "max_completion_length": args.max_completion_length,
            "use_4bit": args.use_4bit,
            "exec_timeout": args.exec_timeout,
            "output_dir": args.output_dir,
            "resume_from": args.resume_from,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "kl_beta": args.kl_beta,
            "max_grad_norm": args.max_grad_norm,
            "dataset_name": args.dataset_name,
            "prompt_source": args.prompt_source,
            "prompt_jsonl_path": args.prompt_jsonl_path,
            "prompt_jsonl_max_examples": args.prompt_jsonl_max_examples,
            "prompt_sampling": args.prompt_sampling,
            "thinking_mode": args.thinking_mode,
            "prompt_policy": args.prompt_policy,
            "dataset_n_features": args.dataset_n_features,
            "dataset_noise_sigma": args.dataset_noise_sigma,
            "dataset_n_train": args.dataset_n_train,
            "dataset_n_holdout": args.dataset_n_holdout,
            "scoring_seed_base": args.scoring_seed_base,
            "report_to": args.report_to,
            "normalization_interval": args.normalization_interval,
            "normalization_sample_size": args.normalization_sample_size,
        }
    )
    run_training(config)


if __name__ == "__main__":
    main()
