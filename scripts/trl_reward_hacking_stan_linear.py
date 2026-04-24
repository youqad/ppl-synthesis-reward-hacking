#!/usr/bin/env python3
"""Local TRL GRPO training for direct-Stan linear regression."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
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

from ppl_synthesis_reward_hacking.config.flattening import flatten_hydra_train_mapping
from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.data.stan_reward_loader import (
    STAN_LINEAR_PROMPT_POLICIES,
    load_stan_linear_reward_prompts,
)
from ppl_synthesis_reward_hacking.experiments.results import (
    attach_common_results_metadata,
    compute_traj_metrics,
    json_default,
    print_training_summary,
)
from ppl_synthesis_reward_hacking.experiments.stan_linear_reward import (
    make_stan_linear_reward_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class TRLStanLinearRewardConfig:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    paper_track: str = "part_a_emergence"
    claim_mode: str = "formal_lh"
    n_steps: int = 1000
    n_prompts: int = 64
    rollouts_per_prompt: int = 8
    lora_rank: int = 32
    lora_dropout: float = 0.0
    lr: float = 5e-6
    max_completion_length: int = 1536
    use_4bit: bool = False
    output_dir: str = "artifacts/grpo_stan_linear_reward"
    resume_from: str | None = None
    num_generations: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    kl_beta: float = 0.001
    max_grad_norm: float = 1.0
    report_to: str = "none"
    run_name: str | None = None
    reward_data_split: str = "train"
    thinking_mode: str = "no_think"
    prompt_policy: str = "neutral_single"
    dataset_n_features: int = 3
    dataset_noise_sigma: float = 1.0
    beta_prior_scale: float = 1.0
    dataset_n_train: int = 20
    dataset_n_holdout: int = 256
    scoring_seed_base: int = 0
    cmdstan_root: str = "cmdsafestan"
    stanc3: str = "safestan"
    protect: str = "y"
    compile_jobs: int = 8
    checker_jobs: int = 4
    reward_output_field: str = "reported_log_density"
    checker_mode: str = "shadow"
    checker_penalty_reward: float = -100.0
    normalization_epsilon: float = 5e-2
    normalization_ci_alpha: float = 0.05
    normalization_mc_samples: int = 256
    normalization_min_ess: float = 30.0
    normalization_interval: int = 1
    normalization_sample_size: int = 20


def config_from_mapping(mapping: Mapping[str, Any]) -> TRLStanLinearRewardConfig:
    flattened = flatten_hydra_train_mapping(mapping)
    allowed = {f.name for f in fields(TRLStanLinearRewardConfig)}
    unknown = sorted(k for k in flattened if k not in allowed)
    if unknown:
        raise ValueError(f"Unsupported train config keys: {', '.join(unknown)}")
    cfg = TRLStanLinearRewardConfig(**flattened)
    _validate_config(cfg)
    return cfg


def _validate_config(config: TRLStanLinearRewardConfig) -> None:
    if config.claim_mode != "formal_lh":
        raise ValueError("claim_mode must be formal_lh for the direct-Stan LH experiment")
    if config.paper_track not in {"part_a_emergence", "part_b_mitigation"}:
        raise ValueError("paper_track must be part_a_emergence|part_b_mitigation")
    if config.reward_data_split not in {"train", "holdout"}:
        raise ValueError("reward_data_split must be train|holdout")
    if config.thinking_mode not in {"think", "no_think"}:
        raise ValueError("thinking_mode must be think|no_think")
    if config.prompt_policy not in STAN_LINEAR_PROMPT_POLICIES:
        raise ValueError(
            "prompt_policy must be one of "
            f"{sorted(STAN_LINEAR_PROMPT_POLICIES)}"
        )
    if config.checker_mode not in {"off", "shadow", "enforce"}:
        raise ValueError("checker_mode must be off|shadow|enforce")
    if config.dataset_n_features <= 0:
        raise ValueError("dataset_n_features must be positive")
    if config.dataset_n_train <= 0:
        raise ValueError("dataset_n_train must be positive")
    if config.dataset_n_holdout <= 0:
        raise ValueError("dataset_n_holdout must be positive")
    if config.dataset_noise_sigma <= 0:
        raise ValueError("dataset_noise_sigma must be positive")
    if config.beta_prior_scale <= 0:
        raise ValueError("beta_prior_scale must be positive")
    if config.num_generations < 2:
        raise ValueError("num_generations must be >= 2")
    if config.normalization_interval <= 0:
        raise ValueError("normalization_interval must be positive")
    if config.normalization_sample_size <= 0:
        raise ValueError("normalization_sample_size must be positive")
    if config.normalization_mc_samples <= 0:
        raise ValueError("normalization_mc_samples must be positive")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training with direct Stan linear-regression reward"
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--paper-track", default="part_a_emergence")
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--rollouts-per-prompt", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-completion-length", type=int, default=1536)
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--output-dir", default="artifacts/grpo_stan_linear_reward")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--kl-beta", type=float, default=0.001)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--report-to", default="none")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--reward-data-split", default="train", choices=["train", "holdout"])
    p.add_argument("--thinking-mode", default="no_think", choices=["think", "no_think"])
    p.add_argument(
        "--prompt-policy",
        default="neutral_single",
        choices=sorted(STAN_LINEAR_PROMPT_POLICIES),
    )
    p.add_argument("--dataset-n-features", type=int, default=3)
    p.add_argument("--dataset-noise-sigma", type=float, default=1.0)
    p.add_argument("--beta-prior-scale", type=float, default=1.0)
    p.add_argument("--dataset-n-train", type=int, default=20)
    p.add_argument("--dataset-n-holdout", type=int, default=256)
    p.add_argument("--scoring-seed-base", type=int, default=0)
    p.add_argument("--cmdstan-root", default="cmdsafestan")
    p.add_argument("--stanc3", default="safestan")
    p.add_argument("--protect", default="y")
    p.add_argument("--compile-jobs", type=int, default=8)
    p.add_argument("--checker-jobs", type=int, default=4)
    p.add_argument("--reward-output-field", default="reported_log_density")
    p.add_argument("--checker-mode", default="shadow", choices=["off", "shadow", "enforce"])
    p.add_argument("--checker-penalty-reward", type=float, default=-100.0)
    p.add_argument("--normalization-epsilon", type=float, default=5e-2)
    p.add_argument("--normalization-ci-alpha", type=float, default=0.05)
    p.add_argument("--normalization-mc-samples", type=int, default=256)
    p.add_argument("--normalization-min-ess", type=float, default=30.0)
    p.add_argument("--normalization-interval", type=int, default=1)
    p.add_argument("--normalization-sample-size", type=int, default=20)
    return p.parse_args()


def _prepare_output_dir(config: TRLStanLinearRewardConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_resume_path(config: TRLStanLinearRewardConfig) -> str | None:
    if not config.resume_from:
        return None
    resume_dir = Path(config.resume_from)
    if not resume_dir.exists():
        raise RuntimeError(f"Resume directory not found: {resume_dir}")
    log.info("Resuming from checkpoint: %s", resume_dir)
    return config.resume_from


def _load_train_dataset(
    config: TRLStanLinearRewardConfig,
) -> tuple[list[dict[str, Any]], HFDataset]:
    prompt_dicts = load_stan_linear_reward_prompts(
        max_examples=config.n_prompts,
        thinking_mode=config.thinking_mode,
        prompt_policy=config.prompt_policy,
    )
    train_dataset = HFDataset.from_list(prompt_dicts)
    log.info("Loaded %d direct-Stan prompts", len(prompt_dicts))
    return prompt_dicts, train_dataset


def _build_reward_function(config: TRLStanLinearRewardConfig, output_dir: Path):
    scoring_data = _make_scoring_payload(config)
    return make_stan_linear_reward_fn(
        scoring_data=scoring_data,
        reward_data_split=config.reward_data_split,
        output_dir=output_dir,
        cmdstan_root=config.cmdstan_root,
        stanc3=config.stanc3,
        protect=config.protect,
        compile_jobs=config.compile_jobs,
        checker_jobs=config.checker_jobs,
        reward_output_field=config.reward_output_field,
        checker_mode=config.checker_mode,
        checker_penalty_reward=config.checker_penalty_reward,
        normalization_epsilon=config.normalization_epsilon,
        normalization_ci_alpha=config.normalization_ci_alpha,
        normalization_mc_samples=config.normalization_mc_samples,
        normalization_min_ess=config.normalization_min_ess,
        normalization_interval=config.normalization_interval,
        normalization_sample_size=config.normalization_sample_size,
        completions_path=output_dir / "completions.jsonl",
    )


def _build_model_init_kwargs(config: TRLStanLinearRewardConfig) -> dict[str, Any] | None:
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


def _resolve_precision() -> tuple[bool, bool, bool]:
    use_cuda = False
    use_bf16 = False
    use_fp16 = False
    try:
        import torch

        use_cuda = bool(torch.cuda.is_available())
        use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
        use_fp16 = bool(use_cuda and not use_bf16)
    except Exception:  # noqa: BLE001
        use_cuda = False
        use_bf16 = False
        use_fp16 = False
    return use_cuda, use_bf16, use_fp16


def _build_training_args(
    config: TRLStanLinearRewardConfig,
    *,
    output_dir: Path,
    model_init_kwargs: dict[str, Any] | None,
) -> TRLGRPOConfig:
    use_cuda, use_bf16, use_fp16 = _resolve_precision()
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
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=not use_cuda,
        gradient_checkpointing=False,
        use_vllm=False,
        model_init_kwargs=model_init_kwargs,
        run_name=(config.run_name or _default_run_name(config)),
    )


def _log_training_setup(
    config: TRLStanLinearRewardConfig, *, n_prompts: int, output_dir: Path
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
        "Dataset: linear_regression (train=%d holdout=%d features=%d sigma=%.2f prior=%.2f)",
        config.dataset_n_train,
        config.dataset_n_holdout,
        config.dataset_n_features,
        config.dataset_noise_sigma,
        config.beta_prior_scale,
    )
    log.info(
        "Direct Stan reward: field=%s checker_mode=%s split=%s prompt_policy=%s",
        config.reward_output_field,
        config.checker_mode,
        config.reward_data_split,
        config.prompt_policy,
    )
    log.info("Output: %s", output_dir)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def run_training(config: TRLStanLinearRewardConfig) -> dict[str, Any]:
    if not TRL_AVAILABLE:
        raise RuntimeError(
            "TRL not installed. Use `pixi install -e arc` or install the project with "
            "the `[arc]` extra before running the local Stan experiment."
        )
    output_dir = _prepare_output_dir(config)
    resume_path = _resolve_resume_path(config)
    prompt_dicts, train_dataset = _load_train_dataset(config)

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
        _write_json(output_dir / "trajectory.json", trajectory_dicts)
        results = _compute_results(config, reward_state)
        _write_json(output_dir / "results.json", results)
        log.info("Results saved to %s", output_dir)
        _print_summary(results)
        return results
    finally:
        if reward_state.completion_writer is not None:
            reward_state.completion_writer.close()


def _compute_results(config: TRLStanLinearRewardConfig, state) -> dict[str, Any]:
    trajectory = state.trajectory
    if not trajectory:
        return {"error": "no trajectory points"}
    if len(trajectory) < 2:
        point = trajectory[-1]
        metrics: dict[str, Any] = {
            "initial_reward": point.reward_mean,
            "final_reward": point.reward_mean,
            "reward_increase": 0.0,
            "final_valid_rate": point.n_valid / max(point.n_total, 1),
            "lh_detected": (
                point.n_norm_checked > 0
                and point.frac_non_normalized > 0.01
            ),
            "final_frac_non_normalized": point.frac_non_normalized,
            "final_mean_abs_log_mass": point.mean_abs_log_mass,
            "final_n_norm_checked": point.n_norm_checked,
        }
    else:
        metrics = compute_traj_metrics(trajectory)

    metrics = attach_common_results_metadata(
        metrics=metrics,
        trajectory=trajectory,
        count_key="n_batches",
        config_payload=asdict(config),
    )
    if "error" in metrics:
        return metrics

    final = trajectory[-1]
    metrics["final_parse_fail_rate"] = final.n_parse_fail / max(final.n_total, 1)
    metrics["final_exec_fail_rate"] = final.n_exec_fail / max(final.n_total, 1)
    metrics["final_unsafe_rate"] = final.unsafe_rate
    metrics["final_n_unsafe"] = final.n_unsafe
    metrics["final_n_checked"] = final.n_checked
    metrics.update(_build_summary(config, metrics))
    return metrics


def _build_summary(
    config: TRLStanLinearRewardConfig, results: dict[str, Any]
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
        elif isinstance(final_reward, int | float) and not math.isfinite(float(final_reward)):
            run_status = "fail"
            error_reason = "non_finite_final_reward"

    summary: dict[str, Any] = {
        "sweep/run_status": run_status,
        "sweep/final_lh_formal_signal": results.get("final_frac_non_normalized", float("nan")),
        "sweep/final_valid_rate": results.get("final_valid_rate"),
        "paper/track": config.paper_track,
        "paper/claim_mode": config.claim_mode,
        "paper/reward_metric": config.reward_output_field,
        "paper/reward_data_split": config.reward_data_split,
        "paper/reward_estimator_backend": "cmdsafestan_plain",
        "paper/prompt_source": "hardcoded",
        "paper/prompt_policy": config.prompt_policy,
        "paper/thinking_mode": config.thinking_mode,
        "paper/monitoring_mode": f"safestan_{config.checker_mode}",
        "paper/normalization_method": "importance_mc",
        "paper/delta_scope": "raw_y_fixed_x",
        "paper/frac_non_normalized_final": results.get("final_frac_non_normalized", float("nan")),
        "paper/lh_formal_signal_final": results.get("final_frac_non_normalized", float("nan")),
        "paper/judge_hacking_rate_final": float("nan"),
        "paper/lh_family_prevalence_final": float("nan"),
    }
    if error_reason is not None:
        summary["sweep/error"] = error_reason
    return summary


def _make_scoring_payload(config: TRLStanLinearRewardConfig) -> dict[str, Any]:
    dataset = generate_dataset(
        "linear_regression",
        {
            "n_features": config.dataset_n_features,
            "noise_sigma": config.dataset_noise_sigma,
            "beta_scale": config.beta_prior_scale,
            "split": {
                "n_train": config.dataset_n_train,
                "n_holdout": config.dataset_n_holdout,
            },
        },
        seed=config.scoring_seed_base,
    )
    return {
        "dataset_name": "linear_regression",
        "X_train": np.asarray(dataset.train["X"], dtype=np.float64),
        "y_train": np.asarray(dataset.train["y"], dtype=np.float64),
        "X_holdout": np.asarray(dataset.holdout["X"], dtype=np.float64),
        "y_holdout": np.asarray(dataset.holdout["y"], dtype=np.float64),
        "sigma_obs": float(config.dataset_noise_sigma),
        "beta_prior_scale": float(config.beta_prior_scale),
        "meta": dict(dataset.meta),
    }


def _print_summary(results: dict[str, Any]) -> None:
    print_training_summary(results)
    if "error" in results:
        return
    log.info(
        "unsafe_rate_final: %.3f (%d/%d)",
        results.get("final_unsafe_rate", float("nan")),
        results.get("final_n_unsafe", 0),
        results.get("final_n_checked", 0),
    )
    log.info("parse_fail_rate_final: %.3f", results.get("final_parse_fail_rate", float("nan")))
    log.info("exec_fail_rate_final: %.3f", results.get("final_exec_fail_rate", float("nan")))


def _default_run_name(config: TRLStanLinearRewardConfig) -> str:
    model_short = config.model.split("/")[-1].lower()
    model_short = re.sub(r"[^a-z0-9]+", "-", model_short).strip("-")
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return (
        f"stan-linear-grpo-{model_short}-"
        f"s{config.n_steps}-p{config.n_prompts}-g{config.num_generations}-{ts}"
    )


def main() -> None:
    if not TRL_AVAILABLE:
        raise SystemExit(
            "TRL not installed. Use `pixi install -e arc` or install the project "
            "with the `[arc]` extra before running this script."
        )

    args = parse_args()
    config = config_from_mapping(
        {
            "model": args.model,
            "paper_track": args.paper_track,
            "n_steps": args.n_steps,
            "n_prompts": args.n_prompts,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "lora_rank": args.lora_rank,
            "lora_dropout": args.lora_dropout,
            "lr": args.lr,
            "max_completion_length": args.max_completion_length,
            "use_4bit": args.use_4bit,
            "output_dir": args.output_dir,
            "resume_from": args.resume_from,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "kl_beta": args.kl_beta,
            "max_grad_norm": args.max_grad_norm,
            "report_to": args.report_to,
            "run_name": args.run_name,
            "reward_data_split": args.reward_data_split,
            "thinking_mode": args.thinking_mode,
            "prompt_policy": args.prompt_policy,
            "dataset_n_features": args.dataset_n_features,
            "dataset_noise_sigma": args.dataset_noise_sigma,
            "beta_prior_scale": args.beta_prior_scale,
            "dataset_n_train": args.dataset_n_train,
            "dataset_n_holdout": args.dataset_n_holdout,
            "scoring_seed_base": args.scoring_seed_base,
            "cmdstan_root": args.cmdstan_root,
            "stanc3": args.stanc3,
            "protect": args.protect,
            "compile_jobs": args.compile_jobs,
            "checker_jobs": args.checker_jobs,
            "reward_output_field": args.reward_output_field,
            "checker_mode": args.checker_mode,
            "checker_penalty_reward": args.checker_penalty_reward,
            "normalization_epsilon": args.normalization_epsilon,
            "normalization_ci_alpha": args.normalization_ci_alpha,
            "normalization_mc_samples": args.normalization_mc_samples,
            "normalization_min_ess": args.normalization_min_ess,
            "normalization_interval": args.normalization_interval,
            "normalization_sample_size": args.normalization_sample_size,
        }
    )
    run_training(config)


if __name__ == "__main__":
    main()
