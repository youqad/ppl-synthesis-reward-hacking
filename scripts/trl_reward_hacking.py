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
from dataclasses import asdict, dataclass, fields
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
from ppl_synthesis_reward_hacking.config.contracts import validate_train_contract
from ppl_synthesis_reward_hacking.config.flattening import flatten_hydra_train_mapping
from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.data.pymc_reward_loader import load_pymc_reward_prompts
from ppl_synthesis_reward_hacking.experiments.pymc_reward import (
    make_pymc_reward_fn,
)
from ppl_synthesis_reward_hacking.experiments.results import (
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


@dataclass
class TRLRewardHackingConfig:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    paper_track: str = "part_a_emergence"
    monitoring_mode: str = "off"
    judge_interval: int = 0
    rubric_evolution_interval: int = 0
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
    predictive_estimator: str = "none"
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
    flattened = flatten_hydra_train_mapping(mapping)
    allowed = {f.name for f in fields(TRLRewardHackingConfig)}
    cfg = TRLRewardHackingConfig(**{k: v for k, v in flattened.items() if k in allowed})
    resolved = validate_train_contract(
        reward_metric=cfg.reward_metric,
        reward_data_split=cfg.reward_data_split,
        predictive_estimator=cfg.predictive_estimator,
        claim_mode=cfg.claim_mode,
        dataset_name=cfg.dataset_name,
        normalization_method=cfg.normalization_method,
        normalization_delta_scope=cfg.normalization_delta_scope,
        normalization_epsilon=cfg.normalization_epsilon,
        normalization_ci_alpha=cfg.normalization_ci_alpha,
        normalization_mc_samples=cfg.normalization_mc_samples,
        normalization_min_ess=cfg.normalization_min_ess,
        normalization_interval=cfg.normalization_interval,
    )
    cfg.normalization_method = resolved.normalization_method
    cfg.normalization_delta_scope = resolved.normalization_delta_scope
    return cfg


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
    p.add_argument("--scoring-seed-base", type=int, default=0)
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--normalization-interval", type=int, default=1)
    p.add_argument("--normalization-sample-size", type=int, default=20)
    return p.parse_args()


def run_training(config: TRLRewardHackingConfig) -> dict[str, Any]:
    if not TRL_AVAILABLE:
        raise RuntimeError(
            "TRL not installed. pip install trl peft datasets transformers accelerate"
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.resume_from:
        resume_dir = Path(config.resume_from)
        if not resume_dir.exists():
            raise RuntimeError(f"Resume directory not found: {resume_dir}")
        log.info("Resuming from checkpoint: %s", resume_dir)

    log.info("Loading PyMC reward prompts")
    prompt_dicts = load_pymc_reward_prompts(
        max_examples=config.n_prompts,
        dataset_name=config.dataset_name,
    )
    train_dataset = HFDataset.from_list(prompt_dicts)
    log.info("Loaded %d prompts", len(prompt_dicts))
    if config.monitoring_mode != "off":
        log.warning(
            "TRL path currently treats judge/rubric fields as metadata only "
            "(monitoring_mode=%s, judge_interval=%d, rubric_evolution_interval=%d).",
            config.monitoring_mode,
            config.judge_interval,
            config.rubric_evolution_interval,
        )

    log.info("Creating scoring dataset and backend")
    scoring_dataset, scoring_data = _make_scoring_payload(config)
    backend = PyMCBackend()

    reward_fn, reward_state = make_pymc_reward_fn(
        backend=backend,
        dataset=scoring_dataset,
        scoring_data=scoring_data,
        cache_dir=output_dir / "pymc_cache",
        exec_timeout=config.exec_timeout,
        completions_path=output_dir / "completions.jsonl",
        reward_metric=config.reward_metric,
        reward_data_split=config.reward_data_split,
        predictive_estimator=config.predictive_estimator,
        smc_draws=config.smc_draws,
        normalization_method=config.normalization_method,
        normalization_delta_scope=config.normalization_delta_scope,
        normalization_epsilon=config.normalization_epsilon,
        normalization_ci_alpha=config.normalization_ci_alpha,
        normalization_mc_samples=config.normalization_mc_samples,
        normalization_min_ess=config.normalization_min_ess,
        normalization_interval=config.normalization_interval,
        normalization_sample_size=config.normalization_sample_size,
    )

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=32,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model_init_kwargs = None
    if config.use_4bit:
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
        except ImportError:
            log.warning("bitsandbytes not available, skipping 4-bit quantization")

    training_args = TRLGRPOConfig(
        output_dir=str(output_dir),
        max_steps=config.n_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.rollouts_per_prompt,
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

    log.info("Model: %s", config.model)
    log.info(
        "Steps: %d, Prompts: %d, Generations/prompt: %d",
        config.n_steps,
        len(prompt_dicts),
        config.num_generations,
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

    trainer = GRPOTrainer(
        model=config.model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    resume_path = config.resume_from if config.resume_from else None
    trainer.train(resume_from_checkpoint=resume_path)

    trajectory_dicts = [asdict(p) for p in reward_state.trajectory]
    with open(output_dir / "trajectory.json", "w") as f:
        json.dump(trajectory_dicts, f, indent=2, default=json_default)

    results = _compute_results(config, reward_state)
    results.update(_build_summary(config, results))
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=json_default)

    log.info("Results saved to %s", output_dir)
    _print_summary(results)

    if reward_state.completion_writer is not None:
        reward_state.completion_writer.close()

    return results


def _compute_results(config: TRLRewardHackingConfig, state) -> dict:
    trajectory = state.trajectory
    metrics = compute_traj_metrics(trajectory)
    if "error" in metrics:
        metrics["n_batches"] = len(trajectory)
        return metrics

    final = trajectory[-1]
    metrics["config"] = {
        "model": config.model,
        "paper_track": config.paper_track,
        "monitoring_mode": config.monitoring_mode,
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
        "predictive_estimator": config.predictive_estimator,
        "normalization_method": config.normalization_method,
        "normalization_delta_scope": config.normalization_delta_scope,
    }
    metrics["final_parse_fail_rate"] = final.n_parse_fail / max(final.n_total, 1)
    metrics["final_exec_fail_rate"] = final.n_exec_fail / max(final.n_total, 1)
    metrics["final_safety_flags"] = final.safety_flags
    metrics["final_reward_mean"] = metrics.get("final_reward")
    metrics["n_batches"] = len(trajectory)
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
        "paper/predictive_estimator": config.predictive_estimator,
        "paper/monitoring_mode": config.monitoring_mode,
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
    if not TRL_AVAILABLE:
        raise SystemExit("TRL not installed. pip install trl peft datasets transformers accelerate")

    args = parse_args()
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
