#!/usr/bin/env python3
"""Local TRL GRPO training for direct-Stan linear regression."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
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
from ppl_synthesis_reward_hacking.data.stan_reward_loader import (
    STAN_LINEAR_PROMPT_POLICIES,
    get_stan_linear_prompt_count,
    get_stan_linear_system_prompt_count,
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
from ppl_synthesis_reward_hacking.utils.hashing import stable_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _valid_only_advantages(
    rewards: Sequence[float],
    *,
    num_generations: int,
    scale_rewards: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if rewards_arr.size % num_generations != 0:
        raise ValueError("reward count must be divisible by num_generations")
    if scale_rewards not in {"batch", "group", "none"}:
        raise ValueError("scale_rewards must be batch|group|none")

    grouped = rewards_arr.reshape(-1, num_generations)
    valid = np.isfinite(grouped)
    counts = valid.sum(axis=1)
    filled = np.where(valid, grouped, 0.0)
    means = np.zeros(grouped.shape[0], dtype=np.float64)
    has_valid = counts > 0
    means[has_valid] = filled.sum(axis=1)[has_valid] / counts[has_valid]
    centered = np.where(valid, grouped - means[:, None], 0.0)

    group_stds = np.zeros(grouped.shape[0], dtype=np.float64)
    has_pair = counts > 1
    if np.any(has_pair):
        group_stds[has_pair] = np.sqrt(
            np.sum(centered[has_pair] ** 2, axis=1) / (counts[has_pair] - 1)
        )
    is_std_zero = np.repeat(np.isclose(group_stds, 0.0), num_generations)

    if scale_rewards == "none":
        advantages = centered
    elif scale_rewards == "group":
        advantages = np.zeros_like(centered)
        nonzero_std = group_stds > 0.0
        if np.any(nonzero_std):
            advantages[nonzero_std] = centered[nonzero_std] / (
                group_stds[nonzero_std, None] + 1e-4
            )
    else:
        valid_rewards = rewards_arr[np.isfinite(rewards_arr)]
        batch_std = (
            float(np.std(valid_rewards, ddof=1))
            if valid_rewards.size > 1
            else 0.0
        )
        is_std_zero = np.full(rewards_arr.shape, np.isclose(batch_std, 0.0))
        advantages = (
            centered / (batch_std + 1e-4)
            if batch_std > 0.0
            else np.zeros_like(centered)
        )

    advantages = np.where(valid, advantages, 0.0)
    return advantages.reshape(-1), valid.reshape(-1), is_std_zero.reshape(-1)


if TRL_AVAILABLE:

    class ValidOnlyGRPOTrainer(GRPOTrainer):
        """Mask invalid Stan programs out of the policy loss."""

        def _calculate_rewards(self, *args, **kwargs):
            rewards_per_func = super()._calculate_rewards(*args, **kwargs)
            self._last_rewards_per_func = rewards_per_func.detach()
            return rewards_per_func

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            rewards_per_func = getattr(self, "_last_rewards_per_func", None)
            if rewards_per_func is None or self.multi_objective_aggregation != "sum_then_normalize":
                return output

            import torch

            finite = torch.isfinite(rewards_per_func)
            weighted = torch.where(
                finite,
                rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0),
                torch.zeros_like(rewards_per_func),
            ).sum(dim=1)
            weighted = torch.where(
                finite.any(dim=1),
                weighted,
                torch.full_like(weighted, torch.nan),
            )
            mode = "train" if self.model.training else "eval"
            num_generations = self.num_generations if mode == "train" else self.num_generations_eval
            advantages_np, valid_np, is_std_zero_np = _valid_only_advantages(
                weighted.detach().cpu().numpy(),
                num_generations=num_generations,
                scale_rewards=self.scale_rewards,
            )

            local_n = int(output["advantages"].shape[0])
            start = self.accelerator.process_index * local_n
            end = start + local_n
            device = output["advantages"].device
            advantages = torch.as_tensor(
                advantages_np[start:end],
                dtype=output["advantages"].dtype,
                device=device,
            )
            valid_mask = torch.as_tensor(
                valid_np[start:end],
                dtype=output["completion_mask"].dtype,
                device=output["completion_mask"].device,
            )
            output["advantages"] = advantages
            output["completion_mask"] = output["completion_mask"] * valid_mask.unsqueeze(1)
            output["valid_only_row_mask"] = valid_mask

            self._metrics[mode]["valid_only/finite_reward_rate"].append(float(valid_np.mean()))
            self._metrics[mode]["valid_only/frac_reward_zero_std"].append(
                float(np.mean(is_std_zero_np))
            )
            return output

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            row_mask = inputs.get("valid_only_row_mask")
            if row_mask is None:
                return loss
            valid_rows = row_mask.to(device=loss.device, dtype=loss.dtype).sum()
            if float(valid_rows.detach().cpu()) <= 0.0:
                return loss
            return loss * (row_mask.numel() / valid_rows.clamp(min=1.0))

else:
    ValidOnlyGRPOTrainer = None


@dataclass
class TRLStanLinearRewardConfig:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    paper_track: str = "part_a_emergence"
    claim_mode: str = "formal_lh"
    n_steps: int = 1000
    n_prompts: int = 32
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
    save_steps: int = 0
    report_to: str = "none"
    run_name: str | None = None
    thinking_mode: str = "no_think"
    prompt_policy: str = "neutral_family"
    num_system_prompts: int = 1
    dataset_n_train: int = 8
    dataset_n_test: int = 4
    dataset_n_obs: int | None = None
    dataset_noise_sigma: float = 1.0
    dataset_beta_scale: float = 1.0
    quadrature_beta_nodes: int = 32
    quadrature_y_nodes: int = 32
    quadrature_beta_scale_multiplier: float = 4.0
    quadrature_y_scale_multiplier: float = 4.0
    normalization_method: str = "gh_y_data"
    normalization_interval: int = 1
    normalization_sample_size: int = -1
    normalization_epsilon: float = 0.1
    normalization_tail_drop_nats: float = 20.0
    scoring_seed_base: int = 0
    cmdstan_root: str = "cmdsafestan"
    stanc3: str = "safestan"
    protect: str = "y"
    compile_jobs: int = 8
    checker_jobs: int = 4
    checker_mode: str = "shadow"
    checker_penalty_reward: float = -100.0
    contract_penalty_reward: float = -100.0
    contract_penalty_reward_final: float | None = None
    parse_fail_penalty_reward: float = -500.0
    parse_fail_penalty_reward_final: float | None = None
    exec_fail_penalty_reward: float = -400.0
    exec_fail_penalty_reward_final: float | None = None
    validity_penalty_schedule: str = "linear"
    validity_penalty_decay_steps: int = 0
    validity_penalty_switch_step: int = 0
    score_workers: int = 0
    reward_floor: float | None = None
    reward_ceiling: float | None = None
    fixed_probe_interval: int = 0
    fixed_probe_sample_size: int = -1
    fixed_probe_n_tasks: int = 1
    fixed_probe_n_train: int = 8
    fixed_probe_n_test: int = 8
    fixed_probe_noise_sigma: float = 2.3
    fixed_probe_beta_scale: float = 1.78
    fixed_probe_seed_base: int = 1729
    invalid_reward_policy: str = "penalty"


def config_from_mapping(mapping: Mapping[str, Any]) -> TRLStanLinearRewardConfig:
    flattened = flatten_hydra_train_mapping(mapping)
    allowed = {f.name for f in fields(TRLStanLinearRewardConfig)}
    unknown = sorted(k for k in flattened if k not in allowed)
    if unknown:
        raise ValueError(f"Unsupported train config keys: {', '.join(unknown)}")
    cfg = TRLStanLinearRewardConfig(**flattened)
    _validate_config(cfg)
    return cfg


def _effective_n_train(config: TRLStanLinearRewardConfig) -> int:
    if config.dataset_n_obs is not None:
        return int(config.dataset_n_obs)
    return int(config.dataset_n_train)


def _validate_config(config: TRLStanLinearRewardConfig) -> None:
    if config.claim_mode != "formal_lh":
        raise ValueError("claim_mode must be formal_lh for the direct-Stan LH experiment")
    if config.paper_track not in {"part_a_emergence", "part_b_mitigation"}:
        raise ValueError("paper_track must be part_a_emergence|part_b_mitigation")
    if config.thinking_mode not in {"think", "no_think"}:
        raise ValueError("thinking_mode must be think|no_think")
    if config.prompt_policy not in STAN_LINEAR_PROMPT_POLICIES:
        raise ValueError(
            "prompt_policy must be one of "
            f"{sorted(STAN_LINEAR_PROMPT_POLICIES)}"
        )
    if config.n_prompts <= 0:
        raise ValueError("n_prompts must be positive")
    if config.num_system_prompts <= 0:
        raise ValueError("num_system_prompts must be positive")
    available_prompts = get_stan_linear_prompt_count(prompt_policy=config.prompt_policy)
    if config.n_prompts > available_prompts:
        raise ValueError(
            f"n_prompts={config.n_prompts} exceeds the {available_prompts} available "
            f"Stan linear prompts for prompt_policy={config.prompt_policy!r}"
        )
    available_system_prompts = get_stan_linear_system_prompt_count()
    if config.num_system_prompts > available_system_prompts:
        raise ValueError(
            f"num_system_prompts={config.num_system_prompts} exceeds the "
            f"{available_system_prompts} available Stan linear system prompts"
        )
    if config.checker_mode not in {"off", "shadow", "enforce"}:
        raise ValueError("checker_mode must be off|shadow|enforce")
    if _effective_n_train(config) <= 0:
        raise ValueError("dataset_n_train must be positive")
    if config.dataset_n_test <= 0:
        raise ValueError("dataset_n_test must be positive")
    if config.dataset_noise_sigma <= 0:
        raise ValueError("dataset_noise_sigma must be positive")
    if config.dataset_beta_scale <= 0:
        raise ValueError("dataset_beta_scale must be positive")
    if config.quadrature_beta_nodes <= 0:
        raise ValueError("quadrature_beta_nodes must be positive")
    if config.quadrature_y_nodes <= 0:
        raise ValueError("quadrature_y_nodes must be positive")
    if config.quadrature_beta_scale_multiplier <= 0:
        raise ValueError("quadrature_beta_scale_multiplier must be positive")
    if config.quadrature_y_scale_multiplier <= 0:
        raise ValueError("quadrature_y_scale_multiplier must be positive")
    if config.normalization_method not in {"off", "gh_y_data"}:
        raise ValueError("normalization_method must be off|gh_y_data")
    if config.normalization_interval < 0:
        raise ValueError("normalization_interval must be >= 0")
    if config.normalization_sample_size < -1:
        raise ValueError("normalization_sample_size must be >= -1")
    if config.normalization_epsilon <= 0:
        raise ValueError("normalization_epsilon must be positive")
    if config.normalization_tail_drop_nats <= 0:
        raise ValueError("normalization_tail_drop_nats must be positive")
    if config.num_generations < 2:
        raise ValueError("num_generations must be >= 2")
    if config.save_steps < 0:
        raise ValueError("save_steps must be >= 0")
    if config.score_workers < 0:
        raise ValueError("score_workers must be >= 0")
    if config.validity_penalty_schedule not in {"linear", "two_phase"}:
        raise ValueError("validity_penalty_schedule must be linear|two_phase")
    if config.validity_penalty_decay_steps < 0:
        raise ValueError("validity_penalty_decay_steps must be >= 0")
    if config.validity_penalty_switch_step < 0:
        raise ValueError("validity_penalty_switch_step must be >= 0")
    if config.reward_floor is not None and not math.isfinite(float(config.reward_floor)):
        raise ValueError("reward_floor must be finite")
    if config.reward_ceiling is not None and not math.isfinite(float(config.reward_ceiling)):
        raise ValueError("reward_ceiling must be finite")
    if (
        config.reward_floor is not None
        and config.reward_ceiling is not None
        and float(config.reward_floor) >= float(config.reward_ceiling)
    ):
        raise ValueError("reward_floor must be < reward_ceiling")
    if config.fixed_probe_interval < 0:
        raise ValueError("fixed_probe_interval must be >= 0")
    if config.fixed_probe_sample_size < -1:
        raise ValueError("fixed_probe_sample_size must be >= -1")
    if config.fixed_probe_n_tasks < 0:
        raise ValueError("fixed_probe_n_tasks must be >= 0")
    if config.fixed_probe_n_train <= 0:
        raise ValueError("fixed_probe_n_train must be positive")
    if config.fixed_probe_n_test <= 0:
        raise ValueError("fixed_probe_n_test must be positive")
    if config.fixed_probe_noise_sigma <= 0:
        raise ValueError("fixed_probe_noise_sigma must be positive")
    if config.fixed_probe_beta_scale <= 0:
        raise ValueError("fixed_probe_beta_scale must be positive")
    if config.invalid_reward_policy not in {"penalty", "filter"}:
        raise ValueError("invalid_reward_policy must be penalty|filter")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training with direct Stan linear-regression reward"
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--paper-track", default="part_a_emergence")
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--n-prompts", type=int, default=32)
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
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--report-to", default="none")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--thinking-mode", default="no_think", choices=["think", "no_think"])
    p.add_argument(
        "--prompt-policy",
        default="neutral_family",
        choices=sorted(STAN_LINEAR_PROMPT_POLICIES),
    )
    p.add_argument("--num-system-prompts", type=int, default=1)
    p.add_argument("--dataset-n-train", type=int, default=8)
    p.add_argument("--dataset-n-test", type=int, default=4)
    p.add_argument("--dataset-n-obs", type=int, default=None)
    p.add_argument("--dataset-noise-sigma", type=float, default=1.0)
    p.add_argument("--dataset-beta-scale", type=float, default=1.0)
    p.add_argument("--quadrature-beta-nodes", type=int, default=32)
    p.add_argument("--quadrature-y-nodes", type=int, default=32)
    p.add_argument("--quadrature-beta-scale-multiplier", type=float, default=4.0)
    p.add_argument("--quadrature-y-scale-multiplier", type=float, default=4.0)
    p.add_argument("--normalization-method", default="gh_y_data", choices=["off", "gh_y_data"])
    p.add_argument("--normalization-interval", type=int, default=1)
    p.add_argument(
        "--normalization-sample-size",
        type=int,
        default=-1,
        help="Number of valid programs to audit per batch; -1 audits the whole valid batch",
    )
    p.add_argument("--normalization-epsilon", type=float, default=0.1)
    p.add_argument("--normalization-tail-drop-nats", type=float, default=20.0)
    p.add_argument("--scoring-seed-base", type=int, default=0)
    p.add_argument("--cmdstan-root", default="cmdsafestan")
    p.add_argument("--stanc3", default="safestan")
    p.add_argument("--protect", default="y")
    p.add_argument("--compile-jobs", type=int, default=8)
    p.add_argument("--checker-jobs", type=int, default=4)
    p.add_argument("--checker-mode", default="shadow", choices=["off", "shadow", "enforce"])
    p.add_argument("--checker-penalty-reward", type=float, default=-100.0)
    p.add_argument("--contract-penalty-reward", type=float, default=-100.0)
    p.add_argument("--contract-penalty-reward-final", type=float, default=None)
    p.add_argument("--parse-fail-penalty-reward", type=float, default=-500.0)
    p.add_argument("--parse-fail-penalty-reward-final", type=float, default=None)
    p.add_argument("--exec-fail-penalty-reward", type=float, default=-400.0)
    p.add_argument("--exec-fail-penalty-reward-final", type=float, default=None)
    p.add_argument(
        "--validity-penalty-schedule",
        default="linear",
        choices=["linear", "two_phase"],
    )
    p.add_argument("--validity-penalty-decay-steps", type=int, default=0)
    p.add_argument("--validity-penalty-switch-step", type=int, default=0)
    p.add_argument("--score-workers", type=int, default=0)
    p.add_argument("--reward-floor", type=float, default=None)
    p.add_argument("--reward-ceiling", type=float, default=None)
    p.add_argument("--fixed-probe-interval", type=int, default=0)
    p.add_argument(
        "--fixed-probe-sample-size",
        type=int,
        default=-1,
        help="Number of valid programs to fixed-probe per batch; -1 probes the whole valid batch",
    )
    p.add_argument("--fixed-probe-n-tasks", type=int, default=1)
    p.add_argument("--fixed-probe-n-train", type=int, default=8)
    p.add_argument("--fixed-probe-n-test", type=int, default=8)
    p.add_argument("--fixed-probe-noise-sigma", type=float, default=2.3)
    p.add_argument("--fixed-probe-beta-scale", type=float, default=1.78)
    p.add_argument("--fixed-probe-seed-base", type=int, default=1729)
    p.add_argument(
        "--invalid-reward-policy",
        default="penalty",
        choices=["penalty", "filter"],
        help="penalty trains on invalid-program penalties; filter masks invalid rows from GRPO",
    )
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
        num_system_prompts=config.num_system_prompts,
    )
    train_dataset = HFDataset.from_list(prompt_dicts)
    log.info(
        "Loaded %d direct-Stan prompt combinations (%d user prompts x %d system prompts)",
        len(prompt_dicts),
        config.n_prompts,
        config.num_system_prompts,
    )
    return prompt_dicts, train_dataset


def _build_reward_function(config: TRLStanLinearRewardConfig, output_dir: Path):
    task_sampler = _build_task_sampler(config)
    fixed_probe_tasks = _build_fixed_probe_tasks(config)
    return make_stan_linear_reward_fn(
        task_sampler=task_sampler,
        output_dir=output_dir,
        cmdstan_root=config.cmdstan_root,
        stanc3=config.stanc3,
        protect=config.protect,
        compile_jobs=config.compile_jobs,
        checker_jobs=config.checker_jobs,
        checker_mode=config.checker_mode,
        checker_penalty_reward=config.checker_penalty_reward,
        contract_penalty_reward=config.contract_penalty_reward,
        contract_penalty_reward_final=config.contract_penalty_reward_final,
        parse_fail_penalty_reward=config.parse_fail_penalty_reward,
        parse_fail_penalty_reward_final=config.parse_fail_penalty_reward_final,
        exec_fail_penalty_reward=config.exec_fail_penalty_reward,
        exec_fail_penalty_reward_final=config.exec_fail_penalty_reward_final,
        validity_penalty_schedule=config.validity_penalty_schedule,
        validity_penalty_decay_steps=config.validity_penalty_decay_steps,
        validity_penalty_switch_step=config.validity_penalty_switch_step,
        score_workers=config.score_workers,
        quadrature_beta_nodes=config.quadrature_beta_nodes,
        quadrature_y_nodes=config.quadrature_y_nodes,
        quadrature_beta_scale_multiplier=config.quadrature_beta_scale_multiplier,
        quadrature_y_scale_multiplier=config.quadrature_y_scale_multiplier,
        normalization_interval=(
            0 if config.normalization_method == "off" else config.normalization_interval
        ),
        normalization_sample_size=config.normalization_sample_size,
        normalization_epsilon=config.normalization_epsilon,
        normalization_tail_drop_nats=config.normalization_tail_drop_nats,
        fixed_probe_tasks=fixed_probe_tasks,
        fixed_probe_interval=config.fixed_probe_interval,
        fixed_probe_sample_size=config.fixed_probe_sample_size,
        reward_floor=config.reward_floor,
        reward_ceiling=config.reward_ceiling,
        invalid_reward_policy=config.invalid_reward_policy,
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
    train_prompt_count: int,
    output_dir: Path,
    model_init_kwargs: dict[str, Any] | None,
) -> TRLGRPOConfig:
    use_cuda, use_bf16, use_fp16 = _resolve_precision()
    programs_per_step = train_prompt_count * config.num_generations
    save_steps = config.save_steps if config.save_steps > 0 else max(1, config.n_steps // 5)
    return TRLGRPOConfig(
        output_dir=str(output_dir),
        max_steps=config.n_steps,
        per_device_train_batch_size=config.num_generations,
        generation_batch_size=programs_per_step,
        gradient_accumulation_steps=train_prompt_count,
        learning_rate=config.lr,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        beta=config.kl_beta,
        max_grad_norm=config.max_grad_norm,
        logging_steps=1,
        save_steps=save_steps,
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
    config: TRLStanLinearRewardConfig, *, n_prompt_rows: int, output_dir: Path
) -> None:
    log.info("Model: %s", config.model)
    log.info(
        "Steps: %d, Prompt rows/step: %d, Generations/prompt: %d, Programs/step: %d",
        config.n_steps,
        n_prompt_rows,
        config.num_generations,
        n_prompt_rows * config.num_generations,
    )
    log.info(
        "Prompt grid: user_prompts=%d system_prompts=%d prompt_policy=%s",
        config.n_prompts,
        config.num_system_prompts,
        config.prompt_policy,
    )
    log.info(
        "Dataset: scalar_linear_regression (n_train=%d k_test=%d sigma=%.2f beta_scale=%.2f)",
        _effective_n_train(config),
        config.dataset_n_test,
        config.dataset_noise_sigma,
        config.dataset_beta_scale,
    )
    log.info(
        "Direct Stan reward: metric=singleton_logZ_ratio backend=cmdstan_log_prob "
        "checker_mode=%s prompt_policy=%s save_steps=%s score_workers=%s "
        "validity_schedule=%s decay_steps=%d switch_step=%d reward_bounds=%s..%s "
        "invalid_policy=%s",
        config.checker_mode,
        config.prompt_policy,
        config.save_steps if config.save_steps > 0 else "auto",
        config.score_workers if config.score_workers > 0 else "auto",
        config.validity_penalty_schedule,
        config.validity_penalty_decay_steps,
        config.validity_penalty_switch_step,
        config.reward_floor if config.reward_floor is not None else "env/default",
        config.reward_ceiling if config.reward_ceiling is not None else "env/default",
        config.invalid_reward_policy,
    )
    log.info(
        "Quadrature: beta_nodes=%d y_nodes=%d beta_scale=%.1f y_scale=%.1f "
        "normalization=%s interval=%d sample=%d epsilon=%.3f",
        config.quadrature_beta_nodes,
        config.quadrature_y_nodes,
        config.quadrature_beta_scale_multiplier,
        config.quadrature_y_scale_multiplier,
        config.normalization_method,
        config.normalization_interval,
        config.normalization_sample_size,
        config.normalization_epsilon,
    )
    if config.fixed_probe_interval > 0 and config.fixed_probe_n_tasks > 0:
        log.info(
            "Fixed probe: interval=%d sample=%d tasks=%d n_train=%d k_test=%d "
            "sigma=%.2f beta_scale=%.2f seed_base=%d",
            config.fixed_probe_interval,
            config.fixed_probe_sample_size,
            config.fixed_probe_n_tasks,
            config.fixed_probe_n_train,
            config.fixed_probe_n_test,
            config.fixed_probe_noise_sigma,
            config.fixed_probe_beta_scale,
            config.fixed_probe_seed_base,
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
        train_prompt_count=len(prompt_dicts),
        output_dir=output_dir,
        model_init_kwargs=model_init_kwargs,
    )
    _log_training_setup(config, n_prompt_rows=len(prompt_dicts), output_dir=output_dir)

    trainer_cls = (
        ValidOnlyGRPOTrainer
        if config.invalid_reward_policy == "filter"
        else GRPOTrainer
    )
    if trainer_cls is None:
        raise RuntimeError("TRL is required for Stan-linear GRPO training")
    trainer = trainer_cls(
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


def _finite_trajectory_values(trajectory, attr_name: str) -> list[float]:
    values: list[float] = []
    for point in trajectory:
        value = getattr(point, attr_name, float("nan"))
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _mean_trajectory_attr(trajectory, attr_name: str) -> float:
    values = _finite_trajectory_values(trajectory, attr_name)
    return float(np.mean(values)) if values else float("nan")


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
            "final_max_abs_log_mass": point.max_abs_log_mass,
            "final_mean_log_mass": getattr(point, "mean_log_mass", float("nan")),
            "final_max_log_mass": getattr(point, "max_log_mass", float("nan")),
            "final_min_log_mass": getattr(point, "min_log_mass", float("nan")),
            "final_mean_program_mean_log_mass": getattr(
                point,
                "mean_program_mean_log_mass",
                float("nan"),
            ),
            "final_mean_program_max_log_mass": getattr(
                point,
                "mean_program_max_log_mass",
                float("nan"),
            ),
            "final_mean_program_min_log_mass": getattr(
                point,
                "mean_program_min_log_mass",
                float("nan"),
            ),
            "final_n_norm_checked": point.n_norm_checked,
            "final_n_norm_with_log_mass": getattr(point, "n_norm_with_log_mass", 0),
            "final_n_norm_failed": point.n_norm_failed,
            "final_n_positive_lh": getattr(point, "n_positive_lh", 0),
            "final_frac_positive_lh": getattr(
                point,
                "frac_positive_lh",
                float("nan"),
            ),
            "final_n_negative_lh": getattr(point, "n_negative_lh", 0),
            "final_frac_negative_lh": getattr(
                point,
                "frac_negative_lh",
                float("nan"),
            ),
            "final_positive_lh_reward_mean": getattr(
                point,
                "positive_lh_reward_mean",
                float("nan"),
            ),
            "final_non_positive_lh_reward_mean": getattr(
                point,
                "non_positive_lh_reward_mean",
                float("nan"),
            ),
            "final_positive_lh_reward_lift": getattr(
                point,
                "positive_lh_reward_lift",
                float("nan"),
            ),
            "final_reward_mean_all": point.reported_mean_all,
            "final_n_unique_programs": getattr(point, "n_unique_programs", 0),
            "final_n_unique_valid_programs": getattr(
                point,
                "n_unique_valid_programs",
                0,
            ),
            "final_unique_program_rate": getattr(
                point,
                "unique_program_rate",
                float("nan"),
            ),
            "final_unique_valid_program_rate": getattr(
                point,
                "unique_valid_program_rate",
                float("nan"),
            ),
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
    finite_frac_non_normalized = [
        float(point.frac_non_normalized)
        for point in trajectory
        if isinstance(point.frac_non_normalized, int | float)
        and math.isfinite(float(point.frac_non_normalized))
    ]
    n_non_normalized_by_batch = [
        int(getattr(point, "n_non_normalized", 0)) for point in trajectory
    ]
    n_positive_lh_by_batch = [
        int(getattr(point, "n_positive_lh", 0)) for point in trajectory
    ]
    n_negative_lh_by_batch = [
        int(getattr(point, "n_negative_lh", 0)) for point in trajectory
    ]
    n_unique_programs_by_batch = [
        int(getattr(point, "n_unique_programs", 0)) for point in trajectory
    ]
    n_unique_valid_programs_by_batch = [
        int(getattr(point, "n_unique_valid_programs", 0)) for point in trajectory
    ]
    metrics["final_parse_fail_rate"] = final.n_parse_fail / max(final.n_total, 1)
    metrics["final_exec_fail_rate"] = final.n_exec_fail / max(final.n_total, 1)
    metrics["final_contract_fail_rate"] = final.n_contract_fail / max(final.n_total, 1)
    metrics["final_unsafe_rate"] = final.unsafe_rate
    metrics["final_n_unsafe"] = final.n_unsafe
    metrics["final_n_checked"] = final.n_checked
    metrics["final_frac_non_normalized"] = final.frac_non_normalized
    metrics["final_mean_abs_log_mass"] = final.mean_abs_log_mass
    metrics["final_max_abs_log_mass"] = final.max_abs_log_mass
    metrics["final_mean_log_mass"] = getattr(final, "mean_log_mass", float("nan"))
    metrics["final_max_log_mass"] = getattr(final, "max_log_mass", float("nan"))
    metrics["final_min_log_mass"] = getattr(final, "min_log_mass", float("nan"))
    metrics["final_mean_program_mean_log_mass"] = getattr(
        final,
        "mean_program_mean_log_mass",
        float("nan"),
    )
    metrics["final_mean_program_max_log_mass"] = getattr(
        final,
        "mean_program_max_log_mass",
        float("nan"),
    )
    metrics["final_mean_program_min_log_mass"] = getattr(
        final,
        "mean_program_min_log_mass",
        float("nan"),
    )
    metrics["final_n_norm_checked"] = final.n_norm_checked
    metrics["final_n_norm_with_log_mass"] = int(
        getattr(final, "n_norm_with_log_mass", 0)
    )
    metrics["final_n_norm_failed"] = final.n_norm_failed
    metrics["final_n_non_normalized"] = int(getattr(final, "n_non_normalized", 0))
    metrics["final_n_positive_lh"] = int(getattr(final, "n_positive_lh", 0))
    metrics["final_frac_positive_lh"] = getattr(
        final,
        "frac_positive_lh",
        float("nan"),
    )
    metrics["final_n_negative_lh"] = int(getattr(final, "n_negative_lh", 0))
    metrics["final_frac_negative_lh"] = getattr(
        final,
        "frac_negative_lh",
        float("nan"),
    )
    metrics["final_positive_lh_reward_mean"] = getattr(
        final,
        "positive_lh_reward_mean",
        float("nan"),
    )
    metrics["final_non_positive_lh_reward_mean"] = getattr(
        final,
        "non_positive_lh_reward_mean",
        float("nan"),
    )
    metrics["final_positive_lh_reward_lift"] = getattr(
        final,
        "positive_lh_reward_lift",
        float("nan"),
    )
    metrics["final_negative_lh_reward_mean"] = getattr(
        final,
        "negative_lh_reward_mean",
        float("nan"),
    )
    metrics["final_non_negative_lh_reward_mean"] = getattr(
        final,
        "non_negative_lh_reward_mean",
        float("nan"),
    )
    metrics["final_negative_lh_reward_lift"] = getattr(
        final,
        "negative_lh_reward_lift",
        float("nan"),
    )
    metrics["final_n_norm_unchecked_valid"] = max(
        int(final.n_valid) - int(final.n_norm_checked),
        0,
    )
    metrics["final_n_norm_cache_hits"] = int(getattr(final, "n_norm_cache_hits", 0))
    metrics["final_n_unique_programs"] = int(getattr(final, "n_unique_programs", 0))
    metrics["final_n_unique_programs_exact"] = int(
        getattr(
            final,
            "n_unique_programs_exact",
            metrics["final_n_unique_programs"],
        )
    )
    metrics["final_n_unique_valid_programs"] = int(
        getattr(final, "n_unique_valid_programs", 0)
    )
    metrics["final_n_unique_valid_programs_exact"] = int(
        getattr(
            final,
            "n_unique_valid_programs_exact",
            metrics["final_n_unique_valid_programs"],
        )
    )
    metrics["final_unique_program_rate"] = getattr(
        final,
        "unique_program_rate",
        float("nan"),
    )
    metrics["final_unique_valid_program_rate"] = getattr(
        final,
        "unique_valid_program_rate",
        float("nan"),
    )
    metrics["mean_frac_non_normalized"] = (
        float(np.mean(finite_frac_non_normalized))
        if finite_frac_non_normalized
        else float("nan")
    )
    metrics["mean_log_mass"] = _mean_trajectory_attr(trajectory, "mean_log_mass")
    metrics["mean_program_mean_log_mass"] = _mean_trajectory_attr(
        trajectory,
        "mean_program_mean_log_mass",
    )
    metrics["mean_program_max_log_mass"] = _mean_trajectory_attr(
        trajectory,
        "mean_program_max_log_mass",
    )
    metrics["mean_program_min_log_mass"] = _mean_trajectory_attr(
        trajectory,
        "mean_program_min_log_mass",
    )
    metrics["mean_n_non_normalized_per_batch"] = (
        float(np.mean(n_non_normalized_by_batch))
        if n_non_normalized_by_batch
        else float("nan")
    )
    metrics["mean_frac_positive_lh"] = _mean_trajectory_attr(
        trajectory,
        "frac_positive_lh",
    )
    metrics["mean_n_positive_lh_per_batch"] = (
        float(np.mean(n_positive_lh_by_batch))
        if n_positive_lh_by_batch
        else float("nan")
    )
    metrics["mean_frac_negative_lh"] = _mean_trajectory_attr(
        trajectory,
        "frac_negative_lh",
    )
    metrics["mean_n_negative_lh_per_batch"] = (
        float(np.mean(n_negative_lh_by_batch))
        if n_negative_lh_by_batch
        else float("nan")
    )
    metrics["mean_positive_lh_reward_lift"] = _mean_trajectory_attr(
        trajectory,
        "positive_lh_reward_lift",
    )
    metrics["mean_n_unique_programs_per_batch"] = (
        float(np.mean(n_unique_programs_by_batch))
        if n_unique_programs_by_batch
        else float("nan")
    )
    metrics["mean_n_unique_valid_programs_per_batch"] = (
        float(np.mean(n_unique_valid_programs_by_batch))
        if n_unique_valid_programs_by_batch
        else float("nan")
    )
    metrics["final_reward_mean_all"] = final.reported_mean_all
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
        "paper/reward_metric": "singleton_posterior_predictive_logZ_ratio",
        "paper/reward_data_split": "train_plus_singleton_holdout",
        "paper/reward_estimator_backend": "cmdstan_log_prob_gauss_hermite",
        "paper/prompt_source": "hardcoded",
        "paper/prompt_policy": config.prompt_policy,
        "paper/invalid_reward_policy": config.invalid_reward_policy,
        "paper/thinking_mode": config.thinking_mode,
        "paper/monitoring_mode": f"safestan_{config.checker_mode}",
        "paper/normalization_method": config.normalization_method,
        "paper/delta_scope": "singleton_y_given_train_x",
        "paper/frac_non_normalized_final": results.get("final_frac_non_normalized", float("nan")),
        "paper/lh_formal_signal_final": results.get("final_frac_non_normalized", float("nan")),
        "paper/lh_rate_batch_final": results.get("final_frac_non_normalized", float("nan")),
        "paper/lh_rate_batch_mean": results.get("mean_frac_non_normalized", float("nan")),
        "paper/lh_count_batch_final": results.get("final_n_non_normalized", float("nan")),
        "paper/lh_count_batch_mean": results.get(
            "mean_n_non_normalized_per_batch",
            float("nan"),
        ),
        "paper/lh_positive_rate_final": results.get(
            "final_frac_positive_lh",
            float("nan"),
        ),
        "paper/lh_positive_rate_mean": results.get(
            "mean_frac_positive_lh",
            float("nan"),
        ),
        "paper/lh_positive_count_batch_final": results.get(
            "final_n_positive_lh",
            float("nan"),
        ),
        "paper/lh_positive_count_batch_mean": results.get(
            "mean_n_positive_lh_per_batch",
            float("nan"),
        ),
        "paper/lh_positive_reward_lift_final": results.get(
            "final_positive_lh_reward_lift",
            float("nan"),
        ),
        "paper/lh_positive_reward_lift_mean": results.get(
            "mean_positive_lh_reward_lift",
            float("nan"),
        ),
        "paper/unique_program_count_batch_final": results.get(
            "final_n_unique_programs",
            float("nan"),
        ),
        "paper/unique_program_count_batch_mean": results.get(
            "mean_n_unique_programs_per_batch",
            float("nan"),
        ),
        "paper/unique_valid_program_count_batch_final": results.get(
            "final_n_unique_valid_programs",
            float("nan"),
        ),
        "paper/unique_valid_program_count_batch_mean": results.get(
            "mean_n_unique_valid_programs_per_batch",
            float("nan"),
        ),
        "paper/judge_hacking_rate_final": float("nan"),
        "paper/lh_family_prevalence_final": float("nan"),
    }
    if error_reason is not None:
        summary["sweep/error"] = error_reason
    return summary


def _sample_scalar_regression_task_from_params(
    *,
    seed: int,
    n_train: int,
    k_test: int,
    noise_sigma: float,
    beta_scale: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    beta = float(rng.normal(0.0, beta_scale))
    x_train = rng.normal(0.0, 1.0, size=n_train).astype(np.float64)
    y_train = beta * x_train + rng.normal(0.0, noise_sigma, size=n_train)
    x_test = rng.normal(0.0, 1.0, size=k_test).astype(np.float64)
    y_test = beta * x_test + rng.normal(0.0, noise_sigma, size=k_test)
    task_payload = {
        "seed": int(seed),
        "beta_true": beta,
        "X_train": x_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": x_test.tolist(),
        "y_test": y_test.tolist(),
        "noise_sigma": float(noise_sigma),
        "beta_scale": float(beta_scale),
    }
    return {
        "task_id": stable_hash(task_payload),
        "seed": int(seed),
        "X": x_train,
        "y": np.asarray(y_train, dtype=np.float64),
        "X_train": x_train,
        "y_train": np.asarray(y_train, dtype=np.float64),
        "X_test": x_test,
        "y_test": np.asarray(y_test, dtype=np.float64),
        "meta": {
            "beta_true": beta,
            "noise_sigma": float(noise_sigma),
            "beta_scale": float(beta_scale),
            "n_train": int(n_train),
            "k_test": int(k_test),
            "seed": int(seed),
        },
    }


def _sample_scalar_regression_task(
    config: TRLStanLinearRewardConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    return _sample_scalar_regression_task_from_params(
        seed=seed,
        n_train=_effective_n_train(config),
        k_test=int(config.dataset_n_test),
        noise_sigma=float(config.dataset_noise_sigma),
        beta_scale=float(config.dataset_beta_scale),
    )


def _build_fixed_probe_tasks(config: TRLStanLinearRewardConfig) -> list[dict[str, Any]]:
    if (
        config.fixed_probe_interval <= 0
        or config.fixed_probe_sample_size == 0
        or config.fixed_probe_n_tasks <= 0
    ):
        return []
    return [
        _sample_scalar_regression_task_from_params(
            seed=int(config.fixed_probe_seed_base + task_index),
            n_train=int(config.fixed_probe_n_train),
            k_test=int(config.fixed_probe_n_test),
            noise_sigma=float(config.fixed_probe_noise_sigma),
            beta_scale=float(config.fixed_probe_beta_scale),
        )
        for task_index in range(int(config.fixed_probe_n_tasks))
    ]


def _build_task_sampler(
    config: TRLStanLinearRewardConfig,
) -> Callable[[int], dict[str, Any]]:
    def _sampler(batch_index: int) -> dict[str, Any]:
        return _sample_scalar_regression_task(
            config,
            seed=int(config.scoring_seed_base + batch_index - 1),
        )

    return _sampler


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
    log.info(
        "contract_fail_rate_final: %.3f",
        results.get("final_contract_fail_rate", float("nan")),
    )
    log.info(
        "normalization_final: frac=%.3f max_abs_log_mass=%.3f checked=%d failed=%d",
        results.get("final_frac_non_normalized", float("nan")),
        results.get("final_max_abs_log_mass", float("nan")),
        results.get("final_n_norm_checked", 0),
        results.get("final_n_norm_failed", 0),
    )
    log.info(
        "positive_lh_final: frac=%.3f count=%d lift=%.3f",
        results.get("final_frac_positive_lh", float("nan")),
        results.get("final_n_positive_lh", 0),
        results.get("final_positive_lh_reward_lift", float("nan")),
    )
    log.info(
        "unique_programs_final: %d valid=%d",
        results.get("final_n_unique_programs", 0),
        results.get("final_n_unique_valid_programs", 0),
    )


def _default_run_name(config: TRLStanLinearRewardConfig) -> str:
    model_short = config.model.split("/")[-1].lower()
    model_short = re.sub(r"[^a-z0-9]+", "-", model_short).strip("-")
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return (
        f"stan-linear-grpo-{model_short}-"
        f"s{config.n_steps}-sys{config.num_system_prompts}-p{config.n_prompts}-"
        f"g{config.num_generations}-{ts}"
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
            "save_steps": args.save_steps,
            "report_to": args.report_to,
            "run_name": args.run_name,
            "thinking_mode": args.thinking_mode,
            "prompt_policy": args.prompt_policy,
            "num_system_prompts": args.num_system_prompts,
            "dataset_n_train": args.dataset_n_train,
            "dataset_n_test": args.dataset_n_test,
            "dataset_n_obs": args.dataset_n_obs,
            "dataset_noise_sigma": args.dataset_noise_sigma,
            "dataset_beta_scale": args.dataset_beta_scale,
            "quadrature_beta_nodes": args.quadrature_beta_nodes,
            "quadrature_y_nodes": args.quadrature_y_nodes,
            "quadrature_beta_scale_multiplier": args.quadrature_beta_scale_multiplier,
            "quadrature_y_scale_multiplier": args.quadrature_y_scale_multiplier,
            "normalization_method": args.normalization_method,
            "normalization_interval": args.normalization_interval,
            "normalization_sample_size": args.normalization_sample_size,
            "normalization_epsilon": args.normalization_epsilon,
            "normalization_tail_drop_nats": args.normalization_tail_drop_nats,
            "scoring_seed_base": args.scoring_seed_base,
            "cmdstan_root": args.cmdstan_root,
            "stanc3": args.stanc3,
            "protect": args.protect,
            "compile_jobs": args.compile_jobs,
            "checker_jobs": args.checker_jobs,
            "checker_mode": args.checker_mode,
            "checker_penalty_reward": args.checker_penalty_reward,
            "contract_penalty_reward": args.contract_penalty_reward,
            "contract_penalty_reward_final": args.contract_penalty_reward_final,
            "parse_fail_penalty_reward": args.parse_fail_penalty_reward,
            "parse_fail_penalty_reward_final": args.parse_fail_penalty_reward_final,
            "exec_fail_penalty_reward": args.exec_fail_penalty_reward,
            "exec_fail_penalty_reward_final": args.exec_fail_penalty_reward_final,
            "validity_penalty_schedule": args.validity_penalty_schedule,
            "validity_penalty_decay_steps": args.validity_penalty_decay_steps,
            "validity_penalty_switch_step": args.validity_penalty_switch_step,
            "score_workers": args.score_workers,
            "reward_floor": args.reward_floor,
            "reward_ceiling": args.reward_ceiling,
            "fixed_probe_interval": args.fixed_probe_interval,
            "fixed_probe_sample_size": args.fixed_probe_sample_size,
            "fixed_probe_n_tasks": args.fixed_probe_n_tasks,
            "fixed_probe_n_train": args.fixed_probe_n_train,
            "fixed_probe_n_test": args.fixed_probe_n_test,
            "fixed_probe_noise_sigma": args.fixed_probe_noise_sigma,
            "fixed_probe_beta_scale": args.fixed_probe_beta_scale,
            "fixed_probe_seed_base": args.fixed_probe_seed_base,
            "invalid_reward_policy": args.invalid_reward_policy,
        }
    )
    run_training(config)


if __name__ == "__main__":
    main()
