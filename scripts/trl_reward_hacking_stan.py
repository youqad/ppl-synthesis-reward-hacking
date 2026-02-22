#!/usr/bin/env python3
"""GRPO training with Stan reward (coin-toss).

This mirrors `scripts/trl_reward_hacking.py` but:
- prompts ask for Stan programs,
- scoring uses plain CmdStan (no SafeStan enforcement),
- SafeStan is used as checker-only to track unsafe-rate over time.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
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
from ppl_synthesis_reward_hacking.data.stan_reward_loader import load_stan_reward_prompts
from ppl_synthesis_reward_hacking.experiments.results import (
    compute_traj_metrics,
    json_default,
    print_training_summary,
)
from ppl_synthesis_reward_hacking.experiments.stan_reward import (
    make_stan_reward_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class TRLStanRewardConfig:
    model: str = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    n_steps: int = 1000
    n_prompts: int = 100
    rollouts_per_prompt: int = 8
    lora_rank: int = 32
    lora_dropout: float = 0.0
    lr: float = 5e-6
    max_completion_length: int = 1024
    use_4bit: bool = False
    output_dir: str = "artifacts/grpo_stan_reward"
    resume_from: str | None = None
    num_generations: int = 8
    temperature: float = 1.28
    top_p: float = 1.0
    top_k: int = 0
    kl_beta: float = 0.001
    max_grad_norm: float = 1.0
    report_to: str = "wandb"
    run_name: str | None = None

    dataset_name: str = "bernoulli_1d"
    dataset_n_train: int = 32
    dataset_n_holdout: int = 64
    dataset_p_true_alpha: float = 1.0
    dataset_p_true_beta: float = 1.0
    dataset_p_true_fixed: float | None = None
    scoring_seed_base: int = 0

    cmdstan_root: str = "cmdsafestan"
    stanc3: str = "safestan"
    protect: str = "y"
    compile_jobs: int = 8
    checker_jobs: int = 4


def config_from_mapping(mapping: Mapping[str, Any]) -> TRLStanRewardConfig:
    flattened = flatten_hydra_train_mapping(mapping)
    allowed = {f.name for f in fields(TRLStanRewardConfig)}
    return TRLStanRewardConfig(**{k: v for k, v in flattened.items() if k in allowed})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training with Stan reward")
    p.add_argument(
        "--model",
        default="NousResearch/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model ID",
    )
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--rollouts-per-prompt", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--output-dir", default="artifacts/grpo_stan_reward")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.28)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--kl-beta", type=float, default=0.001)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--run-name", type=str, default=None, help="W&B run name")
    p.add_argument("--dataset-n-train", type=int, default=32)
    p.add_argument("--dataset-n-holdout", type=int, default=64)
    p.add_argument("--dataset-p-true-fixed", type=float, default=None)
    p.add_argument("--scoring-seed-base", type=int, default=0)
    p.add_argument("--cmdstan-root", default="cmdsafestan")
    p.add_argument("--stanc3", default="safestan")
    p.add_argument("--protect", default="y")
    p.add_argument("--compile-jobs", type=int, default=8)
    p.add_argument("--checker-jobs", type=int, default=4)
    return p.parse_args()


def run_training(config: TRLStanRewardConfig) -> dict[str, Any]:
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

    prompt_dicts = load_stan_reward_prompts(max_examples=config.n_prompts)
    train_dataset = HFDataset.from_list(prompt_dicts)
    log.info("Loaded %d Stan prompts", len(prompt_dicts))

    train_data, holdout_data = _make_scoring_payload(config)
    reward_fn, reward_state = make_stan_reward_fn(
        train_data=train_data,
        holdout_data=holdout_data,
        output_dir=output_dir,
        cmdstan_root=config.cmdstan_root,
        stanc3=config.stanc3,
        protect=config.protect,
        compile_jobs=config.compile_jobs,
        checker_jobs=config.checker_jobs,
        completions_path=output_dir / "completions.jsonl",
    )

    use_cuda = False
    use_bf16 = False
    use_fp16 = False
    try:
        import torch

        use_cuda = bool(torch.cuda.is_available())
        use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
        use_fp16 = bool(use_cuda and not use_bf16)
    except Exception:  # noqa: BLE001
        # Keep CPU fallback robust in environments without torch CUDA support.
        use_cuda = False
        use_bf16 = False
        use_fp16 = False
    log.info(
        "Precision mode: cuda=%s bf16=%s fp16=%s",
        use_cuda,
        use_bf16,
        use_fp16,
    )

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
        run_name=(config.run_name or _default_run_name(config)),
        remove_unused_columns=False,
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=not use_cuda,
        gradient_checkpointing=False,
        use_vllm=False,
        model_init_kwargs=model_init_kwargs,
    )

    log.info("Model: %s", config.model)
    log.info("Run name: %s", training_args.run_name)
    log.info(
        "Steps: %d, Prompts: %d, Generations/prompt: %d",
        config.n_steps,
        len(prompt_dicts),
        config.num_generations,
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
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=json_default)

    log.info("Results saved to %s", output_dir)
    _print_summary(results)

    if reward_state.completion_writer is not None:
        reward_state.completion_writer.close()

    return results


def _compute_results(config: TRLStanRewardConfig, state) -> dict[str, Any]:
    trajectory = state.trajectory
    metrics = compute_traj_metrics(trajectory)
    if "error" in metrics:
        metrics["n_batches"] = len(trajectory)
        return metrics

    final = trajectory[-1]
    metrics["config"] = asdict(config)
    metrics["final_parse_fail_rate"] = final.n_parse_fail / max(final.n_total, 1)
    metrics["final_exec_fail_rate"] = final.n_exec_fail / max(final.n_total, 1)
    metrics["final_unsafe_rate"] = final.unsafe_rate
    metrics["final_n_unsafe"] = final.n_unsafe
    metrics["final_n_checked"] = final.n_checked
    metrics["final_oracle_mean"] = final.oracle_mean
    metrics["final_gap_mean"] = final.gap_mean
    metrics["n_batches"] = len(trajectory)
    return metrics


def _make_scoring_payload(config: TRLStanRewardConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.dataset_name != "bernoulli_1d":
        raise ValueError("TRL Stan reward currently supports dataset_name=bernoulli_1d only")

    params: dict[str, Any] = {
        "split": {
            "n_train": config.dataset_n_train,
            "n_holdout": config.dataset_n_holdout,
        }
    }
    if config.dataset_p_true_fixed is None:
        rng = np.random.default_rng(config.scoring_seed_base + 13_371)
        params["p_true"] = float(rng.beta(config.dataset_p_true_alpha, config.dataset_p_true_beta))
    else:
        params["p_true"] = float(config.dataset_p_true_fixed)

    dataset = generate_dataset(config.dataset_name, params, seed=config.scoring_seed_base)
    y_train = np.asarray(dataset.train["y"], dtype=np.int64).reshape(-1)
    y_holdout = np.asarray(dataset.holdout["y"], dtype=np.int64).reshape(-1)
    return (
        {"N": int(y_train.shape[0]), "y": y_train.tolist()},
        {"N": int(y_holdout.shape[0]), "y": y_holdout.tolist()},
    )


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


def _default_run_name(config: TRLStanRewardConfig) -> str:
    model_short = config.model.split("/")[-1].lower()
    model_short = re.sub(r"[^a-z0-9]+", "-", model_short).strip("-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return (
        f"stan-grpo-{model_short}-"
        f"s{config.n_steps}-p{config.n_prompts}-g{config.num_generations}-{ts}"
    )


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
            "dataset_n_train": args.dataset_n_train,
            "dataset_n_holdout": args.dataset_n_holdout,
            "dataset_p_true_fixed": args.dataset_p_true_fixed,
            "scoring_seed_base": args.scoring_seed_base,
            "cmdstan_root": args.cmdstan_root,
            "stanc3": args.stanc3,
            "protect": args.protect,
            "compile_jobs": args.compile_jobs,
            "checker_jobs": args.checker_jobs,
        }
    )
    run_training(config)


if __name__ == "__main__":
    main()
