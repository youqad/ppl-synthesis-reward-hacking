"""Canonical templates for W&B sweep configs checked into configs/sweeps/wandb."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_SWEEP_METRIC = {
    "name": "paper/lh_formal_signal_final",
    "goal": "maximize",
}

_COMMAND_PREFIX = [
    "${env}",
    "${interpreter}",
    "${program}",
    "launcher=local",
    "train/paper_track=part_a_emergence",
    "train/claim_mode=formal_lh",
]

_MONITORING_ON = [
    "train/monitoring_mode=judge_evolving",
    "train.judge_backend=openai",
    "train.judge_model=gpt-5.4",
    "train.monitoring_mode.judge_interval=10",
    "train.monitoring_mode.rubric_evolution_interval=10",
    "train.judge_sampling_policy=fixed_cap",
    "train.judge_sample_size=20",
    "train.judge_batch_mode=auto",
    "train.judge_batch_max_workers=32",
    "train.track_lh_every_batch=true",
]

_MONITORING_OFF = [
    "train/monitoring_mode=off",
    "train.monitoring_mode.judge_interval=0",
    "train.monitoring_mode.rubric_evolution_interval=0",
    "train.track_lh_every_batch=true",
]


def _command(*segments: list[str]) -> list[str]:
    command = list(_COMMAND_PREFIX)
    for segment in segments:
        command.extend(segment)
    command.append("${args_no_hyphens}")
    return command


def _base_sweep(method: str) -> dict[str, Any]:
    return {
        "program": "scripts/hydra_train_tinker.py",
        "method": method,
        "metric": deepcopy(_SWEEP_METRIC),
    }


def render_tinker_parta_sweeps() -> dict[str, dict[str, Any]]:
    """Return canonical Part A tinker sweep YAML payloads keyed by filename."""
    sweeps: dict[str, dict[str, Any]] = {}

    main = _base_sweep("bayes")
    main["early_terminate"] = {"type": "hyperband", "min_iter": 60, "eta": 2}
    main["command"] = _command(
        [
            "train/reward_metric=log_marginal_likelihood",
            "train/reward_data_split=train",
            "train/reward_estimator_backend=smc",
            "train/normalization=exact_binary",
            "train/data_interface=raw_bernoulli_d3",
            "train.data_interface.dataset_n_train=1",
        ],
        _MONITORING_ON,
    )
    main["parameters"] = {
        "train.base_model": {"values": ["Qwen/Qwen3-4B-Instruct-2507"]},
        "train.n_steps": {"values": [300, 600]},
        "train.n_prompts": {"values": [4, 5]},
        "train.rollouts_per_prompt": {"values": [120, 160]},
        "train.learning_rate": {"values": [5.0e-6, 1.0e-5, 2.0e-5]},
        "train.temperature": {"distribution": "uniform", "min": 1.10, "max": 1.35},
        "train.top_p": {"values": [0.92, 0.95, 0.98]},
        "train.top_k": {"values": [30, 50, 80]},
        "train.data_interface.dataset_n_features": {"values": [3, 5]},
        "train.smc_draws": {"values": [300, 500]},
        "train.normalization.normalization_interval": {"values": [1, 2]},
        "train.normalization.normalization_sample_size": {"values": [20, 40]},
        "train.scoring_workers": {"values": [1, 2]},
    }
    sweeps["tinker_partA_lh_main.yaml"] = main

    linear = _base_sweep("grid")
    linear["command"] = _command(
        [
            "train/reward_metric=log_marginal_likelihood",
            "train/reward_data_split=train",
            "train/reward_estimator_backend=smc",
            "train/data_interface=linear_regression",
            "train.data_interface.dataset_n_features=3",
            "train.data_interface.dataset_n_train=20",
            "train.n_steps=300",
            "train.n_prompts=5",
            "train.rollouts_per_prompt=120",
            "train.learning_rate=1.0e-5",
            "train.temperature=1.2",
            "train.top_p=0.95",
            "train.top_k=50",
            "train/normalization=importance_mc",
            "train.normalization.normalization_mc_samples=512",
            "train.normalization.normalization_interval=2",
            "train.normalization.normalization_sample_size=20",
        ],
        _MONITORING_ON,
    )
    linear["parameters"] = {
        "train.base_model": {"values": ["Qwen/Qwen3-4B-Instruct-2507"]},
        "train.scoring_seed_base": {"values": [0, 10000, 20000]},
    }
    sweeps["tinker_partA_linear_extension.yaml"] = linear

    linear_fast = _base_sweep("grid")
    linear_fast["command"] = _command(
        [
            "train/monitoring_mode=judge_evolving",
            "train/reward_metric=log_marginal_likelihood",
            "train/reward_data_split=train",
            "train/reward_estimator_backend=smc",
            "train/data_interface=linear_regression",
            "train.data_interface.dataset_n_features=3",
            "train.data_interface.dataset_n_train=20",
            "train.n_steps=300",
            "train.n_prompts=4",
            "train.rollouts_per_prompt=80",
            "train.learning_rate=1.0e-5",
            "train.temperature=1.2",
            "train.top_p=0.95",
            "train.top_k=50",
            "train.smc_draws=200",
            "train.scoring_workers=2",
            "train/normalization=importance_mc",
            "train.normalization.normalization_mc_samples=128",
            "train.normalization.normalization_interval=10",
            "train.normalization.normalization_sample_size=8",
            "train.judge_backend=openai",
            "train.judge_model=gpt-5.4",
            "train.monitoring_mode.judge_interval=20",
            "train.monitoring_mode.rubric_evolution_interval=20",
            "train.judge_sampling_policy=fixed_cap",
            "train.judge_sample_size=16",
            "train.judge_batch_mode=auto",
            "train.judge_batch_max_workers=16",
            "train.track_lh_every_batch=true",
        ]
    )
    linear_fast["parameters"] = {
        "train.base_model": {"values": ["Qwen/Qwen3-4B-Instruct-2507"]},
        "train.scoring_seed_base": {"values": [0, 10000, 20000]},
    }
    sweeps["tinker_partA_linear_extension_fast.yaml"] = linear_fast

    for metric, backend, steps, file_name in (
        ("log_marginal_likelihood", "smc", 300, "tinker_partA_metric_ablation_logml.yaml"),
        ("elpd", "psis_loo", 300, "tinker_partA_metric_ablation_elpd.yaml"),
        ("waic", "waic", 100, "tinker_partA_metric_ablation_waic.yaml"),
        ("bic", "bic_approx", 300, "tinker_partA_metric_ablation_bic.yaml"),
    ):
        ablation = _base_sweep("grid")
        ablation["command"] = _command(
            _MONITORING_OFF,
            [
                "train/data_interface=raw_bernoulli_d3",
                "train.data_interface.dataset_n_features=3",
                "train.data_interface.dataset_n_train=1",
                f"train/reward_metric={metric}",
                "train/reward_data_split=train",
                f"train/reward_estimator_backend={backend}",
                "train/normalization=exact_binary",
                f"train.n_steps={steps}",
                "train.n_prompts=5",
                "train.rollouts_per_prompt=120",
                "train.learning_rate=1.0e-5",
                "train.temperature=1.2",
                "train.top_p=0.95",
                "train.top_k=50",
            ],
        )
        ablation["parameters"] = {
            "train.base_model": {"values": ["Qwen/Qwen3-4B-Instruct-2507"]},
            "train.scoring_seed_base": {"values": [0, 10000]},
        }
        sweeps[file_name] = ablation

    return sweeps
