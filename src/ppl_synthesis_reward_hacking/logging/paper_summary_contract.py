from __future__ import annotations

from typing import Any

PAPER_SUMMARY_KEYS_CORE: tuple[str, ...] = (
    "paper/track",
    "paper/claim_mode",
    "paper/reward_metric",
    "paper/reward_data_split",
    "paper/reward_estimator_backend",
    "paper/prompt_source",
    "paper/prompt_policy",
    "paper/thinking_mode",
    "paper/monitoring_mode",
    "paper/normalization_method",
    "paper/delta_scope",
    "paper/frac_non_normalized_final",
    "paper/lh_formal_signal_final",
    "paper/judge_hacking_rate_final",
    "paper/lh_family_prevalence_final",
)

PAPER_SUMMARY_KEYS_TINKER: tuple[str, ...] = (
    "paper/lh_rate_batch_final",
    "paper/lh_rate_batch_mean",
    "paper/lh_other_novel_final",
    "paper/lh_other_novel_mean",
)

PAPER_SUMMARY_KEYS_ALL: tuple[str, ...] = PAPER_SUMMARY_KEYS_CORE + PAPER_SUMMARY_KEYS_TINKER

PAPER_SUMMARY_DEFAULTS: dict[str, Any] = {
    "paper/track": "unknown",
    "paper/claim_mode": "unknown",
    "paper/reward_metric": "unknown",
    "paper/reward_data_split": "unknown",
    "paper/reward_estimator_backend": "unknown",
    "paper/prompt_source": "unknown",
    "paper/prompt_policy": "unknown",
    "paper/thinking_mode": "unknown",
    "paper/monitoring_mode": "unknown",
    "paper/normalization_method": "unknown",
    "paper/delta_scope": "unknown",
    "paper/frac_non_normalized_final": float("nan"),
    "paper/lh_formal_signal_final": float("nan"),
    "paper/judge_hacking_rate_final": float("nan"),
    "paper/lh_family_prevalence_final": float("nan"),
    "paper/lh_rate_batch_final": float("nan"),
    "paper/lh_rate_batch_mean": float("nan"),
    "paper/lh_other_novel_final": float("nan"),
    "paper/lh_other_novel_mean": float("nan"),
}


def paper_summary_keys_for_train_name(train_name: str) -> tuple[str, ...]:
    if str(train_name).strip().lower() == "tinker":
        return PAPER_SUMMARY_KEYS_ALL
    return PAPER_SUMMARY_KEYS_CORE


__all__ = [
    "PAPER_SUMMARY_KEYS_CORE",
    "PAPER_SUMMARY_KEYS_TINKER",
    "PAPER_SUMMARY_KEYS_ALL",
    "PAPER_SUMMARY_DEFAULTS",
    "paper_summary_keys_for_train_name",
]
