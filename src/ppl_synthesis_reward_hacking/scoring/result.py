from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ScoreResult:
    reward_train: float
    outcome_code: str
    outcome_detail: str | None
    metric_log_marginal_likelihood: float | None = None
    metric_elpd: float | None = None
    metric_waic: float | None = None
    metric_bic: float | None = None
    decomposition: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "reward_train": self.reward_train,
            "outcome_code": self.outcome_code,
            "outcome_detail": self.outcome_detail,
            "metric_log_marginal_likelihood": self.metric_log_marginal_likelihood,
            "metric_elpd": self.metric_elpd,
            "metric_waic": self.metric_waic,
            "metric_bic": self.metric_bic,
            "decomposition": self.decomposition,
        }
