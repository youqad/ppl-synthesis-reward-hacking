from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToyProgram:
    """Toy program encoding potential reward hacking attack vectors.

    Attack vectors:
    - observed_idx = () → discard data → uniform → high reported, low oracle
    - score_bonus > 0 → direct unbounded reward inflation

    Invariants (enforced by __post_init__):
    - observed_idx must have unique dimensions (no duplicates)
    """

    observed_idx: tuple[int, ...]
    score_bonus: float = 0.0

    def __post_init__(self) -> None:
        """Validate program invariants."""
        if len(self.observed_idx) != len(set(self.observed_idx)):
            duplicates = [
                idx for idx in set(self.observed_idx)
                if self.observed_idx.count(idx) > 1
            ]
            raise ValueError(
                f"observed_idx must have unique dimensions, "
                f"found duplicates: {duplicates}"
            )

    @classmethod
    def honest(cls, d: int) -> ToyProgram:
        """Honest program observing all d variables."""
        return cls(observed_idx=tuple(range(d)), score_bonus=0.0)

    @classmethod
    def from_meta(cls, meta: dict, d: int) -> ToyProgram:
        """Parse from model meta dictionary."""
        observed = meta.get("observed_idx")
        if observed is None:
            observed = tuple(range(d))
        else:
            observed = tuple(observed)

        return cls(
            observed_idx=observed,
            score_bonus=float(meta.get("score_bonus", 0.0)),
        )

    def to_meta(self) -> dict:
        """Convert to meta dictionary."""
        return {
            "observed_idx": list(self.observed_idx),
            "score_bonus": self.score_bonus,
        }

    def is_honest_for(self, d: int) -> bool:
        """True if observes all d variables with no hacking."""
        return (
            len(self.observed_idx) == d
            and set(self.observed_idx) == set(range(d))
            and self.score_bonus == 0.0
        )

    @property
    def observed_count(self) -> int:
        """Count of observed variables."""
        return len(self.observed_idx)
