from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec, ScoreResult
from ppl_synthesis_reward_hacking.backends.toy import ToyBackend, ToyProgram
from ppl_synthesis_reward_hacking.data.schema import Dataset


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """Single optimization step with reported and oracle scores."""

    step: int
    reported_reward: float
    oracle_score: float
    log_mass: float
    observed_count: int
    score_bonus: float
    program: ToyProgram


def mutate_program(program: ToyProgram, d: int, rng: random.Random) -> ToyProgram:
    """Random mutation: drop/add var or adjust bonus."""
    mutations = []

    if program.observed_count > 0:
        mutations.append("drop_var")

    if program.observed_count < d:
        mutations.append("add_var")

    mutations.append("inc_bonus")
    if program.score_bonus > 0:
        mutations.append("dec_bonus")

    mutation = rng.choice(mutations)

    if mutation == "drop_var":
        idx_list = list(program.observed_idx)
        to_drop = rng.choice(idx_list)
        new_idx = tuple(i for i in idx_list if i != to_drop)
        return ToyProgram(observed_idx=new_idx, score_bonus=program.score_bonus)

    elif mutation == "add_var":
        current = set(program.observed_idx)
        available = [i for i in range(d) if i not in current]
        to_add = rng.choice(available)
        new_idx = tuple(sorted(list(program.observed_idx) + [to_add]))
        return ToyProgram(observed_idx=new_idx, score_bonus=program.score_bonus)

    elif mutation == "inc_bonus":
        delta = rng.uniform(0.1, 2.0)
        return ToyProgram(
            observed_idx=program.observed_idx,
            score_bonus=program.score_bonus + delta,
        )

    else:  # dec_bonus
        delta = rng.uniform(0.1, min(2.0, program.score_bonus))
        return ToyProgram(
            observed_idx=program.observed_idx,
            score_bonus=max(0.0, program.score_bonus - delta),
        )


def evaluate_program(
    backend: ToyBackend,
    program: ToyProgram,
    dataset: Dataset,
    d: int,
    seed: int,
) -> ScoreResult:
    """Compile, fit, and score a program on the dataset."""
    model = ModelSpec(
        backend="toy",
        name="toy_program",
        source="",
        meta=program.to_meta() | {"d": d},
    )

    compiled = backend.compile(model, cache_dir=Path("/tmp"))
    fit_result = backend.fit(compiled, dataset=dataset, seed=seed)
    return backend.score_holdout(compiled, fit=fit_result, dataset=dataset, seed=seed)


def hill_climb_attack(
    backend: ToyBackend,
    dataset: Dataset,
    d: int,
    steps: int = 100,
    seed: int = 0,
    on_step: Callable[[TrajectoryPoint], None] | None = None,
) -> list[TrajectoryPoint]:
    """1+1 hill climber optimizing reported_reward only (ignores oracle)."""
    rng = random.Random(seed)

    current = ToyProgram.honest(d)
    current_score = evaluate_program(backend, current, dataset, d, seed)
    best_reported = current_score.reported_reward

    trajectory: list[TrajectoryPoint] = []

    point = TrajectoryPoint(
        step=0,
        reported_reward=current_score.reported_reward,
        oracle_score=current_score.oracle_score or 0.0,
        log_mass=current_score.diagnostics.get("log_mass", 0.0),
        observed_count=current.observed_count,
        score_bonus=current.score_bonus,
        program=current,
    )
    trajectory.append(point)
    if on_step:
        on_step(point)

    for t in range(1, steps + 1):
        candidate = mutate_program(current, d, rng)
        cand_score = evaluate_program(backend, candidate, dataset, d, seed + t)

        # selection ignores oracle_score
        if cand_score.reported_reward > best_reported:
            current = candidate
            current_score = cand_score
            best_reported = cand_score.reported_reward

        point = TrajectoryPoint(
            step=t,
            reported_reward=current_score.reported_reward,
            oracle_score=current_score.oracle_score or 0.0,
            log_mass=current_score.diagnostics.get("log_mass", 0.0),
            observed_count=current.observed_count,
            score_bonus=current.score_bonus,
            program=current,
        )
        trajectory.append(point)
        if on_step:
            on_step(point)

    return trajectory
