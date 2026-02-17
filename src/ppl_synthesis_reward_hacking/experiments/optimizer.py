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
    step: int
    reported_reward: float
    oracle_score: float
    log_mass: float
    observed_count: int
    score_bonus: float
    program: ToyProgram


MUTATION_DELTA_MIN = 0.1
MUTATION_DELTA_MAX = 2.0


def mutate_program(program: ToyProgram, d: int, rng: random.Random) -> ToyProgram:
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
        delta = rng.uniform(MUTATION_DELTA_MIN, MUTATION_DELTA_MAX)
        return ToyProgram(
            observed_idx=program.observed_idx,
            score_bonus=program.score_bonus + delta,
        )

    else:  # dec_bonus
        delta = rng.uniform(MUTATION_DELTA_MIN, min(MUTATION_DELTA_MAX, program.score_bonus))
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
    model = ModelSpec(
        backend="toy",
        name="toy_program",
        source="",
        meta=program.to_meta() | {"d": d},
    )

    compiled = backend.compile(model, cache_dir=Path("/tmp"))
    fit_result = backend.fit(compiled, dataset=dataset, seed=seed)
    return backend.score_holdout(compiled, fit=fit_result, dataset=dataset, seed=seed)


def _make_trajectory_point(step: int, score: ScoreResult, program: ToyProgram) -> TrajectoryPoint:
    return TrajectoryPoint(
        step=step,
        reported_reward=score.reported_reward,
        oracle_score=score.oracle_score if score.oracle_score is not None else 0.0,
        log_mass=score.diagnostics.get("log_mass", 0.0),
        observed_count=program.observed_count,
        score_bonus=program.score_bonus,
        program=program,
    )


def hill_climb_attack(
    backend: ToyBackend,
    dataset: Dataset,
    d: int,
    steps: int = 100,
    seed: int = 0,
    on_step: Callable[[TrajectoryPoint], None] | None = None,
) -> list[TrajectoryPoint]:
    """Hill climber on reported_reward; ignores oracle."""
    rng = random.Random(seed)

    current = ToyProgram.honest(d)
    current_score = evaluate_program(backend, current, dataset, d, seed)
    best_reported = current_score.reported_reward

    trajectory: list[TrajectoryPoint] = []

    point = _make_trajectory_point(0, current_score, current)
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

        point = _make_trajectory_point(t, current_score, current)
        trajectory.append(point)
        if on_step:
            on_step(point)

    return trajectory
