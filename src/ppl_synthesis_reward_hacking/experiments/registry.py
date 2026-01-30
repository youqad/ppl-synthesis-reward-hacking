from __future__ import annotations

from collections.abc import Callable

from .protocol import Experiment

_EXPERIMENTS: dict[str, Callable[[], Experiment]] = {}


def register_experiment(name: str, factory: Callable[[], Experiment]) -> None:
    _EXPERIMENTS[name] = factory


def get_experiment(name: str) -> Experiment:
    if name not in _EXPERIMENTS:
        raise KeyError(f"Unknown experiment: {name}")
    return _EXPERIMENTS[name]()
