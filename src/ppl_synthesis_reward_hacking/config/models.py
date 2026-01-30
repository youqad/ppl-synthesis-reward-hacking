from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DataConfig:
    generator: str
    params: Mapping[str, Any] = field(default_factory=dict)
    split: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendConfig:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int = 0
    repetitions: int = 1


@dataclass(frozen=True, slots=True)
class AppConfig:
    data: DataConfig
    backend: BackendConfig
    experiment: ExperimentConfig
    run: RunConfig
    raw: Mapping[str, Any] = field(default_factory=dict)
