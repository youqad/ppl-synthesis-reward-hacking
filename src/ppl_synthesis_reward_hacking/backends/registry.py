from __future__ import annotations

import importlib

from .protocol import Backend

_BACKENDS: dict[str, tuple[str, str]] = {
    "toy": (".toy.backend", "ToyBackend"),
    "stan": (".stan.backend", "StanBackend"),
    "sstan": (".sstan.backend", "SStanBackend"),
    "pymc": (".pymc.backend", "PyMCBackend"),
    "pymc_safe": (".pymc_safe.backend", "SafePyMCBackend"),
}


def get_backend(name: str) -> Backend:
    entry = _BACKENDS.get(name)
    if entry is None:
        raise KeyError(f"unknown backend: {name!r} (available: {sorted(_BACKENDS)})")
    module_path, class_name = entry
    module = importlib.import_module(module_path, package=__package__)
    cls = getattr(module, class_name)
    return cls()
