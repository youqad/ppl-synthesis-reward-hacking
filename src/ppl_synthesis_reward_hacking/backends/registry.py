from __future__ import annotations

from .protocol import Backend


def get_backend(name: str) -> Backend:
    if name == "toy":
        from .toy.backend import ToyBackend

        return ToyBackend()
    if name == "stan":
        from .stan.backend import StanBackend

        return StanBackend()
    if name == "sstan":
        from .sstan.backend import SStanBackend

        return SStanBackend()
    if name == "pymc":
        from .pymc.backend import PyMCBackend

        return PyMCBackend()
    if name == "pymc_safe":
        from .pymc_safe.backend import SafePyMCBackend

        return SafePyMCBackend()
    raise KeyError(f"Unknown backend: {name}")
