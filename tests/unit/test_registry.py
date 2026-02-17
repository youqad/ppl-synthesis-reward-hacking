from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.backends.protocol import Backend
from ppl_synthesis_reward_hacking.backends.registry import get_backend

KNOWN_BACKENDS = ("toy", "stan", "sstan", "pymc", "pymc_safe")

EXPECTED_NAMES = {
    "toy": "toy",
    "stan": "stan",
    "sstan": "sstan",
    "pymc": "pymc",
    "pymc_safe": "pymc_safe",
}


@pytest.mark.parametrize("backend_name", KNOWN_BACKENDS)
def test_get_backend_returns_backend_protocol(backend_name):
    backend = get_backend(backend_name)
    assert isinstance(backend, Backend)


@pytest.mark.parametrize("backend_name", KNOWN_BACKENDS)
def test_get_backend_name_attribute_matches(backend_name):
    backend = get_backend(backend_name)
    assert backend.name == EXPECTED_NAMES[backend_name]


def test_get_backend_unknown_raises_key_error():
    with pytest.raises(KeyError, match="Unknown backend"):
        get_backend("nonexistent_backend")


def test_get_backend_empty_string_raises_key_error():
    with pytest.raises(KeyError, match="Unknown backend"):
        get_backend("")


def test_get_backend_returns_fresh_instances():
    a = get_backend("toy")
    b = get_backend("toy")
    assert a is not b


def test_get_backend_toy_has_compile_method():
    backend = get_backend("toy")
    assert callable(getattr(backend, "compile", None))


def test_get_backend_toy_has_fit_method():
    backend = get_backend("toy")
    assert callable(getattr(backend, "fit", None))


def test_get_backend_toy_has_score_holdout_method():
    backend = get_backend("toy")
    assert callable(getattr(backend, "score_holdout", None))
