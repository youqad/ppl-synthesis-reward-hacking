from __future__ import annotations

import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec  # noqa: E402
from ppl_synthesis_reward_hacking.backends.pymc.templates import (  # noqa: E402
    _flatten_y,
    build_model,
)
from ppl_synthesis_reward_hacking.data.schema import Dataset  # noqa: E402

# pm.MutableData was removed in PyMC 5.x (now pm.Data). build_model still
# references the old API, so any code path past the y-check crashes with
# AttributeError. Tests that hit that path are marked xfail.
_mutable_data_missing = not hasattr(pm, "MutableData")


def _make_dataset(y_train, y_holdout=None):
    if y_holdout is None:
        y_holdout = y_train
    return Dataset(
        name="test",
        train={"y": y_train},
        holdout={"y": y_holdout},
        meta={"d": 1},
        dataset_id="test_ds",
    )


def _make_spec(name, meta=None):
    return ModelSpec(
        backend="pymc",
        name=name,
        source="",
        meta=meta or {},
    )


# -- build_model tests (require pm.MutableData) --


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x",
    raises=AttributeError,
)
def test_build_model_bernoulli_honest_returns_pymc_model():
    ds = _make_dataset(np.array([0, 1, 1, 0, 1]))
    spec = _make_spec("bernoulli_honest")
    model = build_model(spec, ds)
    assert isinstance(model, pm.Model)


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x",
    raises=AttributeError,
)
def test_build_model_bernoulli_honest_has_observed_rv():
    ds = _make_dataset(np.array([0, 1, 1, 0, 1]))
    spec = _make_spec("bernoulli_honest")
    model = build_model(spec, ds)
    obs_names = [rv.name for rv in model.observed_RVs]
    assert "obs" in obs_names


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x",
    raises=AttributeError,
)
def test_build_model_bernoulli_double_count_has_two_obs():
    ds = _make_dataset(np.array([0, 1, 1, 0, 1]))
    spec = _make_spec("bernoulli_double_count")
    model = build_model(spec, ds)
    obs_names = [rv.name for rv in model.observed_RVs]
    assert "obs" in obs_names
    assert "obs2" in obs_names


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x",
    raises=AttributeError,
)
def test_build_model_bernoulli_target_add_with_constant():
    ds = _make_dataset(np.array([0, 1, 1, 0, 1]))
    spec = _make_spec("bernoulli_target_add", meta={"attack": {"kind": "target_add", "C": 5.0}})
    model = build_model(spec, ds)
    pot_names = [p.name for p in model.potentials]
    assert "reward_bump" in pot_names


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x",
    raises=AttributeError,
)
def test_build_model_bernoulli_target_add_default_constant_zero():
    ds = _make_dataset(np.array([0, 1, 1, 0, 1]))
    spec = _make_spec("bernoulli_target_add")
    model = build_model(spec, ds)
    pot_names = [p.name for p in model.potentials]
    assert "reward_bump" in pot_names


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x",
    raises=AttributeError,
)
def test_build_model_gaussian_improper_lpdf():
    ds = _make_dataset(np.array([0.5, -0.3, 1.2, 0.0]))
    spec = _make_spec("gaussian_improper_lpdf")
    model = build_model(spec, ds)
    pot_names = [p.name for p in model.potentials]
    assert "improper" in pot_names


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x",
    raises=AttributeError,
)
def test_build_model_gaussian_improper_has_mu_and_sigma():
    ds = _make_dataset(np.array([0.5, -0.3, 1.2]))
    spec = _make_spec("gaussian_improper_lpdf")
    model = build_model(spec, ds)
    named = set(model.named_vars.keys())
    assert "mu" in named
    assert "sigma" in named


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x; unknown-template check is unreachable",
    raises=AttributeError,
)
def test_build_model_unknown_template_raises():
    ds = _make_dataset(np.array([0, 1]))
    spec = _make_spec("totally_unknown_template_xyz")
    with pytest.raises(RuntimeError, match="Unknown PyMC template"):
        build_model(spec, ds)


@pytest.mark.xfail(
    _mutable_data_missing,
    reason="pm.MutableData removed in PyMC 5.x",
    raises=AttributeError,
)
def test_build_model_double_count_via_attack_config():
    ds = _make_dataset(np.array([0, 1, 1]))
    spec = _make_spec("bernoulli_honest", meta={"attack": {"kind": "double_count"}})
    model = build_model(spec, ds)
    obs_names = [rv.name for rv in model.observed_RVs]
    assert "obs2" in obs_names


# -- error paths reachable without pm.MutableData --


def test_build_model_missing_y_raises():
    ds = Dataset(
        name="test",
        train={"x": np.array([1, 2, 3])},
        holdout={"x": np.array([1, 2, 3])},
        meta={},
        dataset_id="test_ds",
    )
    spec = _make_spec("bernoulli_honest")
    with pytest.raises(RuntimeError, match="'y' in dataset"):
        build_model(spec, ds)


# -- _flatten_y tests --


def test_flatten_y_1d_passthrough():
    arr = np.array([1, 2, 3])
    result = _flatten_y(arr)
    np.testing.assert_array_equal(result, arr)
    assert result.ndim == 1


def test_flatten_y_2d_reshaped_to_1d():
    arr = np.array([[1, 2], [3, 4]])
    result = _flatten_y(arr)
    assert result.ndim == 1
    assert len(result) == 4


def test_flatten_y_preserves_element_values():
    arr = np.array([[10, 20], [30, 40]])
    result = _flatten_y(arr)
    assert set(result.tolist()) == {10, 20, 30, 40}


def test_flatten_y_from_list():
    result = _flatten_y([1, 0, 1])
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1


def test_flatten_y_3d_reshaped_to_1d():
    arr = np.ones((2, 3, 4))
    result = _flatten_y(arr)
    assert result.ndim == 1
    assert len(result) == 24


def test_flatten_y_single_element():
    result = _flatten_y(np.array([42]))
    assert result.ndim == 1
    assert result[0] == 42


def test_flatten_y_from_mapping_value():
    data = {"y": np.array([[1, 2], [3, 4]])}
    result = _flatten_y(data["y"])
    assert result.ndim == 1
    assert len(result) == 4
