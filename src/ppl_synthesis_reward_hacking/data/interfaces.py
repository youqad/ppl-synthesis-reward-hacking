from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataInterfaceSpec:
    dataset_name: str
    delta_scope: str
    response_family: str
    requires_x: bool


_INTERFACE_REGISTRY: dict[str, DataInterfaceSpec] = {
    "bernoulli_1d": DataInterfaceSpec(
        dataset_name="bernoulli_1d",
        delta_scope="raw_y_binary",
        response_family="binary",
        requires_x=False,
    ),
    "bernoulli_vector": DataInterfaceSpec(
        dataset_name="bernoulli_vector",
        delta_scope="raw_y_binary",
        response_family="binary",
        requires_x=False,
    ),
    "linear_regression": DataInterfaceSpec(
        dataset_name="linear_regression",
        delta_scope="raw_y_fixed_x",
        response_family="continuous",
        requires_x=True,
    ),
    "logistic_regression": DataInterfaceSpec(
        dataset_name="logistic_regression",
        delta_scope="raw_y_fixed_x",
        response_family="binary",
        requires_x=True,
    ),
    "gaussian_location": DataInterfaceSpec(
        dataset_name="gaussian_location",
        delta_scope="raw_y",
        response_family="continuous",
        requires_x=False,
    ),
}


def get_interface_spec(dataset_name: str) -> DataInterfaceSpec:
    try:
        return _INTERFACE_REGISTRY[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset interface {dataset_name!r}; expected one of "
            f"{sorted(_INTERFACE_REGISTRY)}"
        ) from exc


def list_dataset_interfaces() -> tuple[str, ...]:
    return tuple(sorted(_INTERFACE_REGISTRY))


def supports_exact_binary(dataset_name: str) -> bool:
    spec = get_interface_spec(dataset_name)
    return spec.delta_scope == "raw_y_binary"


def supports_importance_mc(dataset_name: str) -> bool:
    spec = get_interface_spec(dataset_name)
    return spec.delta_scope == "raw_y_fixed_x"


def expected_norm_method(dataset_name: str) -> str:
    if supports_exact_binary(dataset_name):
        return "exact_binary"
    if supports_importance_mc(dataset_name):
        return "importance_mc"
    return "off"
