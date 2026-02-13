from __future__ import annotations

from collections.abc import Iterable, Sequence

from ppl_synthesis_reward_hacking.utils.collections import unique


def check_pymc_model(model) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    data_vars = _collect_data_vars(model)
    data_set = set(data_vars)

    potentials = getattr(model, "potentials", [])
    if potentials:
        reasons.append("potentials not allowed")

    for rv in getattr(model, "basic_RVs", []):
        op = getattr(getattr(rv, "owner", None), "op", None)
        op_name = op.__class__.__name__ if op is not None else ""
        if "DensityDist" in op_name or "CustomDist" in op_name:
            reasons.append("custom logp not allowed")

    observed = list(getattr(model, "observed_RVs", []))
    rvs_to_values = getattr(model, "rvs_to_values", {})
    data_use_counts = {var: 0 for var in data_vars}
    for rv in observed:
        value_var = rvs_to_values.get(rv)
        if value_var is None:
            continue
        used = _vars_in_graph([value_var], data_set)
        for var in used:
            data_use_counts[var] = data_use_counts.get(var, 0) + 1

    for var, count in data_use_counts.items():
        if count > 1:
            var_name = getattr(var, "name", "data")
            reasons.append(f"data variable '{var_name}' observed more than once")

    for rv in _latent_rvs(model):
        try:
            logp = model.logp(vars=[rv])
        except Exception:
            continue
        if _depends_on_data(logp, data_set):
            reasons.append(f"data used in prior for '{getattr(rv, 'name', 'rv')}'")

    if observed:
        try:
            logp_obs = model.logp(vars=observed)
            if _has_data_dependent_branch(logp_obs, data_set):
                reasons.append("data-dependent branching not allowed")
        except Exception:
            pass

    return (len(reasons) == 0), unique(reasons)


def _collect_data_vars(model) -> list[object]:
    data_vars = list(getattr(model, "data_vars", []) or [])
    if data_vars:
        return data_vars
    named = getattr(model, "named_vars", {})
    for var in named.values():
        if getattr(var, "name", None) is None:
            continue
        if var.__class__.__name__ in {"MutableData", "Data"}:
            data_vars.append(var)
    return data_vars


def _latent_rvs(model) -> list[object]:
    observed = set(getattr(model, "observed_RVs", []))
    return [rv for rv in getattr(model, "basic_RVs", []) if rv not in observed]


def _vars_in_graph(outputs: Sequence[object], targets: set[object]) -> set[object]:
    found: set[object] = set()
    for var in _walk_graph(outputs):
        if var in targets:
            found.add(var)
    return found


def _depends_on_data(var: object, data_vars: set[object]) -> bool:
    for node in _walk_graph([var]):
        if node in data_vars:
            return True
    return False


def _has_data_dependent_branch(var: object, data_vars: set[object]) -> bool:
    for node in _walk_graph([var]):
        owner = getattr(node, "owner", None)
        op = getattr(owner, "op", None)
        if op is None:
            continue
        name = op.__class__.__name__
        scalar_op = getattr(op, "scalar_op", None)
        scalar_name = scalar_op.__class__.__name__ if scalar_op else ""
        if name in {"Switch", "IfElse"} or scalar_name in {"Switch", "IfElse"}:
            inputs = getattr(owner, "inputs", [])
            if inputs and _depends_on_data(inputs[0], data_vars):
                return True
    return False


def _walk_graph(outputs: Sequence[object]) -> Iterable[object]:
    stack = list(outputs)
    seen: set[object] = set()
    while stack:
        var = stack.pop()
        if var in seen:
            continue
        seen.add(var)
        yield var
        owner = getattr(var, "owner", None)
        if owner is None:
            continue
        for inp in getattr(owner, "inputs", []):
            stack.append(inp)


