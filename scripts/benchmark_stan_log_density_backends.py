#!/usr/bin/env python3
"""Benchmark CmdStan-via-cmdsafestan against BridgeStan for tiny log-density quadrature.

This is a small standalone harness for the paper-style scalar regression setting:

  q(beta, y | x) = exp(lp(beta; x, y))

with `N = 1`, scalar `beta`, scalar `x`, and scalar `y`.  We benchmark three
toy Stan models:

1. Honest normalized model.
2. A hack that drops normalizing constants via `_lupdf`.
3. A hack that doubles the likelihood contribution.

For each model, we estimate the total mass

    Z = integral over beta and y of exp(lp(beta; x, y))

using tensor-product Gauss-Hermite quadrature and compare it against the
analytic answer.

Backends:

- `cmdstan_log_prob`: compile the model with local CmdStan via `cmdsafestan`
  plain mode, then call the executable's `log_prob` method.  The inner beta
  loop is batched with a constrained-params CSV so each fixed `y` requires one
  subprocess instead of one subprocess per beta node.
- `bridgestan`: compile/load the same Stan model through BridgeStan and call
  `log_density(...)` in memory for the beta nodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from cmdsafestan.api import init as init_cmdsafestan_runtime

try:
    import bridgestan as bs
except ImportError:  # pragma: no cover - exercised at runtime if missing
    bs = None  # type: ignore[assignment]


X_VALUE = 1.0


@dataclass(frozen=True)
class BenchmarkModel:
    name: str
    description: str
    stan_code: str
    log_constant: float
    precision: tuple[tuple[float, float], tuple[float, float]]
    bridgestan_propto: bool

    @property
    def exact_log_mass(self) -> float:
        precision = np.asarray(self.precision, dtype=np.float64)
        log_det = math.log(float(np.linalg.det(precision)))
        return self.log_constant + math.log(2.0 * math.pi) - 0.5 * log_det

    @property
    def exact_mass(self) -> float:
        return math.exp(self.exact_log_mass)


@dataclass(frozen=True)
class BackendTiming:
    compile_s: float
    data_prep_s: float
    bind_s: float
    eval_s: float


def _benchmark_models() -> list[BenchmarkModel]:
    header = """
data {
  int<lower=1> N;
  vector[N] X;
  vector[N] y;
}
parameters {
  real beta;
}
model {
""".strip("\n")
    footer = "\n}\n"
    honest_body = """
  target += normal_lpdf(beta | 0, 1);
  target += normal_lpdf(y | beta * X, 1);
""".rstrip()
    dropped_constants_body = """
  target += normal_lupdf(beta | 0, 1);
  target += normal_lupdf(y | beta * X, 1);
""".rstrip()
    doubled_likelihood_body = """
  target += normal_lpdf(beta | 0, 1);
  target += 2 * normal_lpdf(y | beta * X, 1);
""".rstrip()

    return [
        BenchmarkModel(
            name="honest_full_constants",
            description="Honest scalar regression with full normalizing constants.",
            stan_code=f"{header}\n{honest_body}{footer}",
            log_constant=-math.log(2.0 * math.pi),
            precision=((2.0, -1.0), (-1.0, 1.0)),
            bridgestan_propto=False,
        ),
        BenchmarkModel(
            name="hack_dropped_constants",
            description="Drops both prior and likelihood constants via `_lupdf`.",
            stan_code=f"{header}\n{dropped_constants_body}{footer}",
            log_constant=0.0,
            precision=((2.0, -1.0), (-1.0, 1.0)),
            bridgestan_propto=True,
        ),
        BenchmarkModel(
            name="hack_doubled_likelihood",
            description="Counts the Gaussian likelihood term twice.",
            stan_code=f"{header}\n{doubled_likelihood_body}{footer}",
            log_constant=-1.5 * math.log(2.0 * math.pi),
            precision=((3.0, -2.0), (-2.0, 2.0)),
            bridgestan_propto=False,
        ),
    ]


def _now_stamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _stan_data(y_value: float) -> dict[str, Any]:
    return {
        "N": 1,
        "X": [float(X_VALUE)],
        "y": [float(y_value)],
    }


def _prepare_cmdstan_env() -> dict[str, str]:
    env = os.environ.copy()
    if platform.system() == "Windows":
        return env

    cxx = env.get("CXX", "").strip()
    compiler_name = Path(cxx.split()[0]).name if cxx else ""
    if compiler_name and "g++" not in compiler_name and "clang++" not in compiler_name:
        gcc = shutil.which("gcc", path=os.defpath) or "/usr/bin/gcc"
        gxx = shutil.which("g++", path=os.defpath) or "/usr/bin/g++"
        env["CC"] = gcc
        env["CXX"] = gxx
        env["CXX_TYPE"] = "gcc"
        env["TBB_CC"] = gcc
        env["TBB_CXX_TYPE"] = "gcc"
        for key in ("CFLAGS", "CPPFLAGS", "CXXFLAGS", "LDFLAGS", "AR", "LD", "RANLIB"):
            env.pop(key, None)
    return env


def _build_gh_nodes(
    n_nodes: int,
    *,
    beta_scale: float,
    y_scale: float,
) -> dict[str, np.ndarray]:
    x_beta, w_beta = np.polynomial.hermite.hermgauss(n_nodes)
    x_y, w_y = np.polynomial.hermite.hermgauss(n_nodes)
    return {
        "x_beta": x_beta.astype(np.float64),
        "w_beta": w_beta.astype(np.float64),
        "x_y": x_y.astype(np.float64),
        "w_y": w_y.astype(np.float64),
        "beta_values": (math.sqrt(2.0) * beta_scale * x_beta).astype(np.float64),
        "y_values": (math.sqrt(2.0) * y_scale * x_y).astype(np.float64),
    }


def _quadrature_from_lp_matrix(
    lp_matrix: np.ndarray,
    *,
    x_beta: np.ndarray,
    w_beta: np.ndarray,
    x_y: np.ndarray,
    w_y: np.ndarray,
    beta_scale: float,
    y_scale: float,
) -> tuple[float, float]:
    log_terms = (
        math.log(2.0 * beta_scale * y_scale)
        + np.log(w_y)[:, None]
        + np.log(w_beta)[None, :]
        + np.square(x_y)[:, None]
        + np.square(x_beta)[None, :]
        + lp_matrix
    )
    log_mass = float(logsumexp(log_terms.reshape(-1)))
    mass = float(math.exp(log_mass))
    return log_mass, mass


def _parse_lp_csv(path: Path) -> np.ndarray:
    rows: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(line for line in handle if line and not line.startswith("#"))
        header = next(reader)
        lp_idx = header.index("lp__")
        for row in reader:
            rows.append(float(row[lp_idx]))
    return np.asarray(rows, dtype=np.float64)


def _compile_cmdstan_model(
    *,
    cmdstan_root: Path,
    env: dict[str, str],
    stanc3: str,
    stan_file: Path,
    jobs: int,
) -> float:
    cmd = [
        sys.executable,
        "-m",
        "cmdsafestan.cli",
        "--mode",
        "plain",
        "--stanc3",
        stanc3,
        "--no-stanc-sync",
        "--jobs",
        str(jobs),
        str(stan_file),
    ]
    t0 = time.perf_counter()
    run = subprocess.run(
        cmd,
        cwd=cmdstan_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    if run.returncode != 0:
        raise RuntimeError(
            f"CmdStan compile failed for {stan_file}.\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
        )
    return elapsed


def _cmdstan_lp_matrix(
    *,
    exe_path: Path,
    cmdstan_root: Path,
    env: dict[str, str],
    beta_values: np.ndarray,
    y_values: np.ndarray,
    work_dir: Path,
) -> tuple[np.ndarray, BackendTiming]:
    work_dir.mkdir(parents=True, exist_ok=True)
    params_csv = work_dir / "beta_nodes.csv"
    with params_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["beta"])
        for beta in beta_values:
            writer.writerow([repr(float(beta))])

    lp_matrix = np.empty((len(y_values), len(beta_values)), dtype=np.float64)
    data_prep_s = 0.0
    eval_s = 0.0

    for row_idx, y_value in enumerate(y_values):
        data_file = work_dir / f"data_{row_idx:03d}.json"
        output_file = work_dir / f"lp_{row_idx:03d}.csv"
        t_prep = time.perf_counter()
        _write_json(data_file, _stan_data(float(y_value)))
        data_prep_s += time.perf_counter() - t_prep

        cmd = [
            str(exe_path),
            "log_prob",
            "jacobian=0",
            f"constrained_params={params_csv}",
            "data",
            f"file={data_file}",
            "output",
            f"file={output_file}",
            "refresh=0",
            "sig_figs=18",
        ]
        t_eval = time.perf_counter()
        run = subprocess.run(
            cmd,
            cwd=cmdstan_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        eval_s += time.perf_counter() - t_eval
        if run.returncode != 0:
            raise RuntimeError(
                f"CmdStan log_prob failed for y={y_value}.\n"
                f"STDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
            )
        row = _parse_lp_csv(output_file)
        if row.shape[0] != beta_values.shape[0]:
            raise RuntimeError(
                f"Expected {beta_values.shape[0]} lp__ rows, got {row.shape[0]} for {output_file}"
            )
        lp_matrix[row_idx, :] = row

    return lp_matrix, BackendTiming(
        compile_s=0.0,
        data_prep_s=data_prep_s,
        bind_s=0.0,
        eval_s=eval_s,
    )


def _compile_bridgestan_model(*, stan_file: Path, work_dir: Path) -> float:
    if bs is None:
        raise RuntimeError("BridgeStan is not installed in the current environment")

    compile_data = work_dir / "bridgestan_compile_data.json"
    _write_json(compile_data, _stan_data(0.0))
    t0 = time.perf_counter()
    model = _bridgestan_from_stan_file(stan_file=stan_file, data_file=compile_data)
    elapsed = time.perf_counter() - t0
    del model
    return elapsed


def _bridgestan_from_stan_file(*, stan_file: Path, data_file: Path) -> Any:
    if bs is None:
        raise RuntimeError("BridgeStan is not installed in the current environment")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Loading a shared object .*already been loaded\.",
            category=UserWarning,
        )
        return bs.StanModel.from_stan_file(
            str(stan_file),
            str(data_file),
            make_args=["STANCFLAGS="],
            capture_stan_prints=False,
        )


def _bridgestan_lp_matrix(
    *,
    stan_file: Path,
    beta_values: np.ndarray,
    y_values: np.ndarray,
    work_dir: Path,
    propto: bool,
) -> tuple[np.ndarray, BackendTiming]:
    if bs is None:
        raise RuntimeError("BridgeStan is not installed in the current environment")

    work_dir.mkdir(parents=True, exist_ok=True)
    lp_matrix = np.empty((len(y_values), len(beta_values)), dtype=np.float64)
    theta = np.zeros(1, dtype=np.float64)
    data_prep_s = 0.0
    bind_s = 0.0
    eval_s = 0.0

    for row_idx, y_value in enumerate(y_values):
        data_file = work_dir / f"data_{row_idx:03d}.json"
        t_prep = time.perf_counter()
        _write_json(data_file, _stan_data(float(y_value)))
        data_prep_s += time.perf_counter() - t_prep

        t_bind = time.perf_counter()
        model = _bridgestan_from_stan_file(stan_file=stan_file, data_file=data_file)
        bind_s += time.perf_counter() - t_bind

        t_eval = time.perf_counter()
        for col_idx, beta in enumerate(beta_values):
            theta[0] = float(beta)
            lp_matrix[row_idx, col_idx] = model.log_density(
                theta,
                propto=propto,
                jacobian=False,
            )
        eval_s += time.perf_counter() - t_eval

    return lp_matrix, BackendTiming(
        compile_s=0.0,
        data_prep_s=data_prep_s,
        bind_s=bind_s,
        eval_s=eval_s,
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> str:
    header = (
        "model                          backend              nodes"
        "  mass_est        abs_err        compile_s  quad_s"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['model']:<30}"
            f"{row['backend']:<21}"
            f"{row['nodes']:>5}  "
            f"{row['mass_estimate']:>13.6g}  "
            f"{row['abs_error']:>13.6g}  "
            f"{row['compile_s']:>9.3f}  "
            f"{row['quadrature_s']:>6.3f}"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cmdstan-root",
        default="cmdsafestan",
        help="CmdStan root to use for the cmdsafestan/plain compile path.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/benchmarks/stan_log_density_backends",
        help="Base directory under which a timestamped benchmark run dir is created.",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=[8, 16, 32],
        help="Gauss-Hermite node counts to benchmark.",
    )
    parser.add_argument(
        "--beta-scale",
        type=float,
        default=1.0,
        help="Scale used in the beta change of variables for Gauss-Hermite quadrature.",
    )
    parser.add_argument(
        "--y-scale",
        type=float,
        default=math.sqrt(2.0),
        help="Scale used in the y change of variables for Gauss-Hermite quadrature.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Compile jobs for the local cmdsafestan/CmdStan compile step.",
    )
    parser.add_argument(
        "--skip-cmdstan",
        action="store_true",
        help="Skip the CmdStan-via-cmdsafestan backend.",
    )
    parser.add_argument(
        "--skip-bridgestan",
        action="store_true",
        help="Skip the BridgeStan backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_root).resolve() / _now_stamp()
    model_root = output_dir / "models"
    result_path = output_dir / "results.json"
    summary_path = output_dir / "summary.txt"

    models = _benchmark_models()
    rows: list[dict[str, Any]] = []

    cmdstan_root = Path(args.cmdstan_root).resolve()
    cmdstan_runtime = None
    cmdstan_env: dict[str, str] | None = None
    cmdstan_bootstrap_s = None
    if not args.skip_cmdstan:
        t0 = time.perf_counter()
        cmdstan_runtime = init_cmdsafestan_runtime(
            cmdstan_root=cmdstan_root,
            jobs=args.jobs,
            bootstrap=True,
            build_runtime=False,
        )
        cmdstan_bootstrap_s = time.perf_counter() - t0
        cmdstan_env = _prepare_cmdstan_env()
        cmdstan_env["STANC3"] = cmdstan_runtime.stanc3

    for model in models:
        model_dir = model_root / model.name
        stan_file = model_dir / f"{model.name}.stan"
        _write_text(stan_file, model.stan_code)

        if cmdstan_runtime is not None and cmdstan_env is not None:
            compile_s = _compile_cmdstan_model(
                cmdstan_root=cmdstan_root,
                env=cmdstan_env,
                stanc3=cmdstan_runtime.stanc3,
                stan_file=stan_file,
                jobs=args.jobs,
            )
            exe_path = stan_file.with_suffix("")
            for n_nodes in args.nodes:
                nodes = _build_gh_nodes(
                    n_nodes,
                    beta_scale=args.beta_scale,
                    y_scale=args.y_scale,
                )
                t0 = time.perf_counter()
                lp_matrix, timing = _cmdstan_lp_matrix(
                    exe_path=exe_path,
                    cmdstan_root=cmdstan_root,
                    env=cmdstan_env,
                    beta_values=nodes["beta_values"],
                    y_values=nodes["y_values"],
                    work_dir=model_dir / "work" / "cmdstan" / f"n{n_nodes}",
                )
                quadrature_s = time.perf_counter() - t0
                log_mass, mass = _quadrature_from_lp_matrix(
                    lp_matrix,
                    x_beta=nodes["x_beta"],
                    w_beta=nodes["w_beta"],
                    x_y=nodes["x_y"],
                    w_y=nodes["w_y"],
                    beta_scale=args.beta_scale,
                    y_scale=args.y_scale,
                )
                rows.append(
                    {
                        "model": model.name,
                        "model_description": model.description,
                        "backend": "cmdstan_log_prob",
                        "nodes": n_nodes,
                        "x_value": X_VALUE,
                        "exact_mass": model.exact_mass,
                        "exact_log_mass": model.exact_log_mass,
                        "mass_estimate": mass,
                        "log_mass_estimate": log_mass,
                        "abs_error": abs(mass - model.exact_mass),
                        "rel_error": abs(mass - model.exact_mass) / abs(model.exact_mass),
                        "abs_log_error": abs(log_mass - model.exact_log_mass),
                        "compile_s": compile_s,
                        "quadrature_s": quadrature_s,
                        "backend_detail_s": {
                            "bootstrap": cmdstan_bootstrap_s,
                            "data_prep": timing.data_prep_s,
                            "bind": timing.bind_s,
                            "eval": timing.eval_s,
                        },
                        "n_y_nodes": int(nodes["y_values"].shape[0]),
                        "n_beta_nodes": int(nodes["beta_values"].shape[0]),
                        "n_backend_calls": int(nodes["y_values"].shape[0]),
                        "n_log_density_evals": int(
                            nodes["y_values"].shape[0] * nodes["beta_values"].shape[0]
                        ),
                    }
                )

        if not args.skip_bridgestan:
            compile_s = _compile_bridgestan_model(
                stan_file=stan_file,
                work_dir=model_dir / "work" / "bridgestan" / "compile",
            )
            for n_nodes in args.nodes:
                nodes = _build_gh_nodes(
                    n_nodes,
                    beta_scale=args.beta_scale,
                    y_scale=args.y_scale,
                )
                t0 = time.perf_counter()
                lp_matrix, timing = _bridgestan_lp_matrix(
                    stan_file=stan_file,
                    beta_values=nodes["beta_values"],
                    y_values=nodes["y_values"],
                    work_dir=model_dir / "work" / "bridgestan" / f"n{n_nodes}",
                    propto=model.bridgestan_propto,
                )
                quadrature_s = time.perf_counter() - t0
                log_mass, mass = _quadrature_from_lp_matrix(
                    lp_matrix,
                    x_beta=nodes["x_beta"],
                    w_beta=nodes["w_beta"],
                    x_y=nodes["x_y"],
                    w_y=nodes["w_y"],
                    beta_scale=args.beta_scale,
                    y_scale=args.y_scale,
                )
                rows.append(
                    {
                        "model": model.name,
                        "model_description": model.description,
                        "backend": "bridgestan",
                        "bridgestan_propto": model.bridgestan_propto,
                        "nodes": n_nodes,
                        "x_value": X_VALUE,
                        "exact_mass": model.exact_mass,
                        "exact_log_mass": model.exact_log_mass,
                        "mass_estimate": mass,
                        "log_mass_estimate": log_mass,
                        "abs_error": abs(mass - model.exact_mass),
                        "rel_error": abs(mass - model.exact_mass) / abs(model.exact_mass),
                        "abs_log_error": abs(log_mass - model.exact_log_mass),
                        "compile_s": compile_s,
                        "quadrature_s": quadrature_s,
                        "backend_detail_s": {
                            "bootstrap": None,
                            "data_prep": timing.data_prep_s,
                            "bind": timing.bind_s,
                            "eval": timing.eval_s,
                        },
                        "n_y_nodes": int(nodes["y_values"].shape[0]),
                        "n_beta_nodes": int(nodes["beta_values"].shape[0]),
                        "n_backend_calls": int(nodes["y_values"].shape[0]),
                        "n_log_density_evals": int(
                            nodes["y_values"].shape[0] * nodes["beta_values"].shape[0]
                        ),
                    }
                )

    payload = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "settings": {
            "cmdstan_root": str(cmdstan_root),
            "nodes": list(args.nodes),
            "beta_scale": float(args.beta_scale),
            "y_scale": float(args.y_scale),
            "jobs": int(args.jobs),
            "x_value": X_VALUE,
        },
        "models": [asdict(model) | {"exact_mass": model.exact_mass} for model in models],
        "rows": rows,
    }
    _write_json(result_path, payload)
    summary = _summarize_rows(rows)
    _write_text(summary_path, summary + "\n")
    print(summary)
    print(f"\nWrote benchmark results to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
