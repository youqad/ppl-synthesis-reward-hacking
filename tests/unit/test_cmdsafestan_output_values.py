from __future__ import annotations

from pathlib import Path

from cmdsafestan.api import (
    SafeStanRuntime,
    _extract_output_values,
    _prepare_build_env,
    evaluate_model_string_many_data,
)


def test_extract_output_values_reads_requested_scalars(tmp_path: Path) -> None:
    csv_path = tmp_path / "output.csv"
    csv_path.write_text(
        "# comment\n"
        "lp__,alpha,reported_log_density,log_score\n"
        "-1.5,0.2,-0.7,-0.8\n",
        encoding="utf-8",
    )

    output_values = _extract_output_values(
        csv_path,
        output_variables=["reported_log_density", "log_score"],
    )

    assert output_values == {
        "lp__": -1.5,
        "reported_log_density": -0.7,
        "log_score": -0.8,
    }


def test_extract_output_values_ignores_missing_or_invalid_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "output.csv"
    csv_path.write_text(
        "lp__,reported_log_density\n"
        "-1.5,not-a-number\n",
        encoding="utf-8",
    )

    output_values = _extract_output_values(
        csv_path,
        output_variables=["reported_log_density", "missing_value"],
    )

    assert output_values == {"lp__": -1.5}


def test_prepare_build_env_replaces_conda_wrapped_compilers(monkeypatch) -> None:
    monkeypatch.setenv("CC", "/tmp/x86_64-conda-linux-gnu-cc")
    monkeypatch.setenv("CXX", "/tmp/x86_64-conda-linux-gnu-c++")
    monkeypatch.setenv("CFLAGS", "-I/tmp/conda-include")
    monkeypatch.setenv("CXXFLAGS", "-I/tmp/conda-cxx")
    monkeypatch.setenv("LDFLAGS", "-L/tmp/conda-lib")
    monkeypatch.setattr("cmdsafestan.api._find_system_compiler", lambda name: f"/usr/bin/{name}")

    env = _prepare_build_env()

    assert env["CC"] == "/usr/bin/gcc"
    assert env["CXX"] == "/usr/bin/g++"
    assert env["CXX_TYPE"] == "gcc"
    assert env["TBB_CC"] == "/usr/bin/gcc"
    assert env["TBB_CXX_TYPE"] == "gcc"
    assert "CFLAGS" not in env
    assert "CXXFLAGS" not in env
    assert "LDFLAGS" not in env


def test_evaluate_model_string_many_data_compiles_once(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    (runtime_root / "lib" / "stan_math" / "make").mkdir(parents=True)
    (runtime_root / "lib" / "stan_math" / "make" / "compiler_flags").write_text(
        "",
        encoding="utf-8",
    )
    (runtime_root / "src" / "stan" / "callbacks").mkdir(parents=True)
    (runtime_root / "src" / "stan" / "callbacks" / "writer.hpp").write_text(
        "",
        encoding="utf-8",
    )
    (runtime_root / "lib" / "rapidjson_1.1.0").mkdir(parents=True)

    runtime = SafeStanRuntime(
        cmdstan_root=tmp_path,
        stanc3="safestan",
        runtime_root=runtime_root,
        tmp_root=tmp_path / "tmp",
        no_stanc_sync=True,
    )
    runtime.tmp_root.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []

    def _fake_cmdsafestan_command(**kwargs):
        target = kwargs["target"]
        return [f"compile:{target or 'exe'}"]

    def _fake_run_command(command, *, cwd, env, stream_output):
        del cwd, env, stream_output
        commands.append(command)
        if command[0].startswith("compile:"):
            return 0, "compile ok"
        output_arg = next(
            arg
            for arg in command
            if arg.startswith("file=") and arg.endswith(".csv")
        )
        Path(output_arg.removeprefix("file=")).write_text(
            "lp__,reported_log_density\n-1.5,-2.25\n",
            encoding="utf-8",
        )
        return 0, "run ok"

    monkeypatch.setattr("cmdsafestan.api._cmdsafestan_command", _fake_cmdsafestan_command)
    monkeypatch.setattr("cmdsafestan.api._run_command", _fake_run_command)

    results = evaluate_model_string_many_data(
        "data {} model {}",
        [{"y": [1]}, {"y": [2]}],
        protect="y",
        runtime=runtime,
        jobs=2,
        enforce_safety=False,
        run_sample=True,
        output_variables=["reported_log_density"],
    )

    assert len(results) == 2
    assert [result.output_values for result in results] == [
        {"lp__": -1.5, "reported_log_density": -2.25},
        {"lp__": -1.5, "reported_log_density": -2.25},
    ]
    assert sum(1 for command in commands if command[0] == "compile:hpp") == 1
    assert sum(1 for command in commands if command[0] == "compile:exe") == 1
    assert sum(1 for command in commands if len(command) > 1 and command[1] == "sample") == 2
