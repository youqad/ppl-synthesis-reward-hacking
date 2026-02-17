from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "create_wandb_sweep.py"
    spec = importlib.util.spec_from_file_location("create_wandb_sweep_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_calls_wandb_sweep(monkeypatch, tmp_path) -> None:
    module = _load_module()

    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(
        "method: bayes\nmetric:\n  name: sweep/final_gap_mean\n  goal: maximize\n",
        encoding="utf-8",
    )

    calls: list[dict] = []

    class _FakeWandb:
        def sweep(self, **kwargs):
            calls.append(kwargs)
            return "abc123"

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_wandb_sweep.py",
            "--config",
            str(cfg_path),
            "--project",
            "ppl-synthesis-reward-hacking",
            "--entity",
            "entity",
        ],
    )

    module.main()

    assert len(calls) == 1
    assert calls[0]["project"] == "ppl-synthesis-reward-hacking"
    assert calls[0]["entity"] == "entity"
    assert calls[0]["sweep"]["method"] == "bayes"

