from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_wandb_agent.py"
    spec = importlib.util.spec_from_file_location("run_wandb_agent_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_calls_wandb_agent(monkeypatch) -> None:
    module = _load_module()

    calls: list[dict] = []

    class _FakeWandb:
        def agent(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_wandb_agent.py",
            "--sweep-id",
            "entity/project/abc123",
            "--count",
            "3",
            "--project",
            "ppl-synthesis-reward-hacking",
            "--entity",
            "entity",
        ],
    )

    module.main()

    assert len(calls) == 1
    assert calls[0]["sweep_id"] == "entity/project/abc123"
    assert calls[0]["count"] == 3
    assert calls[0]["project"] == "ppl-synthesis-reward-hacking"
    assert calls[0]["entity"] == "entity"
