from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_slurm_variant_profiles_exist_and_override_launcher() -> None:
    for name in ["slurm", "a100", "h100", "cpu-debug"]:
        cfg = _load_yaml(Path(f"configs/hydra/launcher/{name}.yaml"))
        defaults = cfg.get("defaults")
        assert isinstance(defaults, list)
        assert any(
            isinstance(item, dict) and item.get("override /hydra/launcher") == "submitit_slurm"
            for item in defaults
        )
        launcher = cfg.get("launcher")
        assert isinstance(launcher, dict)
        assert "kind" in launcher
