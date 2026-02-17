from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

from ppl_synthesis_reward_hacking.plotting.styles import apply_publication_style


def test_apply_publication_style_sets_times_new_roman_defaults() -> None:
    plt.rcdefaults()
    apply_publication_style()

    font_family = matplotlib.rcParams["font.family"]
    assert "serif" in font_family
    assert "Times New Roman" in matplotlib.rcParams["font.serif"]
    assert matplotlib.rcParams["mathtext.fontset"] == "stix"
    assert matplotlib.rcParams["axes.grid"] is True


def test_apply_publication_style_prefers_seaborn(monkeypatch) -> None:
    plt.rcdefaults()
    captured: dict[str, Any] = {}

    dummy_sns = types.ModuleType("seaborn")

    def set_theme(*, style: str, font: str, rc: dict[str, Any]) -> None:
        captured["style"] = style
        captured["font"] = font
        captured["rc"] = rc

    dummy_sns.set_theme = set_theme  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "seaborn", dummy_sns)

    apply_publication_style()

    assert captured.get("style") == "whitegrid"
    assert captured.get("font") == "serif"
    assert isinstance(captured.get("rc"), dict)
    assert captured["rc"].get("font.family") == "serif"


def test_apply_publication_style_falls_back_to_matplotlib(monkeypatch) -> None:
    plt.rcdefaults()
    original_import = builtins.__import__

    def fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        if name == "seaborn":
            raise ImportError("seaborn missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("seaborn", None)

    apply_publication_style()

    font_family = matplotlib.rcParams["font.family"]
    assert "serif" in font_family
    assert matplotlib.rcParams["lines.linewidth"] == 1.5
