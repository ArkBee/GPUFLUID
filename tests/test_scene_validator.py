"""Round-17: scene_validator.out_of_domain_warning contract tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ADDON_DIR = Path(__file__).resolve().parent.parent / "addon"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_test_scene_validator",
        _ADDON_DIR / "gpufluid_blender" / "scene_validator.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.out_of_domain_warning


def test_inside_no_warning():
    """Source comfortably inside [eps, 1-eps] — no warning."""
    out = _load()
    w = out("Inflow1", lo=(0.2, 0.2, 0.2), hi=(0.4, 0.4, 0.4),
            kind="Inflow", eps_norm=0.02)
    assert w is None


def test_fully_outside_x_warns():
    out = _load()
    w = out("Inflow1", lo=(1.5, 0.5, 0.5), hi=(2.0, 0.6, 0.6),
            kind="Inflow", eps_norm=0.02)
    assert w is not None
    assert "fully outside" in w
    assert "Inflow1" in w
    assert "x" in w


def test_fully_outside_negative_z_warns():
    out = _load()
    w = out("Plate", lo=(0.3, 0.3, -1.0), hi=(0.4, 0.4, -0.5),
            kind="Obstacle", eps_norm=0.02)
    assert "fully outside" in w
    assert "z" in w
    assert "Obstacle" in w


def test_wall_clip_warns():
    """Inside but too close to the wall."""
    out = _load()
    w = out("Source", lo=(0.0, 0.5, 0.5), hi=(0.1, 0.6, 0.6),
            kind="Fluid source", eps_norm=0.05)
    assert w is not None
    assert "touches/clips" in w
    assert "x" in w


def test_wall_clip_at_upper_bound():
    out = _load()
    w = out("Source", lo=(0.5, 0.5, 0.5), hi=(0.5, 0.5, 1.0),
            kind="Fluid source", eps_norm=0.05)
    assert "touches/clips" in w
    assert "z" in w


def test_returns_first_axis_in_xyz_order():
    """If multiple axes violate, only the FIRST (x → y → z) is reported.
    Caller can re-call after fixing one to surface the next."""
    out = _load()
    w = out("S", lo=(-0.5, -0.5, 0.5), hi=(-0.1, -0.1, 0.6),
            kind="Fluid", eps_norm=0.02)
    # Both x and y are fully outside; expect x to be cited
    assert "fully outside" in w
    assert " x " in w   # space-separated to avoid matching 'xyz' substr


def test_exactly_at_eps_inside_passes():
    """Boundary case: lo == eps_norm exactly → no wall-clip warning."""
    out = _load()
    eps = 0.02
    w = out("S", lo=(eps, eps, eps), hi=(0.5, 0.5, 0.5),
            kind="Fluid", eps_norm=eps)
    assert w is None


def test_just_inside_upper_eps_passes():
    out = _load()
    eps = 0.02
    w = out("S", lo=(0.5, 0.5, 0.5), hi=(1 - eps, 1 - eps, 1 - eps),
            kind="Fluid", eps_norm=eps)
    assert w is None


def test_kind_and_name_interpolated():
    out = _load()
    w = out("MyPlate", lo=(0.5, 0.5, -0.5), hi=(0.6, 0.6, -0.1),
            kind="MyObstacle", eps_norm=0.02)
    assert "MyPlate" in w
    assert "MyObstacle" in w
