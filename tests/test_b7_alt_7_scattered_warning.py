"""B7-alt.7 — scattered-topology warning.

The B7-alt.1 spike documented that bbox-based sub-dense storage is a
no-op when fluid is dispersed across the domain (whitewater, many
droplets). `_rebuild_sub_dense` now computes the bbox/full volume
ratio and fires a one-shot stderr warning when the ratio > 0.8.
"""
import numpy as np
import pytest

from gpufluid.solvers.solver3d import FlipSolver3D


def test_rebuild_does_not_warn_on_connected_blob(capsys):
    """A bbox covering <80% of the domain should NOT warn."""
    s = FlipSolver3D(nx=64, ny=64, nz=64, enable_sub_dense=True,
                     sub_dilation=2)
    s._rebuild_sub_dense((16, 16, 16), (48, 48, 48))  # 32³ = ~12% of 64³
    err = capsys.readouterr().err
    assert "scattered topology" not in err


def test_rebuild_warns_on_scattered_bbox(capsys):
    """A bbox covering nearly the full domain (scattered scene proxy)
    should fire the one-shot warning naming the bbox, full extent, and
    `sub_dilation`."""
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=4)
    s._rebuild_sub_dense((0, 0, 0), (30, 30, 30))  # 30³ ≈ 82.4% of 32³
    err = capsys.readouterr().err
    assert "scattered topology" in err
    assert "82.4%" in err
    assert "sub_dilation`=4" in err


def test_rebuild_warns_only_once(capsys):
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=4)
    s._rebuild_sub_dense((0, 0, 0), (30, 30, 30))
    first = capsys.readouterr().err
    assert "scattered topology" in first
    # Trigger another scattered rebuild — must stay silent.
    s._rebuild_sub_dense((1, 1, 1), (31, 31, 31))
    second = capsys.readouterr().err
    assert second == ""


def test_rebuild_warning_threshold_at_80_percent(capsys):
    """Exactly-at-threshold (0.8) does NOT warn — strict greater-than."""
    s = FlipSolver3D(nx=10, ny=10, nz=10, enable_sub_dense=True,
                     sub_dilation=0)
    # 8 × 10 × 10 / 1000 = 0.8 exactly.
    s._rebuild_sub_dense((0, 0, 0), (8, 10, 10))
    err = capsys.readouterr().err
    assert "scattered topology" not in err
    # Bump to 8.1% over — well, 9×10×10 / 1000 = 0.9
    s2 = FlipSolver3D(nx=10, ny=10, nz=10, enable_sub_dense=True,
                      sub_dilation=0)
    s2._rebuild_sub_dense((0, 0, 0), (9, 10, 10))
    err2 = capsys.readouterr().err
    assert "scattered topology" in err2
