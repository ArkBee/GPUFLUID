"""Round-30 regression tests for round-29 reviewer findings.

  - mpm_initial_velocity is shipped UNSCALED from the addon — the
    CLI does the single correct Z scaling. Source-grep guard.
  - whitewater_potentials rejects v_max<=0 / radius<=0 / dx<=0.
  - cubic_ref tolerates empty + NaN positions defensively.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ─── mpm_initial_velocity addon-side no longer pre-scales ───────────────

def test_addon_does_not_pre_scale_initial_velocity():
    """Round-30: pre-round-30 bake.py multiplied dprops.mpm_initial_velocity
    by inv_size[1] (Y axis!) AND the CLI then re-scaled by inv_dom_z (Z)
    — double-scaled with wrong axis. For a non-cubic domain the
    column velocity was off by `inv_size[1]/inv_size[2]`. Source-grep
    guard prevents the pre-round-30 form drifting back.
    """
    bake_src = (Path(__file__).resolve().parents[1] / "addon"
                / "gpufluid_blender" / "operators" / "bake.py").read_text()
    code_lines = [
        ln for ln in bake_src.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)
    # The buggy form was `... * inv_size[1]` on the mpm_initial_velocity
    # value. Ensure no live (non-comment) line carries that pattern.
    assert "mpm_initial_velocity\": float(dprops.mpm_initial_velocity) * inv_size" not in code, (
        "round-30 contract broken: addon must ship mpm_initial_velocity "
        "UNSCALED; CLI does the single Z rescaling")


# ─── whitewater_potentials guards ───────────────────────────────────────

def test_trapped_air_potential_rejects_zero_v_max():
    from gpufluid.sim.whitewater_potentials import trapped_air_potential
    pos = np.zeros((5, 3), dtype=np.float32)
    vel = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="v_max must be > 0"):
        trapped_air_potential(pos, vel, radius=0.05, v_max=0.0)


def test_trapped_air_potential_rejects_zero_radius():
    from gpufluid.sim.whitewater_potentials import trapped_air_potential
    pos = np.zeros((5, 3), dtype=np.float32)
    vel = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="radius must be > 0"):
        trapped_air_potential(pos, vel, radius=0.0, v_max=10.0)


def test_wave_crest_potential_rejects_zero_dx():
    from gpufluid.sim.whitewater_potentials import wave_crest_potential
    pos = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dx must be > 0"):
        wave_crest_potential(pos, grid_res=16, dx=0.0)


def test_empty_pos_returns_empty_potential():
    """Both wrappers must short-circuit on empty without raising."""
    from gpufluid.sim.whitewater_potentials import (
        trapped_air_potential, wave_crest_potential)
    empty = np.zeros((0, 3), dtype=np.float32)
    assert trapped_air_potential(empty, empty, radius=0.05).shape == (0,)
    assert wave_crest_potential(empty, grid_res=16, dx=1 / 16).shape == (0,)


# ─── cubic_ref empty + NaN guard ────────────────────────────────────────

def test_cubic_ref_empty_returns_zero_grid():
    from gpufluid.meshing.cubic_ref import cubic_isotropic_density_cpu as densify_cubic
    out = densify_cubic(
        np.zeros((0, 3), dtype=np.float32),
        nx=16, ny=16, nz=16, dx=1 / 16, radius_cells=2.0)
    assert out.shape == (16, 16, 16)
    assert out.sum() == 0.0


def test_cubic_ref_filters_nan_positions():
    """Round-30: NaN row used to crash at int(np.floor(nan/dx))."""
    from gpufluid.meshing.cubic_ref import cubic_isotropic_density_cpu as densify_cubic
    pos = np.array([
        [0.5, 0.5, 0.5],
        [np.nan, 0.5, 0.5],
        [0.4, 0.4, 0.4],
        [0.5, np.inf, 0.5],
    ], dtype=np.float32)
    out = densify_cubic(pos, nx=16, ny=16, nz=16, dx=1 / 16,
                        radius_cells=2.0)
    # Should have non-zero contribution from the 2 finite rows.
    assert out.sum() > 0.0
