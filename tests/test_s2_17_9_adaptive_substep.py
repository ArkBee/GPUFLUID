"""S2.17.9 (FU-023) — MPM adaptive substepping.

Deep pools / fast jets push the stress-wave + advection CFL past the fixed
frame dt, so the bake diverges late (the user's "MPM solver diverged at frame
680" report). When cfg.adaptive_substep is on, step() splits the frame dt into
N CFL-sized sub-steps. These tests cover the substep-count arithmetic + the
opt-in default (off = one p2g2p, byte-identical to pre-FU-023).
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("warp")


def _make_solver(**cfg_kw):
    import numpy as np
    from gpufluid.sim.mpm.solver import MpmConfig, MpmSolver
    col = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    cfg = MpmConfig(initial_column=col, **cfg_kw)
    try:
        return MpmSolver(cfg)
    except Exception:
        pytest.skip("no CUDA device for MpmSolver construction")


def test_default_is_single_step():
    """adaptive_substep defaults to False → no substep machinery engaged."""
    from gpufluid.sim.mpm.solver import MpmConfig
    assert MpmConfig().adaptive_substep is False


def test_cfl_substeps_clamped_to_one_when_slow():
    """At rest (v_max≈0) with soft EOS + coarse dt the CFL bound is loose →
    a single substep suffices."""
    s = _make_solver(adaptive_substep=True, dt=1e-4,
                     n_grid=32, grid_lim=1.0, adaptive_cfl=0.6)
    s.cfg.fluid.bulk_modulus = 100.0
    s.cfg.fluid.density = 1000.0
    n, saturated = s._cfl_substeps()
    assert n == 1 and saturated is False


def test_cfl_substeps_scales_with_soundspeed():
    """Stiffer EOS (higher bulk_modulus → higher c_sound) needs more
    substeps for the same dt — the term the divergence message blames."""
    soft = _make_solver(adaptive_substep=True, dt=5e-3, n_grid=96)
    soft.cfg.fluid.bulk_modulus = 100.0
    soft.cfg.fluid.density = 1000.0
    stiff = _make_solver(adaptive_substep=True, dt=5e-3, n_grid=96)
    stiff.cfg.fluid.bulk_modulus = 10000.0
    stiff.cfg.fluid.density = 1000.0
    assert stiff._cfl_substeps()[0] > soft._cfl_substeps()[0]


def test_cfl_substeps_clamped_to_max():
    """A pathologically stiff EOS can't blow up the substep count past the
    configured ceiling (so a runaway frame can't stall the bake)."""
    s = _make_solver(adaptive_substep=True, dt=0.1, n_grid=256,
                     adaptive_max_substeps=8)
    s.cfg.fluid.bulk_modulus = 1e9
    s.cfg.fluid.density = 1.0
    n, saturated = s._cfl_substeps()
    assert n == 8 and saturated is True  # demand >> cap → flagged


def test_cfl_formula_matches_bound():
    """N must equal ceil(dt*(c+v_max)/(cfl*dx)) at rest (v_max=0)."""
    s = _make_solver(adaptive_substep=True, dt=4e-3, n_grid=96,
                     grid_lim=1.0, adaptive_cfl=0.6, adaptive_max_substeps=64)
    s.cfg.fluid.bulk_modulus = 1500.0
    s.cfg.fluid.density = 1000.0
    c = math.sqrt(1500.0 / 1000.0)
    dx = 1.0 / 96
    expected = max(1, math.ceil(4e-3 * c / (0.6 * dx)))
    assert s._cfl_substeps()[0] == expected


def test_cli_threads_cfl_into_mpm():
    """Contract: the MPM CLI branch must pass the [simulation] cfl knobs into
    MpmConfig.adaptive_* (so the existing addon 'CFL Substepping' checkbox
    drives MPM, not just FLIP)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli"
           / "commands.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "adaptive_substep=bool(sim.cfl)" in code
    assert "adaptive_cfl=float(sim.cfl_factor)" in code
    assert "adaptive_max_substeps=int(sim.cfl_max_substeps)" in code
