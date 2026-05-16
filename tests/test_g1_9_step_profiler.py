"""B12 — per-section step profiler (G1.9)."""
import numpy as np
import pytest

from gpufluid.primitives.profiling import StepProfiler
from gpufluid.solvers.solver3d import FlipSolver3D


def _seed_one_step(enable_timing: bool):
    """Build a tiny solver, seed a few particles, run one step.
    Returns the profiler totals (or empty dict)."""
    sol = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16,
                       enable_timing=enable_timing)
    # Seed a small water blob so P2G/G2P actually have work to do.
    sol.seed_box((0.2, 0.2, 0.2), (0.6, 0.6, 0.6), ppc=2)
    sol.step(dt=0.01)
    return sol._prof.summarize(), sol._prof.call_counts


def test_g1_9_disabled_profiler_is_a_noop():
    """When `enable_timing=False`, no per-section dict entries appear.

    This matters because `wp.ScopedTimer(synchronize=True)` blocks on the GPU
    on every section enter/exit — turning it on for non-profiling bakes would
    measurably slow them down. The profiler short-circuits to nullcontext in
    the disabled path; this test guards that the wiring still produces a
    completely empty dict.
    """
    totals, counts = _seed_one_step(enable_timing=False)
    assert totals == {}, f"profiler should be empty when disabled, got {totals}"
    assert counts == {}


def test_g1_9_enabled_profiler_collects_expected_sections():
    """With timing on, every always-on section in step() must fire exactly
    once per step, and report a non-negative duration."""
    totals, counts = _seed_one_step(enable_timing=True)
    # Sections that *always* run (no surface tension, no viscosity in this
    # scene): clear, p2g, normalize, gravity_bc, divergence, pressure,
    # grad_subtract_bc, g2p_advect, color
    expected = {"clear", "p2g", "normalize", "gravity_bc", "divergence",
                "pressure", "grad_subtract_bc", "g2p_advect", "color"}
    assert expected.issubset(totals.keys()), \
        f"missing sections: {expected - totals.keys()}; got {sorted(totals.keys())}"
    # No section that runs only conditionally should appear here.
    assert "viscosity" not in totals, "viscosity ran without being requested"
    assert "surface_tension" not in totals, "CSF ran without being requested"
    for name, ms in totals.items():
        assert ms >= 0.0, f"{name}: negative time {ms}"
    for name in expected:
        assert counts[name] == 1, f"{name} fired {counts[name]} times, expected 1"


def test_g1_9_viscosity_section_fires_when_active():
    """The conditional `viscosity` section should appear only when ν > 0."""
    sol = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16,
                       viscosity=0.05, viscosity_iters=2,
                       enable_timing=True)
    sol.seed_box((0.3, 0.3, 0.3), (0.5, 0.5, 0.5), ppc=2)
    sol.step(dt=0.01)
    totals = sol._prof.summarize()
    assert "viscosity" in totals
    assert totals["viscosity"] >= 0.0


def test_g1_9_surface_tension_section_fires_when_active():
    sol = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16,
                       surface_tension=0.05,
                       enable_timing=True)
    sol.seed_box((0.3, 0.3, 0.3), (0.5, 0.5, 0.5), ppc=2)
    sol.step(dt=0.005)
    totals = sol._prof.summarize()
    assert "surface_tension" in totals


def test_g1_9_reset_clears_state():
    p = StepProfiler(enabled=True)
    # Manually add an entry via the underlying dict to avoid Warp dep here
    p._dict["x"] = [0.5, 0.25]
    assert p.summarize() == {"x": 0.75}
    p.reset()
    assert p.summarize() == {}
