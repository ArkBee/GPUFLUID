"""S2.17.9 (FU-023) — MPM adaptive substepping.

Deep pools / fast jets push the stress-wave + advection CFL past the fixed
frame dt, so the bake diverges late (the user's "MPM solver diverged at frame
680" report). When cfg.adaptive_substep is on, step() splits the frame dt into
N CFL-sized sub-steps. These tests cover the substep-count arithmetic + the
opt-in default (off = one p2g2p, byte-identical to pre-FU-023).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from conftest import HAS_CUDA

pytest.importorskip("warp")


def _make_solver(**cfg_kw):
    import numpy as np
    from gpufluid.sim.mpm.solver import MpmConfig, MpmSolver
    col = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    cfg = MpmConfig(initial_column=col, **cfg_kw)
    try:
        return MpmSolver(cfg)
    except Exception:
        # audit-20260610: skip ONLY when CUDA is genuinely absent. On a GPU
        # box a constructor exception is a real regression and must FAIL —
        # the old blanket `except → skip` reported it as a green skip.
        if HAS_CUDA:
            raise
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


def _spy_solver(adaptive, n_sub, saturated=False):
    """A real MpmSolver instance with no GPU state (object.__new__), wired with
    spy stubs so step() can be exercised on a CPU box. Counts p2g2p +
    _apply_pushback calls."""
    from types import SimpleNamespace
    from gpufluid.sim.mpm.solver import MpmSolver
    s = object.__new__(MpmSolver)
    counts = {"p2g2p": 0, "pushback": 0, "pre": 0, "post": 0}
    s.cfg = SimpleNamespace(adaptive_substep=adaptive, dt=5e-3, device="cpu",
                            dump_every=8, adaptive_max_substeps=16)
    s.mpm = SimpleNamespace(
        p2g2p=lambda *a, **k: counts.__setitem__("p2g2p", counts["p2g2p"] + 1))
    s._cfl_warned = False
    # FU-028: the adaptive loop now drains per sub-step via the REAL
    # _apply_outflows — give the spy an empty drain list so it no-ops
    # (§9.7 mock fidelity: expose the surface production code touches).
    s._outflow_params = []
    s._pre_step = lambda si: counts.__setitem__("pre", counts["pre"] + 1)
    s._post_step = lambda si: counts.__setitem__("post", counts["post"] + 1)
    s._apply_pushback = lambda: counts.__setitem__(
        "pushback", counts["pushback"] + 1)
    # reviewer-velcaps H4: caps now run per sub-step too (sibling of pushback);
    # stub so the spy no-ops on the cfg without tap/anti_splash attrs.
    s._apply_velcaps = lambda: None
    s._cfl_substeps = lambda: (n_sub, saturated)
    return s, counts


def test_step_runs_pushback_every_substep():
    """GUARD for the live-found CUDA-700 fix: in the adaptive branch, step()
    must call _apply_pushback once PER sub-step (not once per frame). A refactor
    moving it out of the loop would silently reintroduce the illegal-access
    crash (a tunnelling particle hits an out-of-bounds grid node)."""
    s, counts = _spy_solver(adaptive=True, n_sub=5)
    s.step(1)
    assert counts["p2g2p"] == 5, "5 sub-steps -> 5 p2g2p calls"
    assert counts["pushback"] == 5, "pushback MUST run every sub-step (CUDA-700 guard)"


def test_step_single_p2g2p_when_off():
    """Default (adaptive off) is exactly one p2g2p and no in-loop pushback —
    byte-identical to pre-S2.17.9 control flow."""
    s, counts = _spy_solver(adaptive=False, n_sub=1)
    s.step(1)
    assert counts["p2g2p"] == 1
    assert counts["pushback"] == 0  # _post_step (stubbed) owns the per-frame one


def test_cfl_saturation_warns_once(capsys):
    """The saturation WARN is one-shot (guarded by _cfl_warned) — must not spam
    every frame, must not stay silent."""
    s, counts = _spy_solver(adaptive=True, n_sub=16, saturated=True)
    s.step(1)
    s.step(2)
    err = capsys.readouterr().err
    assert err.count("CFL demands more than") == 1, "WARN must print exactly once"


def test_cli_threads_cfl_into_mpm():
    """Contract: the MPM CLI branch must pass the [simulation] cfl knobs into
    MpmConfig.adaptive_* (so the existing addon 'CFL Substepping' checkbox
    drives MPM, not just FLIP)."""
    src = (Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli"
           / "commands.py").read_text(encoding="utf-8")
    # §9.12 (audit-20260610): strip docstrings too, not only comment lines.
    src = re.sub(r'("""|\'\'\')[\s\S]*?\1', "", src)
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "adaptive_substep=bool(sim.cfl)" in code
    assert "adaptive_cfl=float(sim.cfl_factor)" in code
    assert "adaptive_max_substeps=int(sim.cfl_max_substeps)" in code
