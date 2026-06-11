"""FU-028 + FU-029 — solver-side follow-ups (2026-06-11).

FU-028: outflow drain ran once per RAW step, after all sub-steps — a fast
particle (the very reason adaptive substepping kicks in) could cross a THIN
drain box between two end-of-step checks and escape out the far side. The
drain now runs per sub-step; the frame window stays clocked in raw steps.

FU-029: `adaptive_cfl` reuses the [simulation] cfl_factor knob whose
FLIP-oriented slider allows 2.0; for the MPM stress-wave bound anything
> 1.0 silently under-substeps. The solver now warns ONCE (latched) and
honours the configured value (no silent clamp).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("warp")  # solver module imports warp at load


# ── FU-028: per-substep drain ───────────────────────────────────────────

def _drain_spy(n_sub, drain_lo, drain_hi, z0, vz):
    """Real MpmSolver control flow (object.__new__) with a 1-particle CPU
    trajectory: p2g2p advances z by vz*sub_dt; _apply_outflows is replaced
    by a recorder that evaluates the kernel's CPU-mirror predicate at the
    particle's CURRENT position (§9.7: same call signature/clock as prod)."""
    from gpufluid.sim.mpm.solver import MpmSolver
    from gpufluid.sim.mpm.outflow import predicate_in_box
    s = object.__new__(MpmSolver)
    state = {"z": z0, "despawned": False, "drain_calls": [], "zs": []}
    s.cfg = SimpleNamespace(adaptive_substep=True, dt=0.05, device="cpu",
                            dump_every=8, adaptive_max_substeps=16)

    def fake_p2g2p(step, sub_dt, device=None):
        state["z"] += vz * sub_dt

    s.mpm = SimpleNamespace(p2g2p=fake_p2g2p)
    s._cfl_warned = True
    s._outflow_params = []          # unused: _apply_outflows replaced below
    s._pre_step = lambda si: None
    s._apply_pushback = lambda: None
    s._cfl_substeps = lambda: (n_sub, False)

    def record_outflows(si):
        state["drain_calls"].append(si)
        state["zs"].append(state["z"])
        if predicate_in_box((0.5, 0.5, state["z"]), drain_lo, drain_hi,
                            si, 0, 1_000_000):
            state["despawned"] = True

    s._apply_outflows = record_outflows
    return s, state


def test_fast_particle_caught_by_per_substep_drain():
    """5 sub-steps, particle rises 0.04/sub-step through a 0.01-thin drain:
    one sub-step sample lands inside the box → must despawn. The OLD
    end-of-frame-only check would have missed it (final position is past
    the box) — asserted explicitly below."""
    from gpufluid.sim.mpm.outflow import predicate_in_box
    lo, hi = (0.0, 0.0, 0.415), (1.0, 1.0, 0.425)
    s, state = _drain_spy(n_sub=5, drain_lo=lo, drain_hi=hi, z0=0.30, vz=4.0)
    s.step(3)
    assert state["despawned"], (
        "FU-028 regressed: fast particle crossing a thin drain within one "
        "raw step must be caught by a per-sub-step check")
    # Honesty cross-check: the pre-FU-028 single end-of-step check would
    # NOT have caught it (final z is beyond the drain box).
    assert not predicate_in_box((0.5, 0.5, state["zs"][-1]), lo, hi,
                                3, 0, 1_000_000), (
        "test setup degenerated: the end-of-step position must lie PAST "
        "the drain for this to demonstrate tunnelling")


def test_drain_runs_once_per_substep_with_constant_raw_step():
    """The drain runs exactly n_sub times (not n_sub+1 — the trailing
    _post_step duplicate was dropped with FU-028) and every call carries
    the RAW step index, so the frame-window gate semantics ([frame_start,
    frame_end] * dump_every, compared against raw steps) are unchanged."""
    s, state = _drain_spy(n_sub=4, drain_lo=(0, 0, 0.9), drain_hi=(1, 1, 1),
                          z0=0.1, vz=0.0)
    s.step(7)
    assert state["drain_calls"] == [7, 7, 7, 7], (
        "FU-028: drain must run once per sub-step, clocked in RAW steps "
        f"(constant across sub-steps); got {state['drain_calls']}")


def test_nonadaptive_path_drains_once_per_step():
    """Adaptive off: exactly one pushback+drain pair per raw step via
    _post_step (pre-FU-028 behaviour preserved bit-for-bit)."""
    from gpufluid.sim.mpm.solver import MpmSolver
    s = object.__new__(MpmSolver)
    counts = {"p2g2p": 0, "drain": [], "pushback": 0}
    s.cfg = SimpleNamespace(adaptive_substep=False, dt=0.05, device="cpu",
                            dump_every=8, adaptive_max_substeps=16)
    s.mpm = SimpleNamespace(
        p2g2p=lambda *a, **k: counts.__setitem__("p2g2p", counts["p2g2p"] + 1))
    s._cfl_warned = True
    s._outflow_params = []
    s._pre_step = lambda si: None
    s._apply_pushback = lambda: counts.__setitem__(
        "pushback", counts["pushback"] + 1)
    s._apply_outflows = lambda si: counts["drain"].append(si)
    s.step(5)
    assert counts["p2g2p"] == 1
    assert counts["pushback"] == 1
    assert counts["drain"] == [5], (
        "non-adaptive path must drain exactly once per raw step")


def test_outflow_window_conversion_unchanged_contract():
    """§9.12 grep: the gate stays step-indexed — _outflow_params still
    converts output frames to raw steps via dump_every, and the despawn
    launch still passes the raw step_index (never a sub-step counter)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "sim"
           / "mpm" / "solver.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "int(of.frame_start) * cfg.dump_every" in code, (
        "FU-028: frame-window → raw-step conversion must stay dump_every-"
        "based")
    assert "self._apply_outflows(step_index)" in code, (
        "FU-028: drain launches must be keyed off the RAW step_index")


# ── FU-029: cfl_factor > 1.0 warning ────────────────────────────────────

def _cfl_spy(adaptive_cfl):
    """Spy exposing exactly the surface _cfl_substeps touches (§9.7)."""
    from gpufluid.sim.mpm.solver import MpmSolver
    s = object.__new__(MpmSolver)
    s.cfg = SimpleNamespace(
        adaptive_cfl=adaptive_cfl, adaptive_max_substeps=16, dt=1e-3,
        dx=lambda: 1.0 / 64,
        fluid=SimpleNamespace(bulk_modulus=1500.0, density=1000.0))
    s.mpm = SimpleNamespace(mpm_state=SimpleNamespace(
        particle_v=SimpleNamespace(
            numpy=lambda: np.zeros((4, 3), dtype=np.float32))))
    s._cfl_factor_warned = False
    return s


def test_cfl_factor_gt1_warns_once(capsys):
    """factor 2.0 (FLIP slider max): warn exactly once across frames, and
    the value must NOT be clamped (user intent wins — N still computed
    from the configured factor)."""
    s = _cfl_spy(2.0)
    n1, _ = s._cfl_substeps()
    n2, _ = s._cfl_substeps()
    err = capsys.readouterr().err
    assert err.count("cfl_factor=2 > 1.0") == 1, (
        f"FU-029: warning must fire exactly once; stderr was: {err!r}")
    assert "<= 1.0" in err, "FU-029: warning must suggest <= 1.0"
    # no clamp: with v_max=0, N = ceil(dt*c / (2.0*dx)) — the 2.0 stays in.
    import math
    c = math.sqrt(1500.0 / 1000.0)
    assert n1 == n2 == max(1, math.ceil(1e-3 * c / (2.0 * (1.0 / 64)))), (
        "FU-029: configured factor must be honoured, not clamped")


def test_cfl_factor_le1_no_warning(capsys):
    """factor at or below 1.0 (incl. the 0.5 default): silent."""
    for v in (1.0, 0.6, 0.5):
        s = _cfl_spy(v)
        s._cfl_substeps()
        s._cfl_substeps()
    assert "cfl_factor" not in capsys.readouterr().err, (
        "FU-029: no warning may fire for cfl_factor <= 1.0")
