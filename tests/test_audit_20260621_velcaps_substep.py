"""Prod-hardening (2026-06-21, reviewer-velcaps H4): velocity caps per sub-step.

The tap (terminal-velocity) + anti-splash vz caps are a *pre-P2G outlier
removal* — they must run before EVERY p2g2p so a sub-step can't seed a
corrupted deformation gradient F from a bad-velocity outlier. Pre-fix they
ran ONCE in `_pre_step` while the colliders' pushback + the drain were already
mirrored into the adaptive sub-step loop (FU-028). Under N-substep CFL that
left the caps ~Nx weaker than intended — a textbook §9.6 mirror gap (one of a
pair of sibling launches got the per-substep treatment, the other didn't).

The fix extracts `_apply_velcaps()` and calls it once per sub-step, right
after the pushback + drain. velcaps now has the SAME call-sites as pushback:
once in `_pre_step` (before the first p2g2p) + once per sub-step in the loop.

These are control-flow tests (no GPU): they spy the real `MpmSolver.step`
built via `object.__new__` (the FU-028 pattern) and count launches. The A/B
bake of demo_cascade2 confirmed the macro outcome is unchanged (drain 100%,
max_speed 1.670->1.665, compression 4.33->4.37) — i.e. the fix hardens the
sub-step path without regressing the tuned cascade, exactly as intended.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("warp")  # solver module imports warp at load

SOLVER = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "sim" / "mpm" / "solver.py"


def _velcaps_spy(adaptive, n_sub):
    """Real MpmSolver.step control flow with the three sibling sub-step
    launches (pushback / outflow / velcaps) replaced by counters (§9.7: same
    call signatures the loop uses). `_pre_step` is stubbed to a no-op so the
    counters isolate the LOOP / _post_step behaviour, not the pre-step launch."""
    from gpufluid.sim.mpm.solver import MpmSolver
    s = object.__new__(MpmSolver)
    counts = {"p2g2p": 0, "pushback": 0, "outflow": [], "velcaps": 0}
    s.cfg = SimpleNamespace(adaptive_substep=adaptive, dt=0.05, device="cpu",
                            dump_every=8, adaptive_max_substeps=16)
    s.mpm = SimpleNamespace(
        p2g2p=lambda *a, **k: counts.__setitem__("p2g2p", counts["p2g2p"] + 1))
    s._cfl_warned = True
    s._pre_step = lambda si: None
    s._apply_pushback = lambda: counts.__setitem__("pushback", counts["pushback"] + 1)
    s._apply_outflows = lambda si: counts["outflow"].append(si)
    s._apply_velcaps = lambda: counts.__setitem__("velcaps", counts["velcaps"] + 1)
    s._cfl_substeps = lambda: (n_sub, False)
    return s, counts


def test_velcaps_runs_once_per_substep():
    """Adaptive path: velcaps fires exactly n_sub times in the loop — the
    same count as pushback and drain. Pre-fix it fired 0 times in the loop
    (only once, in _pre_step) -> ~Nx weaker than the sibling pushback."""
    s, counts = _velcaps_spy(adaptive=True, n_sub=4)
    s.step(7)
    assert counts["p2g2p"] == 4
    assert counts["velcaps"] == 4, (
        "reviewer-velcaps H4 regressed: caps must run once per sub-step, not "
        f"once per raw step; got {counts['velcaps']}")


def test_velcaps_mirrors_pushback_and_drain_per_substep():
    """§9.6 mirror parity: in the adaptive loop velcaps, pushback and drain
    must all fire the SAME number of times. If any future edit moves one out
    of the loop, this catches the drift."""
    for n_sub in (1, 2, 5, 9):
        s, counts = _velcaps_spy(adaptive=True, n_sub=n_sub)
        s.step(3)
        assert counts["velcaps"] == counts["pushback"] == len(counts["outflow"]) == n_sub, (
            f"mirror drift at n_sub={n_sub}: velcaps={counts['velcaps']} "
            f"pushback={counts['pushback']} drain={len(counts['outflow'])}")


def test_nonadaptive_path_does_not_loop_velcaps():
    """Adaptive off: the single p2g2p runs, _post_step does pushback + drain,
    and velcaps is NOT re-launched in this path (the per-frame _pre_step caps
    already preceded the lone p2g2p; the next frame's _pre_step re-clamps
    before its p2g2p). Locks in the 'every p2g2p is preceded by a velcaps'
    invariant without an extra launch on the non-substep path."""
    s, counts = _velcaps_spy(adaptive=False, n_sub=1)
    s.step(5)
    assert counts["p2g2p"] == 1
    assert counts["pushback"] == 1
    assert counts["outflow"] == [5]
    assert counts["velcaps"] == 0, (
        "non-adaptive _post_step must not launch velcaps (it lives in "
        f"_pre_step on this path); got {counts['velcaps']}")


def test_velcaps_in_substep_loop_contract():
    """§9.12 grep: `self._apply_velcaps()` must appear after the per-substep
    `self._apply_outflows(step_index)` inside the adaptive loop, and the
    method itself must gate both kernels on `is not None` (the opt-out)."""
    code = "\n".join(l for l in SOLVER.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    assert "self._apply_outflows(step_index)\n                self._apply_velcaps()" in code, (
        "velcaps must be launched per sub-step, right after the drain")
    assert "def _apply_velcaps" in code
    assert "if cfg.tap is not None:" in code, "tap cap must stay opt-out-gated"
    assert "if cfg.anti_splash is not None:" in code, "anti-splash must stay opt-out-gated"
