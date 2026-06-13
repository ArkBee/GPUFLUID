"""demo30 2026-06-13 — coloured inflow/outflow water lost ALL per-vertex
colour in the meshed PLY (rendered grey).

Root cause: the colour/temperature sidecar dedup wrote frame 0 once and
skipped any per-frame file that already existed (`if not target.exists()`).
On a RE-BAKE into an existing cache dir the PLYs were rewritten but the
sidecars were NOT — they stayed at the previous bake's live counts. The
mesher's `loaded.shape[0] == pts.shape[0]` guard then failed for every
frame where the particle count had shifted (inflow grows it, outflow
shrinks it), so `cols_np` stayed None and the PLY was written uncoloured.

Fix: on any inflow/outflow bake, write the per-frame sidecar EVERY dump,
always overwriting, so it is byte-aligned with that frame's own PLY.
Dedup is kept only for fully-static bakes (selection never mutates).

These tests use the SimpleNamespace spy-solver pattern (no GPU): a real
MpmSolver via object.__new__ with attribute-strict stubs covering exactly
the surface save_frame_ply touches (§9.7 mock fidelity).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("warp")  # solver module imports warp at load

from gpufluid.io.ply import read_points_ply


_BLUE = np.array([0.20, 0.55, 0.90], dtype=np.float32)


def _spy_solver(n_part, inflows, outflows, sel_holder, temps=None):
    """Real MpmSolver control flow, no GPU. colours are all-blue so we can
    assert the source colour survives; selection is read from sel_holder so
    the test can mutate the live set between save_frame_ply calls."""
    from gpufluid.sim.mpm.solver import MpmSolver
    s = object.__new__(MpmSolver)
    s.cfg = SimpleNamespace(dump_every=5, inflows=inflows, outflows=outflows)
    cols = np.tile(_BLUE, (n_part, 1))
    s.positions = lambda: np.tile(
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32), (n_part, 1))
    s.selection = lambda: np.asarray(sel_holder["sel"], dtype=int)
    s.attr_color = SimpleNamespace(numpy=lambda: cols)
    s.attr_temperature = (
        SimpleNamespace(numpy=lambda: np.asarray(temps, dtype=np.float32))
        if temps is not None else None)
    return s


def _live(sel):
    return int((np.asarray(sel) == 0).sum())


def test_rebake_keeps_sidecar_aligned_with_ply(tmp_path):
    """THE demo30 bug: bake, then RE-BAKE into the same dir with a different
    live-count trajectory. Every frame's sidecar must match its PLY row
    count (else the mesher drops colour)."""
    inflow = (SimpleNamespace(color=tuple(_BLUE), temperature=None),)
    outflow = (object(),)
    n = 6
    holder = {"sel": [0, 0, 1, 1, 1, 1]}
    out = tmp_path / "particles_raw"

    # --- first bake: inflow grows the live set 2 -> 4 -> 6
    s = _spy_solver(n, inflow, outflow, holder)
    for f, sel in enumerate([[0, 0, 1, 1, 1, 1],
                             [0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 0]]):
        holder["sel"] = sel
        s.save_frame_ply(out, f * 5)

    # --- RE-BAKE into the SAME dir with a DIFFERENT trajectory (outflow now
    #     drains some, so counts at the same frames differ from bake 1)
    holder2 = {"sel": [0, 0, 1, 1, 1, 1]}
    s2 = _spy_solver(n, inflow, outflow, holder2)
    traj = [[0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],   # fewer live than bake-1 frame 1
            [0, 0, 0, 0, 1, 1]]   # fewer live than bake-1 frame 2
    for f, sel in enumerate(traj):
        holder2["sel"] = sel
        s2.save_frame_ply(out, f * 5)

    colors_dir = tmp_path / "colors"
    for f, sel in enumerate(traj):
        pts = read_points_ply(out / f"sim_{f * 5:010d}.ply")
        cp = colors_dir / f"frame_{f:04d}.npy"
        assert cp.exists(), (
            f"demo30: per-frame sidecar frame_{f:04d}.npy must exist on a "
            f"dynamic re-bake (no stale-skip)")
        loaded = np.load(cp)
        assert loaded.shape[0] == pts.shape[0] == _live(sel), (
            f"demo30 regressed: frame {f} sidecar rows {loaded.shape[0]} != "
            f"PLY pts {pts.shape[0]} — the mesher's count guard would drop "
            f"colour")
        # And the colour is the source blue, not grey/zero.
        assert np.allclose(loaded, _BLUE), (
            "demo30: inflow particles must carry the source colour")


def test_temperature_sidecar_rebake_aligned(tmp_path):
    """Same re-bake invariant for the temperature sidecar (§9.6 mirror)."""
    inflow = (SimpleNamespace(color=None, temperature=42.0),)
    n = 6
    holder = {"sel": [0, 0, 0, 0, 1, 1]}
    out = tmp_path / "particles_raw"
    s = _spy_solver(n, inflow, (object(),), holder,
                    temps=[42.0] * n)
    s.save_frame_ply(out, 0)
    holder["sel"] = [0, 0, 0, 1, 1, 1]   # one drained -> fewer live
    s2 = _spy_solver(n, inflow, (object(),), holder, temps=[42.0] * n)
    s2.save_frame_ply(out, 0)  # re-bake frame 0 with new count
    cp = tmp_path / "temperatures" / "frame_0000.npy"
    pts = read_points_ply(out / "sim_0000000000.ply")
    loaded = np.load(cp)
    assert loaded.shape[0] == pts.shape[0], (
        "demo30: temperature sidecar must be rewritten to match the "
        "re-baked PLY count")


def test_static_bake_still_dedups(tmp_path):
    """No inflow + no outflow → selection never mutates → keep the round-24
    single-file dedup (no per-frame duplicates on disk)."""
    holder = {"sel": [0, 0, 0, 0]}
    s = _spy_solver(4, inflows=(), outflows=(), sel_holder=holder)
    for f in range(3):
        s.save_frame_ply(tmp_path / "particles_raw", f * 5)
    files = sorted(p.name for p in (tmp_path / "colors").glob("frame_*.npy"))
    assert files == ["frame_0000.npy"], (
        f"static bake must keep single-file dedup, got {files}")


def test_inflow_only_writes_per_frame(tmp_path):
    """Inflow with NO outflow is still dynamic (live count grows) → must
    write per-frame sidecars (the audit-20260610 case also covered, but the
    demo30 fix widens the trigger from outflow-only to any inflow OR
    outflow)."""
    inflow = (SimpleNamespace(color=tuple(_BLUE), temperature=None),)
    holder = {"sel": [0, 0, 1, 1]}
    s = _spy_solver(4, inflow, outflows=(), sel_holder=holder)
    s.save_frame_ply(tmp_path / "particles_raw", 0)
    holder["sel"] = [0, 0, 0, 0]
    s.save_frame_ply(tmp_path / "particles_raw", 5)
    assert (tmp_path / "colors" / "frame_0001.npy").exists(), (
        "demo30: an inflow bake (no outflow) must write per-frame sidecars")


def test_no_stale_skip_grep_contract():
    """§9.12: the per-frame write must NOT be gated on file absence — that
    existence check is exactly what reused a stale sidecar on re-bake."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "sim"
           / "mpm" / "solver.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    # The dynamic branch saves unconditionally; ensure no `not target.exists()`
    # guard sits in the sidecar writer.
    assert "not target.exists()" not in code, (
        "demo30 regressed: a `not target.exists()` guard reintroduces "
        "stale-sidecar reuse on re-bake")
