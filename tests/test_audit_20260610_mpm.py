"""audit-20260610 — fixes from the independent MPM/CLI review round.

Covers:
  1. B18 sidecars × S2.17.8 outflow: when a drain is configured, per-frame
     attribute sidecars must be written UNCONDITIONALLY — a drain mutates
     selection row-identity without necessarily changing the row count, so
     the round-24 count-equality dedup would let the mesher apply frame-0
     attribute rows to the wrong particles. Dedup stays for drain-free bakes.
  2. Sphere source seeding 0 particles exits rc=2 with a clean message
     (was: raw ValueError traceback out of MpmSolver.__init__).
  3. Sphere/mesh seed clouds are clamped to [eps, 1-eps]^3 before solver
     init (OOB seeds slip-flatten onto walls + pollute the frame-0 PLY,
     which is dumped before any pushback runs).
  4. Mesh-source 0-particle rc=2 message names the out-of-domain/scaling
     cause (non-cubic domain transform), not only watertightness.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("warp")  # solver/commands modules import warp at load

REPO = Path(__file__).resolve().parents[1]


def _code(relpath: str) -> str:
    """Source with comment lines stripped (§9.12: don't let the fix's own
    comments satisfy/trigger the grep)."""
    src = (REPO / relpath).read_text(encoding="utf-8")
    return "\n".join(
        l for l in src.splitlines() if not l.lstrip().startswith("#")
    )


# ── Fix 1: sidecar/outflow alignment ────────────────────────────────────

def _spy_sidecar_solver(outflows, sel_holder, colors, temps, inflows=()):
    """Real MpmSolver instance with no GPU state (object.__new__), wired
    with attribute-strict stubs covering exactly the surface save_frame_ply
    touches (§9.7 mock fidelity): cfg.dump_every/inflows/outflows,
    selection(), positions(), attr_color.numpy(), attr_temperature.numpy().

    demo30 2026-06-13: a bake is dedup-eligible ("static") only when BOTH
    inflows and outflows are empty; the spy must expose inflows too."""
    from gpufluid.sim.mpm.solver import MpmSolver
    s = object.__new__(MpmSolver)
    s.cfg = SimpleNamespace(dump_every=1, inflows=inflows, outflows=outflows)
    s.selection = lambda: np.asarray(sel_holder["sel"], dtype=int)
    s.positions = lambda: np.zeros(
        (len(sel_holder["sel"]), 3), dtype=np.float32)
    s.attr_color = SimpleNamespace(numpy=lambda: colors)
    s.attr_temperature = (
        SimpleNamespace(numpy=lambda: temps) if temps is not None else None)
    return s


def test_outflow_forces_per_frame_sidecars(tmp_path):
    """Drain configured + live COUNT returns to the frame-0 value with a
    different row IDENTITY (row 3 drained, row 4 spawned) — the per-frame
    sidecar must exist and carry the NEW identity's rows."""
    from gpufluid.sim.mpm.outflow import MpmOutflow
    colors = np.arange(18, dtype=np.float32).reshape(6, 3)
    temps = np.arange(6, dtype=np.float32) + 100.0
    holder = {"sel": [0, 0, 0, 0, 1, 1]}  # frame 0: rows 0-3 live
    s = _spy_sidecar_solver(
        outflows=(MpmOutflow(lo=(0, 0, 0), hi=(1, 1, 0.2)),),
        sel_holder=holder, colors=colors, temps=temps)
    out_dir = tmp_path / "particles_raw"
    s.save_frame_ply(out_dir, 0)
    # frame 1: row 3 drained (0→1), row 4 spawned (1→0) — count still 4
    holder["sel"] = [0, 0, 0, 1, 0, 1]
    s.save_frame_ply(out_dir, 1)
    col_f1 = tmp_path / "colors" / "frame_0001.npy"
    assert col_f1.exists(), (
        "audit-20260610: outflow configured -> per-frame colour sidecar "
        "must be written even when the live count equals frame 0's")
    assert np.array_equal(np.load(col_f1), colors[[0, 1, 2, 4]]), (
        "audit-20260610: per-frame sidecar must carry the CURRENT mask's "
        "rows, not frame 0's")
    tmp_f1 = tmp_path / "temperatures" / "frame_0001.npy"
    assert tmp_f1.exists(), (
        "audit-20260610: temperature sidecar path must mirror the colour "
        "path (same invariant)")
    assert np.array_equal(np.load(tmp_f1), temps[[0, 1, 2, 4]])


def test_dedup_preserved_without_outflow(tmp_path):
    """Drain-free bake with an unchanged mask: round-24 dedup must still
    hold — frame 0 written once, no per-frame duplicates on disk."""
    colors = np.ones((6, 3), dtype=np.float32)
    holder = {"sel": [0, 0, 0, 0, 1, 1]}
    s = _spy_sidecar_solver(
        outflows=(), sel_holder=holder, colors=colors, temps=None)
    out_dir = tmp_path / "particles_raw"
    s.save_frame_ply(out_dir, 0)
    s.save_frame_ply(out_dir, 1)
    assert (tmp_path / "colors" / "frame_0000.npy").exists()
    assert not (tmp_path / "colors" / "frame_0001.npy").exists(), (
        "audit-20260610: drain-free bakes must keep the round-24 dedup "
        "(identical mask -> no per-frame sidecar)")


def test_dedup_still_snapshots_on_count_change_without_outflow(tmp_path):
    """Inflow (no outflow) bake where a spawn grows the live count: a
    per-frame snapshot carrying the CURRENT rows must be written. demo30
    2026-06-13: an inflow alone makes the bake dynamic, so per-frame
    sidecars are written every dump (re-bake-safe)."""
    colors = np.arange(18, dtype=np.float32).reshape(6, 3)
    holder = {"sel": [0, 0, 0, 0, 1, 1]}
    s = _spy_sidecar_solver(
        outflows=(), inflows=(SimpleNamespace(color=None, temperature=None),),
        sel_holder=holder, colors=colors, temps=None)
    out_dir = tmp_path / "particles_raw"
    s.save_frame_ply(out_dir, 0)
    holder["sel"] = [0, 0, 0, 0, 0, 1]  # row 4 spawned -> count 4 -> 5
    s.save_frame_ply(out_dir, 1)
    col_f1 = tmp_path / "colors" / "frame_0001.npy"
    assert col_f1.exists()
    assert np.array_equal(np.load(col_f1), colors[[0, 1, 2, 3, 4]])


def test_solver_sidecar_invariant_contract():
    """§9.12 grep: the dedup must be gated on a fully-static bake (no inflow
    AND no outflow). demo30 2026-06-13 widened the trigger from
    outflow-only to any selection-mutating source."""
    code = _code("src/gpufluid/sim/mpm/solver.py")
    assert "attrs_static = not self.cfg.inflows and not self.cfg.outflows" \
        in code, (
        "demo30: sidecar dedup must apply ONLY to a fully-static bake "
        "(inflow OR outflow makes it dynamic -> per-frame sidecars)")


# ── Fix 2: sphere source 0-particle clean exit ──────────────────────────

def test_sphere_zero_particles_clean_rc2(tmp_path, capsys):
    """Degenerate sphere (radius <= 0) must exit rc=2 with a message, not
    raise ValueError from MpmSolver.__init__."""
    from gpufluid.cli.config import load_scene
    from gpufluid.cli.commands import _cmd_simulate_mpm
    toml = tmp_path / "scene.toml"
    toml.write_text(
        "[domain]\n"
        "resolution = [32, 32, 32]\n"
        "\n"
        "[[fluids]]\n"
        'type = "sphere"\n'
        "center = [0.5, 0.5, 0.5]\n"
        "radius = -0.1\n"
        "\n"
        "[simulation]\n"
        'solver = "mpm"\n'
        "frames = 1\n"
        "\n"
        "[output]\n"
        f'cache_dir = "{(tmp_path / "cache").as_posix()}"\n',
        encoding="utf-8",
    )
    scene = load_scene(toml)
    rc = _cmd_simulate_mpm(SimpleNamespace(), scene)
    assert rc == 2, (
        "audit-20260610: degenerate sphere must exit rc=2, not traceback")
    err = capsys.readouterr().err
    assert "sphere" in err and "seeded 0" in err, (
        "audit-20260610: rc=2 must come with a sphere-specific message")


# ── Fix 3: seed clamp to [eps, 1-eps]^3 ─────────────────────────────────

def test_clamp_to_unit_box_bounds_and_dtype():
    from gpufluid.sim.mpm.seeding import clamp_to_unit_box
    pts = np.array([[-0.2, 0.5, 1.3],
                    [0.5, 0.5, 0.5]], dtype=np.float32)
    out = clamp_to_unit_box(pts, 0.05)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.05 - 1e-7
    assert float(out.max()) <= 0.95 + 1e-7
    # interior points untouched
    assert np.allclose(out[1], [0.5, 0.5, 0.5])


def test_edge_sphere_seeds_clamp_inside():
    """A sphere poking out of the domain top seeds raw points with z > 1;
    after the clamp every point is inside the margin."""
    from gpufluid.sim.mpm.seeding import seed_sphere, clamp_to_unit_box
    raw = seed_sphere((0.5, 0.5, 0.95), 0.2, 1.0 / 32)
    assert float(raw[:, 2].max()) > 1.0  # precondition: OOB without clamp
    eps = 1.5 / 32
    clamped = clamp_to_unit_box(raw, eps)
    assert float(clamped.max()) <= 1.0 - eps + 1e-7
    assert float(clamped.min()) >= eps - 1e-7


def test_cli_clamps_sphere_and_mesh_seeds_contract():
    """§9.12 grep: both non-box branches must clamp before solver init."""
    code = _code("src/gpufluid/cli/commands.py")
    n = code.count("clamp_to_unit_box(column, 1.5 * scene.dx)")
    assert n == 2, (
        f"audit-20260610: sphere AND mesh branches must clamp seeds to the "
        f"1.5-cell margin (found {n} call sites, want 2)")


# ── Fix 4: mesh-source rc=2 message names the OOB cause ─────────────────

def test_mesh_zero_particle_message_names_oob_cause_contract():
    """§9.12 grep: the 0-particle message must mention BOTH known causes —
    bad mesh AND source outside [0,1]^3 after the (non-cubic, uniform)
    scale/translate transform."""
    code = _code("src/gpufluid/cli/commands.py")
    assert "not watertight" in code, (
        "audit-20260610: watertight cause dropped from mesh-source message")
    assert "outside [0,1]^3 after scale/translate" in code, (
        "audit-20260610: mesh-source 0-particle message must name the "
        "out-of-domain cause")
    assert "non-cubic" in code, (
        "audit-20260610: message must point at the non-cubic-domain "
        "uniform-scale trap (FU-019 family)")
