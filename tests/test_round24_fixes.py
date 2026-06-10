"""Round-24 regression tests for deferred follow-ups from round-21/23.

  - sidecar de-dup: solver writes the colour/temp sidecar ONCE at
    frame 0; subsequent frames omit the rewrite when shape unchanged.
  - sidecar fallback: CLI mesher reads frame_0000.npy when the
    per-frame .npy is absent (round-24 contract).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


# ─── solver sidecar de-dup ──────────────────────────────────────────────

def _build_solver_with_attrs(tmp_path: Path):
    """Build a stand-in MpmSolver with attr_color set so save_frame_ply
    exercises the sidecar branch. Skips warp/h5py heavy __init__."""
    from gpufluid.sim.mpm.solver import MpmSolver
    sol = object.__new__(MpmSolver)
    # audit-20260610: save_frame_ply now reads cfg.outflows (sidecar dedup
    # is only valid for drain-free bakes); §9.7 mock fidelity — the spy must
    # expose the full surface the production code touches. Empty tuple =
    # drain-free, preserving the round-24 dedup semantics under test.
    sol.cfg = SimpleNamespace(dump_every=5, n_frames=20, outflows=())

    n_part = 8
    pos = np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (n_part, 1))
    sel = np.zeros(n_part, dtype=np.int32)
    cols = np.tile(np.array([[0.5, 0.5, 0.5]], dtype=np.float32), (n_part, 1))

    sol.positions = lambda: pos
    sol.selection = lambda: sel
    sol.attr_color = SimpleNamespace(numpy=lambda: cols)
    sol.attr_temperature = None
    return sol, tmp_path / "particles_raw"


def test_sidecar_written_once_at_frame_0(tmp_path: Path):
    """Round-24: first dump creates frame_0000.npy."""
    sol, out_dir = _build_solver_with_attrs(tmp_path)
    sol.save_frame_ply(out_dir, 0)
    colors_dir = out_dir.parent / "colors"
    assert (colors_dir / "frame_0000.npy").exists()


def test_sidecar_skipped_on_unchanged_shape(tmp_path: Path):
    """Round-24: subsequent dumps with same particle count must NOT
    create per-frame sidecars (de-dup). Pre-round-24 every frame got
    its own identical .npy → MBs of redundancy."""
    sol, out_dir = _build_solver_with_attrs(tmp_path)
    for step in (0, 5, 10, 15, 20):
        sol.save_frame_ply(out_dir, step)
    colors_dir = out_dir.parent / "colors"
    files = sorted(p.name for p in colors_dir.glob("frame_*.npy"))
    # Only frame_0000 should exist; frame_0001..0004 would be duplicates.
    assert files == ["frame_0000.npy"], (
        f"expected single-file de-dup, got: {files}")


def test_sidecar_written_when_particle_count_changes(tmp_path: Path):
    """Round-24: if the live particle count drifts (mid-bake spawn /
    death), a per-frame snapshot IS written so colours stay aligned."""
    from gpufluid.sim.mpm.solver import MpmSolver
    sol = object.__new__(MpmSolver)
    # audit-20260610: outflows=() — see _build_solver_with_attrs (§9.7).
    sol.cfg = SimpleNamespace(dump_every=5, n_frames=20, outflows=())

    state = {"count": 8}
    cols_full = np.tile(np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
                        (8, 1))

    sol.positions = lambda: np.tile(np.array([[0.1, 0.2, 0.3]],
                                              dtype=np.float32),
                                     (state["count"], 1))
    sol.selection = lambda: np.zeros(state["count"], dtype=np.int32)
    # attr_color sized to MAX particle count; mask filters to live.
    sol.attr_color = SimpleNamespace(numpy=lambda: cols_full[:state["count"]])
    sol.attr_temperature = None
    out_dir = tmp_path / "particles_raw"

    sol.save_frame_ply(out_dir, 0)
    state["count"] = 6   # someone died
    sol.save_frame_ply(out_dir, 5)
    colors_dir = out_dir.parent / "colors"
    files = sorted(p.name for p in colors_dir.glob("frame_*.npy"))
    # Both files must exist; frame_0001 carries the resized snapshot.
    assert files == ["frame_0000.npy", "frame_0001.npy"], (
        f"expected per-frame on shape change, got: {files}")
    # And the second file's content reflects the smaller count.
    a = np.load(colors_dir / "frame_0001.npy")
    assert a.shape[0] == 6
