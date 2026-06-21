"""Prod-hardening (2026-06-21, reviewer-io): read_points_ply layout guard.

read_points_ply assumed 12 bytes/vertex (pure xyz). An xyz+rgb PLY (15 B/vertex)
was SILENTLY misread as float coordinates -> garbage, no error (read_ply got a
vertex-layout guard in a past round; read_points_ply drifted — §9.6). The writer
only ever emits pure xyz, so the normal points pipeline is unaffected, but the
reader must reject coloured/mesh PLYs instead of returning nonsense (it's now
used by the over-compression diagnostic).
"""
from __future__ import annotations

import numpy as np
import pytest

from gpufluid.io.ply import write_points_ply, read_points_ply, write_ply
from gpufluid.blocks import BlockError


def test_pure_xyz_round_trip(tmp_path):
    pts = np.random.RandomState(0).rand(64, 3).astype(np.float32)
    f = tmp_path / "p.ply"
    write_points_ply(f, pts)
    assert np.allclose(read_points_ply(f), pts)


def test_empty_points(tmp_path):
    f = tmp_path / "e.ply"
    write_points_ply(f, np.empty((0, 3), dtype=np.float32))
    assert read_points_ply(f).shape == (0, 3)


def test_rejects_xyz_rgb_mesh_ply(tmp_path):
    """The core fix: a coloured/mesh PLY must RAISE, not silently read rgb bytes
    as coordinates."""
    f = tmp_path / "m.ply"
    verts = np.random.RandomState(1).rand(20, 3).astype(np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    cols = (np.random.RandomState(2).rand(20, 3) * 255).astype(np.uint8)
    write_ply(f, verts, faces, vertex_colors=cols)
    with pytest.raises(BlockError):
        read_points_ply(f)


def test_source_contract_layout_guard():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "io"
           / "ply.py").read_text(encoding="utf-8")
    assert 'names != ["x", "y", "z"]' in src, (
        "read_points_ply must validate the vertex property layout")
