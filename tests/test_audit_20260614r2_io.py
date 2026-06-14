"""Audit 2026-06-14 round 2 — io/ply.py + io/cache.py (#1, #4, #11).

#1  read_ply silently scrambled meshes with a non-triangle (quad) face.
#4  read_ply silently mis-parsed PLYs with extra vertex props (e.g. normals)
    despite the docstring claiming they are rejected.
#11 CacheManifest.mesh_path() crashed (AttributeError) on mesh_pattern=null.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from gpufluid.blocks import BlockError
from gpufluid.io.ply import read_ply, write_ply
from gpufluid.io.cache import CacheManifest


def _ply(vert_props, verts_bytes, n_v, faces_bytes, n_f):
    h = ["ply", "format binary_little_endian 1.0", f"element vertex {n_v}"]
    h += [f"property {t} {n}" for (t, n) in vert_props]
    h += [f"element face {n_f}", "property list uchar int vertex_indices",
          "end_header", ""]
    return ("\n".join(h)).encode("ascii") + verts_bytes + faces_bytes


# ---- #1: a quad face must be rejected, not silently scrambled -------------

def test_read_ply_rejects_quad_face(tmp_path):
    verts = struct.pack("<12f", *([0.0]*12))            # 4 xyz vertices
    quad = struct.pack("<B4i", 4, 0, 1, 2, 3)           # count=4 -> not a triangle
    p = tmp_path / "quad.ply"
    p.write_bytes(_ply([("float", "x"), ("float", "y"), ("float", "z")],
                       verts, 4, quad, 1))
    with pytest.raises(BlockError, match="non-triangle"):
        read_ply(p)


# ---- #4: extra vertex properties must be rejected (docstring contract) ----

def test_read_ply_rejects_xyz_plus_normals(tmp_path):
    vp = [("float", n) for n in ("x", "y", "z", "nx", "ny", "nz")]
    verts = struct.pack("<12f", *([0.0]*12))            # 2 vertices * 6 floats
    p = tmp_path / "normals.ply"
    p.write_bytes(_ply(vp, verts, 2, b"", 0))
    with pytest.raises(BlockError, match="unsupported PLY vertex layout"):
        read_ply(p)


def test_read_ply_still_round_trips_xyz_and_rgb(tmp_path):
    v = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    # plain xyz
    p1 = tmp_path / "xyz.ply"; write_ply(p1, v, f)
    rv, rf = read_ply(p1)
    assert np.allclose(rv, v) and np.array_equal(rf, f)
    # xyz + rgb
    c = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], np.uint8)
    p2 = tmp_path / "rgb.ply"; write_ply(p2, v, f, vertex_colors=c)
    rv2, rf2, rc2 = read_ply(p2, return_colors=True)
    assert np.allclose(rv2, v) and rc2 is not None and np.array_equal(rc2, c)


# ---- #11: mesh_path None-guard -------------------------------------------

def test_cache_manifest_mesh_path_none_when_pattern_null(tmp_path):
    m = CacheManifest(fps=24, frame_count=5, mesh_pattern=None,
                      particles_pattern="particles_raw/sim_{:010d}.ply")
    assert m.mesh_path(tmp_path, 3) is None          # was AttributeError
    assert m.particles_path(tmp_path, 3) is not None  # sibling still works
