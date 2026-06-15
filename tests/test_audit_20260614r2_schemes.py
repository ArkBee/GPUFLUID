"""Audit 2026-06-14 round 2 — BVH max_dist + plane zero-normal (#8, #14)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_CUDA
from gpufluid.blocks import BlockError

MARKER = (Path(__file__).resolve().parents[1] / "src" / "gpufluid"
          / "schemes" / "mesh_marker.py")


# ---- #8: the BVH winding query must not cap the search at 2 world units -----

def test_bvh_launch_uses_domain_spanning_max_dist_not_literal_2():
    code = "\n".join(l for l in MARKER.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    assert "mesh.id, 2.0, marker_wp" not in code, \
        "audit #8 regressed: BVH launch passes the literal 2.0 into max_dist"
    assert "np.linalg.norm(_tb.max(0) - _tb.min(0))" in code, \
        "audit #8: max_dist must be derived from the mesh extent (cover interior)"


# ---- 2026-06-15: wp.Mesh must enable winding-number support ----------------

def test_wp_mesh_built_with_support_winding_number():
    """k3_mesh_to_solid_bvh queries mesh_query_point_sign_winding_number, which
    Warp documents as requiring the wp.Mesh be built with
    support_winding_number=True (ctor default is False). Empirically Warp 1.13
    returns correct signs without it for watertight meshes, but that is
    undocumented — every wp.Mesh ctor in mesh_marker must set the flag so a
    future Warp build can't silently return wrong inside/outside signs (fluid
    leaking through mesh solids)."""
    code = "\n".join(l for l in MARKER.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    n_ctor = code.count("wp.Mesh(points=")
    n_flag = code.count("support_winding_number=True")
    assert n_ctor >= 2, "expected >=2 wp.Mesh constructions in mesh_marker"
    assert n_flag == n_ctor, (
        f"every wp.Mesh must pass support_winding_number=True "
        f"({n_ctor} ctors but {n_flag} flagged)")


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_bvh_marks_deep_interior_of_large_mesh():
    """A cell deep inside a large mesh (nearest face > 2 world units away) must
    be marked solid — the old literal-2.0 max_dist culled it."""
    import warp as wp
    trimesh = pytest.importorskip("trimesh")
    from gpufluid.schemes.mesh_marker import mark_solid_from_mesh_gpu

    box = trimesh.creation.box(extents=(6.0, 6.0, 6.0))   # solid [-3, 3]^3
    tris = np.asarray(box.triangles, dtype=np.float32)
    N, dx = 16, 8.0 / 16.0
    ax = (np.arange(N) + 0.5) * dx - 4.0                  # cell centres over [-4, 4]
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    cc = np.stack([gx, gy, gz], axis=-1).astype(np.float32)
    marker = wp.zeros((N, N, N), dtype=int)
    mark_solid_from_mesh_gpu(marker, cc, tris, use_bvh=True)
    m = marker.numpy()
    c = N // 2                                            # centre cell ~2.75 u from a face
    assert m[c, c, c] == 2, \
        "deep interior cell of a large mesh not marked solid (audit #8 max_dist)"


# ---- #14: a zero plane normal must be rejected, not silently all-solid ------

def test_plane_zero_normal_rejected_at_config_parse(tmp_path):
    from gpufluid.cli.config import load_scene
    body = """
[domain]
resolution = [16, 16, 16]
[[fluids]]
type = "box"
lo = [0.1, 0.1, 0.1]
hi = [0.3, 0.3, 0.3]
[[obstacle]]
type = "plane"
point = [0.5, 0.5, 0.5]
normal = [0.0, 0.0, 0.0]
[simulation]
solver = "mpm"
dt = 0.005
frames = 2
fps = 24
[output]
cache_dir = "out/x"
"""
    p = tmp_path / "s.toml"; p.write_text(body, encoding="utf-8")
    with pytest.raises(BlockError, match="normal must be non-zero"):
        load_scene(p)


def test_sdf_plane_zero_normal_raises():
    from gpufluid.primitives.sdf import sdf_plane
    grid = np.zeros((4, 3), np.float32)
    with pytest.raises(ValueError, match="non-zero"):
        sdf_plane(grid, [0, 0, 0], [0, 0, 0])
    # a real normal still works
    out = sdf_plane(grid, [0, 0, 0], [0, 0, 1])
    assert out.shape == (4,)
