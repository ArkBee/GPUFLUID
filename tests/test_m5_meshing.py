"""Layer M5 tests — particle → density → marching cubes.

Covers [BLK M5.1] density scatter, [BLK M5.2] density box-blur,
[BLK M5.3] CPU marching cubes path, and [BLK M5.7] GPU wall-mask."""
import numpy as np
import pytest
import warp as wp

from gpufluid import MeshExtractor

pytestmark = pytest.mark.gpu


def _uniform_box_particles(lo, hi, n_per_axis):
    xs = np.linspace(lo[0], hi[0], n_per_axis, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], n_per_axis, dtype=np.float32)
    zs = np.linspace(lo[2], hi[2], n_per_axis, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)


def test_m5_3_uniform_box_mesh_bbox_matches():
    """Particles densely filling a box should produce a mesh whose bbox is within
    a couple of cells of the input box."""
    nx, ny, nz, dx = 32, 32, 32, 1.0 / 32
    extractor = MeshExtractor(nx, ny, nz, dx)
    lo = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    hi = np.array([0.75, 0.75, 0.75], dtype=np.float32)
    pos = _uniform_box_particles(lo, hi, n_per_axis=20)
    pos_wp = wp.array(pos, dtype=wp.vec3)
    verts, faces = extractor.extract(pos_wp, iso_level=0.3, smooth_passes=1)
    assert verts is not None and len(verts) > 0
    assert faces is not None and len(faces) > 0
    mesh_lo = verts.min(axis=0)
    mesh_hi = verts.max(axis=0)
    # Tolerance: a few cells of bleeding from trilinear scatter + smoothing.
    tol = 4 * dx
    assert np.all(mesh_lo >= lo - tol)
    assert np.all(mesh_hi <= hi + tol)
    assert np.all(np.abs(mesh_lo - lo) < tol)
    assert np.all(np.abs(mesh_hi - hi) < tol)


def test_m5_3_no_particles_returns_none():
    extractor = MeshExtractor(8, 8, 8, 0.1)
    empty = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)
    # extract with 0 particles → no mesh
    verts, faces = extractor.extract(empty, iso_level=0.5, smooth_passes=0)
    assert verts is None and faces is None
