"""Tests for M5.5 mesh smoothing."""
import numpy as np
import pytest

from gpufluid.meshing.smoothing import smooth_laplacian, smooth_taubin


def _noisy_sphere(n_subdiv=2, noise=0.05, seed=0):
    """Build a faceted sphere and jitter vertices."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=n_subdiv, radius=1.0)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.int32)
    rng = np.random.default_rng(seed)
    verts = verts + noise * rng.standard_normal(verts.shape).astype(np.float32)
    return verts, faces


def _vertex_normals_roughness(verts, faces):
    """Average distance from each vertex to mean of 1-ring neighbours.
    Lower = smoother."""
    n = len(verts)
    sums = np.zeros_like(verts)
    counts = np.zeros(n, dtype=np.int32)
    for tri in faces:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            sums[tri[a]] += verts[tri[b]]
            counts[tri[a]] += 1
            sums[tri[b]] += verts[tri[a]]
            counts[tri[b]] += 1
    counts = np.maximum(counts, 1)
    avg = sums / counts[:, None]
    return float(np.linalg.norm(verts - avg, axis=1).mean())


def test_m5_5_laplacian_reduces_roughness():
    v, f = _noisy_sphere(noise=0.05)
    r0 = _vertex_normals_roughness(v, f)
    v2 = smooth_laplacian(v, f, passes=3, lam=0.5)
    r1 = _vertex_normals_roughness(v2, f)
    assert r1 < r0 * 0.5, f"smoothing should halve roughness; r0={r0:.4f} r1={r1:.4f}"


def test_m5_5_taubin_preserves_volume_roughly():
    """Taubin should not shrink a noisy sphere as aggressively as Laplacian."""
    v, f = _noisy_sphere(n_subdiv=3, noise=0.02)
    r0 = float(np.linalg.norm(v, axis=1).mean())
    v_lap = smooth_laplacian(v, f, passes=10, lam=0.5)
    v_tau = smooth_taubin(v, f, passes=10, lam=0.5, mu=-0.53)
    r_lap = float(np.linalg.norm(v_lap, axis=1).mean())
    r_tau = float(np.linalg.norm(v_tau, axis=1).mean())
    # Laplacian shrinks more than Taubin (relative comparison, robust to
    # particular parameter choices).
    assert r_lap < r0, f"Laplacian should shrink: r0={r0} r_lap={r_lap}"
    assert r_tau > r_lap, f"Taubin should preserve more volume than Laplacian: tau={r_tau} lap={r_lap}"


def test_m5_5_empty_mesh_no_op():
    v = np.zeros((0, 3), dtype=np.float32)
    f = np.zeros((0, 3), dtype=np.int32)
    out = smooth_taubin(v, f, passes=3)
    assert out.shape == v.shape
    out2 = smooth_laplacian(v, f, passes=3)
    assert out2.shape == v.shape
