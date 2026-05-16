"""[Layer M5 / BLK M5.5] Mesh smoothing (Laplacian / Taubin).

These run on CPU (small mesh sizes; trimesh-style). They take and return
the (verts, faces) tuple as produced by :class:`MeshExtractor.extract`.

Two algorithms:

* **Laplacian** — replace each vertex with the average of its 1-ring neighbours.
  Simple, strongly diffusive: it shrinks the mesh and rounds features.
  Good for 1-2 passes on noisy point-cloud surfaces.

* **Taubin** — alternates a Laplacian shrink step (positive λ) with an
  "inflate" step (negative μ), giving a low-pass smooth that does NOT
  shrink. Use when you want many passes without volume loss.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from ..blocks import block


def _build_laplacian(faces: np.ndarray, n_v: int) -> sp.csr_matrix:
    """Row-normalised uniform Laplacian: L = D^-1 A - I (so L @ v = mean_neighbour - v)."""
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                           faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                           faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones_like(rows, dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_v, n_v)).tocsr()
    # de-duplicate (each edge appears twice → values become 2; we want 1)
    A.data = np.minimum(A.data, 1.0)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    D_inv = sp.diags(1.0 / deg)
    return (D_inv @ A) - sp.eye(n_v, format="csr")


# [BLK M5.5]
@block("M5.5", "Laplacian mesh smoothing (uniform 1-ring averaging, N passes)")
def smooth_laplacian(verts: np.ndarray, faces: np.ndarray,
                     passes: int = 1, lam: float = 0.5) -> np.ndarray:
    """Return smoothed vertex positions. faces unchanged."""
    if verts is None or len(verts) == 0 or passes <= 0:
        return verts
    n_v = len(verts)
    L = _build_laplacian(faces, n_v)
    v = verts.astype(np.float32, copy=True)
    for _ in range(passes):
        v = v + lam * (L @ v)
    return v.astype(np.float32)


# [BLK M5.5]
@block("M5.5", "Taubin mesh smoothing (volume-preserving λ/μ alternation)")
def smooth_taubin(verts: np.ndarray, faces: np.ndarray,
                  passes: int = 2, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Return smoothed vertex positions using Taubin lambda/mu smoothing.

    Notes
    -----
    Standard parameters: λ=0.5, μ=−0.53. The kn-product (1+λ)(1+μ) ≈ 0.96
    means little volume change per pass-pair.
    """
    if verts is None or len(verts) == 0 or passes <= 0:
        return verts
    n_v = len(verts)
    L = _build_laplacian(faces, n_v)
    v = verts.astype(np.float32, copy=True)
    for _ in range(passes):
        v = v + lam * (L @ v)
        v = v + mu * (L @ v)
    return v.astype(np.float32)
