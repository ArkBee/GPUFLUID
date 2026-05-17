"""[Layer S2] GPU triangle inside-test for solid-cell marking.

Two implementations, picked automatically by triangle count:

* **D4.3.GPU** — brute-force O(cells × triangles) Moller-Trumbore +X ray-cast.
  Fast for <=256 triangles (lower setup cost than building a BVH); falls back
  to this when a mesh is small or when the caller passes ``use_bvh=False``.

* **D4.3.GPU.BVH** — Warp's built-in ``wp.Mesh`` spatial structure with
  ``mesh_query_point_sign_winding_number``. O(cells * log triangles)
  traversal on the GPU. Mandatory for >=1k-tri obstacles (the brute-force
  path becomes the dominant cost when ``cells * triangles`` exceeds ~2e8).

Relocated from ``domain/mesh_sdf_gpu.py`` on 2026-05-17 as part of
F3.6.A2 — see ``docs/DESIGN.md §3.2.4.2``. The kernels are pure
infrastructure (take cell-centre array + triangle list + marker array,
no FlipSolver3D or scene config dependency); they belong in S2.

Block IDs ``D4.3.GPU`` / ``D4.3.GPU.BVH`` kept unchanged for this PR to
limit blast radius. A follow-up may rename them to ``S2.20.*`` once
the F3.6 macro closes; the current names are grandfathered.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from typing import Sequence

from ..blocks import block
from ..primitives.runtime import init as warp_init, device as default_device

warp_init()


# -----------------------------------------------------------------------------
# Brute-force kernel (kept for small meshes; build cost of wp.Mesh isn't worth
# it below a few hundred triangles).
# -----------------------------------------------------------------------------

@wp.kernel
def k3_mesh_to_solid(
    cell_centers: wp.array3d(dtype=wp.vec3),
    tris: wp.array(dtype=wp.vec3),   # flat: 3 * n_tris vec3
    n_tris: int,
    marker: wp.array3d(dtype=int),
):
    """For each cell, cast +X ray and count triangle crossings. Odd -> inside."""
    i, j, k = wp.tid()
    if i >= cell_centers.shape[0] or j >= cell_centers.shape[1] or k >= cell_centers.shape[2]:
        return
    p = cell_centers[i, j, k]
    crossings = int(0)
    for t in range(n_tris):
        v0 = tris[3 * t + 0]
        v1 = tris[3 * t + 1]
        v2 = tris[3 * t + 2]
        e1x = v1[0] - v0[0]; e1y = v1[1] - v0[1]; e1z = v1[2] - v0[2]
        e2x = v2[0] - v0[0]; e2y = v2[1] - v0[1]; e2z = v2[2] - v0[2]
        hx = float(0.0); hy = -e2z; hz = e2y
        a = e1x * hx + e1y * hy + e1z * hz
        if wp.abs(a) > 1.0e-12:
            f = 1.0 / a
            sx = p[0] - v0[0]; sy = p[1] - v0[1]; sz = p[2] - v0[2]
            u = f * (sx * hx + sy * hy + sz * hz)
            if u >= 0.0 and u <= 1.0:
                qx = sy * e1z - sz * e1y
                v_ = f * qx
                if v_ >= 0.0 and u + v_ <= 1.0:
                    qy = sz * e1x - sx * e1z
                    qz = sx * e1y - sy * e1x
                    t_ = f * (e2x * qx + e2y * qy + e2z * qz)
                    if t_ > 1.0e-6:
                        crossings += 1
    if crossings % 2 == 1:
        if marker[i, j, k] != 2:
            marker[i, j, k] = 2


block("D4.3.GPU", "GPU triangle ray-cast inside test (brute-force, all triangles)")(k3_mesh_to_solid)


# -----------------------------------------------------------------------------
# BVH-accelerated kernel (wp.Mesh winding query)
# -----------------------------------------------------------------------------

@wp.kernel
def k3_mesh_to_solid_bvh(
    cell_centers: wp.array3d(dtype=wp.vec3),
    mesh_id: wp.uint64,
    accuracy: float,
    marker: wp.array3d(dtype=int),
):
    """Per-cell winding-number query against Warp's BVH-backed wp.Mesh."""
    i, j, k = wp.tid()
    if i >= cell_centers.shape[0] or j >= cell_centers.shape[1] or k >= cell_centers.shape[2]:
        return
    p = cell_centers[i, j, k]
    s = wp.mesh_query_point_sign_winding_number(mesh_id, p, accuracy)
    if s.result and s.sign < 0.0:
        if marker[i, j, k] != 2:
            marker[i, j, k] = 2


block("D4.3.GPU.BVH", "BVH-backed inside test via wp.Mesh + winding query")(k3_mesh_to_solid_bvh)


DEFAULT_BVH_THRESHOLD = 256


# -----------------------------------------------------------------------------
# wp.Mesh cache (animated obstacles refresh per frame; rebuilding the BVH each
# frame would dominate cost. Cache by `cache_key` — caller's responsibility).
# -----------------------------------------------------------------------------

class _MeshCache:
    """Tiny LRU-ish cache for wp.Mesh objects keyed by host-provided ID.

    Avoids rebuilding the BVH every frame for animated obstacles. The caller
    passes ``cache_key`` (e.g. obstacle index); ``refit`` is called when the
    same key is seen again, which is cheap (no topology change)."""

    def __init__(self, max_entries: int = 16):
        self._entries: dict = {}
        self.max_entries = max_entries

    def get_or_build(self, triangles_np: np.ndarray, device: str,
                    cache_key: object | None = None) -> "wp.Mesh":
        n_tris = triangles_np.shape[0]
        verts = triangles_np.reshape(-1, 3).astype(np.float32)  # (3*n_tris, 3)
        indices = np.arange(verts.shape[0], dtype=np.int32)

        if cache_key is None:
            pts = wp.array(verts, dtype=wp.vec3, device=device)
            idx = wp.array(indices, dtype=wp.int32, device=device)
            return wp.Mesh(points=pts, indices=idx)

        entry = self._entries.get(cache_key)
        if entry is not None and entry["n_tris"] == n_tris and entry["device"] == device:
            entry["mesh"].points.assign(verts)
            entry["mesh"].refit()
            return entry["mesh"]

        pts = wp.array(verts, dtype=wp.vec3, device=device)
        idx = wp.array(indices, dtype=wp.int32, device=device)
        mesh = wp.Mesh(points=pts, indices=idx)
        if len(self._entries) >= self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._entries[cache_key] = {"mesh": mesh, "n_tris": n_tris, "device": device}
        return mesh


_MESH_CACHE = _MeshCache()


@block("D4.3.GPU.BVH", "Build a synthetic +/-dx signed indicator from mesh "
                      "(GPU BVH inside-test). Cheap stand-in for full SDF "
                      "when only solid marking is needed.")
def mesh_indicator_sdf_gpu(
    cell_centers_np: np.ndarray,
    triangles_np: np.ndarray,
    dx: float,
    device: str | None = None,
    cache_key: object | None = None,
) -> np.ndarray:
    """Returns a (nx, ny, nz) numpy float array where cells inside the mesh
    have value -dx (negative -> solid for add_solid_from_sdf) and outside cells
    have value +dx. This is NOT a true Euclidean SDF — distances near the
    surface are coarse — but it is exactly the input shape that the solver's
    obstacle path needs (it only checks `sdf <= padding`). Bypasses the
    O(cells * tris) CPU `mesh_to_sdf` which OOMs above ~5k tris on 64^3 grids.
    """
    dev = device or default_device()
    nx, ny, nz = cell_centers_np.shape[:3]
    marker = wp.zeros((nx, ny, nz), dtype=int, device=dev)
    mark_solid_from_mesh_gpu(marker, cell_centers_np, triangles_np,
                             device=dev, use_bvh=True, cache_key=cache_key)
    inside = (marker.numpy() == 2)
    sdf = np.full((nx, ny, nz), +dx, dtype=np.float32)
    sdf[inside] = -dx
    return sdf


@block("D4.3.GPU", "Compute solid-cell mask of a triangle mesh on GPU "
                  "(auto-selects BVH for >=256 triangles)")
def mark_solid_from_mesh_gpu(
    marker_wp: wp.array,
    cell_centers_np: np.ndarray,
    triangles_np: np.ndarray,
    device: str | None = None,
    use_bvh: bool | None = None,
    cache_key: object | None = None,
) -> None:
    """Update ``marker_wp`` (Warp int 3d) by setting all cells inside the
    closed mesh to ``2``.

    Parameters
    ----------
    marker_wp : Warp array3d(int), shape (nx, ny, nz)
    cell_centers_np : numpy (nx, ny, nz, 3) cell-centre positions
    triangles_np : numpy (n_tris, 3, 3) — each row is 3 vertex positions
    use_bvh : None -> auto (BVH if n_tris >= DEFAULT_BVH_THRESHOLD), True/False
              to force one path explicitly.
    cache_key : optional hashable key (e.g. obstacle index). If given, the
                wp.Mesh is cached and ``refit()``-ed on subsequent calls with
                the same key + same n_tris — crucial for animated obstacles.
    """
    dev = device or default_device()
    cells_wp = wp.array(cell_centers_np.astype(np.float32), dtype=wp.vec3, device=dev)
    n_tris = triangles_np.shape[0]
    nx, ny, nz = cell_centers_np.shape[:3]

    if use_bvh is None:
        use_bvh = (n_tris >= DEFAULT_BVH_THRESHOLD)

    if use_bvh:
        mesh = _MESH_CACHE.get_or_build(triangles_np, dev, cache_key=cache_key)
        wp.launch(k3_mesh_to_solid_bvh, dim=(nx, ny, nz),
                  inputs=[cells_wp, mesh.id, 2.0, marker_wp], device=dev)
    else:
        flat = triangles_np.reshape(-1, 3).astype(np.float32)
        tris_wp = wp.array(flat, dtype=wp.vec3, device=dev)
        wp.launch(k3_mesh_to_solid, dim=(nx, ny, nz),
                  inputs=[cells_wp, tris_wp, int(n_tris), marker_wp], device=dev)
    wp.synchronize()
