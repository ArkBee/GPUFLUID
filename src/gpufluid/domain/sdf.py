"""[Layer D4] Domain geometry: SDF primitives, cell-centre grid, marker injection.

All SDFs follow the convention: **negative inside the solid**, zero on
surface, positive outside. Functions accept a `grid_xyz` tensor of
cell-center positions (shape ``(nx, ny, nz, 3)``) returning a ``(nx, ny, nz)``
float distance field.
"""
from __future__ import annotations
import numpy as np
from ..blocks import block


# [BLK G1.8]  (lives in D4 module because it's an SDF-companion host helper)
@block("G1.8", "Cell-centre world-position grid (host)")
def cell_centers(nx: int, ny: int, nz: int, dx: float) -> np.ndarray:
    """Return (nx,ny,nz,3) array of cell-centre positions in world units."""
    i = (np.arange(nx) + 0.5) * dx
    j = (np.arange(ny) + 0.5) * dx
    k = (np.arange(nz) + 0.5) * dx
    X, Y, Z = np.meshgrid(i, j, k, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


# [BLK D4.2.1]
@block("D4.2.1", "SDF sphere")
def sdf_sphere(grid_xyz: np.ndarray, center, radius: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    return np.linalg.norm(grid_xyz - c, axis=-1) - float(radius)


# [BLK D4.2.2]
@block("D4.2.2", "SDF axis-aligned box")
def sdf_box(grid_xyz: np.ndarray, center, half_size) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    h = np.asarray(half_size, dtype=np.float32)
    d = np.abs(grid_xyz - c) - h
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


# [BLK D4.2.3]
@block("D4.2.3", "SDF cylinder aligned with Y (gravity) axis")
def sdf_cylinder_y(grid_xyz: np.ndarray, center, radius: float, half_height: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    local = grid_xyz - c
    radial = np.sqrt(local[..., 0] ** 2 + local[..., 2] ** 2)
    axial = np.abs(local[..., 1])
    dr = radial - float(radius)
    da = axial - float(half_height)
    outside = np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(da, 0.0) ** 2)
    inside = np.minimum(np.maximum(dr, da), 0.0)
    return outside + inside


# [BLK D4.2.4]
@block("D4.2.4", "SDF plane (signed distance to half-space)")
def sdf_plane(grid_xyz: np.ndarray, point, normal) -> np.ndarray:
    """Half-space: negative on the ``-normal`` side of ``point``.

    A ramp obstacle is typically constructed as the intersection of a
    plane SDF with a bounding box, so use this with sdf_box for finite slopes.
    """
    p = np.asarray(point, dtype=np.float32)
    n = np.asarray(normal, dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-12)
    return ((grid_xyz - p) * n).sum(axis=-1)


# [BLK D4.2.5]
@block("D4.2.5", "SDF union (min of components)")
def sdf_union(*sdfs: np.ndarray) -> np.ndarray:
    out = sdfs[0]
    for s in sdfs[1:]:
        out = np.minimum(out, s)
    return out


# [BLK D4.4]
@block("D4.4", "Apply SDF as solid marker (marker=2 where sdf<=padding)")
def mark_solid_from_sdf(marker: np.ndarray, sdf: np.ndarray, padding: float = 0.0) -> np.ndarray:
    """In-place set marker=2 where sdf <= padding. Returns marker."""
    marker[sdf <= padding] = 2
    return marker
