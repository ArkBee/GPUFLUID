"""[Layer G1] Analytic signed-distance field primitives + cell-centre grid.

Pure math, no solver or domain dependencies. Relocated 2026-05-17 from
``domain/sdf.py`` as part of F3.6.A1 (see ``docs/DESIGN.md §3.2.4.2``):
these helpers take a position grid and return a distance field. They
have no notion of FlipSolver3D state or domain configuration — moving
them to G1 untangles the F3→D4 import inversion.

All SDFs follow the convention: **negative inside the solid**, zero on
surface, positive outside. Functions accept a ``grid_xyz`` tensor of
cell-centre positions (shape ``(nx, ny, nz, 3)``) returning a
``(nx, ny, nz)`` float distance field.

Block IDs (after the F3.6.A1 relocation):

  G1.8   cell_centers       (unchanged — was already G1 in the registry)
  G1.10  sdf_sphere         (was D4.2.1)
  G1.11  sdf_box            (was D4.2.2)
  G1.12  sdf_cylinder_y     (was D4.2.3)
  G1.13  sdf_plane          (was D4.2.4)
  G1.14  sdf_union          (was D4.2.5)

The solver-coupled ``mark_solid_from_sdf`` host helper stays in
``domain/sdf.py`` — it mutates a solver marker array and is rightly D4.
"""
from __future__ import annotations
import numpy as np
from ..blocks import block


@block("G1.8", "Cell-centre world-position grid (host)")
def cell_centers(nx: int, ny: int, nz: int, dx: float) -> np.ndarray:
    """Return (nx,ny,nz,3) array of cell-centre positions in world units."""
    i = (np.arange(nx) + 0.5) * dx
    j = (np.arange(ny) + 0.5) * dx
    k = (np.arange(nz) + 0.5) * dx
    X, Y, Z = np.meshgrid(i, j, k, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


@block("G1.10", "SDF sphere (analytic, negative inside)")
def sdf_sphere(grid_xyz: np.ndarray, center, radius: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    return np.linalg.norm(grid_xyz - c, axis=-1) - float(radius)


@block("G1.11", "SDF axis-aligned box (analytic, negative inside)")
def sdf_box(grid_xyz: np.ndarray, center, half_size) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    h = np.asarray(half_size, dtype=np.float32)
    d = np.abs(grid_xyz - c) - h
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


@block("G1.12", "SDF cylinder aligned with Y (gravity) axis")
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


@block("G1.13", "SDF plane (signed distance to half-space)")
def sdf_plane(grid_xyz: np.ndarray, point, normal) -> np.ndarray:
    """Half-space: negative on the ``-normal`` side of ``point``.

    A ramp obstacle is typically constructed as the intersection of a
    plane SDF with a bounding box, so use this with sdf_box for finite slopes.
    """
    p = np.asarray(point, dtype=np.float32)
    n = np.asarray(normal, dtype=np.float32)
    nrm = float(np.linalg.norm(n))
    # audit-2026-06-14r2 #14: the +1e-12 below is only a float-noise dodge, NOT
    # a zero-vector handler — a genuine (0,0,0) normal would yield an all-zero
    # half-space (every cell solid). Reject a degenerate normal loudly.
    if nrm < 1e-9:
        raise ValueError("sdf_plane: normal must be non-zero")
    n = n / (nrm + 1e-12)
    return ((grid_xyz - p) * n).sum(axis=-1)


@block("G1.14", "SDF union (min of components)")
def sdf_union(*sdfs: np.ndarray) -> np.ndarray:
    out = sdfs[0]
    for s in sdfs[1:]:
        out = np.minimum(out, s)
    return out
