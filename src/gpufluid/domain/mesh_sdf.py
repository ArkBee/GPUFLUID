"""[Layer D4 / BLK D4.3] Mesh → signed-distance field.

Strategy: load mesh with trimesh, sample cell-centre points, query
``trimesh.proximity.signed_distance`` which returns negative inside.

Limitations (documented):
- Slow for >128^3 grids (CPU + KDTree). One-time init cost; acceptable
  for static obstacles. Future M5.4-style GPU triangle SDF planned.
- Requires watertight or closed mesh for reliable sign. For open meshes
  use ``unsigned=True`` and treat as outside-only (no fluid penetration
  check inside).
"""
from __future__ import annotations
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Sequence, Union

from ..blocks import block, BlockError


# [BLK D4.3]
@block("D4.3", "Mesh → SDF on a regular cell-centre grid")
def mesh_to_sdf(
    mesh_path: Union[str, Path, trimesh.Trimesh],
    grid_xyz: np.ndarray,
    *,
    scale: float = 1.0,
    translate: Sequence[float] = (0.0, 0.0, 0.0),
    rotate_deg: Optional[Sequence[float]] = None,
    unsigned: bool = False,
) -> np.ndarray:
    """Compute SDF of an arbitrary mesh at the given grid sample points.

    Parameters
    ----------
    mesh_path : path to .obj/.stl/.ply/.glb, or an in-memory trimesh.Trimesh
    grid_xyz : (nx, ny, nz, 3) cell-centre positions in world units
    scale : uniform scale applied to the mesh before sampling
    translate : (3,) translation applied after scale
    rotate_deg : optional (rx, ry, rz) Euler degrees applied before translate
    unsigned : if True, all distances are positive (use for non-closed meshes)

    Returns
    -------
    sdf : (nx, ny, nz) float32, negative inside the (closed) mesh
    """
    if isinstance(mesh_path, trimesh.Trimesh):
        mesh = mesh_path.copy()
    else:
        path = Path(mesh_path)
        if not path.exists():
            raise BlockError("D4.3", f"mesh file not found: {path}")
        loaded = trimesh.load(str(path), force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise BlockError("D4.3", f"loaded object is not a single mesh: {type(loaded)}")
        mesh = loaded

    # Apply transform.
    if rotate_deg is not None:
        rx, ry, rz = [float(a) * np.pi / 180.0 for a in rotate_deg]
        R = trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")
        mesh.apply_transform(R)
    if float(scale) != 1.0:
        mesh.apply_scale(float(scale))
    if any(float(t) != 0.0 for t in translate):
        mesh.apply_translation(np.asarray(translate, dtype=np.float64))

    flat = grid_xyz.reshape(-1, 3).astype(np.float64)
    if unsigned:
        # trimesh.proximity.closest_point returns (points, dists, faces)
        _, dists, _ = trimesh.proximity.closest_point(mesh, flat)
        sdf = dists.astype(np.float32)
    else:
        if not mesh.is_watertight:
            # Still try signed_distance — works approximately via winding.
            pass
        sdf = trimesh.proximity.signed_distance(mesh, flat).astype(np.float32)
        # signed_distance: positive inside, negative outside (trimesh convention).
        # Our convention (matches our analytic primitives): negative inside, positive outside.
        sdf = -sdf
    return sdf.reshape(grid_xyz.shape[:3])


# [BLK D4.3]
@block("D4.3", "Mesh → SDF convenience: load + cell-centres in one call")
def mesh_to_sdf_grid(
    mesh_path: Union[str, Path, trimesh.Trimesh],
    nx: int, ny: int, nz: int, dx: float,
    **kwargs,
) -> np.ndarray:
    """Wraps cell_centers + mesh_to_sdf for the common case."""
    from .sdf import cell_centers
    return mesh_to_sdf(mesh_path, cell_centers(nx, ny, nz, dx), **kwargs)
