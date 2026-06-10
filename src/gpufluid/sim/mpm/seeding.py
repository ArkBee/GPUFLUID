"""[Layer S2.17.10] Shape-aware particle seeding for MPM initial fluid sources.

The MPM solver seeds its initial fluid from an ``(N,3)`` point cloud
(``MpmConfig.initial_column``). Historically the CLI only built a rectangular
column via :func:`gpufluid.sim.mpm.solver.make_column`, so a fluid SOURCE was
always a box. These helpers generate the point cloud for any of box / sphere /
mesh so "mark this object as a source" works for non-cube shapes.

All coordinates are in the solver's normalised ``[0,1]³`` space (the addon
normalises world → unit-cube before emitting; a hand-written TOML uses the same
convention). Point spacing defaults to ``dx`` (one particle per grid cell),
matching ``make_column``'s density so fluid mass / stability is unchanged.

Pure numpy (+ trimesh for the mesh case) — no GPU/warp, CPU-unit-testable.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


def _grid_points(lo: Sequence[float], hi: Sequence[float],
                 spacing: float) -> np.ndarray:
    """Regular grid covering the AABB [lo, hi] at ~``spacing`` apart
    (>=2 samples per axis). Returns (N,3) float32."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    spacing = max(float(spacing), 1e-6)
    axes = []
    for a in range(3):
        n = max(2, int(np.ceil((hi[a] - lo[a]) / spacing)) + 1)
        axes.append(np.linspace(lo[a], hi[a], n))
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()],
                    axis=-1).astype(np.float32)


def seed_box(lo: Vec3, hi: Vec3, spacing: float) -> np.ndarray:
    """Fill the axis-aligned box [lo, hi] with a regular point grid."""
    return _grid_points(lo, hi, spacing)


def clamp_to_unit_box(pts: np.ndarray, eps: float) -> np.ndarray:
    """Clamp seeded positions into ``[eps, 1-eps]³`` (audit-20260610).

    Box sources get this margin addon-side (the addon clamps the box
    lo/hi to its 1.5-cell ``eps_norm`` before emitting the TOML), but
    sphere/mesh sources bypass that path entirely — their point clouds
    come straight from :func:`seed_sphere`/:func:`seed_mesh` and can
    stick outside the solver's [0,1]³ space when the shape touches the
    domain edge (or a non-cubic-domain transform lands it outside).
    """
    # dodge: out-of-box seeds are not a crash (k_wall_pushback slip-clamps
    # them at the first _pre_step) but they ARE silent artefacts — the
    # OOB portion gets flattened onto the wall (mangled geometry) and the
    # frame-0 PLY is dumped BEFORE any pushback runs, so it contains the
    # raw OOB cloud. Clamping here mirrors the box path's addon-side
    # eps clamp for the shapes that bypass it.
    eps = float(eps)
    return np.clip(pts, eps, 1.0 - eps).astype(np.float32, copy=False)


def seed_sphere(center: Vec3, radius: float, spacing: float) -> np.ndarray:
    """Fill a sphere by grid-sampling its bounding box and keeping points
    within ``radius`` of ``center``."""
    c = np.asarray(center, dtype=np.float64)
    r = float(radius)
    pts = _grid_points(c - r, c + r, spacing)
    keep = np.linalg.norm(pts.astype(np.float64) - c, axis=1) <= r
    return pts[keep]


def seed_mesh(mesh_path: str, spacing: float,
              scale: float = 1.0,
              translate: Vec3 = (0.0, 0.0, 0.0),
              rotate_deg: Optional[Vec3] = None) -> np.ndarray:
    """Fill an arbitrary watertight mesh: load it, apply rotate→scale→translate
    (the same order/convention as a MESH obstacle), then keep grid points that
    fall INSIDE the mesh (trimesh ray-cast containment).

    ``scale``/``translate`` map the OBJ's own coordinates into the solver's
    normalised space (the addon exports the mesh in world metres and supplies
    scale = avg inverse domain size + translate, exactly like a mesh obstacle).
    """
    import trimesh  # local import: heavy, only needed for mesh sources
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if rotate_deg is not None:
        from trimesh.transformations import euler_matrix
        rx, ry, rz = (float(np.deg2rad(a)) for a in rotate_deg)
        mesh.apply_transform(euler_matrix(rx, ry, rz, "sxyz"))
    if scale != 1.0:
        mesh.apply_scale(float(scale))
    if any(t != 0.0 for t in translate):
        mesh.apply_translation(np.asarray(translate, dtype=np.float64))
    lo, hi = mesh.bounds  # post-transform AABB
    pts = _grid_points(lo, hi, spacing)
    inside = mesh.contains(pts)
    return pts[inside].astype(np.float32)
