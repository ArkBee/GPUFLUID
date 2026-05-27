"""[Round-17] World↔unit-cube transform for the addon's normalised scene_dict.

Senior architect (round-12) code smell #2: `bake.collect_scene` was a
250-line bpy→dict adapter mixing (a) bpy traversal, (b) world→unit-cube
math, (c) validation warnings, (d) OBJ export side-effects, and (e)
per-solver param mapping. Round-17 extracts the **math** into this
file — pure, bpy-free, unit-testable.

The MPM solver runs in a fixed [0,1]³ unit cube (see `MpmDomainWalls`
in `src/gpufluid/sim/mpm/solver.py` — hardcoded lo=0.05, hi=0.95). The
mesher outputs vertices in normalised [0,1]³. So every position the
solver receives must be normalised:

    world_xyz → translate by domain origin → divide by domain extent

When the addon attaches the cache, it scales the cache object by
`dom_size` and translates it by `origin`, so the inverse transform
makes mesh world coords match emitter world coords exactly. Domains
scaled to ≠ 1×1×1 used to silently desync — everything was in metres
for the solver, in [0,1] for the mesh. This module is the single
source of truth for that mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class DomainTransform:
    """Encodes the world ↔ [0,1]³ mapping for a single Domain bake.

    Fields:
      origin   — world-space corner where unit-cube (0,0,0) lands
      dom_size — world-space extent of the domain (hi - lo)
      eps_norm — 1.5-cell margin in NORMALISED units, used to warn
                 when a source touches the domain wall. Stored here
                 (not on validator) because it's a function of
                 resolution that the collector decides per-bake.
      resolution — per-axis cell count (rx, ry, rz), proportional to
                   world extent so non-cubic domains get equal dx
                   in all axes.
      dx       — 1.0 / max(resolution); same unit-cube basis as the
                 geometry. FLIP uses dx for kernel sizes; MPM ignores
                 it.
    """
    origin: Vec3
    dom_size: Vec3
    eps_norm: float
    resolution: Tuple[int, int, int]
    dx: float

    @property
    def inv_size(self) -> Vec3:
        return (1.0 / self.dom_size[0],
                1.0 / self.dom_size[1],
                1.0 / self.dom_size[2])

    @property
    def avg_inv(self) -> float:
        return (self.inv_size[0] + self.inv_size[1] + self.inv_size[2]) / 3.0

    def to_sim(self, world: Vec3) -> Vec3:
        """World-coord position → [0,1]³ normalised position."""
        ox, oy, oz = self.origin
        ix, iy, iz = self.inv_size
        return (float((world[0] - ox) * ix),
                float((world[1] - oy) * iy),
                float((world[2] - oz) * iz))

    def normalize_velocity(self, world_v: Vec3) -> Vec3:
        """Velocity in m/s → normalised velocity (same scale factor as
        positions: divide by domain extent per axis)."""
        ix, iy, iz = self.inv_size
        return (float(world_v[0]) * ix,
                float(world_v[1]) * iy,
                float(world_v[2]) * iz)

    @classmethod
    def from_world_aabb(cls, world_lo: Vec3, world_hi: Vec3,
                        resolution: int) -> "DomainTransform":
        """Build a DomainTransform from a world-space AABB + a target
        resolution (cells per side of the longest axis).

        Round-13/14/15+ extracted invariant: resolution is per-axis,
        proportional to world extent — so a 2×1×1 domain at res=64
        gets (64, 32, 32) cells, all with the same dx. Without this,
        non-cubic domains had anisotropic dx and the solver fired
        kernels at wrong scales.
        """
        dom_size = (world_hi[0] - world_lo[0],
                    world_hi[1] - world_lo[1],
                    world_hi[2] - world_lo[2])
        nmax_w = max(dom_size[0], dom_size[1], dom_size[2])
        if nmax_w <= 0:
            raise ValueError(
                f"DomainTransform: degenerate world AABB {world_lo} → "
                f"{world_hi} (extent {dom_size})")
        if resolution < 1:
            raise ValueError(
                f"DomainTransform: resolution must be ≥ 1, got {resolution}")
        res = (
            max(8, int(round(dom_size[0] / nmax_w * resolution))),
            max(8, int(round(dom_size[1] / nmax_w * resolution))),
            max(8, int(round(dom_size[2] / nmax_w * resolution))),
        )
        dx = 1.0 / float(max(res))
        # 1.5-cell margin in normalised units. Independent of which axis
        # was the longest because dx is uniform after normalisation.
        eps_norm = 1.5 / float(resolution)
        return cls(
            origin=(float(world_lo[0]), float(world_lo[1]), float(world_lo[2])),
            dom_size=(float(dom_size[0]), float(dom_size[1]), float(dom_size[2])),
            eps_norm=eps_norm,
            resolution=res,
            dx=dx,
        )
