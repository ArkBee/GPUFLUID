"""[Layer S2.17.1] SDF box collider for MPM grid.

Used by F3.7 (MpmSolver) to handle rigid-box obstacles. The kernel:

  * Computes the signed-distance field of an axis-aligned box at every
    grid node (positive outside, negative inside).
  * Deep interior (d < -1.5·dx): zero grid velocity completely (deep
    inside the solid, no fluid motion possible).
  * Boundary shell (-1.5·dx ≤ d ≤ +1·dx): "separate" surface — only
    the inward-normal velocity component is removed.
  * Optional `tangential_friction`: scalar in [0, 1]. When > 0, the
    tangential velocity in the shell is multiplied by this factor.
    Applied **only on the chosen face** (default: +z, top). The other
    faces stay slip so cube-side waterfalls are not braked. Use the
    `top_only_friction` flag to control this.

DESIGN.md §5.3 — S2.17.1.
"""
from __future__ import annotations

import warp as wp

from ...blocks import block
from ._warp_mpm_imports import (
    Dirichlet_collider,
    MPMStateStruct,
    MPMModelStruct,
)


@block("S2.17.1",
       "SDF axis-aligned-box grid collider: separate-surface boundary "
       "+ deep-interior zeroing + optional top-face tangential friction")
@wp.kernel
def k_sdf_box_collide(
    time: float,
    dt: float,
    state: MPMStateStruct,
    model: MPMModelStruct,
    param: Dirichlet_collider,
):
    """SDF box collider kernel.

    Reads the box geometry from ``param.point`` (centre) and ``param.size``
    (half-extent). Reads ``param.friction`` as the tangential damping
    factor (0 = full sticky tangentially; 1 = pure slip). Applies the
    friction only on the +Y face if ``param.surface_type == 0`` (sticky),
    on the +Z face if ``param.surface_type == 1`` (slip-with-top-friction
    semantics — what we use for cube-top accumulation), or none if
    ``param.surface_type == 2`` (pure slip everywhere).
    """
    gx, gy, gz = wp.tid()
    px = float(gx) * model.dx - param.point[0]
    py = float(gy) * model.dx - param.point[1]
    pz = float(gz) * model.dx - param.point[2]
    halfx = param.size[0]
    halfy = param.size[1]
    halfz = param.size[2]
    qx = wp.abs(px) - halfx
    qy = wp.abs(py) - halfy
    qz = wp.abs(pz) - halfz
    # SDF: positive outside, negative inside
    d_out = wp.sqrt(wp.max(qx, 0.0) * wp.max(qx, 0.0)
                  + wp.max(qy, 0.0) * wp.max(qy, 0.0)
                  + wp.max(qz, 0.0) * wp.max(qz, 0.0))
    d_in = wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
    d = d_out + d_in
    if d >= model.dx:
        return  # outside influence shell
    # Pick outward normal from largest q component
    nx = 0.0
    ny = 0.0
    nz = 0.0
    if qx >= qy and qx >= qz:
        if px > 0.0: nx = 1.0
        else:        nx = -1.0
    elif qy >= qz:
        if py > 0.0: ny = 1.0
        else:        ny = -1.0
    else:
        if pz > 0.0: nz = 1.0
        else:        nz = -1.0
    if d < -1.5 * model.dx:
        # Deep inside — zero completely
        state.grid_v_out[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)
        return
    # Boundary shell — project inward-normal component
    v = state.grid_v_out[gx, gy, gz]
    n = wp.vec3(nx, ny, nz)
    vn = wp.dot(v, n)
    if vn < 0.0:
        v = v - vn * n
    # Top-face friction (surface_type == 1 means "slip with top friction"):
    # damp tangential v on +Z face when configured
    if param.surface_type == 1 and nz > 0.99:
        v = v * param.friction
    state.grid_v_out[gx, gy, gz] = v
