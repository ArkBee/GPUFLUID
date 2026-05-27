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
    # Round-57: oriented bounding box. Box-local axes (world coords)
    # are stored in `param.x_unit`, `param.y_unit`, `param.direction`.
    # For axis-aligned obstacles solver.py loads identity columns
    # (1,0,0)/(0,1,0)/(0,0,1) so the math reduces to the pre-round-57
    # AABB form bit-exactly.
    gx, gy, gz = wp.tid()
    rel = wp.vec3(
        float(gx) * model.dx - param.point[0],
        float(gy) * model.dx - param.point[1],
        float(gz) * model.dx - param.point[2],
    )
    ex = param.x_unit
    ey = param.y_unit
    ez = param.direction
    # World→local: dot with each basis vector.
    plx = wp.dot(rel, ex)
    ply = wp.dot(rel, ey)
    plz = wp.dot(rel, ez)
    halfx = param.size[0]
    halfy = param.size[1]
    halfz = param.size[2]
    qx = wp.abs(plx) - halfx
    qy = wp.abs(ply) - halfy
    qz = wp.abs(plz) - halfz
    # SDF: positive outside, negative inside (in local frame; for OBB
    # the SDF is identical in local — rotation preserves distance).
    d_out = wp.sqrt(wp.max(qx, 0.0) * wp.max(qx, 0.0)
                  + wp.max(qy, 0.0) * wp.max(qy, 0.0)
                  + wp.max(qz, 0.0) * wp.max(qz, 0.0))
    d_in = wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
    d = d_out + d_in
    if d >= model.dx:
        return  # outside influence shell
    # Outward normal in box-LOCAL frame, then rotate to world.
    nlx = float(0.0)
    nly = float(0.0)
    nlz = float(0.0)
    if qx >= qy and qx >= qz:
        if plx > 0.0: nlx = 1.0
        else:         nlx = -1.0
    elif qy >= qz:
        if ply > 0.0: nly = 1.0
        else:         nly = -1.0
    else:
        if plz > 0.0: nlz = 1.0
        else:         nlz = -1.0
    if d < -1.5 * model.dx:
        # Deep inside — zero completely
        state.grid_v_out[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)
        return
    # Local normal → world normal: n_w = nlx*ex + nly*ey + nlz*ez
    n = nlx * ex + nly * ey + nlz * ez
    # Boundary shell — project inward-normal component (world frame).
    v = state.grid_v_out[gx, gy, gz]
    vn = wp.dot(v, n)
    if vn < 0.0:
        v = v - vn * n
    # Top-face friction: "top" = box-local +Z (so for tilted obstacles
    # friction follows the tilt — fluid pooling on the tilted top face,
    # which is what users expect). nlz > 0.99 selects the +Z LOCAL face.
    # Round-34: tangential-only damp post-projection — see history.
    if param.surface_type == 1 and nlz > 0.99:
        vn_post = wp.dot(v, n)
        v_normal = vn_post * n
        v_tangent = v - v_normal
        v = v_normal + param.friction * v_tangent
    state.grid_v_out[gx, gy, gz] = v
