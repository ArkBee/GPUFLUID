"""[Layer S2.17.2 + S2.17.3] Particle pushback kernels for MPM.

Two kernels:

  S2.17.2 — :func:`k_cube_pushback`: particles inside the cube body are
            snapped to the nearest face and their inward normal velocity
            component is zeroed. F, F_trial, C are reset to identity/zero
            because deformation history at a rigid contact is ill-defined
            (standard production MPM trick).

  S2.17.3 — :func:`k_wall_pushback`: particles outside the safe interior
            ``[lo, hi]³`` are slip-clamped back to the boundary. Same
            F/C reset on contact.

Both kernels run twice per p2g2p call (pre and post) — see F3.7 pipeline
in DESIGN.md §6.7.
"""
from __future__ import annotations

import warp as wp

from ...blocks import block
from ._warp_mpm_imports import MPMStateStruct


@block("S2.17.2",
       "Particle pushback inside (oriented) cube body: snap to nearest "
       "face in local frame, zero inward normal velocity, reset "
       "F/F_trial/C. Round-57: accepts a rotation matrix R whose "
       "columns are world-space box-local axes — identity for AABB.")
@wp.kernel
def k_cube_pushback(
    state: MPMStateStruct,
    cx: float, cy: float, cz: float,
    hx: float, hy: float, hz: float,
    snap_eps: float,
    R: wp.mat33,  # columns = world-space box-local +X/+Y/+Z axes
):
    p = wp.tid()
    # Round-44: skip held inflow particles (selection == 1).
    if state.particle_selection[p] == 1:
        return
    pos = state.particle_x[p]
    rel_world = wp.vec3(pos[0] - cx, pos[1] - cy, pos[2] - cz)
    # World→local: R is orthonormal so R^T = R^-1.
    Rt = wp.transpose(R)
    rl = Rt * rel_world  # rel in box-local frame
    if not (wp.abs(rl[0]) < hx and wp.abs(rl[1]) < hy
            and wp.abs(rl[2]) < hz):
        return
    # Inside cube body (local frame) — pick nearest face.
    qx = hx - wp.abs(rl[0])
    qy = hy - wp.abs(rl[1])
    qz = hz - wp.abs(rl[2])
    new_rl = rl
    v = state.particle_v[p]
    vl = Rt * v  # local-frame velocity
    new_vl = vl
    if qz <= qx and qz <= qy:
        if rl[2] > 0.0:
            new_rl = wp.vec3(rl[0], rl[1], hz + snap_eps)
            if vl[2] < 0.0:
                new_vl = wp.vec3(vl[0], vl[1], 0.0)
        else:
            new_rl = wp.vec3(rl[0], rl[1], -hz - snap_eps)
            if vl[2] > 0.0:
                new_vl = wp.vec3(vl[0], vl[1], 0.0)
    elif qx <= qy:
        if rl[0] > 0.0:
            new_rl = wp.vec3(hx + snap_eps, rl[1], rl[2])
            if vl[0] < 0.0:
                new_vl = wp.vec3(0.0, vl[1], vl[2])
        else:
            new_rl = wp.vec3(-hx - snap_eps, rl[1], rl[2])
            if vl[0] > 0.0:
                new_vl = wp.vec3(0.0, vl[1], vl[2])
    else:
        if rl[1] > 0.0:
            new_rl = wp.vec3(rl[0], hy + snap_eps, rl[2])
            if vl[1] < 0.0:
                new_vl = wp.vec3(vl[0], 0.0, vl[2])
        else:
            new_rl = wp.vec3(rl[0], -hy - snap_eps, rl[2])
            if vl[1] > 0.0:
                new_vl = wp.vec3(vl[0], 0.0, vl[2])
    # Local → world, then add centre back.
    new_rel_world = R * new_rl
    new_v_world = R * new_vl
    state.particle_x[p] = wp.vec3(
        cx + new_rel_world[0],
        cy + new_rel_world[1],
        cz + new_rel_world[2],
    )
    state.particle_v[p] = new_v_world
    # F/C reset — accumulated deformation at rigid contact is artifact
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    state.particle_F[p] = I
    state.particle_F_trial[p] = I
    state.particle_C[p] = Z


@block("S2.17.3",
       "Particle pushback at domain walls: slip-clamp to [lo, hi]³, "
       "zero normal velocity, preserve tangential, reset F/C")
@wp.kernel
def k_wall_pushback(
    state: MPMStateStruct,
    lo_x: float, lo_y: float, lo_z: float,
    hi_x: float, hi_y: float, hi_z: float,
):
    p = wp.tid()
    # Round-44: same selection gate as k_cube_pushback. Held inflow
    # particles sit at hold_pos which may be just outside [lo, hi];
    # pre-round-44 we clamped them every step (wasted writes) and the
    # gate kernel re-bound next step.
    if state.particle_selection[p] == 1:
        return
    pos = state.particle_x[p]
    v = state.particle_v[p]
    vx = v[0]; vy = v[1]; vz = v[2]
    px = pos[0]; py = pos[1]; pz = pos[2]
    moved = False
    if px < lo_x:
        px = lo_x;  vx = wp.max(vx, 0.0); moved = True
    elif px > hi_x:
        px = hi_x;  vx = wp.min(vx, 0.0); moved = True
    if py < lo_y:
        py = lo_y;  vy = wp.max(vy, 0.0); moved = True
    elif py > hi_y:
        py = hi_y;  vy = wp.min(vy, 0.0); moved = True
    if pz < lo_z:
        pz = lo_z;  vz = wp.max(vz, 0.0); moved = True
    elif pz > hi_z:
        pz = hi_z;  vz = wp.min(vz, 0.0); moved = True
    if not moved:
        return
    state.particle_x[p] = wp.vec3(px, py, pz)
    state.particle_v[p] = wp.vec3(vx, vy, vz)
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    state.particle_F[p] = I
    state.particle_F_trial[p] = I
    state.particle_C[p] = Z
