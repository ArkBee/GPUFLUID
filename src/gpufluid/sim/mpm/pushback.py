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
       "Particle pushback inside cube body: snap to nearest face, "
       "zero inward normal velocity, reset F/F_trial/C")
@wp.kernel
def k_cube_pushback(
    state: MPMStateStruct,
    cx: float, cy: float, cz: float,
    hx: float, hy: float, hz: float,
    snap_eps: float,
):
    p = wp.tid()
    pos = state.particle_x[p]
    rx = pos[0] - cx
    ry = pos[1] - cy
    rz = pos[2] - cz
    if not (wp.abs(rx) < hx and wp.abs(ry) < hy and wp.abs(rz) < hz):
        return
    # Inside cube body — pick nearest face (smallest "depth to face")
    qx = hx - wp.abs(rx)
    qy = hy - wp.abs(ry)
    qz = hz - wp.abs(rz)
    new_pos = pos
    new_v = state.particle_v[p]
    if qz <= qx and qz <= qy:
        if rz > 0.0:
            new_pos = wp.vec3(pos[0], pos[1], cz + hz + snap_eps)
            if new_v[2] < 0.0:
                new_v = wp.vec3(new_v[0], new_v[1], 0.0)
        else:
            new_pos = wp.vec3(pos[0], pos[1], cz - hz - snap_eps)
            if new_v[2] > 0.0:
                new_v = wp.vec3(new_v[0], new_v[1], 0.0)
    elif qx <= qy:
        if rx > 0.0:
            new_pos = wp.vec3(cx + hx + snap_eps, pos[1], pos[2])
            if new_v[0] < 0.0:
                new_v = wp.vec3(0.0, new_v[1], new_v[2])
        else:
            new_pos = wp.vec3(cx - hx - snap_eps, pos[1], pos[2])
            if new_v[0] > 0.0:
                new_v = wp.vec3(0.0, new_v[1], new_v[2])
    else:
        if ry > 0.0:
            new_pos = wp.vec3(pos[0], cy + hy + snap_eps, pos[2])
            if new_v[1] < 0.0:
                new_v = wp.vec3(new_v[0], 0.0, new_v[2])
        else:
            new_pos = wp.vec3(pos[0], cy - hy - snap_eps, pos[2])
            if new_v[1] > 0.0:
                new_v = wp.vec3(new_v[0], 0.0, new_v[2])
    state.particle_x[p] = new_pos
    state.particle_v[p] = new_v
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
