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


@block("S2.17.2.SPH",
       "Particle pushback out of a sphere obstacle (analytic SDF |p-c|-r). "
       "FU-038: sphere had a grid collider but no particle pushback twin, so a "
       "fast particle could tunnel inside between substeps.")
@wp.kernel
def k_sphere_pushback(
    state: MPMStateStruct,
    cx: float, cy: float, cz: float,
    radius: float,
    snap_eps: float,
):
    """Push any particle inside a sphere back out to its surface, mirroring
    :func:`k_cube_pushback`: outward normal = unit radial, snap to
    ``|p-c| = radius + snap_eps``, remove inward-normal velocity, reset
    F/F_trial/C (deformation history at a rigid contact is ill-defined)."""
    p = wp.tid()
    if state.particle_selection[p] == 1:
        return  # held inflow particle
    pos = state.particle_x[p]
    rel = wp.vec3(pos[0] - cx, pos[1] - cy, pos[2] - cz)
    dist = wp.length(rel)
    d = dist - radius  # signed distance to the surface (negative inside)
    if d >= snap_eps:
        return  # outside the solid
    n = wp.vec3(0.0, 0.0, 1.0)  # fallback for a particle exactly at the centre
    if dist > 1.0e-8:
        n = rel / dist
    state.particle_x[p] = pos + n * (snap_eps - d)
    v = state.particle_v[p]
    vn = wp.dot(v, n)
    if vn < 0.0:
        state.particle_v[p] = v - vn * n
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    state.particle_F[p] = I
    state.particle_F_trial[p] = I
    state.particle_C[p] = Z


@block("S2.17.2.CYL",
       "Particle pushback out of a Y-axis cylinder obstacle (radial X-Z + "
       "axial Y, nearest-surface select like the cube). FU-038 sibling of the "
       "grid collider.")
@wp.kernel
def k_cylinder_pushback(
    state: MPMStateStruct,
    cx: float, cy: float, cz: float,
    radius: float, half_h: float,
    snap_eps: float,
):
    """Push any particle inside a Y-axis cylinder back out, mirroring
    :func:`k_cube_pushback`'s nearest-face select but on the cylinder's
    radial side / axial caps. Same SDF split as ``k_sdf_cylinder_collide``."""
    p = wp.tid()
    if state.particle_selection[p] == 1:
        return
    pos = state.particle_x[p]
    rx = pos[0] - cx
    ry = pos[1] - cy
    rz = pos[2] - cz
    radial = wp.sqrt(rx * rx + rz * rz)
    dr = radial - radius
    dy = wp.abs(ry) - half_h
    d_out = wp.sqrt(wp.max(dr, 0.0) * wp.max(dr, 0.0)
                    + wp.max(dy, 0.0) * wp.max(dy, 0.0))
    d_in = wp.min(wp.max(dr, dy), 0.0)
    d = d_out + d_in
    if d >= snap_eps:
        return  # outside the solid
    # nearest surface: radial side vs axial cap (same test as the collider).
    n = wp.vec3(0.0, 0.0, 1.0)
    if dr >= dy:
        if radial > 1.0e-8:
            n = wp.vec3(rx / radial, 0.0, rz / radial)
    else:
        if ry > 0.0:
            n = wp.vec3(0.0, 1.0, 0.0)
        else:
            n = wp.vec3(0.0, -1.0, 0.0)
    state.particle_x[p] = pos + n * (snap_eps - d)
    v = state.particle_v[p]
    vn = wp.dot(v, n)
    if vn < 0.0:
        state.particle_v[p] = v - vn * n
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    state.particle_F[p] = I
    state.particle_F_trial[p] = I
    state.particle_C[p] = Z


@block("S2.17.2.MESH",
       "Particle pushback out of a MESH obstacle via its precomputed signed-"
       "distance grid + gradient normal. Goal 2026-06-14: mesh obstacles were "
       "rendered (FU-033) but never collided — water fell straight through.")
@wp.kernel
def k_mesh_sdf_pushback(
    state: MPMStateStruct,
    sdf: wp.array3d(dtype=float),   # signed distance, NEGATIVE inside the mesh
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    snap_eps: float,
):
    """Push any particle that lies inside an arbitrary mesh obstacle back out
    to its surface, mirroring :func:`k_cube_pushback` but using a precomputed
    SDF grid (the mesh has no closed analytic SDF). Outward normal = the SDF
    gradient (central differences); pushes to ``sdf ~ snap_eps`` and removes
    the inward-normal velocity. Resets F/F_trial/C like the cube path."""
    p = wp.tid()
    if state.particle_selection[p] == 1:
        return  # held inflow particle
    pos = state.particle_x[p]
    ix = int(pos[0] / dx)
    iy = int(pos[1] / dx)
    iz = int(pos[2] / dx)
    if (ix < 1 or ix >= nx - 1 or iy < 1 or iy >= ny - 1
            or iz < 1 or iz >= nz - 1):
        return
    s = sdf[ix, iy, iz]
    if s >= snap_eps:
        return  # outside the solid
    gx = sdf[ix + 1, iy, iz] - sdf[ix - 1, iy, iz]
    gy = sdf[ix, iy + 1, iz] - sdf[ix, iy - 1, iz]
    gz = sdf[ix, iy, iz + 1] - sdf[ix, iy, iz - 1]
    g = wp.vec3(gx, gy, gz)
    gl = wp.length(g)
    if gl < 1.0e-8:
        return  # degenerate gradient (flat region) — leave it
    n = g / gl  # outward (toward increasing SDF)
    state.particle_x[p] = pos + n * (snap_eps - s)
    v = state.particle_v[p]
    vn = wp.dot(v, n)
    if vn < 0.0:
        state.particle_v[p] = v - vn * n
    I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    state.particle_F[p] = I
    state.particle_F_trial[p] = I
    state.particle_C[p] = Z
