"""[Layer F3] 3D FLIP/PIC solver orchestration.

MAC grid:
    u (nx+1, ny, nz)  v (nx, ny+1, nz)  w (nx, ny, nz+1)
    p, marker  (nx, ny, nz)   marker: 0=air, 1=fluid, 2=solid

Step pipeline (F3.3) calls S2.x kernels in fixed order:

    clear → P2G(S2.1) → normalize(S2.2) → gravity(S2.3) → bc(S2.4)
      → div(S2.5) → pressure(S2.6.1) ×N → grad(S2.7) → bc(S2.4)
      → G2P+advect(S2.8+S2.9)

Warp kernels for each S2 scheme live in this file (their @block tag
makes them addressable individually). Kept monolithic for the v0.1
sprint; will split into `schemes/*.py` files once unit-tested.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from typing import Optional

from typing import Callable, List, Optional, Sequence
from dataclasses import dataclass, field

from ..blocks import block
from ..primitives.runtime import init as warp_init, device as default_device, zeros, zeros_int
from ..primitives.gridmath import clamp_int, sample3, scatter_face
from ..domain.regions import InflowBox, OutflowBox, apply_inflows, apply_outflows
from ..domain.animation import Motion, evaluate_center
from ..domain.sdf import sdf_sphere, sdf_box, sdf_cylinder_y, sdf_union

warp_init()


# =============================================================================
# Grid clear (housekeeping, not a numbered scheme)
# =============================================================================

@wp.kernel
def k3_clear_grid(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    uw: wp.array3d(dtype=float),
    vw: wp.array3d(dtype=float),
    ww: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
):
    i, j, k = wp.tid()
    if i < u.shape[0] and j < u.shape[1] and k < u.shape[2]:
        u[i, j, k] = 0.0; uw[i, j, k] = 0.0
    if i < v.shape[0] and j < v.shape[1] and k < v.shape[2]:
        v[i, j, k] = 0.0; vw[i, j, k] = 0.0
    if i < w.shape[0] and j < w.shape[1] and k < w.shape[2]:
        w[i, j, k] = 0.0; ww[i, j, k] = 0.0
    if i < marker.shape[0] and j < marker.shape[1] and k < marker.shape[2]:
        if marker[i, j, k] != 2:
            marker[i, j, k] = 0


# =============================================================================
# S2.1 — P2G transfer
# =============================================================================

@wp.kernel
def k3_p2g(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    uw: wp.array3d(dtype=float),
    vw: wp.array3d(dtype=float),
    ww: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    dx: float, nx: int, ny: int, nz: int,
):
    pid = wp.tid()
    p = pos[pid]
    velp = vel[pid]
    ci = clamp_int(int(p[0] / dx), 0, nx - 1)
    cj = clamp_int(int(p[1] / dx), 0, ny - 1)
    ck = clamp_int(int(p[2] / dx), 0, nz - 1)
    if marker[ci, cj, ck] != 2:
        marker[ci, cj, ck] = 1
    scatter_face(u, uw, velp[0],
                 p[0] / dx,        p[1] / dx - 0.5, p[2] / dx - 0.5,
                 nx + 1, ny, nz)
    scatter_face(v, vw, velp[1],
                 p[0] / dx - 0.5,  p[1] / dx,       p[2] / dx - 0.5,
                 nx, ny + 1, nz)
    scatter_face(w, ww, velp[2],
                 p[0] / dx - 0.5,  p[1] / dx - 0.5, p[2] / dx,
                 nx, ny, nz + 1)


block("S2.1", "P2G: scatter particle velocities → MAC faces (trilinear)")(k3_p2g)


# =============================================================================
# S2.2 — Normalize faces (divide by accumulated weight, save pre-pressure)
# =============================================================================

@wp.kernel
def k3_normalize(
    u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
    uw: wp.array3d(dtype=float), vw: wp.array3d(dtype=float), ww: wp.array3d(dtype=float),
    us: wp.array3d(dtype=float), vs: wp.array3d(dtype=float), ws: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    if i < u.shape[0] and j < u.shape[1] and k < u.shape[2]:
        if uw[i, j, k] > 1.0e-8: u[i, j, k] = u[i, j, k] / uw[i, j, k]
        else:                    u[i, j, k] = 0.0
        us[i, j, k] = u[i, j, k]
    if i < v.shape[0] and j < v.shape[1] and k < v.shape[2]:
        if vw[i, j, k] > 1.0e-8: v[i, j, k] = v[i, j, k] / vw[i, j, k]
        else:                    v[i, j, k] = 0.0
        vs[i, j, k] = v[i, j, k]
    if i < w.shape[0] and j < w.shape[1] and k < w.shape[2]:
        if ww[i, j, k] > 1.0e-8: w[i, j, k] = w[i, j, k] / ww[i, j, k]
        else:                    w[i, j, k] = 0.0
        ws[i, j, k] = w[i, j, k]


block("S2.2", "Normalize MAC faces and save pre-pressure copy")(k3_normalize)


# =============================================================================
# S2.3 — Gravity
# =============================================================================

@wp.kernel
def k3_add_gravity(v: wp.array3d(dtype=float), g: float, dt: float):
    i, j, k = wp.tid()
    if i < v.shape[0] and j < v.shape[1] and k < v.shape[2]:
        v[i, j, k] = v[i, j, k] + g * dt


block("S2.3", "Add gravity g·dt to v faces")(k3_add_gravity)


# =============================================================================
# S2.13 — Viscosity (implicit Jacobi diffusion of velocity components)
# =============================================================================
# Standard semi-implicit viscosity step (Bridson §8.4):
#     u^(n+1) - dt·ν·∇²u^(n+1) = u^n
# Discretised per face with Jacobi: each face's new velocity blends old self
# with neighbours, weighted by `r = dt·ν/dx²`.

@wp.func
def _safe_get_u(u: wp.array3d(dtype=float), i: int, j: int, k: int,
                fallback: float) -> float:
    if i < 0 or j < 0 or k < 0:
        return fallback
    if i >= u.shape[0] or j >= u.shape[1] or k >= u.shape[2]:
        return fallback
    return u[i, j, k]


@wp.kernel
def k3_jacobi_visc(
    u_old: wp.array3d(dtype=float),
    u_in: wp.array3d(dtype=float),
    u_out: wp.array3d(dtype=float),
    r: float,
):
    """One Jacobi sweep of (I - r·Lap) u_new = u_old, in-place safe."""
    i, j, k = wp.tid()
    if i >= u_in.shape[0] or j >= u_in.shape[1] or k >= u_in.shape[2]:
        return
    centre = u_in[i, j, k]
    nb_sum = float(0.0); nb_count = float(0.0)
    if i > 0:                      nb_sum += u_in[i - 1, j, k]; nb_count += 1.0
    if i < u_in.shape[0] - 1:      nb_sum += u_in[i + 1, j, k]; nb_count += 1.0
    if j > 0:                      nb_sum += u_in[i, j - 1, k]; nb_count += 1.0
    if j < u_in.shape[1] - 1:      nb_sum += u_in[i, j + 1, k]; nb_count += 1.0
    if k > 0:                      nb_sum += u_in[i, j, k - 1]; nb_count += 1.0
    if k < u_in.shape[2] - 1:      nb_sum += u_in[i, j, k + 1]; nb_count += 1.0
    # diagonal stencil: 1 + r*nb_count; rhs: u_old + r*nb_sum
    u_out[i, j, k] = (u_old[i, j, k] + r * nb_sum) / (1.0 + r * nb_count)


block("S2.13", "Implicit viscosity Jacobi sweep on one MAC component")(k3_jacobi_visc)


# =============================================================================
# S2.14 — Surface tension (Brackbill-Kothe Continuum Surface Force)
# =============================================================================
# Builds a smoothed fluid indicator χ̃, derives unit normal n̂ = ∇χ̃/|∇χ̃|,
# curvature κ = −∇·n̂, then applies face force σ·κ·∇χ̃ as a velocity impulse.
# Bridson §8.5; Brackbill, Kothe & Zemach 1992.


@wp.kernel
def k3_indicator_from_marker(
    marker: wp.array3d(dtype=int),
    chi: wp.array3d(dtype=float),
):
    """[BLK S2.14.1a] Raw fluid indicator: 1 where marker==1 (fluid), else 0."""
    i, j, k = wp.tid()
    if i >= marker.shape[0] or j >= marker.shape[1] or k >= marker.shape[2]:
        return
    if marker[i, j, k] == 1:
        chi[i, j, k] = 1.0
    else:
        chi[i, j, k] = 0.0


block("S2.14.1", "Build raw fluid indicator field from marker (pre-blur)")(k3_indicator_from_marker)


@wp.kernel
def k3_box_blur_centred(
    src: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
):
    """3x3x3 box-blur, boundary-aware (divides by valid-voxel count).
    Local copy of G1.7 — used for χ → χ̃ smoothing inside CSF."""
    i, j, k = wp.tid()
    if i >= src.shape[0] or j >= src.shape[1] or k >= src.shape[2]:
        return
    s = float(0.0); n = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                ii = i + di; jj = j + dj; kk = k + dk
                if (0 <= ii and ii < src.shape[0]
                    and 0 <= jj and jj < src.shape[1]
                    and 0 <= kk and kk < src.shape[2]):
                    s += src[ii, jj, kk]
                    n += 1.0
    dst[i, j, k] = s / n


block("S2.14.1", "Box-blur smoothing pass for χ → χ̃")(k3_box_blur_centred)


@wp.kernel
def k3_csf_normal(
    chi: wp.array3d(dtype=float),
    n_x: wp.array3d(dtype=float),
    n_y: wp.array3d(dtype=float),
    n_z: wp.array3d(dtype=float),
    dx: float,
):
    """[BLK S2.14.2] Unit normal n̂ = ∇χ̃ / (|∇χ̃|+ε) at cell centres
    via central differences. Skips a 1-cell boundary (writes zero)."""
    i, j, k = wp.tid()
    if i >= chi.shape[0] or j >= chi.shape[1] or k >= chi.shape[2]:
        return
    nxs = chi.shape[0]; nys = chi.shape[1]; nzs = chi.shape[2]
    if i == 0 or j == 0 or k == 0 or i == nxs - 1 or j == nys - 1 or k == nzs - 1:
        n_x[i, j, k] = 0.0; n_y[i, j, k] = 0.0; n_z[i, j, k] = 0.0
        return
    gx = (chi[i + 1, j, k] - chi[i - 1, j, k]) / (2.0 * dx)
    gy = (chi[i, j + 1, k] - chi[i, j - 1, k]) / (2.0 * dx)
    gz = (chi[i, j, k + 1] - chi[i, j, k - 1]) / (2.0 * dx)
    mag = wp.sqrt(gx * gx + gy * gy + gz * gz) + 1.0e-8
    n_x[i, j, k] = gx / mag
    n_y[i, j, k] = gy / mag
    n_z[i, j, k] = gz / mag


block("S2.14.2", "Unit-normal field n̂ = ∇χ̃/|∇χ̃| on cell centres")(k3_csf_normal)


@wp.kernel
def k3_csf_curvature(
    n_x: wp.array3d(dtype=float),
    n_y: wp.array3d(dtype=float),
    n_z: wp.array3d(dtype=float),
    kappa: wp.array3d(dtype=float),
    dx: float,
):
    """[BLK S2.14.3] κ = −∇·n̂ at cell centres (central diff)."""
    i, j, k = wp.tid()
    if i >= kappa.shape[0] or j >= kappa.shape[1] or k >= kappa.shape[2]:
        return
    nxs = kappa.shape[0]; nys = kappa.shape[1]; nzs = kappa.shape[2]
    if i == 0 or j == 0 or k == 0 or i == nxs - 1 or j == nys - 1 or k == nzs - 1:
        kappa[i, j, k] = 0.0
        return
    dnx = (n_x[i + 1, j, k] - n_x[i - 1, j, k]) / (2.0 * dx)
    dny = (n_y[i, j + 1, k] - n_y[i, j - 1, k]) / (2.0 * dx)
    dnz = (n_z[i, j, k + 1] - n_z[i, j, k - 1]) / (2.0 * dx)
    kappa[i, j, k] = -(dnx + dny + dnz)


block("S2.14.3", "Curvature κ = −∇·n̂")(k3_csf_curvature)


@wp.kernel
def k3_csf_apply_u(
    u: wp.array3d(dtype=float),
    chi: wp.array3d(dtype=float),
    kappa: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    sigma_dt_over_rho: float,
    dx: float,
    nx: int, ny: int, nz: int,
):
    """[BLK S2.14.4a] u-face: Δu = (σ dt/ρ) · κ_face · (∂χ̃/∂x)_face.
    Skips domain-boundary and solid-adjacent faces."""
    i, j, k = wp.tid()
    if i <= 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return
    if marker[i - 1, j, k] == 2 or marker[i, j, k] == 2:
        return
    grad = (chi[i, j, k] - chi[i - 1, j, k]) / dx
    kf = 0.5 * (kappa[i, j, k] + kappa[i - 1, j, k])
    u[i, j, k] = u[i, j, k] + sigma_dt_over_rho * kf * grad


@wp.kernel
def k3_csf_apply_v(
    v: wp.array3d(dtype=float),
    chi: wp.array3d(dtype=float),
    kappa: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    sigma_dt_over_rho: float,
    dx: float,
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i < 0 or i >= nx or j <= 0 or j >= ny or k < 0 or k >= nz:
        return
    if marker[i, j - 1, k] == 2 or marker[i, j, k] == 2:
        return
    grad = (chi[i, j, k] - chi[i, j - 1, k]) / dx
    kf = 0.5 * (kappa[i, j, k] + kappa[i, j - 1, k])
    v[i, j, k] = v[i, j, k] + sigma_dt_over_rho * kf * grad


@wp.kernel
def k3_csf_apply_w(
    w_: wp.array3d(dtype=float),
    chi: wp.array3d(dtype=float),
    kappa: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    sigma_dt_over_rho: float,
    dx: float,
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i < 0 or i >= nx or j < 0 or j >= ny or k <= 0 or k >= nz:
        return
    if marker[i, j, k - 1] == 2 or marker[i, j, k] == 2:
        return
    grad = (chi[i, j, k] - chi[i, j, k - 1]) / dx
    kf = 0.5 * (kappa[i, j, k] + kappa[i, j, k - 1])
    w_[i, j, k] = w_[i, j, k] + sigma_dt_over_rho * kf * grad


block("S2.14.4", "Apply CSF impulse (σ·κ·∇χ̃ · dt/ρ) to MAC u/v/w faces")(k3_csf_apply_u)


# -----------------------------------------------------------------------------
# S2.14.6 — Force-balancing (kill parasitic currents from discrete κ noise)
# -----------------------------------------------------------------------------
# CSF on a discrete grid does not respect Newton's 3rd law exactly: the sum of
# σ·κ·∇χ̃ over a closed fluid blob should integrate to zero (internal cohesive
# force), but the finite-difference κ has O(dx) noise that biases the sum.
# Without correction the blob drifts (parasitic currents); after enough steps
# it walks into a wall and smears. The cure is to project out the bulk mean
# impulse over the fluid region — done as a separate pass so the per-face
# kernel above stays simple and visually inspectable.

@wp.kernel
def k3_csf_sum_u(
    u_before: wp.array3d(dtype=float),
    u_after: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    sum_du: wp.array(dtype=float),
    count: wp.array(dtype=float),
    nx: int, ny: int, nz: int,
):
    """Σ Δu and face-count over fluid-adjacent u-faces (skip walls)."""
    i, j, k = wp.tid()
    if i <= 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return
    if marker[i - 1, j, k] == 2 or marker[i, j, k] == 2:
        return
    if marker[i - 1, j, k] != 1 and marker[i, j, k] != 1:
        return
    wp.atomic_add(sum_du, 0, u_after[i, j, k] - u_before[i, j, k])
    wp.atomic_add(count, 0, 1.0)


@wp.kernel
def k3_csf_sum_v(
    v_before: wp.array3d(dtype=float),
    v_after: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    sum_dv: wp.array(dtype=float),
    count: wp.array(dtype=float),
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i < 0 or i >= nx or j <= 0 or j >= ny or k < 0 or k >= nz:
        return
    if marker[i, j - 1, k] == 2 or marker[i, j, k] == 2:
        return
    if marker[i, j - 1, k] != 1 and marker[i, j, k] != 1:
        return
    wp.atomic_add(sum_dv, 0, v_after[i, j, k] - v_before[i, j, k])
    wp.atomic_add(count, 0, 1.0)


@wp.kernel
def k3_csf_sum_w(
    w_before: wp.array3d(dtype=float),
    w_after: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    sum_dw: wp.array(dtype=float),
    count: wp.array(dtype=float),
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i < 0 or i >= nx or j < 0 or j >= ny or k <= 0 or k >= nz:
        return
    if marker[i, j, k - 1] == 2 or marker[i, j, k] == 2:
        return
    if marker[i, j, k - 1] != 1 and marker[i, j, k] != 1:
        return
    wp.atomic_add(sum_dw, 0, w_after[i, j, k] - w_before[i, j, k])
    wp.atomic_add(count, 0, 1.0)


@wp.kernel
def k3_csf_subtract_bias_u(
    u: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    bias: float,
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i <= 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return
    if marker[i - 1, j, k] == 2 or marker[i, j, k] == 2:
        return
    if marker[i - 1, j, k] != 1 and marker[i, j, k] != 1:
        return
    u[i, j, k] = u[i, j, k] - bias


@wp.kernel
def k3_csf_subtract_bias_v(
    v: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    bias: float,
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i < 0 or i >= nx or j <= 0 or j >= ny or k < 0 or k >= nz:
        return
    if marker[i, j - 1, k] == 2 or marker[i, j, k] == 2:
        return
    if marker[i, j - 1, k] != 1 and marker[i, j, k] != 1:
        return
    v[i, j, k] = v[i, j, k] - bias


@wp.kernel
def k3_csf_subtract_bias_w(
    w_: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    bias: float,
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i < 0 or i >= nx or j < 0 or j >= ny or k <= 0 or k >= nz:
        return
    if marker[i, j, k - 1] == 2 or marker[i, j, k] == 2:
        return
    if marker[i, j, k - 1] != 1 and marker[i, j, k] != 1:
        return
    w_[i, j, k] = w_[i, j, k] - bias


block("S2.14.6", "Force-balance CSF: subtract per-axis mean impulse over the fluid blob")(k3_csf_sum_u)


# =============================================================================
# S2.15 — Per-particle color attribute (RGB)
# =============================================================================
# Same trilinear P2G/G2P scheme used for velocity, but on cell-centred grids
# (no MAC offsets). Three channels scattered in one kernel to amortise launch
# overhead. Linear RGB blend — physically additive light mixing. Mixbox is a
# future drop-in on G2P.


@wp.kernel
def k3_p2g_color(
    pos: wp.array(dtype=wp.vec3),
    color: wp.array(dtype=wp.vec3),
    cgrid_r: wp.array3d(dtype=float),
    cgrid_g: wp.array3d(dtype=float),
    cgrid_b: wp.array3d(dtype=float),
    cgrid_w: wp.array3d(dtype=float),
    dx: float, nx: int, ny: int, nz: int,
):
    """[BLK S2.15.1] Trilinear scatter of per-particle RGB into cell-centred grid.
    Uses the same 8-corner weights as velocity P2G; cell-centred (fx = px/dx - 0.5)."""
    p = wp.tid()
    pp = pos[p]
    c = color[p]
    fx = pp[0] / dx - 0.5
    fy = pp[1] / dx - 0.5
    fz = pp[2] / dx - 0.5
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii >= 0 and ii < nx and jj >= 0 and jj < ny and kk >= 0 and kk < nz:
                    wx = float(1.0) - sx
                    if di == 1: wx = sx
                    wy = float(1.0) - sy
                    if dj == 1: wy = sy
                    wz = float(1.0) - sz
                    if dk == 1: wz = sz
                    w = wx * wy * wz
                    wp.atomic_add(cgrid_r, ii, jj, kk, c[0] * w)
                    wp.atomic_add(cgrid_g, ii, jj, kk, c[1] * w)
                    wp.atomic_add(cgrid_b, ii, jj, kk, c[2] * w)
                    wp.atomic_add(cgrid_w, ii, jj, kk, w)


block("S2.15.1", "P2G scatter of per-particle RGB color to cell-centred grid")(k3_p2g_color)


@wp.kernel
def k3_normalize_color(
    cgrid_r: wp.array3d(dtype=float),
    cgrid_g: wp.array3d(dtype=float),
    cgrid_b: wp.array3d(dtype=float),
    cgrid_w: wp.array3d(dtype=float),
):
    """[BLK S2.15.2] Divide accumulated RGB by scalar weight (per-cell)."""
    i, j, k = wp.tid()
    if i >= cgrid_r.shape[0] or j >= cgrid_r.shape[1] or k >= cgrid_r.shape[2]:
        return
    w = cgrid_w[i, j, k]
    if w > 1.0e-8:
        inv = 1.0 / w
        cgrid_r[i, j, k] = cgrid_r[i, j, k] * inv
        cgrid_g[i, j, k] = cgrid_g[i, j, k] * inv
        cgrid_b[i, j, k] = cgrid_b[i, j, k] * inv
    else:
        cgrid_r[i, j, k] = 0.0
        cgrid_g[i, j, k] = 0.0
        cgrid_b[i, j, k] = 0.0


block("S2.15.2", "Normalize grid color by deposited weight")(k3_normalize_color)


@wp.kernel
def k3_g2p_color(
    pos: wp.array(dtype=wp.vec3),
    color: wp.array(dtype=wp.vec3),
    cgrid_r: wp.array3d(dtype=float),
    cgrid_g: wp.array3d(dtype=float),
    cgrid_b: wp.array3d(dtype=float),
    cgrid_w: wp.array3d(dtype=float),
    dx: float, nx: int, ny: int, nz: int,
):
    """[BLK S2.15.3] Trilinear gather of grid RGB back to particle.
    Particles in empty cells keep their previous color (no overwrite)."""
    p = wp.tid()
    pp = pos[p]
    fx = pp[0] / dx - 0.5
    fy = pp[1] / dx - 0.5
    fz = pp[2] / dx - 0.5
    # Use sample3 on each channel + check weight to decide if we update.
    r = sample3(cgrid_r, fx, fy, fz, nx, ny, nz)
    g = sample3(cgrid_g, fx, fy, fz, nx, ny, nz)
    b = sample3(cgrid_b, fx, fy, fz, nx, ny, nz)
    w = sample3(cgrid_w, fx, fy, fz, nx, ny, nz)
    if w > 1.0e-6:
        color[p] = wp.vec3(r, g, b)


block("S2.15.3", "G2P gather grid color back to particle")(k3_g2p_color)


# =============================================================================
# S2.18 — Per-particle SCALAR attribute transfer (temperature, age, density…)
#         Same P2G/normalize/G2P pattern as S2.15 colour, one channel.
#         B11: generalises S2.15 for non-RGB attributes.
# =============================================================================

@wp.kernel
def k3_p2g_scalar(
    pos: wp.array(dtype=wp.vec3),
    attr: wp.array(dtype=float),
    sgrid_v: wp.array3d(dtype=float),
    sgrid_w: wp.array3d(dtype=float),
    dx: float, nx: int, ny: int, nz: int,
):
    """[BLK S2.18.1] Trilinear scatter of per-particle scalar onto a
    cell-centred grid. Mirrors k3_p2g_color but for one channel."""
    p = wp.tid()
    pp = pos[p]
    v = attr[p]
    fx = pp[0] / dx - 0.5
    fy = pp[1] / dx - 0.5
    fz = pp[2] / dx - 0.5
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii >= 0 and ii < nx and jj >= 0 and jj < ny and kk >= 0 and kk < nz:
                    wx = float(1.0) - sx
                    if di == 1: wx = sx
                    wy = float(1.0) - sy
                    if dj == 1: wy = sy
                    wz = float(1.0) - sz
                    if dk == 1: wz = sz
                    w = wx * wy * wz
                    wp.atomic_add(sgrid_v, ii, jj, kk, v * w)
                    wp.atomic_add(sgrid_w, ii, jj, kk, w)


block("S2.18.1", "P2G scatter of per-particle scalar onto grid")(k3_p2g_scalar)


@wp.kernel
def k3_normalize_scalar(
    sgrid_v: wp.array3d(dtype=float),
    sgrid_w: wp.array3d(dtype=float),
):
    """[BLK S2.18.2] Divide accumulated scalar by deposited weight."""
    i, j, k = wp.tid()
    if i >= sgrid_v.shape[0] or j >= sgrid_v.shape[1] or k >= sgrid_v.shape[2]:
        return
    w = sgrid_w[i, j, k]
    if w > 1.0e-8:
        sgrid_v[i, j, k] = sgrid_v[i, j, k] / w
    else:
        sgrid_v[i, j, k] = 0.0


block("S2.18.2", "Normalize grid scalar by deposited weight")(k3_normalize_scalar)


@wp.kernel
def k3_g2p_scalar(
    pos: wp.array(dtype=wp.vec3),
    attr: wp.array(dtype=float),
    sgrid_v: wp.array3d(dtype=float),
    sgrid_w: wp.array3d(dtype=float),
    dx: float, nx: int, ny: int, nz: int,
):
    """[BLK S2.18.3] Trilinear gather of grid scalar back to particle.
    Particles in empty cells keep their previous value (no overwrite)."""
    p = wp.tid()
    pp = pos[p]
    fx = pp[0] / dx - 0.5
    fy = pp[1] / dx - 0.5
    fz = pp[2] / dx - 0.5
    v = sample3(sgrid_v, fx, fy, fz, nx, ny, nz)
    w = sample3(sgrid_w, fx, fy, fz, nx, ny, nz)
    if w > 1.0e-6:
        attr[p] = v


block("S2.18.3", "G2P gather grid scalar back to particle")(k3_g2p_scalar)


# =============================================================================
# S2.4 — Solid boundary conditions
# =============================================================================

@wp.kernel
def k3_enforce_solid_bc(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    solid_u: wp.array3d(dtype=float),
    solid_v: wp.array3d(dtype=float),
    solid_w: wp.array3d(dtype=float),
    nx: int, ny: int, nz: int,
):
    """Enforce normal velocity at solid faces. For static obstacles this is 0;
    for moving obstacles it is the obstacle's velocity component (solid_u/v/w)."""
    i, j, k = wp.tid()
    if i <= nx and j < ny and k < nz:
        left = (i == 0) or (i > 0 and marker[i - 1, j, k] == 2)
        right = (i == nx) or (i < nx and marker[i, j, k] == 2)
        if left or right:
            u[i, j, k] = solid_u[i, j, k]
    if i < nx and j <= ny and k < nz:
        bot = (j == 0) or (j > 0 and marker[i, j - 1, k] == 2)
        top = (j == ny) or (j < ny and marker[i, j, k] == 2)
        if bot or top:
            v[i, j, k] = solid_v[i, j, k]
    if i < nx and j < ny and k <= nz:
        back = (k == 0) or (k > 0 and marker[i, j, k - 1] == 2)
        front = (k == nz) or (k < nz and marker[i, j, k] == 2)
        if back or front:
            w[i, j, k] = solid_w[i, j, k]


block("S2.4", "Solid boundary condition: faces = obstacle velocity (zero for static)")(k3_enforce_solid_bc)


@wp.kernel
def k3_clear_solid_vel(
    solid_u: wp.array3d(dtype=float),
    solid_v: wp.array3d(dtype=float),
    solid_w: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    if i < solid_u.shape[0] and j < solid_u.shape[1] and k < solid_u.shape[2]:
        solid_u[i, j, k] = 0.0
    if i < solid_v.shape[0] and j < solid_v.shape[1] and k < solid_v.shape[2]:
        solid_v[i, j, k] = 0.0
    if i < solid_w.shape[0] and j < solid_w.shape[1] and k < solid_w.shape[2]:
        solid_w[i, j, k] = 0.0


@wp.kernel
def k3_write_solid_face_vel(
    mask: wp.array3d(dtype=int),
    vx: float, vy: float, vz: float,
    solid_u: wp.array3d(dtype=float),
    solid_v: wp.array3d(dtype=float),
    solid_w: wp.array3d(dtype=float),
):
    """For each cell where mask==1, write (vx,vy,vz) into all 6 surrounding faces."""
    i, j, k = wp.tid()
    if i >= mask.shape[0] or j >= mask.shape[1] or k >= mask.shape[2]:
        return
    if mask[i, j, k] != 1:
        return
    solid_u[i, j, k] = vx
    solid_u[i + 1, j, k] = vx
    solid_v[i, j, k] = vy
    solid_v[i, j + 1, k] = vy
    solid_w[i, j, k] = vz
    solid_w[i, j, k + 1] = vz


block("D4.6", "Write moving-obstacle face velocity into solid_u/v/w")(k3_write_solid_face_vel)


# =============================================================================
# S2.5 — Divergence
# =============================================================================

@wp.kernel
def k3_compute_divergence(
    u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    dx: float, dt: float, rho: float,
):
    i, j, k = wp.tid()
    if i >= div.shape[0] or j >= div.shape[1] or k >= div.shape[2]:
        return
    if marker[i, j, k] != 1:
        div[i, j, k] = 0.0
        return
    d = (u[i + 1, j, k] - u[i, j, k]
         + v[i, j + 1, k] - v[i, j, k]
         + w[i, j, k + 1] - w[i, j, k])
    # rhs for Poisson: Lap(p)=(rho/dt)·div(u); scaled to match Jacobi update.
    div[i, j, k] = d * (rho * dx / dt)


block("S2.5", "Cell-wise divergence × rho·dx/dt (Poisson RHS)")(k3_compute_divergence)


# =============================================================================
# S2.6.1 — Jacobi pressure iteration
# =============================================================================

@wp.kernel
def k3_jacobi_pressure(
    p_in: wp.array3d(dtype=float),
    p_out: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
):
    i, j, k = wp.tid()
    nx = p_in.shape[0]; ny = p_in.shape[1]; nz = p_in.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if marker[i, j, k] != 1:
        p_out[i, j, k] = 0.0
        return
    sum_nb = float(0.0); diag = float(0.0)
    if i > 0:
        m = marker[i - 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i - 1, j, k]
    if i < nx - 1:
        m = marker[i + 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i + 1, j, k]
    if j > 0:
        m = marker[i, j - 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j - 1, k]
    if j < ny - 1:
        m = marker[i, j + 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j + 1, k]
    if k > 0:
        m = marker[i, j, k - 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j, k - 1]
    if k < nz - 1:
        m = marker[i, j, k + 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j, k + 1]
    if diag < 0.5:
        p_out[i, j, k] = 0.0
        return
    p_out[i, j, k] = (sum_nb - div[i, j, k]) / diag


block("S2.6.1", "Jacobi pressure iteration (one sweep)")(k3_jacobi_pressure)


# -----------------------------------------------------------------------------
# S2.16 — Active-block bitmask
# -----------------------------------------------------------------------------
# For each 8³ block of cells, set block_active=1 if any cell has marker==1
# (fluid). A block-aligned skip in the pressure solve is safe because the
# pressure of non-fluid cells stays at 0 across iterations (initial zero) —
# so neighbours of an "inactive" block read 0 from p_in just as they would
# from the dense path. See DESIGN.md §5.5.

BLOCK_SIZE = 8


@wp.kernel
def k_mark_active_blocks(
    marker: wp.array3d(dtype=int),
    block_active: wp.array3d(dtype=int),
    block_size: int,
):
    """[BLK S2.16] Per cell, if marker==fluid set the host 8³ block to 1.
    Uses atomic_or-style assignment via atomic_max (1 wins over 0)."""
    i, j, k = wp.tid()
    if i >= marker.shape[0] or j >= marker.shape[1] or k >= marker.shape[2]:
        return
    if marker[i, j, k] == 1:
        bi = i // block_size
        bj = j // block_size
        bk = k // block_size
        # atomic_max with target=1 is equivalent to "set to 1 if not already".
        wp.atomic_max(block_active, bi, bj, bk, 1)


block("S2.16", "Active-block bitmask builder (8³ tiles with any fluid cell)")(k_mark_active_blocks)


@wp.kernel
def k3_jacobi_pressure_blocksparse(
    p_in: wp.array3d(dtype=float),
    p_out: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    block_active: wp.array3d(dtype=int),
    block_size: int,
):
    """[BLK S2.6.4] Block-sparse Jacobi: identical to k3_jacobi_pressure but
    skips the entire 8³ tile of threads when block_active[bi,bj,bk]==0.

    The early-exit cost on a skipped block is a single atomic-free read of
    block_active + branch — far cheaper than the 12+ marker reads of the
    full stencil. At low fill ratios the kernel barely touches the empty
    regions of the domain."""
    i, j, k = wp.tid()
    nx = p_in.shape[0]; ny = p_in.shape[1]; nz = p_in.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    bi = i // block_size
    bj = j // block_size
    bk = k // block_size
    if block_active[bi, bj, bk] == 0:
        # Maintain dense semantics: non-fluid cells get p=0. Inactive blocks
        # are by definition all non-fluid (marker != 1 for every cell inside),
        # so this write is the same as the dense kernel's "marker != 1" branch.
        p_out[i, j, k] = 0.0
        return
    if marker[i, j, k] != 1:
        p_out[i, j, k] = 0.0
        return
    sum_nb = float(0.0); diag = float(0.0)
    if i > 0:
        m = marker[i - 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i - 1, j, k]
    if i < nx - 1:
        m = marker[i + 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i + 1, j, k]
    if j > 0:
        m = marker[i, j - 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j - 1, k]
    if j < ny - 1:
        m = marker[i, j + 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j + 1, k]
    if k > 0:
        m = marker[i, j, k - 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j, k - 1]
    if k < nz - 1:
        m = marker[i, j, k + 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j, k + 1]
    if diag < 0.5:
        p_out[i, j, k] = 0.0
        return
    p_out[i, j, k] = (sum_nb - div[i, j, k]) / diag


block("S2.6.4", "Block-sparse Jacobi pressure (skip inactive 8³ blocks)")(k3_jacobi_pressure_blocksparse)


# -----------------------------------------------------------------------------
# Active-block compaction → "per-tile launch" Jacobi.
# The per-cell early-exit version above is correct but only marginally faster
# than the dense path because dense kernel already short-circuits non-fluid
# cells in O(1). The real win comes from *not launching threads at all* in
# dead regions. We compact the active-block coordinates into a flat list and
# launch exactly `n_active × 512` threads, one per cell in the active blocks.
# -----------------------------------------------------------------------------

@wp.kernel
def k_compact_active_blocks(
    block_active: wp.array3d(dtype=int),
    prefix: wp.array(dtype=int),
    active_coords: wp.array(dtype=wp.vec3i),
):
    """If block (bi,bj,bk) is active, write its coords to slot prefix-1.
    `prefix` is inclusive prefix-sum of the flattened bitmask."""
    bi, bj, bk = wp.tid()
    nbx = block_active.shape[0]; nby = block_active.shape[1]; nbz = block_active.shape[2]
    if bi >= nbx or bj >= nby or bk >= nbz:
        return
    if block_active[bi, bj, bk] == 0:
        return
    flat = bi * (nby * nbz) + bj * nbz + bk
    dst = prefix[flat] - 1
    active_coords[dst] = wp.vec3i(bi, bj, bk)


@wp.kernel
def k3_jacobi_pressure_per_tile(
    p_in: wp.array3d(dtype=float),
    p_out: wp.array3d(dtype=float),
    div: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    active_coords: wp.array(dtype=wp.vec3i),
    block_size: int,
):
    """Launched with dim = n_active * block_size³. Each thread covers one
    cell within an active block. Inactive blocks are *not iterated over*."""
    tid = wp.tid()
    cells_per_block = block_size * block_size * block_size
    blk = tid // cells_per_block
    rem = tid - blk * cells_per_block
    di = rem // (block_size * block_size)
    rem2 = rem - di * (block_size * block_size)
    dj = rem2 // block_size
    dk = rem2 - dj * block_size
    c = active_coords[blk]
    i = c[0] * block_size + di
    j = c[1] * block_size + dj
    k = c[2] * block_size + dk
    nx = p_in.shape[0]; ny = p_in.shape[1]; nz = p_in.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if marker[i, j, k] != 1:
        p_out[i, j, k] = 0.0
        return
    sum_nb = float(0.0); diag = float(0.0)
    if i > 0:
        m = marker[i - 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i - 1, j, k]
    if i < nx - 1:
        m = marker[i + 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i + 1, j, k]
    if j > 0:
        m = marker[i, j - 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j - 1, k]
    if j < ny - 1:
        m = marker[i, j + 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j + 1, k]
    if k > 0:
        m = marker[i, j, k - 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j, k - 1]
    if k < nz - 1:
        m = marker[i, j, k + 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p_in[i, j, k + 1]
    if diag < 0.5:
        p_out[i, j, k] = 0.0
        return
    p_out[i, j, k] = (sum_nb - div[i, j, k]) / diag


block("S2.6.4", "Per-tile Jacobi pressure: launch only on active 8³ blocks")(k3_jacobi_pressure_per_tile)


# =============================================================================
# S2.6.2 — Gauss–Seidel red-black (in-place updates → ~2× faster convergence)
# =============================================================================

@wp.kernel
def k3_gauss_seidel_rb(
    p: wp.array3d(dtype=float),         # in-place
    div: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    color: int,                          # 0=red, 1=black
):
    """In-place GS sweep on cells where (i+j+k)%2 == color."""
    i, j, k = wp.tid()
    nx = p.shape[0]; ny = p.shape[1]; nz = p.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if (i + j + k) % 2 != color:
        return
    if marker[i, j, k] != 1:
        p[i, j, k] = 0.0
        return
    sum_nb = float(0.0); diag = float(0.0)
    if i > 0:
        m = marker[i - 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i - 1, j, k]
    if i < nx - 1:
        m = marker[i + 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i + 1, j, k]
    if j > 0:
        m = marker[i, j - 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i, j - 1, k]
    if j < ny - 1:
        m = marker[i, j + 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i, j + 1, k]
    if k > 0:
        m = marker[i, j, k - 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i, j, k - 1]
    if k < nz - 1:
        m = marker[i, j, k + 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i, j, k + 1]
    if diag < 0.5:
        p[i, j, k] = 0.0
    else:
        p[i, j, k] = (sum_nb - div[i, j, k]) / diag


block("S2.6.2", "Gauss-Seidel red-black sweep (in-place)")(k3_gauss_seidel_rb)


# =============================================================================
# S2.6.5 — Per-tile GS-RB (B4.1): same in-place red-black sweep but launched
#          ONLY on the active 8³ blocks compacted by k_mark_active_blocks +
#          k_compact_active_blocks (S2.16). For sparse-fill scenes this skips
#          the dense (nx·ny·nz) launch grid and visits only fluid tiles.
# =============================================================================

@wp.kernel
def k3_gauss_seidel_rb_per_tile(
    p: wp.array3d(dtype=float),         # in-place
    div: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    active_coords: wp.array(dtype=wp.vec3i),
    block_size: int,
    color: int,                          # 0=red, 1=black
):
    """Launched with dim = n_active * block_size³. Each thread maps a
    (block_index, cell_in_block) pair to a global (i,j,k). Cells where
    (i+j+k)%2 != color are skipped.

    Updates p in place — same red/black ordering as `k3_gauss_seidel_rb`.
    Outside-active-set cells are not touched; they keep whatever value
    they had from initialisation (zeros) or a prior sweep.
    """
    tid = wp.tid()
    cells_per_block = block_size * block_size * block_size
    blk = tid // cells_per_block
    rem = tid - blk * cells_per_block
    di = rem // (block_size * block_size)
    rem2 = rem - di * (block_size * block_size)
    dj = rem2 // block_size
    dk = rem2 - dj * block_size
    c = active_coords[blk]
    i = c[0] * block_size + di
    j = c[1] * block_size + dj
    k = c[2] * block_size + dk
    nx = p.shape[0]; ny = p.shape[1]; nz = p.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if (i + j + k) % 2 != color:
        return
    if marker[i, j, k] != 1:
        p[i, j, k] = 0.0
        return
    sum_nb = float(0.0); diag = float(0.0)
    if i > 0:
        m = marker[i - 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i - 1, j, k]
    if i < nx - 1:
        m = marker[i + 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i + 1, j, k]
    if j > 0:
        m = marker[i, j - 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i, j - 1, k]
    if j < ny - 1:
        m = marker[i, j + 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i, j + 1, k]
    if k > 0:
        m = marker[i, j, k - 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i, j, k - 1]
    if k < nz - 1:
        m = marker[i, j, k + 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += p[i, j, k + 1]
    if diag < 0.5:
        p[i, j, k] = 0.0
    else:
        p[i, j, k] = (sum_nb - div[i, j, k]) / diag


block("S2.6.5", "Per-tile GS-RB: red-black sweep on active 8³ blocks only")(k3_gauss_seidel_rb_per_tile)


# =============================================================================
# S2.7 — Subtract pressure gradient
# =============================================================================

@wp.kernel
def k3_subtract_pressure_grad(
    u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    dx: float, dt: float, rho: float,
):
    i, j, k = wp.tid()
    nx = marker.shape[0]; ny = marker.shape[1]; nz = marker.shape[2]
    scale = dt / (rho * dx)
    if i >= 1 and i <= nx - 1 and j < ny and k < nz:
        ml = marker[i - 1, j, k]; mr = marker[i, j, k]
        if (ml == 1 or mr == 1) and ml != 2 and mr != 2:
            pl = float(0.0); pr = float(0.0)
            if ml == 1: pl = p[i - 1, j, k]
            if mr == 1: pr = p[i, j, k]
            u[i, j, k] = u[i, j, k] - scale * (pr - pl)
    if j >= 1 and j <= ny - 1 and i < nx and k < nz:
        mb = marker[i, j - 1, k]; mt = marker[i, j, k]
        if (mb == 1 or mt == 1) and mb != 2 and mt != 2:
            pb = float(0.0); pt = float(0.0)
            if mb == 1: pb = p[i, j - 1, k]
            if mt == 1: pt = p[i, j, k]
            v[i, j, k] = v[i, j, k] - scale * (pt - pb)
    if k >= 1 and k <= nz - 1 and i < nx and j < ny:
        mba = marker[i, j, k - 1]; mfr = marker[i, j, k]
        if (mba == 1 or mfr == 1) and mba != 2 and mfr != 2:
            pba = float(0.0); pfr = float(0.0)
            if mba == 1: pba = p[i, j, k - 1]
            if mfr == 1: pfr = p[i, j, k]
            w[i, j, k] = w[i, j, k] - scale * (pfr - pba)


block("S2.7", "Subtract pressure gradient from MAC faces")(k3_subtract_pressure_grad)


# =============================================================================
# S2.8 + S2.9 — G2P + FLIP/PIC blend + advection (fused, per particle)
# =============================================================================

@wp.kernel
def k3_g2p_and_advect(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    us: wp.array3d(dtype=float),
    vs: wp.array3d(dtype=float),
    ws: wp.array3d(dtype=float),
    dx: float, dt: float,
    nx: int, ny: int, nz: int,
    flip_blend: float,
    dom: wp.vec3,
):
    pid = wp.tid()
    p = pos[pid]
    ov = vel[pid]
    pic_u = sample3(u, p[0]/dx,        p[1]/dx - 0.5, p[2]/dx - 0.5, nx + 1, ny,     nz)
    old_u = sample3(us, p[0]/dx,       p[1]/dx - 0.5, p[2]/dx - 0.5, nx + 1, ny,     nz)
    pic_v = sample3(v, p[0]/dx - 0.5,  p[1]/dx,       p[2]/dx - 0.5, nx,     ny + 1, nz)
    old_v = sample3(vs, p[0]/dx - 0.5, p[1]/dx,       p[2]/dx - 0.5, nx,     ny + 1, nz)
    pic_w = sample3(w, p[0]/dx - 0.5,  p[1]/dx - 0.5, p[2]/dx,       nx,     ny,     nz + 1)
    old_w = sample3(ws, p[0]/dx - 0.5, p[1]/dx - 0.5, p[2]/dx,       nx,     ny,     nz + 1)
    flip_u = ov[0] + (pic_u - old_u)
    flip_v = ov[1] + (pic_v - old_v)
    flip_w = ov[2] + (pic_w - old_w)
    nu = flip_blend * flip_u + (1.0 - flip_blend) * pic_u
    nv = flip_blend * flip_v + (1.0 - flip_blend) * pic_v
    nw = flip_blend * flip_w + (1.0 - flip_blend) * pic_w
    nvel = wp.vec3(nu, nv, nw)
    vel[pid] = nvel
    npos = p + nvel * dt
    eps = dx * 1.001
    npos[0] = wp.clamp(npos[0], eps, dom[0] - eps)
    npos[1] = wp.clamp(npos[1], eps, dom[1] - eps)
    npos[2] = wp.clamp(npos[2], eps, dom[2] - eps)
    pos[pid] = npos


block("S2.8", "G2P sample + FLIP/PIC blend (fused with S2.9)")(k3_g2p_and_advect)


# =============================================================================
# S2.12 — APIC transfer (Affine Particle-In-Cell, Jiang et al. 2015)
# =============================================================================
# Each particle stores a 3x3 affine matrix C. Rows = gradients of (u, v, w)
# at the particle. P2G adds affine extension term, G2P recovers C from the
# grid (with the linear-B-spline D = dx²/3 · I approximation, so D⁻¹ = 3/dx²).
# Drop-in replacement for FLIP/PIC: less noise, less particle clumping,
# better angular-momentum conservation.

@wp.kernel
def k3_p2g_apic(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    affine_C: wp.array(dtype=wp.mat33),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    uw: wp.array3d(dtype=float),
    vw: wp.array3d(dtype=float),
    ww: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    dx: float, nx: int, ny: int, nz: int,
):
    pid = wp.tid()
    p = pos[pid]
    vp = vel[pid]
    C = affine_C[pid]
    ci = clamp_int(int(p[0] / dx), 0, nx - 1)
    cj = clamp_int(int(p[1] / dx), 0, ny - 1)
    ck = clamp_int(int(p[2] / dx), 0, nz - 1)
    if marker[ci, cj, ck] != 2:
        marker[ci, cj, ck] = 1

    # ---- u faces at (i*dx, (j+0.5)dx, (k+0.5)dx) -------------------------
    fx = p[0] / dx; fy = p[1] / dx - 0.5; fz = p[2] / dx - 0.5
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii >= 0 and ii <= nx and jj >= 0 and jj < ny and kk >= 0 and kk < nz:
                    wxi = float(0.0); wyi = float(0.0); wzi = float(0.0)
                    if di == 0: wxi = 1.0 - sx
                    else:       wxi = sx
                    if dj == 0: wyi = 1.0 - sy
                    else:       wyi = sy
                    if dk == 0: wzi = 1.0 - sz
                    else:       wzi = sz
                    wt = wxi * wyi * wzi
                    ox = float(ii) * dx - p[0]
                    oy = (float(jj) + 0.5) * dx - p[1]
                    oz = (float(kk) + 0.5) * dx - p[2]
                    # APIC affine extension: u_eff = vp.x + C row 0 · offset
                    u_eff = vp[0] + C[0, 0] * ox + C[0, 1] * oy + C[0, 2] * oz
                    wp.atomic_add(u, ii, jj, kk, u_eff * wt)
                    wp.atomic_add(uw, ii, jj, kk, wt)

    # ---- v faces at ((i+0.5)dx, j*dx, (k+0.5)dx) -------------------------
    fx = p[0] / dx - 0.5; fy = p[1] / dx; fz = p[2] / dx - 0.5
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii >= 0 and ii < nx and jj >= 0 and jj <= ny and kk >= 0 and kk < nz:
                    wxi = float(0.0); wyi = float(0.0); wzi = float(0.0)
                    if di == 0: wxi = 1.0 - sx
                    else:       wxi = sx
                    if dj == 0: wyi = 1.0 - sy
                    else:       wyi = sy
                    if dk == 0: wzi = 1.0 - sz
                    else:       wzi = sz
                    wt = wxi * wyi * wzi
                    ox = (float(ii) + 0.5) * dx - p[0]
                    oy = float(jj) * dx - p[1]
                    oz = (float(kk) + 0.5) * dx - p[2]
                    v_eff = vp[1] + C[1, 0] * ox + C[1, 1] * oy + C[1, 2] * oz
                    wp.atomic_add(v, ii, jj, kk, v_eff * wt)
                    wp.atomic_add(vw, ii, jj, kk, wt)

    # ---- w faces at ((i+0.5)dx, (j+0.5)dx, k*dx) -------------------------
    fx = p[0] / dx - 0.5; fy = p[1] / dx - 0.5; fz = p[2] / dx
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii >= 0 and ii < nx and jj >= 0 and jj < ny and kk >= 0 and kk <= nz:
                    wxi = float(0.0); wyi = float(0.0); wzi = float(0.0)
                    if di == 0: wxi = 1.0 - sx
                    else:       wxi = sx
                    if dj == 0: wyi = 1.0 - sy
                    else:       wyi = sy
                    if dk == 0: wzi = 1.0 - sz
                    else:       wzi = sz
                    wt = wxi * wyi * wzi
                    ox = (float(ii) + 0.5) * dx - p[0]
                    oy = (float(jj) + 0.5) * dx - p[1]
                    oz = float(kk) * dx - p[2]
                    w_eff = vp[2] + C[2, 0] * ox + C[2, 1] * oy + C[2, 2] * oz
                    wp.atomic_add(w, ii, jj, kk, w_eff * wt)
                    wp.atomic_add(ww, ii, jj, kk, wt)


block("S2.12", "APIC P2G: scatter (vel + affine·offset) per face")(k3_p2g_apic)


@wp.kernel
def k3_g2p_apic_advect(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    affine_C: wp.array(dtype=wp.mat33),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    dx: float, dt: float,
    nx: int, ny: int, nz: int,
    dom: wp.vec3,
):
    """G2P + advect with APIC. Pure PIC sample for velocity; reconstruct C
    via linear-B-spline approximation D⁻¹ = 3/dx² · I."""
    pid = wp.tid()
    p = pos[pid]
    inv_dx2_3 = 3.0 / (dx * dx)

    # accumulators
    nu = float(0.0); nv = float(0.0); nw = float(0.0)
    cu0 = float(0.0); cu1 = float(0.0); cu2 = float(0.0)
    cv0 = float(0.0); cv1 = float(0.0); cv2 = float(0.0)
    cw0 = float(0.0); cw1 = float(0.0); cw2 = float(0.0)

    # ---- u faces ---------------------------------------------------------
    fx = p[0] / dx; fy = p[1] / dx - 0.5; fz = p[2] / dx - 0.5
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii < 0: ii = 0
                if jj < 0: jj = 0
                if kk < 0: kk = 0
                if ii > nx: ii = nx
                if jj >= ny: jj = ny - 1
                if kk >= nz: kk = nz - 1
                wxi = float(0.0); wyi = float(0.0); wzi = float(0.0)
                if di == 0: wxi = 1.0 - sx
                else:       wxi = sx
                if dj == 0: wyi = 1.0 - sy
                else:       wyi = sy
                if dk == 0: wzi = 1.0 - sz
                else:       wzi = sz
                wt = wxi * wyi * wzi
                u_val = u[ii, jj, kk]
                nu += wt * u_val
                ox = float(ii) * dx - p[0]
                oy = (float(jj) + 0.5) * dx - p[1]
                oz = (float(kk) + 0.5) * dx - p[2]
                fu = wt * u_val * inv_dx2_3
                cu0 += fu * ox; cu1 += fu * oy; cu2 += fu * oz

    # ---- v faces ---------------------------------------------------------
    fx = p[0] / dx - 0.5; fy = p[1] / dx; fz = p[2] / dx - 0.5
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii < 0: ii = 0
                if jj < 0: jj = 0
                if kk < 0: kk = 0
                if ii >= nx: ii = nx - 1
                if jj > ny: jj = ny
                if kk >= nz: kk = nz - 1
                wxi = float(0.0); wyi = float(0.0); wzi = float(0.0)
                if di == 0: wxi = 1.0 - sx
                else:       wxi = sx
                if dj == 0: wyi = 1.0 - sy
                else:       wyi = sy
                if dk == 0: wzi = 1.0 - sz
                else:       wzi = sz
                wt = wxi * wyi * wzi
                v_val = v[ii, jj, kk]
                nv += wt * v_val
                ox = (float(ii) + 0.5) * dx - p[0]
                oy = float(jj) * dx - p[1]
                oz = (float(kk) + 0.5) * dx - p[2]
                fv = wt * v_val * inv_dx2_3
                cv0 += fv * ox; cv1 += fv * oy; cv2 += fv * oz

    # ---- w faces ---------------------------------------------------------
    fx = p[0] / dx - 0.5; fy = p[1] / dx - 0.5; fz = p[2] / dx
    i0 = int(wp.floor(fx)); j0 = int(wp.floor(fy)); k0 = int(wp.floor(fz))
    sx = fx - float(i0); sy = fy - float(j0); sz = fz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if ii < 0: ii = 0
                if jj < 0: jj = 0
                if kk < 0: kk = 0
                if ii >= nx: ii = nx - 1
                if jj >= ny: jj = ny - 1
                if kk > nz: kk = nz
                wxi = float(0.0); wyi = float(0.0); wzi = float(0.0)
                if di == 0: wxi = 1.0 - sx
                else:       wxi = sx
                if dj == 0: wyi = 1.0 - sy
                else:       wyi = sy
                if dk == 0: wzi = 1.0 - sz
                else:       wzi = sz
                wt = wxi * wyi * wzi
                w_val = w[ii, jj, kk]
                nw += wt * w_val
                ox = (float(ii) + 0.5) * dx - p[0]
                oy = (float(jj) + 0.5) * dx - p[1]
                oz = float(kk) * dx - p[2]
                fw = wt * w_val * inv_dx2_3
                cw0 += fw * ox; cw1 += fw * oy; cw2 += fw * oz

    new_vel = wp.vec3(nu, nv, nw)
    vel[pid] = new_vel
    affine_C[pid] = wp.mat33(cu0, cu1, cu2,
                              cv0, cv1, cv2,
                              cw0, cw1, cw2)
    new_pos = p + new_vel * dt
    eps = dx * 1.001
    new_pos[0] = wp.clamp(new_pos[0], eps, dom[0] - eps)
    new_pos[1] = wp.clamp(new_pos[1], eps, dom[1] - eps)
    new_pos[2] = wp.clamp(new_pos[2], eps, dom[2] - eps)
    pos[pid] = new_pos


block("S2.12", "APIC G2P + advect (recovers C, no FLIP/PIC blend)")(k3_g2p_apic_advect)


# =============================================================================
# S2.6.3 — PCG pressure solve (diagonal-preconditioned conjugate gradient)
# =============================================================================
# Standard CG on the Laplacian discretisation that Jacobi solves. Diagonal
# preconditioner = 1/diag(L) where diag(L) is the cell's neighbour count.
# Converges in ~sqrt(N) iterations vs N for Jacobi; typically 15-40 PCG
# iterations match 100+ Jacobi iterations on equivalent grids.

@wp.kernel
def k3_apply_A(  # A·x — same stencil as Jacobi: diag*x - sum_nb (on fluid cells; 0 elsewhere)
    x: wp.array3d(dtype=float),
    y: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
):
    i, j, k = wp.tid()
    nx = x.shape[0]; ny = x.shape[1]; nz = x.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if marker[i, j, k] != 1:
        y[i, j, k] = 0.0
        return
    diag = float(0.0); sum_nb = float(0.0)
    if i > 0:
        m = marker[i - 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i - 1, j, k]
    if i < nx - 1:
        m = marker[i + 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i + 1, j, k]
    if j > 0:
        m = marker[i, j - 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i, j - 1, k]
    if j < ny - 1:
        m = marker[i, j + 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i, j + 1, k]
    if k > 0:
        m = marker[i, j, k - 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i, j, k - 1]
    if k < nz - 1:
        m = marker[i, j, k + 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i, j, k + 1]
    y[i, j, k] = diag * x[i, j, k] - sum_nb


block("S2.6.3", "PCG: apply Laplacian operator A to x")(k3_apply_A)


@wp.kernel
def k3_compute_diag(marker: wp.array3d(dtype=int), diag: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    nx = marker.shape[0]; ny = marker.shape[1]; nz = marker.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if marker[i, j, k] != 1:
        diag[i, j, k] = 1.0
        return
    d = float(0.0)
    if i > 0          and marker[i - 1, j, k] != 2: d += 1.0
    if i < nx - 1     and marker[i + 1, j, k] != 2: d += 1.0
    if j > 0          and marker[i, j - 1, k] != 2: d += 1.0
    if j < ny - 1     and marker[i, j + 1, k] != 2: d += 1.0
    if k > 0          and marker[i, j, k - 1] != 2: d += 1.0
    if k < nz - 1     and marker[i, j, k + 1] != 2: d += 1.0
    diag[i, j, k] = wp.max(d, 1.0)


@wp.kernel
def k3_apply_invM(r: wp.array3d(dtype=float), diag: wp.array3d(dtype=float),
                  z: wp.array3d(dtype=float), marker: wp.array3d(dtype=int)):
    i, j, k = wp.tid()
    if i >= r.shape[0] or j >= r.shape[1] or k >= r.shape[2]:
        return
    if marker[i, j, k] != 1:
        z[i, j, k] = 0.0
    else:
        z[i, j, k] = r[i, j, k] / diag[i, j, k]


@wp.kernel
def k3_axpy(a: float, x: wp.array3d(dtype=float), y: wp.array3d(dtype=float),
            out: wp.array3d(dtype=float), marker: wp.array3d(dtype=int)):
    """out = y + a * x, restricted to fluid cells."""
    i, j, k = wp.tid()
    if i >= x.shape[0] or j >= x.shape[1] or k >= x.shape[2]:
        return
    if marker[i, j, k] != 1:
        out[i, j, k] = 0.0
    else:
        out[i, j, k] = y[i, j, k] + a * x[i, j, k]


@wp.kernel
def k3_dot_fluid(a: wp.array3d(dtype=float), b: wp.array3d(dtype=float),
                 out: wp.array(dtype=float), marker: wp.array3d(dtype=int)):
    i, j, k = wp.tid()
    if i >= a.shape[0] or j >= a.shape[1] or k >= a.shape[2]:
        return
    if marker[i, j, k] == 1:
        wp.atomic_add(out, 0, a[i, j, k] * b[i, j, k])


@wp.kernel
def k3_zero_scalar(s: wp.array(dtype=float)):
    s[0] = 0.0


# v0.6 — GPU-resident scalar ops to keep PCG off the host fast path.

@wp.kernel
def k3_div_scalar(numer: wp.array(dtype=float),
                  denom: wp.array(dtype=float),
                  out: wp.array(dtype=float)):
    """out[0] = numer[0] / denom[0] (with epsilon to avoid div by zero)."""
    d = denom[0]
    if wp.abs(d) < 1.0e-30:
        out[0] = 0.0
    else:
        out[0] = numer[0] / d


@wp.kernel
def k3_copy_scalar(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    dst[0] = src[0]


# v0.6 — D4.7 GPU compaction kernels (stream compaction via prefix scan)

@wp.kernel
def k3_mark_alive_outflows(
    pos: wp.array(dtype=wp.vec3),
    out_lo: wp.array(dtype=wp.vec3),
    out_hi: wp.array(dtype=wp.vec3),
    n_out: int,
    alive: wp.array(dtype=int),
):
    """alive[i] = 0 if particle is inside ANY outflow box, else 1."""
    i = wp.tid()
    p = pos[i]
    a = int(1)
    for j in range(n_out):
        lo = out_lo[j]; hi = out_hi[j]
        if p[0] >= lo[0] and p[0] <= hi[0] and \
           p[1] >= lo[1] and p[1] <= hi[1] and \
           p[2] >= lo[2] and p[2] <= hi[2]:
            a = int(0)
    alive[i] = a


@wp.kernel
def k3_scatter_alive(
    pos_in: wp.array(dtype=wp.vec3),
    vel_in: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=int),
    prefix: wp.array(dtype=int),
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
):
    """If alive[i], copy pos_in[i] / vel_in[i] to pos_out[prefix[i]-1]."""
    i = wp.tid()
    if alive[i] == 1:
        dst = prefix[i] - 1
        pos_out[dst] = pos_in[i]
        vel_out[dst] = vel_in[i]


block("D4.7.GPU", "GPU stream compaction for outflow (mark + scan + scatter)")(k3_mark_alive_outflows)


@wp.kernel
def k3_axpy_devscalar(
    a_dev: wp.array(dtype=float), sign: float,
    x: wp.array3d(dtype=float), y_in: wp.array3d(dtype=float),
    y_out: wp.array3d(dtype=float), marker: wp.array3d(dtype=int),
):
    """y_out = y_in + sign * a_dev[0] * x  (on fluid cells)."""
    i, j, k = wp.tid()
    if i >= x.shape[0] or j >= x.shape[1] or k >= x.shape[2]:
        return
    if marker[i, j, k] != 1:
        y_out[i, j, k] = 0.0
        return
    a = a_dev[0] * sign
    y_out[i, j, k] = y_in[i, j, k] + a * x[i, j, k]


block("S2.6.3", "GPU-resident PCG scalar ops (div/copy/axpy from device array)")(k3_axpy_devscalar)


# =============================================================================
# S2.6.6 — Per-tile variants of the PCG hot kernels (B4.2). One thread per
#          cell of an active 8³ block. Inactive blocks are not iterated.
# =============================================================================

@wp.func
def _tile_to_ijk(tid: int, active_coords: wp.array(dtype=wp.vec3i),
                 block_size: int) -> wp.vec3i:
    """Map a per-tile thread id to a global (i,j,k) cell coord. Returned
    coord may be out-of-grid; the caller bounds-checks against x.shape."""
    cells_per_block = block_size * block_size * block_size
    blk = tid // cells_per_block
    rem = tid - blk * cells_per_block
    di = rem // (block_size * block_size)
    rem2 = rem - di * (block_size * block_size)
    dj = rem2 // block_size
    dk = rem2 - dj * block_size
    c = active_coords[blk]
    return wp.vec3i(c[0] * block_size + di,
                     c[1] * block_size + dj,
                     c[2] * block_size + dk)


@wp.kernel
def k3_apply_A_per_tile(
    x: wp.array3d(dtype=float),
    y: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    active_coords: wp.array(dtype=wp.vec3i),
    block_size: int,
):
    """[BLK S2.6.6] Apply Laplacian A·x on active tiles only."""
    ijk = _tile_to_ijk(wp.tid(), active_coords, block_size)
    i = ijk[0]; j = ijk[1]; k = ijk[2]
    nx = x.shape[0]; ny = x.shape[1]; nz = x.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    if marker[i, j, k] != 1:
        return  # leave y alone; per-iter caller zero-fills before use
    diag = float(0.0); sum_nb = float(0.0)
    if i > 0:
        m = marker[i - 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i - 1, j, k]
    if i < nx - 1:
        m = marker[i + 1, j, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i + 1, j, k]
    if j > 0:
        m = marker[i, j - 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i, j - 1, k]
    if j < ny - 1:
        m = marker[i, j + 1, k]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i, j + 1, k]
    if k > 0:
        m = marker[i, j, k - 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i, j, k - 1]
    if k < nz - 1:
        m = marker[i, j, k + 1]
        if m != 2:
            diag += 1.0
            if m == 1: sum_nb += x[i, j, k + 1]
    y[i, j, k] = diag * x[i, j, k] - sum_nb


block("S2.6.6", "Per-tile PCG: apply A on active blocks")(k3_apply_A_per_tile)


@wp.kernel
def k3_apply_invM_per_tile(
    r: wp.array3d(dtype=float),
    diag: wp.array3d(dtype=float),
    z: wp.array3d(dtype=float),
    marker: wp.array3d(dtype=int),
    active_coords: wp.array(dtype=wp.vec3i),
    block_size: int,
):
    """[BLK S2.6.6] z = M⁻¹ r on active tiles. Non-fluid cells of an active
    tile get zeroed (matches the dense kernel's contract); cells outside
    the active set keep their previous (zero-init) value."""
    ijk = _tile_to_ijk(wp.tid(), active_coords, block_size)
    i = ijk[0]; j = ijk[1]; k = ijk[2]
    if i >= r.shape[0] or j >= r.shape[1] or k >= r.shape[2]:
        return
    if marker[i, j, k] != 1:
        z[i, j, k] = 0.0
    else:
        z[i, j, k] = r[i, j, k] / diag[i, j, k]


block("S2.6.6", "Per-tile PCG: apply M⁻¹ on active blocks")(k3_apply_invM_per_tile)


@wp.kernel
def k3_dot_fluid_per_tile(
    a: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    out: wp.array(dtype=float),
    marker: wp.array3d(dtype=int),
    active_coords: wp.array(dtype=wp.vec3i),
    block_size: int,
):
    """[BLK S2.6.6] Inner product over fluid cells, per-tile dispatched.
    Caller zeroes `out[0]` before launch."""
    ijk = _tile_to_ijk(wp.tid(), active_coords, block_size)
    i = ijk[0]; j = ijk[1]; k = ijk[2]
    if i >= a.shape[0] or j >= a.shape[1] or k >= a.shape[2]:
        return
    if marker[i, j, k] == 1:
        wp.atomic_add(out, 0, a[i, j, k] * b[i, j, k])


block("S2.6.6", "Per-tile PCG: fluid-cell dot product on active blocks")(k3_dot_fluid_per_tile)


@wp.kernel
def k3_axpy_devscalar_per_tile(
    a_dev: wp.array(dtype=float), sign: float,
    x: wp.array3d(dtype=float), y_in: wp.array3d(dtype=float),
    y_out: wp.array3d(dtype=float), marker: wp.array3d(dtype=int),
    active_coords: wp.array(dtype=wp.vec3i),
    block_size: int,
):
    """[BLK S2.6.6] y_out = y_in + sign · a_dev[0] · x on active tiles."""
    ijk = _tile_to_ijk(wp.tid(), active_coords, block_size)
    i = ijk[0]; j = ijk[1]; k = ijk[2]
    if i >= x.shape[0] or j >= x.shape[1] or k >= x.shape[2]:
        return
    if marker[i, j, k] != 1:
        y_out[i, j, k] = 0.0
        return
    a = a_dev[0] * sign
    y_out[i, j, k] = y_in[i, j, k] + a * x[i, j, k]


block("S2.6.6", "Per-tile PCG: AXPY with device-side scalar on active blocks")(k3_axpy_devscalar_per_tile)


# =============================================================================
# S2.10 — CFL helper: compute v_max on host (small reduction)
# =============================================================================

# [BLK S2.10]
def cfl_substep_count(vel_np: np.ndarray, dx: float, target_dt: float,
                      cfl: float = 0.5, max_substeps: int = 32) -> int:
    """Return how many substeps of ≤ CFL·dx/|v_max| fit inside ``target_dt``.

    Pure-host helper (small reduction). Called once per frame.
    """
    if vel_np.size == 0:
        return 1
    vmax = float(np.linalg.norm(vel_np, axis=1).max())
    if vmax < 1e-6:
        return 1
    sub_dt = cfl * dx / vmax
    n = max(1, int(np.ceil(target_dt / sub_dt)))
    return min(n, max_substeps)


# -----------------------------------------------------------------------------
# S2.10.GPU — GPU |v|_max reduction (atomic_max on |v|² then sqrt on host).
# Avoids the per-frame vel.numpy() roundtrip that dominated step_cfl at scale.
# -----------------------------------------------------------------------------

@wp.kernel
def k3_vmax_sq_reduce(
    vel: wp.array(dtype=wp.vec3),
    out_max_sq: wp.array(dtype=float),
):
    p = wp.tid()
    v = vel[p]
    s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    wp.atomic_max(out_max_sq, 0, s)


block("S2.10.GPU", "GPU reduction: atomic_max over |v|² (CFL helper)")(k3_vmax_sq_reduce)


def cfl_substep_count_gpu(vel_wp: "wp.array", dx: float, target_dt: float,
                           cfl: float = 0.5, max_substeps: int = 32,
                           scratch: Optional["wp.array"] = None) -> int:
    """[BLK S2.10.GPU] Device-resident variant of cfl_substep_count.

    Computes |v|_max on the GPU via atomic-max reduction into a single-element
    array, then transfers only that one float to host. Eliminates the
    O(n_particles) `vel.numpy()` roundtrip of the host path.

    ``scratch`` (optional): a 1-element float wp.array re-used across calls.
    """
    n = int(vel_wp.shape[0])
    if n == 0:
        return 1
    dev = vel_wp.device
    if scratch is None:
        scratch = wp.zeros(1, dtype=float, device=dev)
    else:
        scratch.zero_()
    wp.launch(k3_vmax_sq_reduce, dim=n, inputs=[vel_wp, scratch], device=dev)
    vmax = float(np.sqrt(scratch.numpy()[0]))
    if vmax < 1e-6:
        return 1
    sub_dt = cfl * dx / vmax
    n_sub = max(1, int(np.ceil(target_dt / sub_dt)))
    return min(n_sub, max_substeps)


# [BLK S2.14.5]
def csf_max_stable_dt(rho: float, dx: float, surface_tension: float,
                      safety: float = 0.9) -> float:
    """Capillary-wave CFL bound for the explicit Brackbill-Kothe CSF.

    Derivation (Brackbill, Kothe & Zemach 1992 §IV.C; Bridson §8.5): the
    fastest capillary wave on the discrete grid has period
        T = 2π · √(ρ·dx³ / σ)
    Explicit time integration of σ·κ·∇χ̃ is stable only when dt ≤ T/(2π),
    giving

        dt_max = safety · √(ρ·dx³ / (2π · σ))

    Above this bound, surface impulses don't decay → parasitic currents
    grow, the blob drifts, eventually smears against walls. Below it,
    CSF is bounded and the projection step keeps things div-free.

    Returns +inf when σ ≤ 0 (no constraint).
    """
    if surface_tension <= 0.0:
        return float("inf")
    import math
    return float(safety) * math.sqrt(
        max(rho, 1e-12) * (dx ** 3) / (2.0 * math.pi * surface_tension)
    )


block("S2.14.5", "Host helper: capillary-wave dt_max for explicit CSF")(csf_max_stable_dt)


# =============================================================================
# F3.2 — Solver class
# =============================================================================

class FlipSolver3D:
    """[BLK F3.2] 3D FLIP/PIC solver. Holds all state on GPU.

    Parameters
    ----------
    nx, ny, nz : grid resolution in cells
    dx : cell size (world units)
    gravity : signed acceleration on y, m/s²
    flip_blend : 0..1, 1 = pure FLIP (noisy/energetic), 0 = pure PIC (smooth/diffusive)
    rho : fluid density (only matters as a scale)
    """

    def __init__(self, nx: int = 48, ny: int = 48, nz: int = 48,
                 dx: Optional[float] = None,
                 gravity: float = -9.81, flip_blend: float = 0.95,
                 rho: float = 1.0, viscosity: float = 0.0,
                 viscosity_iters: int = 12,
                 surface_tension: float = 0.0,
                 csf_smoothing_passes: int = 2,
                 transfer_mode: str = "flip",
                 device: Optional[str] = None,
                 enable_timing: bool = False,
                 enable_cuda_graphs: bool = False):
        from ..primitives.profiling import StepProfiler
        self._prof = StepProfiler(enabled=enable_timing)
        # B5.2 — CUDA-graph cache. Lazily captures step() for the current
        # topology + launch args, replays until the topology changes.
        self._enable_cuda_graphs = bool(enable_cuda_graphs)
        self._cuda_graph = None
        self._cuda_graph_key = None
        self._cuda_graph_hits = 0
        self._cuda_graph_misses = 0
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx if dx is not None else 1.0 / max(nx, ny, nz)
        self.dom = wp.vec3(nx * self.dx, ny * self.dx, nz * self.dx)
        self.gravity = gravity
        self.flip_blend = flip_blend
        self.rho = rho
        self.viscosity = viscosity
        self.viscosity_iters = viscosity_iters
        self.surface_tension = surface_tension
        self.csf_smoothing_passes = max(1, int(csf_smoothing_passes))
        self.transfer_mode = transfer_mode   # "flip" | "pic" | "apic"
        self.device = device or default_device()

        self.u  = zeros((nx + 1, ny, nz), dev=self.device)
        self.v  = zeros((nx, ny + 1, nz), dev=self.device)
        self.w  = zeros((nx, ny, nz + 1), dev=self.device)
        # moving-obstacle target velocities (set per frame for animated obstacles)
        self.solid_u = zeros((nx + 1, ny, nz), dev=self.device)
        self.solid_v = zeros((nx, ny + 1, nz), dev=self.device)
        self.solid_w = zeros((nx, ny, nz + 1), dev=self.device)
        # scratch for marking which cells a moving obstacle occupies this frame
        self.anim_mask = zeros_int((nx, ny, nz), dev=self.device)
        self.uw = zeros((nx + 1, ny, nz), dev=self.device)
        self.vw = zeros((nx, ny + 1, nz), dev=self.device)
        self.ww = zeros((nx, ny, nz + 1), dev=self.device)
        self.us = zeros((nx + 1, ny, nz), dev=self.device)
        self.vs = zeros((nx, ny + 1, nz), dev=self.device)
        self.ws = zeros((nx, ny, nz + 1), dev=self.device)
        self.p     = zeros((nx, ny, nz), dev=self.device)
        self.p_tmp = zeros((nx, ny, nz), dev=self.device)
        self.div   = zeros((nx, ny, nz), dev=self.device)
        self.marker = zeros_int((nx, ny, nz), dev=self.device)
        # S2.16 active-block bitmask (8³ blocks). Lazy: allocated on first
        # pressure_block_sparse=True call.
        self._block_active = None
        # S2.14 CSF scratch (allocated lazily on first surface_tension>0 step)
        self._csf_chi = None     # smoothed indicator
        self._csf_tmp = None     # ping-pong for blur passes
        self._csf_nx = None      # unit normal x
        self._csf_ny = None      # unit normal y
        self._csf_nz = None      # unit normal z
        self._csf_kappa = None   # curvature
        # S2.15 per-particle color (vec3 RGB in [0,1]). None until first colored seed.
        self.attr_color = None
        self._cgrid_r = None
        self._cgrid_g = None
        self._cgrid_b = None
        self._cgrid_w = None
        # S2.18/B11 per-particle scalar attribute (temperature). None until set.
        # Reuses the same P2G/normalize/G2P pattern as colour, single channel.
        # Future B11 extensions can stamp `age`, `density`, etc. by adding more
        # parallel attribute slots; the kernels themselves stay generic-shaped.
        self.attr_temperature = None
        self._sgrid_t = None
        self._sgrid_tw = None

        # [BLK D4.1] solid wall shell — one-cell solid border
        m = np.zeros((nx, ny, nz), dtype=np.int32)
        m[0, :, :] = 2; m[-1, :, :] = 2
        m[:, 0, :] = 2; m[:, -1, :] = 2
        m[:, :, 0] = 2; m[:, :, -1] = 2
        wp.copy(self.marker, wp.array(m, dtype=int, device=self.device))
        self._marker_host = m
        self._wall_marker = m.copy()   # walls only; reused for anim obstacle rebuild

        self.pos = None
        self.vel = None
        self.affine_C = None     # APIC: per-particle 3x3 affine matrix
        self.n_particles = 0

        # v0.5 — runtime regions and animated obstacles
        self._static_obstacle_sdfs: list = []   # SDFs applied once at init
        self._anim_specs: list = []              # animated obstacle descriptors
        self.inflows: list = []
        self.outflows: list = []
        self._rng = np.random.default_rng(0)

    # ---------------------------------------------------------- D4.4 obstacles
    @block("D4.4", "Inject SDF obstacle into marker grid")
    def add_solid_from_sdf(self, sdf, padding: float = 0.0):
        m = self._marker_host
        m[sdf <= padding] = 2
        wp.copy(self.marker, wp.array(m, dtype=int, device=self.device))
        self._static_obstacle_sdfs.append(np.asarray(sdf, dtype=np.float32))

    def cell_centers_np(self):
        from ..domain.sdf import cell_centers
        return cell_centers(self.nx, self.ny, self.nz, self.dx)

    # -------- D4.6 animated obstacle registration -------------------------
    @block("D4.6", "Register an analytic obstacle that moves per frame")
    def add_animated_obstacle(self, kind: str, base_center, motion, **params):
        """kind: 'sphere'|'box'|'cylinder_y'; params: radius / half_size / half_height."""
        self._anim_specs.append({"kind": kind, "base_center": tuple(base_center),
                                 "motion": motion, "params": dict(params)})

    def _obstacle_velocity_at(self, spec, frame_idx: int):
        """World-space velocity (m/s) of an animated obstacle this frame.
        Linear motion → spec velocity. Keyframe motion → finite difference."""
        m = spec.get("motion")
        if m is None:
            return (0.0, 0.0, 0.0)
        # LinearMotion
        if hasattr(m, "velocity") and m.kind == "linear":
            return tuple(float(c) for c in m.velocity)
        # KeyframeMotion → centered difference of centre across one frame
        if m.kind == "keyframes":
            c1 = evaluate_center(spec["base_center"], m, frame_idx + 1)
            c0 = evaluate_center(spec["base_center"], m, frame_idx - 1)
            return tuple(float(c) for c in (c1 - c0) * (m.fps if hasattr(m, "fps") else 24) / 2.0)
        return (0.0, 0.0, 0.0)

    def _build_anim_sdf_per_obstacle(self, frame_idx: int):
        """Return (list_of_(sdf, velocity)_for_analytic_obstacles, mesh_specs)."""
        if not self._anim_specs:
            return [], []
        grid = self.cell_centers_np()
        analytic = []
        mesh_specs = []
        for spec in self._anim_specs:
            c = evaluate_center(spec["base_center"], spec["motion"], frame_idx)
            kind = spec["kind"]; p = spec["params"]
            vel = self._obstacle_velocity_at(spec, frame_idx)
            if kind == "sphere":
                analytic.append((sdf_sphere(grid, c, p["radius"]), vel))
            elif kind == "box":
                analytic.append((sdf_box(grid, c, p["half_size"]), vel))
            elif kind == "cylinder_y":
                analytic.append((sdf_cylinder_y(grid, c, p["radius"], p["half_height"]), vel))
            elif kind == "mesh":
                mesh_specs.append((spec, c, vel))
            else:
                raise ValueError(f"unsupported animated obstacle kind: {kind!r}")
        return analytic, mesh_specs

    def _stamp_solid_vel_from_sdf(self, sdf, velocity):
        """For cells with sdf<=0, write velocity into surrounding solid faces."""
        mask_host = (sdf <= 0.0).astype(np.int32)
        if not mask_host.any():
            return
        wp.copy(self.anim_mask, wp.array(mask_host, dtype=int, device=self.device))
        wp.launch(k3_write_solid_face_vel, dim=mask_host.shape,
                  inputs=[self.anim_mask,
                          float(velocity[0]), float(velocity[1]), float(velocity[2]),
                          self.solid_u, self.solid_v, self.solid_w],
                  device=self.device)

    def _build_anim_sdf(self, frame_idx: int):
        """Legacy helper — returns (analytic_sdf, mesh_specs) without velocities.
        Kept for backward-compat; prepare_frame now uses _build_anim_sdf_per_obstacle."""
        if not self._anim_specs:
            return None, []
        grid = self.cell_centers_np()
        parts = []
        mesh_specs_active = []
        for spec in self._anim_specs:
            c = evaluate_center(spec["base_center"], spec["motion"], frame_idx)
            k = spec["kind"]; p = spec["params"]
            if k == "sphere":
                parts.append(sdf_sphere(grid, c, p["radius"]))
            elif k == "box":
                parts.append(sdf_box(grid, c, p["half_size"]))
            elif k == "cylinder_y":
                parts.append(sdf_cylinder_y(grid, c, p["radius"], p["half_height"]))
            elif k == "mesh":
                mesh_specs_active.append((spec, c))
            else:
                raise ValueError(f"unsupported animated obstacle kind: {k!r}")
        sdf_combined = None
        if parts:
            sdf_combined = sdf_union(*parts) if len(parts) > 1 else parts[0]
        return sdf_combined, mesh_specs_active

    def _apply_anim_mesh_obstacles_after_upload(self, mesh_specs, grid, frame_idx):
        """Mark cells inside each animated mesh obstacle as solid + write the
        obstacle's velocity into the surrounding solid faces (moving-boundary BC)."""
        import trimesh
        from ..domain.mesh_sdf_gpu import mark_solid_from_mesh_gpu
        marker_before = self.marker.numpy()
        for entry in mesh_specs:
            spec, c, vel = entry  # may be 3-tuple from the new path
            p = spec["params"]
            mesh = trimesh.load(str(p["mesh_path"]), force="mesh", process=False)
            mesh.apply_scale(float(p.get("scale", 1.0)))
            base = np.asarray(spec["base_center"], dtype=np.float32)
            delta = np.asarray(c, dtype=np.float32) - base
            t_total = np.asarray(p.get("translate", (0, 0, 0)), dtype=np.float32) + delta
            mesh.apply_translation(t_total)
            tris = np.asarray(mesh.triangles, dtype=np.float32)
            if len(tris) == 0:
                continue
            # Mark on GPU.
            mark_solid_from_mesh_gpu(self.marker, grid, tris, device=self.device)
            # Build the cells-introduced-by-this-mesh mask (diff: was-not-solid → now-solid).
            marker_after = self.marker.numpy()
            this_mask = ((marker_after == 2) & (marker_before != 2)).astype(np.int32)
            marker_before = marker_after
            if not this_mask.any():
                continue
            # Write obstacle velocity into faces around these cells.
            wp.copy(self.anim_mask, wp.array(this_mask, dtype=int, device=self.device))
            wp.launch(k3_write_solid_face_vel, dim=this_mask.shape,
                      inputs=[self.anim_mask,
                              float(vel[0]), float(vel[1]), float(vel[2]),
                              self.solid_u, self.solid_v, self.solid_w],
                      device=self.device)
        self._marker_host = self.marker.numpy()

    # -------- D4.7 inflow / outflow registration --------------------------
    @block("D4.7", "Register an inflow region (continuous particle emission)")
    def add_inflow(self, region):
        self.inflows.append(region)

    @block("D4.7", "Register an outflow region (particle removal)")
    def add_outflow(self, region):
        self.outflows.append(region)

    # -------- D4.7.GPU stream-compaction outflow path ---------------------
    @block("D4.7.GPU", "GPU outflow compaction (mark + scan + scatter)")
    def _apply_outflows_gpu(self, frame_idx: int) -> int:
        """Compact ``self.pos/vel`` in place by removing particles inside any
        active outflow box. Returns the new particle count."""
        import warp.utils as wputils
        if self.pos is None or self.n_particles == 0:
            return 0
        active = [o for o in self.outflows
                  if o.frame_start <= frame_idx <= o.frame_end]
        if not active:
            return self.n_particles
        n = self.n_particles
        dev = self.device
        lo_np = np.array([o.lo for o in active], dtype=np.float32)
        hi_np = np.array([o.hi for o in active], dtype=np.float32)
        lo_wp = wp.array(lo_np, dtype=wp.vec3, device=dev)
        hi_wp = wp.array(hi_np, dtype=wp.vec3, device=dev)

        # lazy buffers
        cap = max(n, getattr(self, "_compact_cap", 0))
        if not hasattr(self, "_alive") or self._alive.shape[0] < cap:
            self._alive = wp.zeros(cap, dtype=int, device=dev)
            self._prefix = wp.zeros(cap, dtype=int, device=dev)
            self._pos_alt = wp.zeros(cap, dtype=wp.vec3, device=dev)
            self._vel_alt = wp.zeros(cap, dtype=wp.vec3, device=dev)
            self._compact_cap = cap

        wp.launch(k3_mark_alive_outflows, dim=n,
                  inputs=[self.pos, lo_wp, hi_wp, len(active), self._alive],
                  device=dev)
        # inclusive prefix-sum
        wputils.array_scan(self._alive[:n], self._prefix[:n], inclusive=True)
        # last prefix gives new count
        new_n = int(self._prefix[n - 1:n].numpy()[0])
        if new_n == n:
            return n
        wp.launch(k3_scatter_alive, dim=n,
                  inputs=[self.pos, self.vel, self._alive, self._prefix,
                          self._pos_alt, self._vel_alt],
                  device=dev)
        # swap arrays (slice to the compacted size)
        self.pos = self._pos_alt[:new_n]
        self.vel = self._vel_alt[:new_n]
        # APIC C buffer becomes stale — drop it; will be rebuilt fresh in step()
        self.affine_C = None
        self.n_particles = new_n
        # rotate scratch buffers (so next frame's compaction writes to a fresh slab)
        self._pos_alt, self.pos_keep_ref = wp.zeros(self._compact_cap, dtype=wp.vec3, device=dev), self._pos_alt
        self._vel_alt = wp.zeros(self._compact_cap, dtype=wp.vec3, device=dev)
        return new_n

    # ------------------------------------------------------- D4.5.2 mesh seeder
    @block("D4.5.2", "Mesh volumetric seeder (fill arbitrary triangle mesh with particles)")
    def seed_mesh(self, mesh_path, ppc: int = 8, scale: float = 1.0,
                  translate=(0.0, 0.0, 0.0), rotate_deg=None, color=None):
        """Fill the interior of a closed triangle mesh with particles.

        Uses the same GPU ray-cast (D4.3.GPU) used for animated mesh
        obstacles to determine which cells are inside the mesh, then emits
        a jittered cube of particles per inside cell.
        """
        import trimesh
        from ..domain.mesh_sdf_gpu import mark_solid_from_mesh_gpu
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        if rotate_deg is not None:
            from trimesh.transformations import euler_matrix
            rx, ry, rz = [float(a) * np.pi / 180.0 for a in rotate_deg]
            mesh.apply_transform(euler_matrix(rx, ry, rz, "sxyz"))
        if float(scale) != 1.0:
            mesh.apply_scale(float(scale))
        if any(float(t) != 0.0 for t in translate):
            mesh.apply_translation(np.asarray(translate, dtype=np.float64))
        tris = np.asarray(mesh.triangles, dtype=np.float32)

        # build a scratch marker (only the mesh test marks cells -> walls excluded)
        nx, ny, nz, dx = self.nx, self.ny, self.nz, self.dx
        scratch_host = np.zeros((nx, ny, nz), dtype=np.int32)
        scratch_wp = wp.array(scratch_host, dtype=int, device=self.device)
        grid = self.cell_centers_np()
        mark_solid_from_mesh_gpu(scratch_wp, grid, tris, device=self.device)
        inside_mask = (scratch_wp.numpy() == 2)

        per_axis = max(1, int(round(ppc ** (1.0 / 3.0))))
        idx = np.argwhere(inside_mask)   # (M, 3)
        if len(idx) == 0:
            return
        rng = np.random.default_rng(0)
        positions = []
        for (i, j, k) in idx:
            for sx in range(per_axis):
                for sy in range(per_axis):
                    for sz in range(per_axis):
                        px = (i + (sx + 0.5 + 0.3 * (rng.random() - 0.5)) / per_axis) * dx
                        py = (j + (sy + 0.5 + 0.3 * (rng.random() - 0.5)) / per_axis) * dx
                        pz = (k + (sz + 0.5 + 0.3 * (rng.random() - 0.5)) / per_axis) * dx
                        positions.append([px, py, pz])
        positions = np.array(positions, dtype=np.float32)
        velocities = np.zeros_like(positions)
        # append to any existing seeded particles
        if self.pos is None or self.n_particles == 0:
            self.pos = wp.array(positions, dtype=wp.vec3, device=self.device)
            self.vel = wp.array(velocities, dtype=wp.vec3, device=self.device)
        else:
            cur_pos = self.pos.numpy(); cur_vel = self.vel.numpy()
            cur_pos = np.concatenate([cur_pos, positions], axis=0)
            cur_vel = np.concatenate([cur_vel, velocities], axis=0)
            self.pos = wp.array(cur_pos, dtype=wp.vec3, device=self.device)
            self.vel = wp.array(cur_vel, dtype=wp.vec3, device=self.device)
        self.affine_C = None
        prev_n = self.n_particles
        self.n_particles = len(self.pos.numpy())
        # S2.15 color attribute maintenance (mirrors seed_box semantics)
        if color is not None:
            new_col = np.tile(np.asarray(color, dtype=np.float32),
                              (self.n_particles - prev_n, 1))
            if self.attr_color is not None and prev_n > 0:
                prev_col = self.attr_color.numpy()
                col = np.concatenate([prev_col, new_col], axis=0)
            else:
                col = new_col
            self.attr_color = wp.array(col, dtype=wp.vec3, device=self.device)
        elif self.attr_color is not None:
            pad = np.ones((self.n_particles - prev_n, 3), dtype=np.float32)
            prev_col = self.attr_color.numpy() if prev_n > 0 else np.zeros((0, 3), dtype=np.float32)
            self.attr_color = wp.array(np.concatenate([prev_col, pad], axis=0),
                                       dtype=wp.vec3, device=self.device)

    # ------------------------------------------------------- D4.5.1 seeders
    @block("D4.5.1", "Box seeder (uniform jittered, ppc particles per cell)")
    def seed_box(self, lo, hi, ppc: int = 8, color=None, temperature=None):
        """[BLK D4.5.1] Seed a uniform grid of particles inside [lo, hi].

        color : optional (R,G,B) tuple in [0,1]. If provided, sets per-particle
        attr_color. If existing particles already have color, this batch is
        appended with the given color; previously-seeded particles keep theirs.
        Mixing colored and uncolored seeds in one solver is not supported."""
        i0 = int(lo[0] / self.dx); i1 = int(hi[0] / self.dx)
        j0 = int(lo[1] / self.dx); j1 = int(hi[1] / self.dx)
        k0 = int(lo[2] / self.dx); k1 = int(hi[2] / self.dx)
        per_axis = int(round(ppc ** (1.0 / 3.0)))
        rng = np.random.default_rng(0)
        positions = []
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    for sx in range(per_axis):
                        for sy in range(per_axis):
                            for sz in range(per_axis):
                                px = (i + (sx + 0.5 + 0.3 * (rng.random() - 0.5)) / per_axis) * self.dx
                                py = (j + (sy + 0.5 + 0.3 * (rng.random() - 0.5)) / per_axis) * self.dx
                                pz = (k + (sz + 0.5 + 0.3 * (rng.random() - 0.5)) / per_axis) * self.dx
                                positions.append([px, py, pz])
        positions = np.array(positions, dtype=np.float32)
        velocities = np.zeros_like(positions)
        # Append-or-replace semantics. Concatenate with existing particles
        # whenever either side carries a per-particle attribute we'd lose
        # by replacing — that covers multi-source scenes for both colour
        # and the B11 scalar attribute (temperature). Otherwise we keep
        # the v0.6 "second seed_box replaces" behaviour.
        prev_pos = self.pos.numpy() if self.pos is not None else np.zeros((0, 3), dtype=np.float32)
        prev_vel = self.vel.numpy() if self.vel is not None else np.zeros((0, 3), dtype=np.float32)
        have_attrs = (color is not None or temperature is not None
                      or self.attr_color is not None
                      or self.attr_temperature is not None)
        if have_attrs and len(prev_pos) > 0:
            positions = np.concatenate([prev_pos, positions], axis=0)
            velocities = np.concatenate([prev_vel, velocities], axis=0)
        self.pos = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.vel = wp.array(velocities, dtype=wp.vec3, device=self.device)
        zeros_C = np.zeros((len(positions), 3, 3), dtype=np.float32)
        self.affine_C = wp.array(zeros_C, dtype=wp.mat33, device=self.device)
        # Per-particle color: maintain a vec3 attribute, appended in lockstep.
        if color is not None:
            new_col = np.tile(np.asarray(color, dtype=np.float32), (len(positions) - len(prev_pos), 1))
            if self.attr_color is not None and len(prev_pos) > 0:
                prev_col = self.attr_color.numpy()
                col = np.concatenate([prev_col, new_col], axis=0)
            else:
                col = new_col
            self.attr_color = wp.array(col, dtype=wp.vec3, device=self.device)
        elif self.attr_color is not None:
            # New uncolored seed in a colored scene: pad with white.
            pad = np.ones((len(positions) - len(prev_pos), 3), dtype=np.float32)
            prev_col = self.attr_color.numpy() if len(prev_pos) > 0 else np.zeros((0, 3), dtype=np.float32)
            self.attr_color = wp.array(np.concatenate([prev_col, pad], axis=0),
                                       dtype=wp.vec3, device=self.device)
        # Per-particle scalar (temperature) — same append-in-lockstep semantics.
        new_n = len(positions) - len(prev_pos)
        if temperature is not None:
            new_t = np.full(new_n, float(temperature), dtype=np.float32)
            if self.attr_temperature is not None and len(prev_pos) > 0:
                prev_t = self.attr_temperature.numpy()
                t = np.concatenate([prev_t, new_t], axis=0)
            else:
                t = new_t
            self.attr_temperature = wp.array(t, dtype=float, device=self.device)
        elif self.attr_temperature is not None:
            # Seeded without temperature into a temperatured scene: pad with NaN-ish
            # sentinel 0.0 (callers can re-stamp afterwards if they need a default).
            pad = np.zeros(new_n, dtype=np.float32)
            prev_t = self.attr_temperature.numpy() if len(prev_pos) > 0 else np.zeros(0, dtype=np.float32)
            self.attr_temperature = wp.array(np.concatenate([prev_t, pad], axis=0),
                                              dtype=float, device=self.device)
        self.n_particles = len(positions)

    # ---------- internal: PCG pressure solve (S2.6.3) -----------------------
    # GPU-resident variant: alpha, beta, rz_old, rz_new, pAp, r_norm² all live
    # on the device. Host reads happen ONLY for the convergence check, and
    # only every `check_every` iterations (default 10). This collapses ~3
    # CUDA syncs per iter (the v0.4 implementation) down to ~0.1.
    def _pressure_pcg(self, max_iter: int, tol: float = 1e-4,
                      check_every: int = 10) -> int:
        nx, ny, nz, dev = self.nx, self.ny, self.nz, self.device
        if not hasattr(self, "_pcg_r"):
            self._pcg_r = zeros((nx, ny, nz), dev=dev)
            self._pcg_z = zeros((nx, ny, nz), dev=dev)
            self._pcg_p = zeros((nx, ny, nz), dev=dev)
            self._pcg_Ap = zeros((nx, ny, nz), dev=dev)
            self._pcg_diag = zeros((nx, ny, nz), dev=dev)
            self._pcg_alpha = wp.zeros(1, dtype=float, device=dev)
            self._pcg_beta = wp.zeros(1, dtype=float, device=dev)
            self._pcg_rz_old = wp.zeros(1, dtype=float, device=dev)
            self._pcg_rz_new = wp.zeros(1, dtype=float, device=dev)
            self._pcg_pAp = wp.zeros(1, dtype=float, device=dev)
            self._pcg_r_norm2 = wp.zeros(1, dtype=float, device=dev)
            self._pcg_r0_norm2 = wp.zeros(1, dtype=float, device=dev)

        wp.launch(k3_compute_diag, dim=(nx, ny, nz),
                  inputs=[self.marker, self._pcg_diag], device=dev)
        # x = 0, r = b
        self.p.zero_()
        wp.copy(self._pcg_r, self.div)
        wp.launch(k3_apply_invM, dim=(nx, ny, nz),
                  inputs=[self._pcg_r, self._pcg_diag, self._pcg_z, self.marker], device=dev)
        wp.copy(self._pcg_p, self._pcg_z)

        def dev_dot(a, b, out):
            wp.launch(k3_zero_scalar, dim=1, inputs=[out], device=dev)
            wp.launch(k3_dot_fluid, dim=(nx, ny, nz),
                      inputs=[a, b, out, self.marker], device=dev)

        # rz_old = r·z, r0_norm² = r·r  — both on device
        dev_dot(self._pcg_r, self._pcg_z, self._pcg_rz_old)
        dev_dot(self._pcg_r, self._pcg_r, self._pcg_r0_norm2)

        # one upfront host read so we can early-exit on already-zero residual
        r0_norm2_host = float(self._pcg_r0_norm2.numpy()[0])
        if r0_norm2_host < 1e-30:
            return 0
        tol_sq = (tol ** 2) * r0_norm2_host

        for it in range(max_iter):
            wp.launch(k3_apply_A, dim=(nx, ny, nz),
                      inputs=[self._pcg_p, self._pcg_Ap, self.marker], device=dev)
            dev_dot(self._pcg_p, self._pcg_Ap, self._pcg_pAp)
            wp.launch(k3_div_scalar, dim=1,
                      inputs=[self._pcg_rz_old, self._pcg_pAp, self._pcg_alpha], device=dev)
            # x += α p
            wp.launch(k3_axpy_devscalar, dim=(nx, ny, nz),
                      inputs=[self._pcg_alpha, 1.0, self._pcg_p, self.p, self.p, self.marker],
                      device=dev)
            # r -= α Ap
            wp.launch(k3_axpy_devscalar, dim=(nx, ny, nz),
                      inputs=[self._pcg_alpha, -1.0, self._pcg_Ap, self._pcg_r, self._pcg_r, self.marker],
                      device=dev)
            # convergence check: only every `check_every` iters
            if (it + 1) % check_every == 0 or it == max_iter - 1:
                dev_dot(self._pcg_r, self._pcg_r, self._pcg_r_norm2)
                if float(self._pcg_r_norm2.numpy()[0]) < tol_sq:
                    return it + 1
            # z = M⁻¹ r
            wp.launch(k3_apply_invM, dim=(nx, ny, nz),
                      inputs=[self._pcg_r, self._pcg_diag, self._pcg_z, self.marker], device=dev)
            # rz_new = r·z
            dev_dot(self._pcg_r, self._pcg_z, self._pcg_rz_new)
            # β = rz_new / rz_old
            wp.launch(k3_div_scalar, dim=1,
                      inputs=[self._pcg_rz_new, self._pcg_rz_old, self._pcg_beta], device=dev)
            # p = z + β p
            wp.launch(k3_axpy_devscalar, dim=(nx, ny, nz),
                      inputs=[self._pcg_beta, 1.0, self._pcg_p, self._pcg_z, self._pcg_p, self.marker],
                      device=dev)
            # rz_old ← rz_new
            wp.launch(k3_copy_scalar, dim=1,
                      inputs=[self._pcg_rz_new, self._pcg_rz_old], device=dev)
        return max_iter

    # ---------- internal: per-tile (block-sparse) PCG (S2.6.6 / B4.2) -------
    def _build_active_blocks(self):
        """Compact the active-block list into (n_active, coords). Shared by
        sparse Jacobi / GS-RB / PCG. Returns the host-int count + the
        device-side coord array."""
        import warp.utils as wputils
        nx, ny, nz, dev = self.nx, self.ny, self.nz, self.device
        nbx = (nx + BLOCK_SIZE - 1) // BLOCK_SIZE
        nby = (ny + BLOCK_SIZE - 1) // BLOCK_SIZE
        nbz = (nz + BLOCK_SIZE - 1) // BLOCK_SIZE
        n_blocks = nbx * nby * nbz
        if (self._block_active is None
                or self._block_active.shape != (nbx, nby, nbz)):
            self._block_active = zeros_int((nbx, nby, nbz), dev=dev)
            self._block_active_flat = wp.zeros(n_blocks, dtype=int, device=dev)
            self._block_prefix = wp.zeros(n_blocks, dtype=int, device=dev)
            self._block_coords = wp.zeros(n_blocks, dtype=wp.vec3i, device=dev)
        self._block_active.zero_()
        wp.launch(k_mark_active_blocks, dim=(nx, ny, nz),
                  inputs=[self.marker, self._block_active, BLOCK_SIZE], device=dev)
        self._block_active_flat = self._block_active.flatten()
        wputils.array_scan(self._block_active_flat, self._block_prefix,
                           inclusive=True)
        n_active = int(self._block_prefix[n_blocks - 1: n_blocks].numpy()[0])
        if n_active > 0:
            wp.launch(k_compact_active_blocks, dim=(nbx, nby, nbz),
                      inputs=[self._block_active, self._block_prefix,
                              self._block_coords], device=dev)
        return n_active, self._block_coords

    def _pressure_pcg_sparse(self, max_iter: int, tol: float = 1e-4,
                              check_every: int = 10) -> int:
        """Block-sparse PCG: identical algorithm to `_pressure_pcg`, but each
        per-cell kernel (apply A, axpy, dot, invM) launches only on cells
        of active 8³ blocks. The active list is built ONCE per call, then
        reused across all PCG iterations.
        """
        nx, ny, nz, dev = self.nx, self.ny, self.nz, self.device
        if not hasattr(self, "_pcg_r"):
            self._pcg_r = zeros((nx, ny, nz), dev=dev)
            self._pcg_z = zeros((nx, ny, nz), dev=dev)
            self._pcg_p = zeros((nx, ny, nz), dev=dev)
            self._pcg_Ap = zeros((nx, ny, nz), dev=dev)
            self._pcg_diag = zeros((nx, ny, nz), dev=dev)
            self._pcg_alpha = wp.zeros(1, dtype=float, device=dev)
            self._pcg_beta = wp.zeros(1, dtype=float, device=dev)
            self._pcg_rz_old = wp.zeros(1, dtype=float, device=dev)
            self._pcg_rz_new = wp.zeros(1, dtype=float, device=dev)
            self._pcg_pAp = wp.zeros(1, dtype=float, device=dev)
            self._pcg_r_norm2 = wp.zeros(1, dtype=float, device=dev)
            self._pcg_r0_norm2 = wp.zeros(1, dtype=float, device=dev)

        # Diag: still cheap, launch dense. (One-shot per step.)
        wp.launch(k3_compute_diag, dim=(nx, ny, nz),
                  inputs=[self.marker, self._pcg_diag], device=dev)

        n_active, coords = self._build_active_blocks()
        if n_active == 0:
            self.p.zero_()
            return 0
        bs = BLOCK_SIZE
        tile_dim = n_active * bs * bs * bs

        self.p.zero_()
        wp.copy(self._pcg_r, self.div)
        wp.launch(k3_apply_invM_per_tile, dim=tile_dim,
                  inputs=[self._pcg_r, self._pcg_diag, self._pcg_z, self.marker,
                          coords, bs], device=dev)
        wp.copy(self._pcg_p, self._pcg_z)

        def dev_dot(a, b, out):
            wp.launch(k3_zero_scalar, dim=1, inputs=[out], device=dev)
            wp.launch(k3_dot_fluid_per_tile, dim=tile_dim,
                      inputs=[a, b, out, self.marker, coords, bs], device=dev)

        dev_dot(self._pcg_r, self._pcg_z, self._pcg_rz_old)
        dev_dot(self._pcg_r, self._pcg_r, self._pcg_r0_norm2)
        r0_norm2_host = float(self._pcg_r0_norm2.numpy()[0])
        if r0_norm2_host < 1e-30:
            return 0
        tol_sq = (tol ** 2) * r0_norm2_host

        for it in range(max_iter):
            wp.launch(k3_apply_A_per_tile, dim=tile_dim,
                      inputs=[self._pcg_p, self._pcg_Ap, self.marker,
                              coords, bs], device=dev)
            dev_dot(self._pcg_p, self._pcg_Ap, self._pcg_pAp)
            wp.launch(k3_div_scalar, dim=1,
                      inputs=[self._pcg_rz_old, self._pcg_pAp, self._pcg_alpha],
                      device=dev)
            wp.launch(k3_axpy_devscalar_per_tile, dim=tile_dim,
                      inputs=[self._pcg_alpha, 1.0, self._pcg_p, self.p, self.p,
                              self.marker, coords, bs], device=dev)
            wp.launch(k3_axpy_devscalar_per_tile, dim=tile_dim,
                      inputs=[self._pcg_alpha, -1.0, self._pcg_Ap, self._pcg_r,
                              self._pcg_r, self.marker, coords, bs], device=dev)
            if (it + 1) % check_every == 0 or it == max_iter - 1:
                dev_dot(self._pcg_r, self._pcg_r, self._pcg_r_norm2)
                if float(self._pcg_r_norm2.numpy()[0]) < tol_sq:
                    return it + 1
            wp.launch(k3_apply_invM_per_tile, dim=tile_dim,
                      inputs=[self._pcg_r, self._pcg_diag, self._pcg_z,
                              self.marker, coords, bs], device=dev)
            dev_dot(self._pcg_r, self._pcg_z, self._pcg_rz_new)
            wp.launch(k3_div_scalar, dim=1,
                      inputs=[self._pcg_rz_new, self._pcg_rz_old, self._pcg_beta],
                      device=dev)
            wp.launch(k3_axpy_devscalar_per_tile, dim=tile_dim,
                      inputs=[self._pcg_beta, 1.0, self._pcg_p, self._pcg_z,
                              self._pcg_p, self.marker, coords, bs], device=dev)
            wp.launch(k3_copy_scalar, dim=1,
                      inputs=[self._pcg_rz_new, self._pcg_rz_old], device=dev)
        return max_iter

    # ----------------------------------------------------------- S2.14 CSF
    @block("S2.14", "Surface tension: build χ̃, normal, curvature; apply CSF impulse")
    def _apply_surface_tension(self, dt: float):
        nx, ny, nz, dx = self.nx, self.ny, self.nz, self.dx
        dev = self.device
        if self._csf_chi is None:
            self._csf_chi   = zeros((nx, ny, nz), dev=dev)
            self._csf_tmp   = zeros((nx, ny, nz), dev=dev)
            self._csf_nx    = zeros((nx, ny, nz), dev=dev)
            self._csf_ny    = zeros((nx, ny, nz), dev=dev)
            self._csf_nz    = zeros((nx, ny, nz), dev=dev)
            self._csf_kappa = zeros((nx, ny, nz), dev=dev)
            self._csf_u_pre = zeros((nx + 1, ny, nz), dev=dev)
            self._csf_v_pre = zeros((nx, ny + 1, nz), dev=dev)
            self._csf_w_pre = zeros((nx, ny, nz + 1), dev=dev)
            self._csf_sum   = zeros((1,), dev=dev)
            self._csf_count = zeros((1,), dev=dev)
        # S2.14.1 — raw indicator + smoothing passes (ping-pong)
        wp.launch(k3_indicator_from_marker, dim=(nx, ny, nz),
                  inputs=[self.marker, self._csf_chi], device=dev)
        for _ in range(self.csf_smoothing_passes):
            wp.launch(k3_box_blur_centred, dim=(nx, ny, nz),
                      inputs=[self._csf_chi, self._csf_tmp], device=dev)
            self._csf_chi, self._csf_tmp = self._csf_tmp, self._csf_chi
        # S2.14.2 — unit normal
        wp.launch(k3_csf_normal, dim=(nx, ny, nz),
                  inputs=[self._csf_chi, self._csf_nx, self._csf_ny, self._csf_nz, dx],
                  device=dev)
        # S2.14.3 — curvature
        wp.launch(k3_csf_curvature, dim=(nx, ny, nz),
                  inputs=[self._csf_nx, self._csf_ny, self._csf_nz, self._csf_kappa, dx],
                  device=dev)
        # snapshot pre-CSF face velocities (for S2.14.6 balancing)
        wp.copy(self._csf_u_pre, self.u)
        wp.copy(self._csf_v_pre, self.v)
        wp.copy(self._csf_w_pre, self.w)
        # S2.14.4 — apply σ·κ·∇χ̃·dt/ρ to each MAC component
        coeff = self.surface_tension * dt / max(self.rho, 1.0e-8)
        wp.launch(k3_csf_apply_u, dim=self.u.shape,
                  inputs=[self.u, self._csf_chi, self._csf_kappa, self.marker,
                          coeff, dx, nx, ny, nz], device=dev)
        wp.launch(k3_csf_apply_v, dim=self.v.shape,
                  inputs=[self.v, self._csf_chi, self._csf_kappa, self.marker,
                          coeff, dx, nx, ny, nz], device=dev)
        wp.launch(k3_csf_apply_w, dim=self.w.shape,
                  inputs=[self.w, self._csf_chi, self._csf_kappa, self.marker,
                          coeff, dx, nx, ny, nz], device=dev)
        # S2.14.6 — subtract per-axis mean impulse (kills parasitic drift)
        for sum_kern, sub_kern, vel, vel_pre in [
            (k3_csf_sum_u, k3_csf_subtract_bias_u, self.u, self._csf_u_pre),
            (k3_csf_sum_v, k3_csf_subtract_bias_v, self.v, self._csf_v_pre),
            (k3_csf_sum_w, k3_csf_subtract_bias_w, self.w, self._csf_w_pre),
        ]:
            self._csf_sum.zero_(); self._csf_count.zero_()
            wp.launch(sum_kern, dim=vel.shape,
                      inputs=[vel_pre, vel, self.marker,
                              self._csf_sum, self._csf_count,
                              nx, ny, nz], device=dev)
            s = float(self._csf_sum.numpy()[0])
            c = float(self._csf_count.numpy()[0])
            if c > 0.5:
                bias = s / c
                wp.launch(sub_kern, dim=vel.shape,
                          inputs=[vel, self.marker, bias, nx, ny, nz], device=dev)

    # ----------------------------------------------------------- S2.18 scalar attrs (B11)
    @block("S2.18", "Per-particle scalar attribute: P2G → normalize → G2P (B11)")
    def _apply_scalar_transfer(self, attr_wp):
        """Run the P2G → normalize → G2P round-trip for a per-particle
        float attribute. Allocates the scratch grids on first call.

        ``attr_wp`` is the `wp.array(dtype=float)` we want to advect.
        Lockstep with the particle array — caller maintains that.
        """
        if attr_wp is None or self.n_particles == 0:
            return
        nx, ny, nz, dx, dev = self.nx, self.ny, self.nz, self.dx, self.device
        if self._sgrid_t is None:
            self._sgrid_t = zeros((nx, ny, nz), dev=dev)
            self._sgrid_tw = zeros((nx, ny, nz), dev=dev)
        self._sgrid_t.zero_(); self._sgrid_tw.zero_()
        wp.launch(k3_p2g_scalar, dim=self.n_particles,
                  inputs=[self.pos, attr_wp, self._sgrid_t, self._sgrid_tw,
                          dx, nx, ny, nz], device=dev)
        wp.launch(k3_normalize_scalar, dim=(nx, ny, nz),
                  inputs=[self._sgrid_t, self._sgrid_tw], device=dev)
        wp.launch(k3_g2p_scalar, dim=self.n_particles,
                  inputs=[self.pos, attr_wp, self._sgrid_t, self._sgrid_tw,
                          dx, nx, ny, nz], device=dev)

    # ----------------------------------------------------------- S2.15 color
    @block("S2.15", "Per-particle color: P2G → normalize → G2P (linear RGB blend)")
    def _apply_color_transfer(self):
        if self.attr_color is None or self.n_particles == 0:
            return
        nx, ny, nz, dx, dev = self.nx, self.ny, self.nz, self.dx, self.device
        if self._cgrid_r is None:
            self._cgrid_r = zeros((nx, ny, nz), dev=dev)
            self._cgrid_g = zeros((nx, ny, nz), dev=dev)
            self._cgrid_b = zeros((nx, ny, nz), dev=dev)
            self._cgrid_w = zeros((nx, ny, nz), dev=dev)
        self._cgrid_r.zero_(); self._cgrid_g.zero_()
        self._cgrid_b.zero_(); self._cgrid_w.zero_()
        wp.launch(k3_p2g_color, dim=self.n_particles,
                  inputs=[self.pos, self.attr_color,
                          self._cgrid_r, self._cgrid_g, self._cgrid_b, self._cgrid_w,
                          dx, nx, ny, nz], device=dev)
        wp.launch(k3_normalize_color, dim=(nx, ny, nz),
                  inputs=[self._cgrid_r, self._cgrid_g, self._cgrid_b, self._cgrid_w],
                  device=dev)
        wp.launch(k3_g2p_color, dim=self.n_particles,
                  inputs=[self.pos, self.attr_color,
                          self._cgrid_r, self._cgrid_g, self._cgrid_b, self._cgrid_w,
                          dx, nx, ny, nz], device=dev)

    # ----------------------------------------------------------- B5.2 graph cache
    def _cuda_graph_eligible(self, pressure_solver: str,
                              pressure_block_sparse: bool) -> bool:
        """Which solver configurations can be captured into a CUDA graph
        today. Anything that reads device→host inside step() is excluded."""
        if pressure_solver not in ("jacobi", "gsrb"):
            return False        # PCG has r_norm² convergence check
        if pressure_block_sparse:
            return False        # n_active.numpy() inside the build
        if self.surface_tension > 0.0:
            return False        # CSF S2.14.6 force-balance reads 3 sums
        return True

    def _cuda_graph_make_key(self, dt: float, pressure_iters: int,
                              pressure_solver: str,
                              pressure_block_sparse: bool):
        """Hashable cache key. Anything that changes the kernel-launch
        sequence or constants must appear here, otherwise replaying the
        cached graph would silently use stale args."""
        return (
            self.transfer_mode,
            pressure_solver,
            bool(pressure_block_sparse),
            self.surface_tension > 0.0,
            self.viscosity > 0.0,
            self.attr_color is not None,
            self.attr_temperature is not None,
            int(self.n_particles),
            float(dt),
            int(pressure_iters),
        )

    def _cuda_graph_invalidate(self):
        """Called whenever topology changes (prepare_frame, set_particles,
        seed_box on an existing solver, etc.)."""
        self._cuda_graph = None
        self._cuda_graph_key = None

    # ----------------------------------------------------------- F3.3 step
    @block("F3.3", "Per-step pipeline: S2.1..S2.9 fixed order")
    def step(self, dt: float, pressure_iters: int = 80,
             pressure_solver: str = "jacobi",
             pressure_block_sparse: bool = False):
        # B5.2 — auto-replay through a captured graph when the config is
        # eligible AND the cached graph's key still matches. First call
        # in a topology pays the capture cost; subsequent calls just
        # replay (~2× faster at 64³, validated by the B5.1 spike).
        if self._enable_cuda_graphs and self._cuda_graph_eligible(
                pressure_solver, pressure_block_sparse):
            key = self._cuda_graph_make_key(
                dt, pressure_iters, pressure_solver, pressure_block_sparse)
            if self._cuda_graph is not None and self._cuda_graph_key == key:
                wp.capture_launch(self._cuda_graph)
                self._cuda_graph_hits += 1
                return
            # Recapture: ask the inner _step_impl to execute, and capture
            # all launches it emits onto the graph.
            wp.capture_begin(device=self.device, force_module_load=False)
            try:
                self._step_impl(dt, pressure_iters, pressure_solver,
                                pressure_block_sparse)
            except Exception:
                try: wp.capture_end(device=self.device)
                except Exception: pass
                self._cuda_graph_invalidate()
                raise
            self._cuda_graph = wp.capture_end(device=self.device)
            self._cuda_graph_key = key
            self._cuda_graph_misses += 1
            return
        # Direct path — either user opted out of graphs or this config has
        # an in-step host sync (PCG / CSF / block-sparse).
        self._step_impl(dt, pressure_iters, pressure_solver,
                        pressure_block_sparse)

    def _step_impl(self, dt: float, pressure_iters: int,
                   pressure_solver: str, pressure_block_sparse: bool):
        """Original per-step pipeline. step() may call this directly or
        wrap it in a CUDA-graph capture. Keep the body sync-free so the
        capture-path stays valid (see B5.1 audit)."""
        nx, ny, nz, dx = self.nx, self.ny, self.nz, self.dx
        full = (max(nx + 1, nx), max(ny + 1, ny), max(nz + 1, nz))
        dev = self.device
        prof = self._prof
        with prof.section("clear"):
            wp.launch(k3_clear_grid, dim=full,
                      inputs=[self.u, self.v, self.w, self.uw, self.vw, self.ww, self.marker], device=dev)
        with prof.section("p2g"):
            if self.transfer_mode == "apic":
                if self.affine_C is None or self.affine_C.shape[0] != self.n_particles:
                    # rebuild after inflow/outflow compaction
                    zeros_C = np.zeros((self.n_particles, 3, 3), dtype=np.float32)
                    self.affine_C = wp.array(zeros_C, dtype=wp.mat33, device=dev)
                wp.launch(k3_p2g_apic, dim=self.n_particles,
                          inputs=[self.pos, self.vel, self.affine_C,
                                  self.u, self.v, self.w, self.uw, self.vw, self.ww,
                                  self.marker, dx, nx, ny, nz], device=dev)
            else:
                wp.launch(k3_p2g, dim=self.n_particles,
                          inputs=[self.pos, self.vel,
                                  self.u, self.v, self.w, self.uw, self.vw, self.ww,
                                  self.marker, dx, nx, ny, nz], device=dev)
        with prof.section("normalize"):
            wp.launch(k3_normalize, dim=full,
                      inputs=[self.u, self.v, self.w, self.uw, self.vw, self.ww,
                              self.us, self.vs, self.ws], device=dev)
        with prof.section("gravity_bc"):
            wp.launch(k3_add_gravity, dim=self.v.shape,
                      inputs=[self.v, self.gravity, dt], device=dev)
            wp.launch(k3_enforce_solid_bc, dim=(nx + 1, ny + 1, nz + 1),
                      inputs=[self.u, self.v, self.w, self.marker,
                              self.solid_u, self.solid_v, self.solid_w,
                              nx, ny, nz], device=dev)
        # S2.13 viscosity (semi-implicit). Solve (I - r·Lap) u_new = u for each
        # MAC component using Jacobi. Re-uses uw/vw/ww as scratch buffers.
        if self.viscosity > 0.0:
          with prof.section("viscosity"):
            r = dt * self.viscosity / (dx * dx)
            for old, scratch in [(self.u, self.uw), (self.v, self.vw), (self.w, self.ww)]:
                wp.copy(scratch, old)
                # `old` plays role of u_old (rhs); we iterate in `scratch` ↔ `old`
                # alternately. Keep u_old in `scratch` and iterate values in `old`.
                # We need a separate temp buffer; reuse `us`/`vs`/`ws` if free.
            # 3 separate iterations to avoid aliasing
            for _ in range(self.viscosity_iters):
                wp.launch(k3_jacobi_visc, dim=self.u.shape,
                          inputs=[self.uw, self.u, self.us, r], device=dev); self.u, self.us = self.us, self.u
                wp.launch(k3_jacobi_visc, dim=self.v.shape,
                          inputs=[self.vw, self.v, self.vs, r], device=dev); self.v, self.vs = self.vs, self.v
                wp.launch(k3_jacobi_visc, dim=self.w.shape,
                          inputs=[self.ww, self.w, self.ws, r], device=dev); self.w, self.ws = self.ws, self.w
            wp.launch(k3_enforce_solid_bc, dim=(nx + 1, ny + 1, nz + 1),
                      inputs=[self.u, self.v, self.w, self.marker,
                              self.solid_u, self.solid_v, self.solid_w,
                              nx, ny, nz], device=dev)
        # S2.14 surface tension (Brackbill-Kothe CSF). Apply *before* divergence
        # so the impulse is projected to a divergence-free field by the pressure
        # solve. Allocations are lazy: only paid when surface_tension>0.
        if self.surface_tension > 0.0:
          with prof.section("surface_tension"):
            self._apply_surface_tension(dt)
            wp.launch(k3_enforce_solid_bc, dim=(nx + 1, ny + 1, nz + 1),
                      inputs=[self.u, self.v, self.w, self.marker,
                              self.solid_u, self.solid_v, self.solid_w,
                              nx, ny, nz], device=dev)
        with prof.section("divergence"):
            wp.launch(k3_compute_divergence, dim=(nx, ny, nz),
                      inputs=[self.u, self.v, self.w, self.div, self.marker, dx, dt, self.rho], device=dev)
        with prof.section("pressure"):
            if pressure_solver == "pcg":
                if pressure_block_sparse:
                    self.last_pressure_iters = self._pressure_pcg_sparse(
                        max_iter=pressure_iters)
                else:
                    self.last_pressure_iters = self._pressure_pcg(
                        max_iter=pressure_iters)
            elif pressure_solver == "jacobi":
                self.p.zero_(); self.p_tmp.zero_()
                if pressure_block_sparse:
                    # S2.16 — rebuild the 8³ active-block bitmask + compact list.
                    import warp.utils as wputils
                    nbx = (nx + BLOCK_SIZE - 1) // BLOCK_SIZE
                    nby = (ny + BLOCK_SIZE - 1) // BLOCK_SIZE
                    nbz = (nz + BLOCK_SIZE - 1) // BLOCK_SIZE
                    n_blocks = nbx * nby * nbz
                    if (self._block_active is None
                            or self._block_active.shape != (nbx, nby, nbz)):
                        self._block_active = zeros_int((nbx, nby, nbz), dev=dev)
                        self._block_active_flat = wp.zeros(n_blocks, dtype=int, device=dev)
                        self._block_prefix = wp.zeros(n_blocks, dtype=int, device=dev)
                        self._block_coords = wp.zeros(n_blocks, dtype=wp.vec3i, device=dev)
                    self._block_active.zero_()
                    wp.launch(k_mark_active_blocks, dim=(nx, ny, nz),
                              inputs=[self.marker, self._block_active, BLOCK_SIZE],
                              device=dev)
                    # Flatten bitmask (alias view) and prefix-sum to compact coords.
                    # array_scan needs a 1D input — alias self._block_active.
                    self._block_active_flat = self._block_active.flatten()
                    wputils.array_scan(self._block_active_flat, self._block_prefix,
                                       inclusive=True)
                    n_active = int(self._block_prefix[n_blocks - 1:n_blocks].numpy()[0])
                    if n_active == 0:
                        # No fluid → pressure stays zero
                        pass
                    else:
                        wp.launch(k_compact_active_blocks, dim=(nbx, nby, nbz),
                                  inputs=[self._block_active, self._block_prefix,
                                          self._block_coords], device=dev)
                        # S2.6.4 per-tile launch (n_active × 512 threads per sweep)
                        cells_per_block = BLOCK_SIZE ** 3
                        for _ in range(pressure_iters):
                            wp.launch(k3_jacobi_pressure_per_tile,
                                      dim=n_active * cells_per_block,
                                      inputs=[self.p, self.p_tmp, self.div, self.marker,
                                              self._block_coords, BLOCK_SIZE],
                                      device=dev)
                            self.p, self.p_tmp = self.p_tmp, self.p
                else:
                    for _ in range(pressure_iters):
                        wp.launch(k3_jacobi_pressure, dim=(nx, ny, nz),
                                  inputs=[self.p, self.p_tmp, self.div, self.marker], device=dev)
                        self.p, self.p_tmp = self.p_tmp, self.p
                self.last_pressure_iters = pressure_iters
            elif pressure_solver == "gsrb":
                self.p.zero_()
                if pressure_block_sparse:
                    # B4.1 — per-tile GS-RB. Build the active-block list the
                    # same way the per-tile Jacobi branch does.
                    import warp.utils as wputils
                    nbx = (nx + BLOCK_SIZE - 1) // BLOCK_SIZE
                    nby = (ny + BLOCK_SIZE - 1) // BLOCK_SIZE
                    nbz = (nz + BLOCK_SIZE - 1) // BLOCK_SIZE
                    n_blocks = nbx * nby * nbz
                    if (self._block_active is None
                            or self._block_active.shape != (nbx, nby, nbz)):
                        self._block_active = zeros_int((nbx, nby, nbz), dev=dev)
                        self._block_active_flat = wp.zeros(n_blocks, dtype=int, device=dev)
                        self._block_prefix = wp.zeros(n_blocks, dtype=int, device=dev)
                        self._block_coords = wp.zeros(n_blocks, dtype=wp.vec3i, device=dev)
                    self._block_active.zero_()
                    wp.launch(k_mark_active_blocks, dim=(nx, ny, nz),
                              inputs=[self.marker, self._block_active, BLOCK_SIZE],
                              device=dev)
                    self._block_active_flat = self._block_active.flatten()
                    wputils.array_scan(self._block_active_flat, self._block_prefix,
                                       inclusive=True)
                    n_active = int(self._block_prefix[n_blocks - 1:n_blocks].numpy()[0])
                    if n_active > 0:
                        wp.launch(k_compact_active_blocks, dim=(nbx, nby, nbz),
                                  inputs=[self._block_active, self._block_prefix,
                                          self._block_coords], device=dev)
                        cells_per_block = BLOCK_SIZE ** 3
                        for _ in range(pressure_iters):
                            wp.launch(k3_gauss_seidel_rb_per_tile,
                                      dim=n_active * cells_per_block,
                                      inputs=[self.p, self.div, self.marker,
                                              self._block_coords, BLOCK_SIZE, 0],
                                      device=dev)
                            wp.launch(k3_gauss_seidel_rb_per_tile,
                                      dim=n_active * cells_per_block,
                                      inputs=[self.p, self.div, self.marker,
                                              self._block_coords, BLOCK_SIZE, 1],
                                      device=dev)
                else:
                    for _ in range(pressure_iters):
                        wp.launch(k3_gauss_seidel_rb, dim=(nx, ny, nz),
                                  inputs=[self.p, self.div, self.marker, 0], device=dev)
                        wp.launch(k3_gauss_seidel_rb, dim=(nx, ny, nz),
                                  inputs=[self.p, self.div, self.marker, 1], device=dev)
                self.last_pressure_iters = pressure_iters
            else:
                raise ValueError(f"unknown pressure_solver: {pressure_solver!r}")
        with prof.section("grad_subtract_bc"):
            wp.launch(k3_subtract_pressure_grad, dim=(nx + 1, ny + 1, nz + 1),
                      inputs=[self.u, self.v, self.w, self.p, self.marker, dx, dt, self.rho], device=dev)
            wp.launch(k3_enforce_solid_bc, dim=(nx + 1, ny + 1, nz + 1),
                      inputs=[self.u, self.v, self.w, self.marker,
                              self.solid_u, self.solid_v, self.solid_w,
                              nx, ny, nz], device=dev)
        with prof.section("g2p_advect"):
            if self.transfer_mode == "apic":
                wp.launch(k3_g2p_apic_advect, dim=self.n_particles,
                          inputs=[self.pos, self.vel, self.affine_C,
                                  self.u, self.v, self.w,
                                  dx, dt, nx, ny, nz, self.dom], device=dev)
            else:
                wp.launch(k3_g2p_and_advect, dim=self.n_particles,
                          inputs=[self.pos, self.vel,
                                  self.u, self.v, self.w, self.us, self.vs, self.ws,
                                  dx, dt, nx, ny, nz, self.flip_blend, self.dom], device=dev)
        # S2.15 — color transfer at end of step (positions are now current)
        with prof.section("color"):
            self._apply_color_transfer()
        # S2.18 / B11 — generic scalar attribute transfer (temperature for now).
        # Same pipeline; cheap when attr is None (early-return inside the helper).
        if self.attr_temperature is not None:
            with prof.section("temperature"):
                self._apply_scalar_transfer(self.attr_temperature)
        # No host-sync here. step() queues ops on the default stream and
        # downstream consumers (mesh extraction, particles readback) sync
        # implicitly when they call .numpy(). Removing the explicit sync
        # is what makes step() capturable into a CUDA graph (B5.1 spike).

    # --------------------------------------------------- F3.6 prepare_frame
    @block("F3.6", "Per-frame hook: rebuild marker for anim obstacles, emit "
                  "inflow particles, drop outflow particles")
    def prepare_frame(self, frame_idx: int, frame_dt: float):
        # B5.2 — topology may change here (marker rebuild, inflow/outflow
        # particle count delta). Drop any cached graph; next step() will
        # recapture for the new topology.
        self._cuda_graph_invalidate()
        # ---- animated obstacles: rebuild marker = walls ∪ static ∪ analytic_anim
        # then overlay mesh-anim via GPU ray-cast (after upload, see comment in _build_anim_sdf).
        # Also: for each moving obstacle, write its world-space velocity into
        # solid_u/v/w on the surrounding faces so that the boundary condition
        # makes fluid see a *moving* wall (creates wake, displacement).
        if self._anim_specs:
            # clear last-frame solid velocities
            full = (max(self.nx + 1, self.nx), max(self.ny + 1, self.ny), max(self.nz + 1, self.nz))
            wp.launch(k3_clear_solid_vel, dim=full,
                      inputs=[self.solid_u, self.solid_v, self.solid_w], device=self.device)

            m = self._wall_marker.copy()
            for static_sdf in self._static_obstacle_sdfs:
                m[static_sdf <= 0.0] = 2
            # split analytic into per-obstacle (need their velocities)
            analytic_perobs, mesh_specs_active = self._build_anim_sdf_per_obstacle(frame_idx)
            for sdf, vel in analytic_perobs:
                m[sdf <= 0.0] = 2
            self._marker_host = m
            wp.copy(self.marker, wp.array(m, dtype=int, device=self.device))
            # write per-obstacle solid velocity into faces (analytic obstacles)
            for sdf, vel in analytic_perobs:
                self._stamp_solid_vel_from_sdf(sdf, vel)
            # overlay mesh-anim markers + their velocities
            if mesh_specs_active:
                self._apply_anim_mesh_obstacles_after_upload(
                    mesh_specs_active, self.cell_centers_np(), frame_idx)

        # ---- inflow + outflow
        if not (self.inflows or self.outflows):
            return
        # outflow on GPU (stream compaction); falls through to no-op if no active
        if self.outflows:
            self._apply_outflows_gpu(frame_idx)
        # inflow: small per-frame batch — append host-side and re-bind
        if self.inflows:
            emit_pos, emit_vel = apply_inflows(self.inflows, frame_idx, frame_dt, self._rng)
            if len(emit_pos) > 0:
                cur_pos = self.pos.numpy() if self.pos is not None else np.zeros((0, 3), dtype=np.float32)
                cur_vel = self.vel.numpy() if self.vel is not None else np.zeros((0, 3), dtype=np.float32)
                cur_pos = np.concatenate([cur_pos, emit_pos], axis=0)
                cur_vel = np.concatenate([cur_vel, emit_vel], axis=0)
                self.pos = wp.array(cur_pos, dtype=wp.vec3, device=self.device)
                self.vel = wp.array(cur_vel, dtype=wp.vec3, device=self.device)
                self.affine_C = None
                self.n_particles = len(cur_pos)

    # ----------------------------------------------------- F3.4 step_cfl
    @block("F3.4", "CFL-adaptive substepping: clamps by advection CFL "
                  "and (when σ>0) by explicit-CSF capillary-wave CFL")
    def step_cfl(self, target_dt: float, pressure_iters: int = 80,
                 cfl: float = 0.5, max_substeps: int = 16,
                 pressure_solver: str = "jacobi") -> int:
        """Run one frame's worth of simulation, automatically substepping by
        the tighter of (a) the advection CFL `CFL·dx/|v_max|` and
        (b) the capillary-wave CFL `√(ρ·dx³/(2π·σ))` when σ>0.

        Returns
        -------
        n_substeps : how many sub-steps were taken (>=1).
        """
        # S2.10.GPU — device-resident |v|_max; only transfers 1 float to host.
        # Saves an O(n_particles) D→H roundtrip per frame at scale.
        if not hasattr(self, "_cfl_vmax_scratch"):
            self._cfl_vmax_scratch = wp.zeros(1, dtype=float, device=self.device)
        n_adv = cfl_substep_count_gpu(self.vel, self.dx, target_dt,
                                      cfl=cfl, max_substeps=max_substeps,
                                      scratch=self._cfl_vmax_scratch)
        # S2.14.5 — capillary-wave bound (explicit CSF is conditionally stable)
        if self.surface_tension > 0.0:
            dt_csf = csf_max_stable_dt(self.rho, self.dx, self.surface_tension)
            n_csf_needed = max(1, int(np.ceil(target_dt / dt_csf)))
            n_csf = min(n_csf_needed, max_substeps)
            # Honest: if the substep ceiling truncates below the CFL requirement,
            # CSF will still produce parasitic currents. Surface this loudly so
            # the user can raise cfl_max_substeps rather than blame the algo.
            if n_csf_needed > max_substeps and not getattr(self, "_csf_cfl_warned", False):
                import sys
                print(
                    f"[gpufluid] WARNING: CSF capillary-wave CFL needs "
                    f"{n_csf_needed} substeps for dt={target_dt:.4g}s "
                    f"(σ={self.surface_tension}, dx={self.dx:.4g}, "
                    f"dt_csf={dt_csf:.4g}s) but max_substeps={max_substeps}. "
                    f"Expect parasitic currents / drift. Raise "
                    f"cfl_max_substeps to ≥{n_csf_needed} or lower σ.",
                    file=sys.stderr, flush=True,
                )
                self._csf_cfl_warned = True
        else:
            n_csf = 1
        n = max(n_adv, n_csf)
        sub_dt = target_dt / n
        for _ in range(n):
            self.step(sub_dt, pressure_iters=pressure_iters, pressure_solver=pressure_solver)
        return n

    def get_particles(self):
        return self.pos.numpy(), self.vel.numpy()

    # ------------------------------------------------------ F3.5 checkpoint
    @block("F3.5", "Save solver state to .npz (resumable)")
    def save_checkpoint(self, path) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pos = self.pos.numpy() if self.pos is not None else np.zeros((0, 3), dtype=np.float32)
        vel = self.vel.numpy() if self.vel is not None else np.zeros((0, 3), dtype=np.float32)
        marker = self.marker.numpy()
        affine = self.affine_C.numpy() if self.affine_C is not None else np.zeros((0, 3, 3), dtype=np.float32)
        np.savez_compressed(
            str(path),
            pos=pos, vel=vel, affine_C=affine, marker=marker,
            nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx,
            gravity=self.gravity, flip_blend=self.flip_blend, rho=self.rho,
            viscosity=self.viscosity, transfer_mode=str(self.transfer_mode),
            surface_tension=self.surface_tension,
        )

    @block("F3.5", "Load checkpoint into a fresh solver (state-only; topology must match)")
    def load_checkpoint(self, path) -> None:
        data = np.load(str(path), allow_pickle=False)
        # validate topology
        nx, ny, nz = int(data["nx"]), int(data["ny"]), int(data["nz"])
        if (nx, ny, nz) != (self.nx, self.ny, self.nz):
            raise RuntimeError(f"checkpoint grid {(nx,ny,nz)} != solver {(self.nx,self.ny,self.nz)}")
        pos = data["pos"]; vel = data["vel"]; marker = data["marker"]; affine = data["affine_C"]
        self.pos = wp.array(pos, dtype=wp.vec3, device=self.device) if len(pos) else None
        self.vel = wp.array(vel, dtype=wp.vec3, device=self.device) if len(vel) else None
        self.affine_C = wp.array(affine, dtype=wp.mat33, device=self.device) if len(affine) else None
        wp.copy(self.marker, wp.array(marker.astype(np.int32), dtype=int, device=self.device))
        self._marker_host = marker.astype(np.int32)
        self.n_particles = len(pos)
