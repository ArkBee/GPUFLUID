"""[BLK W7.7] Ihmsen-2012 trapped-air potential for whitewater emission.

The legacy emit selector ``|v| > threshold`` (whitewater.py:127) overcounts
calm fast-moving bulk fluid and undercounts true turbulence pockets. Ihmsen
et al. 2012 §3 identify trapped air via a per-particle scalar that fires
strongly only where neighbour velocity vectors **diverge** within a small
neighbourhood — i.e. where air can actually be entrained.

This module ports W7.7 to GPU:

    I_ta(i) = sum_{j != i, |x_ij| < h}  (1 - cos θ_ij) · (1 - |v_ij|/v_max)

where ``θ_ij`` is the angle between v_i and v_j and ``v_ij = v_i - v_j``.
The (1 - cos θ) factor is 0 for aligned (laminar) neighbours and grows
toward 2 for anti-aligned (diverging) pairs. The (1 - |v_ij|/v_max) factor
caps the contribution of pairs with extreme relative speed so a single
outlier doesn't dominate the sum. Both factors are clamped to [0, 1] inside
the kernel.

Per BACKLOG B3.1. The wave-crest companion [BLK W7.8] (B3.2) lives
further down in this module and uses a different algorithm — particle
density indicator + box blur + normal divergence.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from typing import Optional

from ..blocks import block
from ..primitives.runtime import device as default_device, zeros


# Kernel parameter type for a HashGrid handle. Recent Warp accepts
# `wp.uint64`; we accept the HashGrid object directly at launch time and
# Warp converts.
_HG = wp.uint64


@wp.kernel
def k3_trapped_air_potential(
    grid: _HG,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    radius: float,
    v_max: float,
    out: wp.array(dtype=float),
):
    """[BLK W7.7] Per-particle trapped-air potential.

    One thread per particle. Iterates neighbours via wp.hash_grid_query.
    Self is skipped via ``index == i`` check.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    pi = pos[i]
    vi = vel[i]
    speed_i = wp.length(vi)
    accum = float(0.0)

    for j in wp.hash_grid_query(grid, pi, radius):
        if j == i:
            continue
        pj = pos[j]
        d = wp.length(pi - pj)
        if d > radius or d < 1.0e-6:
            continue
        vj = vel[j]
        speed_j = wp.length(vj)
        # cos(θ) between v_i and v_j; treat near-zero velocities as aligned
        # (no contribution from quiescent neighbours).
        if speed_i < 1.0e-6 or speed_j < 1.0e-6:
            continue
        cos_theta = wp.dot(vi, vj) / (speed_i * speed_j)
        # Clamp to [-1, 1] to absorb fp noise from the dot product.
        if cos_theta > 1.0:
            cos_theta = 1.0
        if cos_theta < -1.0:
            cos_theta = -1.0
        ang_factor = 1.0 - cos_theta            # in [0, 2]; spec caps SUM, not factor
        # (1 - |v_ij|/v_max), clamped to [0, 1]
        vij_mag = wp.length(vi - vj)
        vel_factor = 1.0 - vij_mag / v_max
        if vel_factor < 0.0:
            vel_factor = 0.0
        if vel_factor > 1.0:
            vel_factor = 1.0
        accum += ang_factor * vel_factor

    # Final clamp per-particle to [0, 1] — spec: "capped at 1". A bulk
    # neighbour count of N converging pairs could otherwise push the sum
    # arbitrarily high.
    if accum > 1.0:
        accum = 1.0
    out[i] = accum


block("W7.7", "Trapped-air potential (Ihmsen 2012 §3.1) — GPU HashGrid")(k3_trapped_air_potential)


@block("W7.7.H", "Host wrapper: numpy pos/vel → numpy trapped-air potential")
def trapped_air_potential(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: float,
    v_max: float = 10.0,
    grid_cells: int = 64,
) -> np.ndarray:
    """Compute I_ta per particle on GPU and return as a numpy float32 array.

    Parameters
    ----------
    pos : (N, 3) float32   — particle positions in sim space (m).
    vel : (N, 3) float32   — particle velocities (m/s).
    radius : float         — neighbour search radius. Typical 2..3·dx.
    v_max : float          — velocity normaliser. Pairs with |v_i - v_j| ≥ v_max
                             contribute 0 (the (1 - |v_ij|/v_max) factor floors).
    grid_cells : int       — HashGrid resolution per axis. Build cost is
                             O(N + grid_cells³); 64 covers 0..1 sim space at
                             radius ~ 1/64. Increase for larger domains.

    Returns
    -------
    I_ta : (N,) float32, values in [0, 1].
    """
    n = pos.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if pos.shape != vel.shape or pos.shape[1] != 3:
        raise ValueError(f"pos/vel must both be (N,3); got {pos.shape}, {vel.shape}")
    dev = default_device()
    pos_wp = wp.array(pos.astype(np.float32), dtype=wp.vec3, device=dev)
    vel_wp = wp.array(vel.astype(np.float32), dtype=wp.vec3, device=dev)
    grid = wp.HashGrid(grid_cells, grid_cells, grid_cells, device=dev)
    grid.build(pos_wp, radius)
    out = zeros((n,), dev=dev)
    wp.launch(k3_trapped_air_potential, dim=n,
              inputs=[grid.id, pos_wp, vel_wp, float(radius), float(v_max), out],
              device=dev)
    return out.numpy()


# ---------------------------------------------------------------------------
# [BLK W7.8] — wave-crest potential I_wc = |∇·n̂|
#
# Spec: DESIGN.md §11.5.1. Standalone P2G→blur→∇→∇· pipeline; does NOT
# depend on CSF (S2.14.*) so it works on scenes without σ>0 and on
# offline particle dumps.
# ---------------------------------------------------------------------------

# Small floor under |∇χ̃| to avoid 0/0 normalisation in plateau regions.
# Interior cells of a dense cluster have ∇χ̃ ≈ 0; we need |n̂| ≈ 0 there,
# not a noisy unit vector, so the divergence stays low for bulk particles.
_GRAD_FLOOR = 1.0e-6


@wp.kernel
def k3_p2g_indicator(
    pos: wp.array(dtype=wp.vec3),
    chi: wp.array3d(dtype=float),
    inv_dx: float,
    nx: int, ny: int, nz: int,
):
    """[BLK W7.8] Trilinear scatter of unit indicator per particle.

    One thread per particle. Atomic-adds the eight corner weights of the
    cell containing pos[p] into chi. After all launches, chi[i,j,k]
    holds the fractional particle count in cell (i,j,k) — a smoothed
    density field.
    """
    p = wp.tid()
    px = pos[p][0] * inv_dx
    py = pos[p][1] * inv_dx
    pz = pos[p][2] * inv_dx
    i0 = int(wp.floor(px)); j0 = int(wp.floor(py)); k0 = int(wp.floor(pz))
    sx = px - float(i0); sy = py - float(j0); sz = pz - float(k0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ii = i0 + di; jj = j0 + dj; kk = k0 + dk
                if 0 <= ii and ii < nx and 0 <= jj and jj < ny and 0 <= kk and kk < nz:
                    wxi = (1.0 - sx) if di == 0 else sx
                    wyi = (1.0 - sy) if dj == 0 else sy
                    wzi = (1.0 - sz) if dk == 0 else sz
                    wp.atomic_add(chi, ii, jj, kk, wxi * wyi * wzi)


block("W7.8", "Wave-crest indicator P2G scatter (unit weight, trilinear)")(k3_p2g_indicator)


@wp.kernel
def k3_compute_normal_field(
    chi: wp.array3d(dtype=float),
    n_field: wp.array3d(dtype=wp.vec3),
    grad_mag: wp.array3d(dtype=float),
):
    """[BLK W7.8] Cell-centred unit normal n_hat and gradient magnitude.

    Writes two outputs: n_hat = grad_chi / |grad_chi| (unit), and the
    raw |grad_chi| in `grad_mag`. The downstream curvature step uses
    n_hat; the per-particle gather uses `grad_mag` as a "surface-ness"
    gate so bulk-interior particles (where grad is zero) don't pick up
    spurious curvature from noisy normals in the plateau.

    Below `_GRAD_FLOOR` the normal is forced to zero rather than
    arbitrary — keeps the divergence step well-conditioned.
    """
    i, j, k = wp.tid()
    nx = chi.shape[0]; ny = chi.shape[1]; nz = chi.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    ip = i + 1 if i + 1 < nx else i
    im = i - 1 if i > 0 else i
    jp = j + 1 if j + 1 < ny else j
    jm = j - 1 if j > 0 else j
    kp = k + 1 if k + 1 < nz else k
    km = k - 1 if k > 0 else k
    gx = 0.5 * (chi[ip, j, k] - chi[im, j, k])
    gy = 0.5 * (chi[i, jp, k] - chi[i, jm, k])
    gz = 0.5 * (chi[i, j, kp] - chi[i, j, km])
    mag = wp.sqrt(gx * gx + gy * gy + gz * gz)
    grad_mag[i, j, k] = mag
    if mag > _GRAD_FLOOR:
        inv = 1.0 / mag
        n_field[i, j, k] = wp.vec3(gx * inv, gy * inv, gz * inv)
    else:
        n_field[i, j, k] = wp.vec3(0.0, 0.0, 0.0)


block("W7.8", "Wave-crest normal n_hat and gradient magnitude (plateau zeros out)")(k3_compute_normal_field)


@wp.kernel
def k3_compute_div_normal(
    n_field: wp.array3d(dtype=wp.vec3),
    div_abs: wp.array3d(dtype=float),
    dx: float,
):
    """[BLK W7.8] |∇·n̂| via central differences, raw 1/length units.

    Stored as |∇·n̂| so the downstream sample doesn't need to take an
    abs (and so the grid stays non-negative, which simplifies the
    [0,1] clamp in the gather kernel). Boundary cells use one-sided
    differences.
    """
    i, j, k = wp.tid()
    nx = n_field.shape[0]; ny = n_field.shape[1]; nz = n_field.shape[2]
    if i >= nx or j >= ny or k >= nz:
        return
    ip = i + 1 if i + 1 < nx else i
    im = i - 1 if i > 0 else i
    jp = j + 1 if j + 1 < ny else j
    jm = j - 1 if j > 0 else j
    kp = k + 1 if k + 1 < nz else k
    km = k - 1 if k > 0 else k
    inv_2dx = 0.5 / dx
    dnx_dx = (n_field[ip, j, k][0] - n_field[im, j, k][0]) * inv_2dx
    dny_dy = (n_field[i, jp, k][1] - n_field[i, jm, k][1]) * inv_2dx
    dnz_dz = (n_field[i, j, kp][2] - n_field[i, j, km][2]) * inv_2dx
    d = dnx_dx + dny_dy + dnz_dz
    div_abs[i, j, k] = d if d >= 0.0 else -d


block("W7.8", "Wave-crest |∇·n̂| central differences on cell centres")(k3_compute_div_normal)


@wp.kernel
def k3_g2p_gate_clamp(
    pos: wp.array(dtype=wp.vec3),
    div_abs: wp.array3d(dtype=float),
    grad_mag: wp.array3d(dtype=float),
    inv_dx: float,
    curv_norm: float,
    grad_norm: float,
    out: wp.array(dtype=float),
):
    """[BLK W7.8] Per-particle gather of |div n_hat| GATED by |grad chi|.

    Two trilinear samples (curvature, surface-ness gate). I_wc(p) is
    the product, both normalised to [0,1]:

        I_wc = clip01(|div n_hat| * dx / curv_norm) * clip01(|grad chi| / grad_norm)

    The gate is what kills bulk-interior particles: they sit on the
    chi_tilde plateau where |grad chi| is near zero, so even if the
    normal field has noise nearby and produces a non-trivial divergence
    sample, the gate clamps it to zero. Only particles ON the surface
    AND in a curving region can score high.
    """
    p = wp.tid()
    px = pos[p][0] * inv_dx
    py = pos[p][1] * inv_dx
    pz = pos[p][2] * inv_dx
    nx = div_abs.shape[0]; ny = div_abs.shape[1]; nz = div_abs.shape[2]
    i0 = int(wp.floor(px)); j0 = int(wp.floor(py)); k0 = int(wp.floor(pz))
    if i0 < 0: i0 = 0
    if j0 < 0: j0 = 0
    if k0 < 0: k0 = 0
    if i0 >= nx - 1: i0 = nx - 2
    if j0 >= ny - 1: j0 = ny - 2
    if k0 >= nz - 1: k0 = nz - 2
    sx = px - float(i0); sy = py - float(j0); sz = pz - float(k0)
    if sx < 0.0: sx = 0.0
    if sx > 1.0: sx = 1.0
    if sy < 0.0: sy = 0.0
    if sy > 1.0: sy = 1.0
    if sz < 0.0: sz = 0.0
    if sz > 1.0: sz = 1.0
    # Sample curvature |div n_hat|.
    c000 = div_abs[i0, j0, k0]; c100 = div_abs[i0 + 1, j0, k0]
    c010 = div_abs[i0, j0 + 1, k0]; c110 = div_abs[i0 + 1, j0 + 1, k0]
    c001 = div_abs[i0, j0, k0 + 1]; c101 = div_abs[i0 + 1, j0, k0 + 1]
    c011 = div_abs[i0, j0 + 1, k0 + 1]; c111 = div_abs[i0 + 1, j0 + 1, k0 + 1]
    c00 = c000 * (1.0 - sx) + c100 * sx
    c10 = c010 * (1.0 - sx) + c110 * sx
    c01 = c001 * (1.0 - sx) + c101 * sx
    c11 = c011 * (1.0 - sx) + c111 * sx
    c0 = c00 * (1.0 - sy) + c10 * sy
    c1 = c01 * (1.0 - sy) + c11 * sy
    curv = c0 * (1.0 - sz) + c1 * sz
    # Sample gate |grad chi|.
    g000 = grad_mag[i0, j0, k0]; g100 = grad_mag[i0 + 1, j0, k0]
    g010 = grad_mag[i0, j0 + 1, k0]; g110 = grad_mag[i0 + 1, j0 + 1, k0]
    g001 = grad_mag[i0, j0, k0 + 1]; g101 = grad_mag[i0 + 1, j0, k0 + 1]
    g011 = grad_mag[i0, j0 + 1, k0 + 1]; g111 = grad_mag[i0 + 1, j0 + 1, k0 + 1]
    g00 = g000 * (1.0 - sx) + g100 * sx
    g10 = g010 * (1.0 - sx) + g110 * sx
    g01 = g001 * (1.0 - sx) + g101 * sx
    g11 = g011 * (1.0 - sx) + g111 * sx
    g0 = g00 * (1.0 - sy) + g10 * sy
    g1 = g01 * (1.0 - sy) + g11 * sy
    grad = g0 * (1.0 - sz) + g1 * sz
    # Normalise both to [0,1] and multiply.
    curv_n = curv / curv_norm
    if curv_n < 0.0: curv_n = 0.0
    if curv_n > 1.0: curv_n = 1.0
    gate = grad / grad_norm
    if gate < 0.0: gate = 0.0
    if gate > 1.0: gate = 1.0
    out[p] = curv_n * gate


block("W7.8", "Wave-crest G2P: gather |div n| gated by |grad chi|, clamped to [0,1]")(k3_g2p_gate_clamp)


@block("W7.8.H", "Host wrapper: numpy pos → numpy wave-crest potential I_wc")
def wave_crest_potential(
    pos: np.ndarray,
    grid_res: int = 32,
    dx: float = 1.0 / 32,
    n_blur: int = 2,
) -> np.ndarray:
    """Compute I_wc = |∇·n̂| per particle on GPU.

    Parameters
    ----------
    pos : (N, 3) float32 — particle positions in sim space (m).
    grid_res : int       — cell-centred grid resolution per axis. The
                           full grid covers [0, grid_res*dx] in each
                           axis; particles outside this box are clamped
                           to the boundary.
    dx : float           — cell size (m). Together with grid_res
                           determines the domain extent.
    n_blur : int         — number of 3×3×3 box-blur passes on χ before
                           normal/divergence. More passes = smoother
                           normals = wider crest detection band.
                           Default 2 matches the typical CSF
                           csf_smoothing_passes.

    Returns
    -------
    I_wc : (N,) float32, values in [0, 1]. Flat / interior particles
           sit near 0; wave crests and breaking ridges approach 1.
    """
    n = pos.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if pos.shape[1] != 3:
        raise ValueError(f"pos must be (N,3); got {pos.shape}")
    if grid_res < 4:
        raise ValueError(f"grid_res {grid_res} too small; need ≥4 for "
                         f"central differences to make sense.")
    dev = default_device()
    pos_wp = wp.array(pos.astype(np.float32), dtype=wp.vec3, device=dev)
    chi_a = zeros((grid_res, grid_res, grid_res), dev=dev)
    chi_b = zeros((grid_res, grid_res, grid_res), dev=dev)
    n_field = wp.zeros(shape=(grid_res, grid_res, grid_res),
                       dtype=wp.vec3, device=dev)
    grad_mag = zeros((grid_res, grid_res, grid_res), dev=dev)
    div_abs = zeros((grid_res, grid_res, grid_res), dev=dev)

    inv_dx = 1.0 / float(dx)

    # 1. Scatter unit indicator (one weight per particle, smeared trilinearly).
    wp.launch(k3_p2g_indicator, dim=n,
              inputs=[pos_wp, chi_a, inv_dx, grid_res, grid_res, grid_res],
              device=dev)

    # 2. Box-blur (G1.7) N passes, ping-pong.
    from ..primitives.gridmath import k_box_blur_3d
    src, dst = chi_a, chi_b
    for _ in range(int(n_blur)):
        wp.launch(k_box_blur_3d, dim=(grid_res, grid_res, grid_res),
                  inputs=[src, dst], device=dev)
        src, dst = dst, src

    # 3. Normal field + raw |grad chi| (the future surface-gate).
    wp.launch(k3_compute_normal_field,
              dim=(grid_res, grid_res, grid_res),
              inputs=[src, n_field, grad_mag], device=dev)

    # 4. |div n_hat| — raw curvature, units 1/length.
    wp.launch(k3_compute_div_normal,
              dim=(grid_res, grid_res, grid_res),
              inputs=[n_field, div_abs, float(dx)], device=dev)

    # 5. Gated sample per-particle.
    # Normalisation policy:
    #   curv_norm — sharp crest |div n_hat| scales as 1/dx, so
    #              curv_norm = 1/dx puts saturation near 1.
    #   grad_norm — depends on how many particles per cell scatter into chi.
    #              We take the field's max so the gate is per-scene
    #              relative. Empirically robust across density regimes.
    grad_mag_max = float(grad_mag.numpy().max())
    if grad_mag_max < 1.0e-9:
        # All particles in one cell → no surface at all → no crest emit.
        return np.zeros((n,), dtype=np.float32)

    out = zeros((n,), dev=dev)
    curv_norm = inv_dx
    wp.launch(k3_g2p_gate_clamp, dim=n,
              inputs=[pos_wp, div_abs, grad_mag,
                      inv_dx, float(curv_norm), float(grad_mag_max), out],
              device=dev)
    return out.numpy()
