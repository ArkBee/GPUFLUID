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

Per BACKLOG B3.1. B3.2 (wave-crest, |∇·n̂|) and the emit-rate fold-in
(B3.3) plug into this potential later.
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
