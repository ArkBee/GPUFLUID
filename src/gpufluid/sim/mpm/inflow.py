"""[Layer S2.17.7] Per-frame inflow injection for MPM.

warp-mpm pre-allocates particles at init — there is no native "spawn" path.
We work around that by pre-seeding the *total* expected inflow particle count
at t=0, parking them at the inflow source position with a per-particle
``spawn_step``. The :func:`k_inflow_gate` kernel runs every pre-step:

* Before ``spawn_step``: clamp position to source AABB, zero velocity, reset
  F/F_trial/C, and mark ``particle_selection=1`` so the particle is excluded
  from the PLY dump (matches warp-mpm's "skip" convention, see S2.17.6).
* At/after ``spawn_step``: release the particle with the inflow's prescribed
  velocity and flip ``particle_selection=0`` so it appears in subsequent
  dumps. Normal MPM physics carries it from then on.

This lets a single MPM bake express a finite-duration tap whose rate is set
by ``rate_per_sec`` over ``[frame_start, frame_end]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...blocks import block
from ._warp_mpm_imports import MPMStateStruct


@dataclass
class MpmInflow:
    """Continuous emission from an axis-aligned box.

    ``frame_start``/``frame_end`` are in output-frame units (i.e. ticks of
    ``fps``). Conversion to sim steps uses ``dump_every`` from the solver.
    ``rate_per_sec`` is the particle injection rate while active.

    ``keyframes`` (optional, S2.17.7.MOVING) carries per-frame AABBs for an
    animated emitter. Each row is ``(frame, lo_x, lo_y, lo_z, hi_x, hi_y,
    hi_z)``. When present, :func:`seed_inflow_particles` samples each
    particle's hold position from the AABB that's active at that particle's
    spawn frame (linear interpolation between adjacent keyframes).
    """
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, -1.0)
    rate_per_sec: float = 5000.0
    frame_start: int = 0
    frame_end: int = 240
    keyframes: tuple = ()   # tuple of (frame:int, lo3, hi3) rows
    # B18 — per-source per-particle attributes. When set, every particle
    # spawned from this inflow inherits the same RGB / scalar value; the
    # solver concatenates them in lockstep with positions during init.
    color: tuple[float, float, float] | None = None
    temperature: float | None = None


@block("S2.17.7",
       "MPM inflow gate: hold pre-allocated particles at source position "
       "until their spawn_step, then release with prescribed velocity")
@wp.kernel
def k_inflow_gate(
    state: MPMStateStruct,
    current_step: int,
    base_idx: int,
    spawn_step: wp.array(dtype=int),
    hold_pos: wp.array(dtype=wp.vec3),
    vx: float, vy: float, vz: float,
):
    p = wp.tid()
    idx = base_idx + p
    ss = spawn_step[p]
    if current_step < ss:
        state.particle_x[idx] = hold_pos[p]
        state.particle_v[idx] = wp.vec3(0.0, 0.0, 0.0)
        I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state.particle_F[idx] = I
        state.particle_F_trial[idx] = I
        state.particle_C[idx] = Z
        state.particle_selection[idx] = 1
    elif current_step == ss:
        state.particle_v[idx] = wp.vec3(vx, vy, vz)
        state.particle_selection[idx] = 0


def seed_inflow_particles(
    inflow: MpmInflow,
    fps: int,
    dump_every: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-generate (hold_positions, spawn_steps) for one inflow.

    Returns
    -------
    positions : (N, 3) float32 — uniform random in the inflow AABB
    spawn_steps : (N,) int32 — sim step at which each particle releases
    """
    if rng is None:
        rng = np.random.default_rng(42)
    duration_frames = max(1, int(inflow.frame_end - inflow.frame_start))
    n = max(1, int(inflow.rate_per_sec * duration_frames / max(fps, 1)))
    # Distribute spawn frames uniformly across the active window.
    frame_floats = rng.uniform(
        float(inflow.frame_start), float(inflow.frame_end), size=n
    )
    spawn_steps = (frame_floats * float(dump_every)).astype(np.int32)

    if inflow.keyframes:
        # S2.17.7.MOVING — animated emitter: per-particle AABB sampled from
        # the keyframe at this particle's spawn time (linear interpolation
        # between adjacent keyframes).
        kf = np.asarray(inflow.keyframes, dtype=np.float32)  # (K, 7)
        kf_frames = kf[:, 0]
        kf_lo = kf[:, 1:4]
        kf_hi = kf[:, 4:7]
        # For each particle, find the interpolation bracket. np.interp handles
        # out-of-range by clamping to endpoints, which is exactly what we
        # want for spawn frames slightly outside the keyframe span.
        lo_p = np.empty((n, 3), dtype=np.float32)
        hi_p = np.empty((n, 3), dtype=np.float32)
        for axis in range(3):
            lo_p[:, axis] = np.interp(frame_floats, kf_frames, kf_lo[:, axis])
            hi_p[:, axis] = np.interp(frame_floats, kf_frames, kf_hi[:, axis])
        # rng.uniform supports broadcast over (low, high) of shape (n, 3)
        u = rng.random((n, 3), dtype=np.float32)
        positions = (lo_p + u * (hi_p - lo_p)).astype(np.float32)
    else:
        lo = np.asarray(inflow.lo, dtype=np.float32)
        hi = np.asarray(inflow.hi, dtype=np.float32)
        positions = rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)
    return positions, spawn_steps
