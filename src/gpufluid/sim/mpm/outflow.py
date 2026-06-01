"""[Layer S2.17.8] Per-frame outflow (drain) for MPM.

The MPM workaround pre-allocates a fixed particle array (see :mod:`inflow`),
so there is no native "remove particle" path — array compaction would
invalidate the inflow gates' base indices. Instead we DESPAWN: when a live
particle (``particle_selection == 0``) enters an active outflow AABB, we

  * set ``particle_selection = 1`` so it is excluded from every PLY dump and
    every collider/wall/inflow kernel (they all early-return on selection==1),
  * zero its velocity and reset F/F_trial/C (a parked particle carries no
    deformation history), and
  * park it at a fixed sink point well inside the box centre.

This is the exact mirror of the inflow gate's "held" state, reusing warp-mpm's
selection convention. It bounds the live particle count for continuous-flow
scenes (waterfalls, fountains, rivers) that would otherwise fill the domain.

A despawned particle stays despawned (one-way): once selection flips to 1 here
it is never re-released (only the inflow gate releases, and only for its own
pre-allocated slice that never overlaps a drained particle). Frame-gated by
``frame_start``/``frame_end`` in output-frame units, converted to sim steps
via ``dump_every`` (same convention as :class:`MpmInflow`).
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ...blocks import block
from ._warp_mpm_imports import MPMStateStruct


@dataclass
class MpmOutflow:
    """Axis-aligned drain box. Particles entering it while active are
    despawned (removed from the live set and all dumps).

    ``frame_start``/``frame_end`` are in output-frame units (ticks of
    ``fps``); the solver converts them to sim steps with ``dump_every``.
    """
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    frame_start: int = 0
    frame_end: int = 1_000_000


@block("S2.17.8",
       "MPM outflow despawn: live particles inside an active drain AABB are "
       "marked selection=1 (excluded from dumps + all kernels), velocity and "
       "deformation reset, parked at the box centre. Mirror of the inflow gate.")
@wp.kernel
def k_outflow_despawn(
    state: MPMStateStruct,
    current_step: int,
    step_start: int,
    step_end: int,
    lo_x: float, lo_y: float, lo_z: float,
    hi_x: float, hi_y: float, hi_z: float,
):
    p = wp.tid()
    # Only drain currently-live particles. Held inflow particles
    # (selection==1) and already-drained particles are skipped — the latter
    # makes the despawn one-way and idempotent.
    if state.particle_selection[p] == 1:
        return
    # Frame gate: inclusive [step_start, step_end].
    if current_step < step_start or current_step > step_end:
        return
    pos = state.particle_x[p]
    if (pos[0] >= lo_x and pos[0] <= hi_x
            and pos[1] >= lo_y and pos[1] <= hi_y
            and pos[2] >= lo_z and pos[2] <= hi_z):
        # Despawn: park at box centre, freeze, reset deformation, hide.
        cx = 0.5 * (lo_x + hi_x)
        cy = 0.5 * (lo_y + hi_y)
        cz = 0.5 * (lo_z + hi_z)
        state.particle_x[p] = wp.vec3(cx, cy, cz)
        state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
        I = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        Z = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state.particle_F[p] = I
        state.particle_F_trial[p] = I
        state.particle_C[p] = Z
        state.particle_selection[p] = 1
