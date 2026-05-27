"""[Layer S2.17.4 + S2.17.5] Velocity-cap kernels for MPM.

  S2.17.4 — :func:`k_tap_terminal_velocity`: inside a narrow tap column
            above the obstacle, downward `v_z` is capped to a terminal
            value (e.g. -1.0 m/s). Mimics air drag on a laminar stream so
            the falling water arrives at the obstacle at a controlled
            speed instead of free-fall ~2.5 m/s.

  S2.17.5 — :func:`k_anti_splash_vz`: above the obstacle top, `|v_z|` is
            hard-clamped to a small splash budget (e.g. ±0.3 m/s).
            Suppresses APIC stress-launch artifacts that fling individual
            particles upward at the rigid edge.

Both kernels operate on **particles** (not grid). They run pre-P2G in the
F3.7 pipeline so the bad-velocity outliers never seed corrupted F.
"""
from __future__ import annotations

import warp as wp

from ...blocks import block
from ._warp_mpm_imports import MPMStateStruct


@block("S2.17.4",
       "Tap-zone terminal velocity cap (downward only): "
       "mimics air drag on a laminar stream")
@wp.kernel
def k_tap_terminal_velocity(
    state: MPMStateStruct,
    tap_lo_x: float, tap_hi_x: float,
    tap_lo_y: float, tap_hi_y: float,
    tap_z_min: float,
    v_terminal: float,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if pos[2] < tap_z_min:
        return
    if pos[0] < tap_lo_x or pos[0] > tap_hi_x:
        return
    if pos[1] < tap_lo_y or pos[1] > tap_hi_y:
        return
    v = state.particle_v[p]
    if v[2] < v_terminal:
        state.particle_v[p] = wp.vec3(v[0], v[1], v_terminal)


@block("S2.17.5",
       "Above-obstacle anti-splash |v_z| clamp: caps stress-launched "
       "upward outliers at a physical hydraulic-jump bound")
@wp.kernel
def k_anti_splash_vz(
    state: MPMStateStruct,
    z_threshold: float,
    vz_min: float,
    vz_max: float,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if pos[2] < z_threshold:
        return
    v = state.particle_v[p]
    vz = wp.clamp(v[2], vz_min, vz_max)
    state.particle_v[p] = wp.vec3(v[0], v[1], vz)
