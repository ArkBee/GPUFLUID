"""[Layer F3.7] MPM solver shell-out adapter around third-party warp-mpm.

The :class:`MpmSolver` consumes a scene config (the same TOML the FLIP
solver reads) and produces a PLY-per-frame cache that the M5.11 mesher
and the A8 render bridge can read unchanged.

See DESIGN.md §6.7 for the pipeline order. See §5.3 for the
``S2.17.*`` helper kernels invoked here.

Stability contract (added round-20, 2026-05-26):
  :meth:`MpmSolver.run` checks for NaN in particle positions on every
  step. On detection, it raises :class:`MpmDivergenceError` instead of
  the pre-round-20 silent ``print + break`` that left the CLI thinking
  rc=0 and the addon's auto-attach pulling a truncated cache silently.

  Callers (CLI ``_cmd_simulate_mpm``) catch this, write a
  ``cache.json`` with ``truncated_at_frame: k``, and exit non-zero
  (rc=2 — distinct from generic failure rc=1). The addon's
  ``ModalSubprocessRunner`` recognises rc=2 specifically and surfaces
  the truncation step in the user-visible ERROR report.
"""
from __future__ import annotations

import os
import tempfile
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import warp as wp

from ...blocks import block
from ...io.ply import write_points_ply


class MpmDivergenceError(RuntimeError):
    """Raised by :meth:`MpmSolver.run` when NaN appears in particle
    positions mid-bake. Means the constitutive model blew up
    (typically dt × sqrt(bulk_modulus/rho) > 1, violating the CFL on
    stress wavespeed).

    Carries ``step`` (the failing inner step), ``last_dumped_frame``
    (the last frame the dump_every cadence successfully wrote to
    disk), and ``requested_frames`` (the cfg.n_frames target). CLI
    uses these to write a truthful ``cache.json`` with
    ``truncated_at_frame`` and exit rc=2 (round-20 contract — distinct
    from generic failure rc=1 so callers can give specific UX).

    Recovery: lower ``bulk_modulus`` (default 1500 is water-like),
    raise ``dt`` denominator (i.e. lower dt or raise fps), or shrink
    domain resolution. See ``docs/BACKLOG.md`` MPM stability entry
    for the original symptom report.
    """

    def __init__(
        self,
        step: int,
        last_dumped_frame: int,
        requested_frames: int,
        detail: str = "",
    ) -> None:
        super().__init__(
            detail or f"MPM solver diverged at step {step} "
                     f"(last dumped: {last_dumped_frame}/{requested_frames})"
        )
        self.step = int(step)
        self.last_dumped_frame = int(last_dumped_frame)
        self.requested_frames = int(requested_frames)
from ._warp_mpm_imports import (
    Dirichlet_collider,
    MPM_Simulator_WARP,
)
from .colliders import k_sdf_box_collide
from .inflow import MpmInflow, k_inflow_gate, seed_inflow_particles
from .outflow import MpmOutflow, k_outflow_despawn
from .pushback import k_cube_pushback, k_wall_pushback
from .velcaps import k_anti_splash_vz, k_tap_terminal_velocity


# ─── config dataclasses ────────────────────────────────────────────────

@dataclass
class MpmFluidParams:
    """Fluid material — defaults model water with mild viscosity."""
    bulk_modulus: float = 1500.0
    density: float = 1000.0
    rpic_damping: float = 0.15
    grid_v_damping_scale: float = 0.998


@dataclass
class MpmCubeCollider:
    """A single rigid box obstacle.

    Round-57: optional ``rotation`` makes this an oriented bounding box
    (OBB). The rotation is the world-space orientation expressed as a
    3×3 matrix whose **columns** are the box-local axes (+X, +Y, +Z) in
    world coordinates. ``None`` ⇒ identity (axis-aligned, pre-round-57
    behaviour). Must be orthonormal (no scale folded in); use the
    addon-side ``matrix_world.decompose()`` rotation quaternion to build
    it. half_size is in box-local frame.
    """
    centre: tuple[float, float, float]
    half_size: tuple[float, float, float]
    tangential_friction: float = 0.6  # keep-fraction on top face; 1.0 = slip
    # 3×3 with columns = world-space box-local +X/+Y/+Z axes.
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None


@dataclass
class MpmDomainWalls:
    """Slip-clamp interior. Particles snapping outside get pushed back."""
    lo: tuple[float, float, float] = (0.05, 0.05, 0.105)
    hi: tuple[float, float, float] = (0.95, 0.95, 0.95)
    floor_z: float = 0.10  # slip plane height for warp-mpm's add_surface_collider


@dataclass
class MpmTap:
    """Inflow column geometry + terminal velocity."""
    lo_x: float = 0.485
    hi_x: float = 0.515
    lo_y: float = 0.485
    hi_y: float = 0.515
    z_min: float = 0.61
    v_terminal: float = -1.0


@dataclass
class MpmAntiSplash:
    z_threshold: float = 0.50
    vz_min: float = -2.0
    vz_max: float = 0.3


@dataclass
class MpmConfig:
    """Full scene config consumed by :class:`MpmSolver`."""
    initial_column: np.ndarray = field(  # (N, 3) float32 starting positions
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    # B18.1 — per-particle attributes for the initial column. None means
    # "no attribute" (zero-overhead path: nothing allocated, no sidecars
    # written). When set, must match `initial_column` row count.
    initial_color: np.ndarray | None = None        # (N, 3) float32 in [0, 1]
    initial_temperature: np.ndarray | None = None  # (N,)   float32
    # B18.5 — colour blending used by the mesher when computing per-vertex
    # colour from particle attributes. "off" disables vertex-colour output;
    # "linear" uses weighted RGB mean (default); "mixbox" uses pigment-space
    # mixing via the vendored LUT.
    color_mix_mode: str = "linear"
    initial_velocity_z: float = -0.3
    n_grid: int = 160
    grid_lim: float = 1.0
    dt: float = 0.001
    n_frames: int = 1500
    dump_every: int = 5
    particle_volume: float = 5.0e-8
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    fluid: MpmFluidParams = field(default_factory=MpmFluidParams)
    cubes: Sequence[MpmCubeCollider] = field(default_factory=tuple)
    walls: MpmDomainWalls = field(default_factory=MpmDomainWalls)
    tap: MpmTap | None = field(default_factory=MpmTap)
    anti_splash: MpmAntiSplash | None = field(default_factory=MpmAntiSplash)
    # S2.17.7: continuous inflow zones. Particles are pre-allocated at t=0
    # and time-gated until their spawn_step (see :mod:`inflow`).
    inflows: Sequence[MpmInflow] = field(default_factory=tuple)
    # S2.17.8: drain zones. Live particles entering an active outflow AABB are
    # despawned (selection=1), bounding the count for continuous-flow scenes.
    outflows: Sequence[MpmOutflow] = field(default_factory=tuple)
    fps: int = 60
    device: str = "cuda:0"
    # S2.17.9 (FU-023): adaptive substepping. Deep pools / fast jets push the
    # stress-wave + advection CFL past the fixed `dt`, so the bake diverges
    # (NaN) late — the user only sees "lower bulk/dt and re-bake". When enabled,
    # step() splits each frame dt into N sub-steps sized by the CFL bound
    #   dt_sub <= cfl_target * dx / (c_sound + v_max),  c_sound=sqrt(K/rho)
    # so stability holds regardless of pool depth, with no hand-tuning. Default
    # OFF → step() makes exactly one p2g2p(dt) call, byte-identical to pre-FU-023
    # (reference bakes / determinism tests unaffected). N is clamped to
    # adaptive_max_substeps so a runaway frame can't stall the bake.
    adaptive_substep: bool = False
    adaptive_cfl: float = 0.6
    adaptive_max_substeps: int = 16
    # Round-22: seed for the inflow-particle RNG. Default 42 preserves
    # pre-round-22 determinism (regression tests + repro of bake hashes);
    # set to a different int per-bake when comparing variability, or
    # set to None for non-deterministic (system-entropy) inflows.
    seed: int | None = 42

    def dx(self) -> float:
        return self.grid_lim / self.n_grid


# ─── attribute helpers (B18.1) ─────────────────────────────────────────

def build_attribute_arrays(
    cfg: "MpmConfig",
    n_initial: int,
    n_particles: int,
    inflow_gates: list,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build per-particle ``(colors, temperatures)`` numpy arrays in the
    same row order as ``[initial_column | inflow_0 | inflow_1 | ...]``.

    Returns ``(None, None)`` when no source carries the respective
    attribute — letting :class:`MpmSolver` skip GPU allocation and PLY
    sidecar emission entirely (zero-overhead path).

    Defaults applied to "silent" sources when at least one other source
    sets the attribute: ``(1, 1, 1)`` white for colour, ``20.0`` for
    temperature. These are chosen to be visually / physically neutral
    rather than zero, which would otherwise read as "black water" or
    "absolute zero" downstream.
    """
    has_color = cfg.initial_color is not None or any(
        inf.color is not None for inf in cfg.inflows
    )
    has_temp = cfg.initial_temperature is not None or any(
        inf.temperature is not None for inf in cfg.inflows
    )
    colors_np: np.ndarray | None = None
    temps_np: np.ndarray | None = None
    if n_particles > 0 and has_color:
        colors_np = np.full((n_particles, 3), 1.0, dtype=np.float32)
        if cfg.initial_color is not None:
            ic = np.asarray(cfg.initial_color, dtype=np.float32)
            if ic.shape != (n_initial, 3):
                raise ValueError(
                    f"initial_color shape {ic.shape} != ({n_initial}, 3)"
                )
            colors_np[:n_initial] = ic
        for inf, g in zip(cfg.inflows, inflow_gates):
            if inf.color is None:
                continue
            b, n = g["base"], g["n"]
            colors_np[b: b + n] = np.asarray(inf.color, dtype=np.float32)
    if n_particles > 0 and has_temp:
        temps_np = np.full((n_particles,), 20.0, dtype=np.float32)
        if cfg.initial_temperature is not None:
            it = np.asarray(cfg.initial_temperature, dtype=np.float32)
            if it.shape != (n_initial,):
                raise ValueError(
                    f"initial_temperature shape {it.shape} != ({n_initial},)"
                )
            temps_np[:n_initial] = it
        for inf, g in zip(cfg.inflows, inflow_gates):
            if inf.temperature is None:
                continue
            b, n = g["base"], g["n"]
            temps_np[b: b + n] = float(inf.temperature)
    return colors_np, temps_np


# ─── solver class ──────────────────────────────────────────────────────

@block("F3.7", "MPM shell-out solver around third-party warp-mpm")
class MpmSolver:
    """Run an MPM bake driven by an :class:`MpmConfig`.

    Pipeline per step (DESIGN.md §6.7):

        pre  : cube_pushback, wall_pushback, tap_velocity_cap,
               anti_splash_vz
        core : warp_mpm.p2g2p(dt)   ← S2.17.PATCH.{SLIP,EOS} active
        post : cube_pushback, wall_pushback
    """

    def __init__(self, cfg: MpmConfig):
        if cfg.initial_column.size == 0 and not cfg.inflows:
            raise ValueError(
                "MpmConfig: provide either initial_column or at least one inflow"
            )
        self.cfg = cfg
        # Lay out particles: [initial_column | inflow_0 | inflow_1 | ...].
        # Each inflow slice gets its own `spawn_step` & `hold_pos` arrays so
        # k_inflow_gate can be launched per-inflow with `base_idx`.
        initial = (
            np.asarray(cfg.initial_column, dtype=np.float32)
            if cfg.initial_column.size
            else np.empty((0, 3), dtype=np.float32)
        )
        if initial.ndim != 2 or initial.shape[1] != 3:
            raise ValueError(
                f"initial_column must be (N, 3); got {initial.shape}"
            )
        self.n_initial = initial.shape[0]
        # Round-22: respect cfg.seed (defaults to 42 for backwards-compat
        # determinism; pre-round-22 this was a hardcoded literal).
        rng = np.random.default_rng(cfg.seed)
        inflow_chunks: list[np.ndarray] = []
        # gate metadata, keyed by inflow index: (base_idx, n, hold_wp, spawn_wp, velocity)
        self._inflow_gates: list = []
        running_base = self.n_initial
        for inf in cfg.inflows:
            pos_i, spawn_i = seed_inflow_particles(
                inf, fps=cfg.fps, dump_every=cfg.dump_every, rng=rng,
            )
            n_i = pos_i.shape[0]
            inflow_chunks.append(pos_i)
            self._inflow_gates.append({
                "base": running_base, "n": n_i,
                "hold_np": pos_i,    # keep numpy ref for re-upload after init
                "spawn_np": spawn_i,
                "velocity": tuple(inf.velocity),
            })
            running_base += n_i
        if inflow_chunks:
            positions = np.concatenate([initial, *inflow_chunks], axis=0)
        else:
            positions = initial
        self.n_particles = positions.shape[0]

        # ── B18.1 per-particle attributes (colour + temperature) ──────────
        # Pure-numpy build is factored out (`build_attribute_arrays`) so it
        # can be unit-tested without firing up warp-mpm / CUDA.
        colors_np, temps_np = build_attribute_arrays(
            cfg, self.n_initial, self.n_particles, self._inflow_gates,
        )
        self._colors_np = colors_np
        self._temps_np = temps_np
        self.attr_color: wp.array | None = None
        self.attr_temperature: wp.array | None = None
        # GPU upload happens after warp-mpm's state is constructed; see below.
        # warp-mpm wants particle data via h5; emit a temp file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            self._tmp_h5 = tmp.name
        with h5py.File(self._tmp_h5, "w") as h5:
            h5.create_dataset("x", data=positions.T.astype(np.float32))
            h5.create_dataset(
                "particle_volume",
                data=np.full((1, self.n_particles), cfg.particle_volume,
                             dtype=np.float32),
            )
        self.mpm = MPM_Simulator_WARP(10)
        self.mpm.load_from_sampling(self._tmp_h5, n_grid=cfg.n_grid,
                                    grid_lim=cfg.grid_lim, device=cfg.device)
        self.mpm.set_parameters_dict({
            "bulk_modulus":   cfg.fluid.bulk_modulus,
            "material":       "fluid",
            "friction_angle": 35,
            "g":              list(cfg.gravity),
            "density":        cfg.fluid.density,
            "rpic_damping":         cfg.fluid.rpic_damping,
            "grid_v_damping_scale": cfg.fluid.grid_v_damping_scale,
        })
        # Initial column gets the prescribed downward velocity; inflow
        # particles stay at v=0 — the gate kernel injects their velocity
        # at their individual spawn_step.
        if cfg.initial_velocity_z != 0.0 or self._inflow_gates:
            init_v = np.zeros((self.n_particles, 3), dtype=np.float32)
            init_v[: self.n_initial, 2] = cfg.initial_velocity_z
            self.mpm.mpm_state.particle_v = wp.from_numpy(
                init_v, dtype=wp.vec3, device=cfg.device
            )
        # Upload gate arrays + mark inflow particles as hidden in PLY until
        # their spawn step (selection=1 → write_points_ply filters them out).
        if self._inflow_gates:
            sel = self.mpm.mpm_state.particle_selection.numpy().copy()
            for g in self._inflow_gates:
                b, n = g["base"], g["n"]
                sel[b: b + n] = 1
                g["hold_wp"] = wp.from_numpy(
                    g["hold_np"], dtype=wp.vec3, device=cfg.device
                )
                g["spawn_wp"] = wp.from_numpy(
                    g["spawn_np"], dtype=int, device=cfg.device
                )
            self.mpm.mpm_state.particle_selection = wp.from_numpy(
                sel, dtype=int, device=cfg.device
            )
        # B18.1 — upload per-particle attributes to GPU. Kept alongside
        # warp-mpm's particle_x; their lifetime is tied to MpmSolver.
        if self._colors_np is not None:
            self.attr_color = wp.from_numpy(
                self._colors_np, dtype=wp.vec3, device=cfg.device
            )
        if self._temps_np is not None:
            self.attr_temperature = wp.from_numpy(
                self._temps_np, dtype=float, device=cfg.device
            )
        # Floor as slip plane (uses warp-mpm's built-in patched collider)
        self.mpm.add_surface_collider(
            (0.0, 0.0, cfg.walls.floor_z), (0.0, 0.0, 1.0),
            surface="slip", friction=0.0,
        )
        # Cube colliders — register our SDF box kernel + a Dirichlet_collider
        # struct per cube to carry the box geometry into the kernel.
        # Round-57: optional OBB. The 3 basis-vector fields x_unit /
        # y_unit / direction of Dirichlet_collider carry the box-local
        # axes in world coordinates. For AABB we fill with identity so
        # the kernel reduces to pre-round-57 behaviour bit-exactly.
        for cube in cfg.cubes:
            param = Dirichlet_collider()
            param.point = wp.vec3(*cube.centre)
            param.size = wp.vec3(*cube.half_size)
            if cube.rotation is None:
                ex_c = (1.0, 0.0, 0.0)
                ey_c = (0.0, 1.0, 0.0)
                ez_c = (0.0, 0.0, 1.0)
            else:
                # rotation is row-major: rotation[i] is the i-th ROW.
                # Columns = box-local axes in world (cube.rotation
                # contract). Extract columns by indexing each row.
                ex_c = (cube.rotation[0][0], cube.rotation[1][0],
                        cube.rotation[2][0])
                ey_c = (cube.rotation[0][1], cube.rotation[1][1],
                        cube.rotation[2][1])
                ez_c = (cube.rotation[0][2], cube.rotation[1][2],
                        cube.rotation[2][2])
            param.x_unit = wp.vec3(*ex_c)
            param.y_unit = wp.vec3(*ey_c)
            param.direction = wp.vec3(*ez_c)
            # surface_type=1 is our convention for "slip with top friction"
            param.surface_type = 1
            param.friction = cube.tangential_friction
            param.start_time = 0.0
            param.end_time = 1.0e9
            self.mpm.grid_postprocess.append(k_sdf_box_collide)
            self.mpm.collider_params.append(param)
            self.mpm.modify_bc.append(None)
        # Cube geometry caches for pushback launches (avoid re-allocating).
        # Round-32: snap_eps was hardcoded 0.0001 in the kernel; for
        # sub-millimetre dx (high-res MPM in a small domain) this
        # offset spanned multiple cells, causing visible particle
        # popping near obstacle faces. Now: scale to dx, capped at the
        # pre-round-32 0.0001 so default-resolution behaviour is bit-
        # exact with prior bakes.
        _snap_eps = min(1e-4, cfg.dx() * 0.1)
        # Round-57: pushback kernel now takes a mat33 rotation as the
        # last arg (identity for axis-aligned cubes). Build it from
        # cube.rotation (row-major) or default to identity.
        self._cube_params = []
        for cube in cfg.cubes:
            if cube.rotation is None:
                R = wp.mat33(1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0)
            else:
                r = cube.rotation
                R = wp.mat33(
                    r[0][0], r[0][1], r[0][2],
                    r[1][0], r[1][1], r[1][2],
                    r[2][0], r[2][1], r[2][2],
                )
            self._cube_params.append((
                cube.centre[0], cube.centre[1], cube.centre[2],
                cube.half_size[0], cube.half_size[1], cube.half_size[2],
                _snap_eps,
                R,
            ))
        self._wall_lo = cfg.walls.lo
        self._wall_hi = cfg.walls.hi

        # S2.17.8 outflow drains. Convert output-frame gates to sim steps
        # (same dump_every convention as the inflow gate). Pre-resolve each
        # box to a flat tuple for the per-step kernel launch.
        self._outflow_params = []
        for of in cfg.outflows:
            self._outflow_params.append((
                int(of.frame_start) * cfg.dump_every,
                int(of.frame_end) * cfg.dump_every,
                of.lo[0], of.lo[1], of.lo[2],
                of.hi[0], of.hi[1], of.hi[2],
            ))
        # S2.17.9: one-shot guard so the CFL-saturation warning prints once.
        self._cfl_warned = False
        # S2.17.8: per-particle "drained" flag (1 = removed by an outflow,
        # never to be re-released by the inflow gate). Always allocated; stays
        # all-zero when there are no outflows, so the inflow gate behaves
        # byte-identically to pre-S2.17.8 for non-drain scenes.
        self._despawned = wp.zeros(self.n_particles, dtype=int,
                                   device=cfg.device)

    # ── pipeline -----------------------------------------------------

    def _pre_step(self, step_index: int = 0) -> None:
        cfg = self.cfg
        # S2.17.7 inflow gate: hold not-yet-spawned particles at source,
        # release them with prescribed velocity at their spawn_step.
        for g in self._inflow_gates:
            vx, vy, vz = g["velocity"]
            wp.launch(
                k_inflow_gate, dim=g["n"],
                inputs=[
                    self.mpm.mpm_state,
                    int(step_index), int(g["base"]),
                    g["spawn_wp"], g["hold_wp"],
                    float(vx), float(vy), float(vz),
                    self._despawned,
                ],
                device=cfg.device,
            )
        for cube in self._cube_params:
            wp.launch(k_cube_pushback, dim=self.n_particles,
                      inputs=[self.mpm.mpm_state, *cube],
                      device=cfg.device)
        wp.launch(k_wall_pushback, dim=self.n_particles,
                  inputs=[self.mpm.mpm_state, *self._wall_lo, *self._wall_hi],
                  device=cfg.device)
        if cfg.tap is not None:
            wp.launch(k_tap_terminal_velocity, dim=self.n_particles,
                      inputs=[self.mpm.mpm_state,
                              cfg.tap.lo_x, cfg.tap.hi_x,
                              cfg.tap.lo_y, cfg.tap.hi_y,
                              cfg.tap.z_min, cfg.tap.v_terminal],
                      device=cfg.device)
        if cfg.anti_splash is not None:
            wp.launch(k_anti_splash_vz, dim=self.n_particles,
                      inputs=[self.mpm.mpm_state,
                              cfg.anti_splash.z_threshold,
                              cfg.anti_splash.vz_min,
                              cfg.anti_splash.vz_max],
                      device=cfg.device)

    def _apply_pushback(self) -> None:
        """Cube + wall pushback. Shared by _post_step (per frame) and the
        adaptive-substep inner loop (per sub-step) so a tunnelling particle is
        clamped back inside the domain/colliders before the next P2G."""
        cfg = self.cfg
        for cube in self._cube_params:
            wp.launch(k_cube_pushback, dim=self.n_particles,
                      inputs=[self.mpm.mpm_state, *cube],
                      device=cfg.device)
        wp.launch(k_wall_pushback, dim=self.n_particles,
                  inputs=[self.mpm.mpm_state, *self._wall_lo, *self._wall_hi],
                  device=cfg.device)

    def _post_step(self, step_index: int = 0) -> None:
        cfg = self.cfg
        self._apply_pushback()
        # S2.17.8: drain AFTER physics has moved particles this step, so a
        # particle that crossed into the drain box this frame is caught before
        # the next dump.
        for of in self._outflow_params:
            wp.launch(k_outflow_despawn, dim=self.n_particles,
                      inputs=[self.mpm.mpm_state, int(step_index), *of,
                              self._despawned],
                      device=cfg.device)

    def _cfl_substeps(self) -> tuple[int, bool]:
        """S2.17.9: (n_substeps, saturated) for this frame from the CFL bound.

        dt_sub <= cfl * dx / (c_sound + v_max), so
            N = ceil(dt / dt_sub) = ceil(dt * (c_sound + v_max) / (cfl * dx)).
        c_sound = sqrt(bulk_modulus / density) is the EOS stress wavespeed
        (the term the divergence message blames). v_max is the current peak
        particle speed (advection term) — read once per frame (cheap GPU→CPU
        copy at 10⁴-10⁵ particles, same cost as the existing per-step NaN
        check). Clamped to [1, adaptive_max_substeps].

        ``saturated`` is True when the unclamped CFL demand exceeded the cap —
        i.e. substepping is *under*-resolving and the frame may still diverge.
        The caller surfaces this so the user gets an honest "raise
        cfl_max_substeps / lower dt" signal instead of a silent under-substep
        followed by a cryptic CUDA fault.
        """
        cfg = self.cfg
        import math
        c_sound = math.sqrt(max(cfg.fluid.bulk_modulus, 0.0)
                            / max(cfg.fluid.density, 1e-9))
        vel = self.mpm.mpm_state.particle_v.numpy()
        v_max = float(np.abs(vel).max()) if vel.size else 0.0
        denom = max(cfg.adaptive_cfl * cfg.dx(), 1e-12)
        n_demand = int(math.ceil(cfg.dt * (c_sound + v_max) / denom))
        n_cap = int(cfg.adaptive_max_substeps)
        return max(1, min(n_demand, n_cap)), (n_demand > n_cap)

    def step(self, step_index: int) -> None:
        """Advance the simulation by `dt`.

        Default: one p2g2p(dt) call (byte-identical to pre-FU-023). When
        ``cfg.adaptive_substep`` is set, the frame dt is split into N CFL-sized
        sub-steps so deep-pool / fast-jet scenes stay stable without the user
        lowering bulk_modulus or dt by hand (FU-023).

        CRITICAL (live-found 2026-06-02): the pushback colliders (cube + wall)
        MUST run every sub-step, not just once per frame. A sub-step advects
        particles; if a fast particle tunnels through a cube/wall and isn't
        pushed back before the next sub-step's P2G, it lands outside the grid →
        out-of-bounds node index → CUDA illegal-access (error 700), NOT a clean
        NaN. So the inner loop brackets each p2g2p with pushback. The inflow
        gate + outflow drain stay frame-clocked (they key off step_index, which
        is constant across the frame's sub-steps) and run once in
        _pre_step/_post_step.
        """
        self._pre_step(step_index)
        cfg = self.cfg
        if cfg.adaptive_substep:
            n_sub, saturated = self._cfl_substeps()
            if saturated and not self._cfl_warned:
                self._cfl_warned = True
                print(f"[mpm] WARN: CFL demands more than "
                      f"cfl_max_substeps={cfg.adaptive_max_substeps} sub-steps "
                      f"at frame {step_index // max(1, cfg.dump_every)} — the "
                      f"bake is under-resolving and may still diverge. Raise "
                      f"cfl_max_substeps, lower dt, or lower bulk_modulus.",
                      file=sys.stderr)
            sub_dt = cfg.dt / float(n_sub)
            for _ in range(n_sub):
                self.mpm.p2g2p(step_index, sub_dt, device=cfg.device)
                # Bracket each sub-step with the pushbacks so tunnelling
                # particles are clamped before the next sub-step's P2G.
                self._apply_pushback()
        else:
            self.mpm.p2g2p(step_index, cfg.dt, device=cfg.device)
        self._post_step(step_index)

    # ── output -------------------------------------------------------

    def positions(self) -> np.ndarray:
        return self.mpm.mpm_state.particle_x.numpy()

    def selection(self) -> np.ndarray:
        return self.mpm.mpm_state.particle_selection.numpy()

    def colors(self) -> np.ndarray | None:
        """Per-particle RGB attribute or None if not configured (B18.1)."""
        if self.attr_color is None:
            return None
        return self.attr_color.numpy()

    def temperatures(self) -> np.ndarray | None:
        """Per-particle scalar attribute or None if not configured (B18.1)."""
        if self.attr_temperature is None:
            return None
        return self.attr_temperature.numpy()

    def save_frame_ply(self, out_dir: Path | str, frame_index: int) -> int:
        """Write a points-only PLY for the given frame.

        Returns the count of particles actually written. When colour or
        temperature attributes are configured, also writes
        ``../colors/frame_NNNN.npy`` and ``../temperatures/frame_NNNN.npy``
        sidecars (B18.2). The frame index NNNN in the sidecar filename
        is derived from `frame_index // dump_every` to match the
        per-output-frame naming used by the FLIP cache convention.

        Selection-mask filtering applies uniformly: a particle hidden in
        the PLY (selection==1, e.g. pre-spawn inflow particles) is also
        omitted from the sidecars, so the row indices align.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sel = self.selection()
        n_written = write_points_ply(
            out_dir / f"sim_{frame_index:010d}.ply",
            self.positions(),
            selection_mask=sel,
        )
        if self.attr_color is None and self.attr_temperature is None:
            return n_written
        # Sidecars live next to the PLY directory, not inside it, so the
        # mesher can find them without globbing through point clouds.
        # Frame index is the OUTPUT frame number; commands.py passes
        # `step` (raw sim step) into save_frame_ply, so divide by
        # dump_every to recover the per-output index.
        out_frame = int(frame_index // max(1, self.cfg.dump_every))
        mask = (np.asarray(sel).ravel() == 0)
        sidecar_dir = out_dir.parent
        # Round-24: pre-round-24 we wrote the entire attribute array
        # (whole-particle, never mutated post-init) every frame —
        # an MPM bake with 200k particles × 300 frames × 2 attrs
        # produced ~120 MB of literally-identical data on disk.
        # Now: write each sidecar once (when out_frame == 0 OR file
        # absent — covers the case where someone restarts mid-bake)
        # and mesh-pass reads frame 0 for every frame's lookup.
        #
        # audit-20260610 — PRECISE dedup invariant (§9.11): the
        # frame-0-only write is valid ONLY while the selection mask's
        # row identity is a pure function of its row count. That holds
        # for inflow-only bakes (selection flips 1→0 monotonically:
        # the live count strictly grows, so equal count ⇒ identical
        # mask). An S2.17.8 outflow BREAKS it: a drain flips arbitrary
        # rows 0→1, so a later frame can return to an earlier live
        # COUNT with a different row IDENTITY — the mesher's
        # shape-equality fallback would then apply frame-0 attribute
        # rows to the wrong particles (silently wrong vertex colours /
        # temperatures). So: when any outflow is configured, write the
        # per-frame snapshot unconditionally; keep the count-diff
        # dedup only for drain-free bakes.
        sel_identity_from_count = not self.cfg.outflows
        if self.attr_color is not None:
            colors_dir = sidecar_dir / "colors"
            colors_dir.mkdir(parents=True, exist_ok=True)
            target = colors_dir / f"frame_{out_frame:04d}.npy"
            once = colors_dir / "frame_0000.npy"
            if not once.exists():
                cols = self.attr_color.numpy().astype(np.float32)
                np.save(once, cols[mask])
            if target != once and not target.exists():
                # Per-frame snapshot: unconditional when a drain can
                # mutate selection (see invariant above); otherwise
                # only when the live count differs from frame 0.
                try:
                    cols = self.attr_color.numpy().astype(np.float32)
                    cur = cols[mask]
                    if not sel_identity_from_count:
                        np.save(target, cur)
                    else:
                        base = np.load(once)
                        if base.shape != cur.shape:
                            np.save(target, cur)
                except Exception:
                    pass
        if self.attr_temperature is not None:
            temps_dir = sidecar_dir / "temperatures"
            temps_dir.mkdir(parents=True, exist_ok=True)
            target = temps_dir / f"frame_{out_frame:04d}.npy"
            once = temps_dir / "frame_0000.npy"
            if not once.exists():
                temps = self.attr_temperature.numpy().astype(np.float32)
                np.save(once, temps[mask])
            if target != once and not target.exists():
                try:
                    temps = self.attr_temperature.numpy().astype(np.float32)
                    cur = temps[mask]
                    if not sel_identity_from_count:
                        np.save(target, cur)
                    else:
                        base = np.load(once)
                        if base.shape != cur.shape:
                            np.save(target, cur)
                except Exception:
                    pass
        return n_written

    # ── high-level entry point --------------------------------------

    def run(self, out_dir: Path | str, progress: bool = False) -> Path:
        """Run the full bake. Returns the output directory path on
        normal completion. Raises :class:`MpmDivergenceError` if NaN
        appears in particle positions — meaning the constitutive model
        blew up (typically dt × sqrt(bulk_modulus/rho) > 1, i.e.
        violated the CFL bound on stress wavespeed). Caller catches +
        decides salvage strategy.

        Round-20 changes vs pre-round-20:
          - NaN check runs every step (was: progress=True AND k%200==0
            → never fired on bakes shorter than 200 frames).
          - Detection raises typed exception instead of silent
            ``print + break`` (was: caller saw rc=0 with truncated cache).
          - Periodic progress prints stay gated on ``progress=True``
            and the same 200-step cadence (separated concern).
        """
        cfg = self.cfg
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.save_frame_ply(out_dir, 0)
        for k in range(1, cfg.n_frames + 1):
            self.step(k)
            if k % cfg.dump_every == 0:
                self.save_frame_ply(out_dir, k)
            # NaN check — every step. The positions() call is a
            # GPU→CPU copy; cheap enough at typical particle counts
            # (~10⁴-10⁵) and the failure-mode protection is critical.
            # If profiling later shows this is a hot-path cost,
            # demote to every-N-steps with N tuned by total step count.
            pos = self.positions()
            if pos.size and np.any(np.isnan(pos)):
                # Compute how many full dump_every frames actually
                # landed on disk so caller can write a truthful
                # cache.json with truncated_at_frame.
                last_dumped = (k // cfg.dump_every) * cfg.dump_every
                raise MpmDivergenceError(
                    step=k,
                    last_dumped_frame=last_dumped,
                    requested_frames=cfg.n_frames,
                    detail=f"NaN in particle positions at step {k} "
                           f"(last successfully dumped frame: {last_dumped} "
                           f"/ {cfg.n_frames} requested)",
                )
            if progress and k % cfg.dump_every == 0:
                # Round-62: emit progress on the DUMP cadence in OUTPUT-
                # FRAME units ("frame N/M"), not every 200 raw steps in
                # "step" units. Two pre-round-62 bugs starved the Blender
                # addon's modal progress bar: (1) the line said "step", but
                # the addon parser only matches "frame N/M" → the counter
                # never advanced for MPM bakes; (2) the 200-step cadence
                # never fired on bakes shorter than 200 steps. Now it ticks
                # once per written frame, so the UI shows real progress
                # (the "no bake indicator / nothing happens" complaint).
                out_frame = k // cfg.dump_every
                total_frames = max(1, cfg.n_frames // cfg.dump_every)
                print(f"  [MPM] frame {out_frame}/{total_frames}  "
                      f"z_min={float(pos[:, 2].min()):.3f}  "
                      f"z_max={float(pos[:, 2].max()):.3f}", flush=True)
        return out_dir

    def __del__(self):
        # Round-22: catch-all. During Python interpreter shutdown the
        # `os` module's attributes (or `os` itself in extreme cases)
        # can become None — `os.unlink` then raises TypeError, not
        # the AttributeError/OSError family we used to catch. Broad
        # `Exception` is justified: this is best-effort cleanup of a
        # tempfile path; never propagate from a destructor.
        try:
            os.unlink(self._tmp_h5)
        except Exception:
            pass


# ─── helpers ───────────────────────────────────────────────────────────

def make_column(
    xy_centre: tuple[float, float],
    xy_half_size: float,
    z_lo: float,
    z_hi: float,
    n_xy: int,
    n_z: int,
) -> np.ndarray:
    """Build a regular column of particles for use as MpmConfig.initial_column."""
    xs = np.linspace(xy_centre[0] - xy_half_size,
                     xy_centre[0] + xy_half_size, n_xy)
    ys = np.linspace(xy_centre[1] - xy_half_size,
                     xy_centre[1] + xy_half_size, n_xy)
    zs = np.linspace(z_lo, z_hi, n_z)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    return pts.astype(np.float32)
