"""[Layer F3 / BLK W7.x] Whitewater (foam / spray / bubble).

v0.7 upgrade: differentiates three classes by density-grid lookup at
emit time, then runs *per-class* dynamics (different gravity, drag,
lifetime, and buoyancy/pop rules). Class assignments:

* ``kind == 0`` **foam** — near surface (density ~0.5). Buoyancy
  ~= gravity (floats), high drag (sticks to surface flow), long life.
* ``kind == 1`` **spray** — above surface (density < 0.2). Full gravity,
  low drag, short life. Ballistic droplets that fall back.
* ``kind == 2`` **bubble** — submerged (density > 0.8). Rises at
  ~0.3·|g| upward, pops on reaching surface (density drops below
  ``pop_threshold``).

If no density grid is provided at emit time the system falls back to
all-foam (the v0.6 behaviour), so existing scenes keep working.

Block layout:
* ``W7.1``  state container (kind/pos/vel/age arrays in lockstep)
* ``W7.2``  emit hook (now classifies by density sample)
* ``W7.3``  legacy single-class advect (kept for back-compat)
* ``W7.4``  ``classify_kinds`` — density-based classifier
* ``W7.5``  ``step_per_class`` — per-class advection rules
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from ..blocks import block


# Public kind constants — also used by render/cache layers.
KIND_FOAM   = 0
KIND_SPRAY  = 1
KIND_BUBBLE = 2


@dataclass
class WhitewaterConfig:
    speed_threshold: float = 4.0     # m/s — emit when |vel| exceeds this
    emit_per_frame_max: int = 4000   # cap new emissions per frame
    total_cap: int = 80000           # absolute particle cap
    # Per-class lifetimes
    lifetime_foam: float = 3.0
    lifetime_spray: float = 1.0
    lifetime_bubble: float = 2.0
    # Per-class drag (1/s, exponential)
    drag_foam: float = 2.5           # high — foam sticks
    drag_spray: float = 0.4          # low — spray flies
    drag_bubble: float = 1.5
    # Per-class effective gravity (m/s²). Negative = downward.
    gravity_foam: float = -0.5       # near-zero net vertical force
    gravity_spray: float = -9.81     # full gravity
    gravity_bubble: float = +3.0     # buoyant rise
    # Classification thresholds against the smoothed density field [0..1]:
    density_above_surface: float = 0.2   # below this → spray
    density_below_surface: float = 0.8   # above this → bubble
    pop_threshold: float = 0.4           # bubble pops when it reaches < this
    spawn_offset: float = 0.0
    # Legacy single-class fields (kept for v0.6 back-compat callers)
    drag: float = 0.6
    gravity: float = -9.81
    lifetime_sec: float = 1.5


def _world_to_cell(pos: np.ndarray, dx: float, shape):
    """World positions -> CELL-CENTRED grid indices, matching the density grid.

    audit-2026-06-15r9 #3: extractor.dens (mesh_method='trilinear') is filled by
    k_density_scatter with the cell-centred convention `i = floor(p/dx - 0.5)`
    (index i <-> world (i+0.5)*dx; same as the MC vertex fix audit-2026-06-14
    #12). The classifier + bubble-pop read this grid, so they MUST use the same
    `- 0.5`. The old `floor(p/dx)` was node-centred — a systematic +half-cell
    sampling bias on every axis that flips foam/spray/bubble thresholds in the
    steep surface band. One helper so the two read-sites can't drift again (§9.6).
    """
    nx, ny, nz = shape
    i = np.clip(np.floor(pos[:, 0] / dx - 0.5).astype(np.int32), 0, nx - 1)
    j = np.clip(np.floor(pos[:, 1] / dx - 0.5).astype(np.int32), 0, ny - 1)
    k = np.clip(np.floor(pos[:, 2] / dx - 0.5).astype(np.int32), 0, nz - 1)
    return i, j, k


def calibrate_density_field(dens: np.ndarray) -> np.ndarray:
    """Normalise a particle-count density grid to ~[0,1] for the classifier.

    audit-2026-06-21 (reviewer-whitewater H2): the M5 trilinear density grid
    (MeshExtractor.dens) holds the raw scattered particle COUNT per cell —
    ``k_density_scatter`` deposits weight 1 per particle, so a fully-packed cell
    reads ≈ ppc (≈8 by default), NOT the [0,1] field the classifier's 0.2/0.8
    thresholds expect. Passed raw, every submerged sample read > 0.8 → everything
    was misclassified BUBBLE and foam vanished (a prior comment wrongly called
    the trilinear field "calibrated [0,1]").

    Calibrate by the BULK density (90th percentile of occupied cells → 1.0). This
    is self-calibrating: independent of ppc, the number of sources, and the
    trilinear+box-blur smoothing, so submerged bulk → ~1 (bubble), the surface
    band → 0.2–0.8 (foam), and isolated cells → < 0.2 (spray), as intended.
    Returns the input unchanged if it is empty/all-zero.
    """
    if dens is None:
        return dens
    occupied = dens[dens > 1e-6]
    if occupied.size == 0:
        return dens
    ref = float(np.percentile(occupied, 90))
    if ref <= 1e-6:
        return dens
    return dens / ref


# [BLK W7.1]
@block("W7.1", "Whitewater state container + per-frame update (foam/spray/bubble)")
class WhitewaterSystem:
    """Manages flat arrays of whitewater particles. Each particle carries
    a ``kind`` (foam=0, spray=1, bubble=2) and the per-frame ``step()``
    applies class-specific gravity / drag / pop rules."""

    def __init__(self, cfg: Optional[WhitewaterConfig] = None, seed: int = 0):
        self.cfg = cfg or WhitewaterConfig()
        self.pos = np.zeros((0, 3), dtype=np.float32)
        self.vel = np.zeros((0, 3), dtype=np.float32)
        self.age = np.zeros((0,), dtype=np.float32)
        self.kind = np.zeros((0,), dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    # ----------------------------- W7.4 — density-based classifier
    @block("W7.4", "Classify whitewater kind by density-grid lookup at emit")
    def classify_kinds(self, pos: np.ndarray, density: Optional[np.ndarray],
                       dx: float, vel: Optional[np.ndarray] = None,
                       lookahead_dt: float = 0.02) -> np.ndarray:
        """Return an int32 array of kinds (length == len(pos)).

        Classification uses the *projected* position ``pos + vel·lookahead_dt``
        when velocity is given. This matters for spray: a fluid particle is
        always *inside* the fluid (density high) at emit time, but a particle
        flying upward will be in air in 0.1 s. Without the lookahead the
        classifier would label everything foam or bubble and spray would never
        appear in the output. lookahead_dt=0.1 s is ~2 frames at 24 fps.

        If ``density`` is None (no grid supplied), everything is foam — this
        is the v0.6 behaviour and gives a sane visible result for scenes
        that don't expose a density field."""
        n = len(pos)
        if density is None or n == 0:
            return np.zeros(n, dtype=np.int32)
        sample_pos = pos if vel is None else pos + vel * float(lookahead_dt)
        i, j, k = _world_to_cell(sample_pos, dx, density.shape)
        d_at = density[i, j, k]
        kind = np.full(n, KIND_FOAM, dtype=np.int32)
        kind[d_at < self.cfg.density_above_surface] = KIND_SPRAY
        kind[d_at > self.cfg.density_below_surface] = KIND_BUBBLE
        return kind

    # ----------------------------- W7.2 — emit from a fluid particle batch
    @block("W7.2", "Emit new whitewater from fast-moving fluid particles")
    def emit_from_fluid(self, fluid_pos: np.ndarray, fluid_vel: np.ndarray,
                        density: Optional[np.ndarray] = None,
                        dx: Optional[float] = None,
                        potential: Optional[np.ndarray] = None):
        """Emit new whitewater particles seeded from fast or turbulent fluid.

        Selector modes:
          * ``potential=None`` (legacy / v0.7): emit any particle whose speed
            exceeds ``cfg.speed_threshold``, uniform random subsample to the
            emit cap.
          * ``potential`` provided (B3.1+, expects values in [0, 1]): sample
            with probability proportional to potential, so genuine turbulence
            pockets dominate and laminar bulk fluid contributes ~0 — this is
            the Ihmsen 2012 §3 approach. Particles with potential==0 are
            never selected. The speed-threshold gate still applies, so a
            stationary fluid never emits regardless of potential.

        Optionally pass ``density`` (the M5 smoothed indicator grid) + ``dx``
        to classify emissions into foam/spray/bubble at spawn time.
        """
        if len(fluid_pos) == 0:
            return
        speeds = np.linalg.norm(fluid_vel, axis=1)
        mask = speeds > self.cfg.speed_threshold
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        cap = self.cfg.emit_per_frame_max
        if potential is not None:
            if potential.shape != speeds.shape:
                raise ValueError(
                    f"potential must be shape ({speeds.shape[0]},), got {potential.shape}")
            # Add a tiny floor so even "perfectly laminar" particles get
            # a non-zero (but much lower) emit probability. Without this a
            # waterfall column — laminar relative to itself — would emit
            # almost no whitewater because every neighbour has the same
            # velocity vector. Floor magnitude (1e-3) is small enough that
            # turbulent zones still dominate by ~3 orders of magnitude.
            w = np.clip(potential[idx].astype(np.float64), 0.0, None) + 1.0e-3
            p = w / w.sum()
            n_pick = min(cap, len(idx))
            idx = self.rng.choice(idx, size=n_pick, replace=False, p=p)
        elif len(idx) > cap:
            idx = self.rng.choice(idx, size=cap, replace=False)
        new_pos = fluid_pos[idx].copy()
        if self.cfg.spawn_offset != 0.0:
            # Round-23: was index 1 (Y) — entire project is Z-up
            # (MpmDomainWalls.floor_z, scene.simulation.gravity → Z,
            # FLIP solver Z-up). Pre-round-23 every whitewater particle
            # spawned offset on the WRONG axis, then advected with
            # gravity also on the wrong axis, then exit-tested against
            # the wrong dom face — bubbles "rose" horizontally and never
            # reached the surface, spray fell sideways.
            new_pos[:, 2] += self.cfg.spawn_offset
        kick = self.rng.standard_normal(new_pos.shape).astype(np.float32) * 0.2
        # Round-23: upward kick component must hit Z (vertical), not Y.
        kick[:, 2] = np.abs(kick[:, 2]) * 0.5
        new_vel = fluid_vel[idx].copy() + kick
        new_age = np.zeros(len(idx), dtype=np.float32)
        new_kind = self.classify_kinds(new_pos, density,
                                       dx if dx is not None else 1.0,
                                       vel=new_vel)
        # cap (drop oldest first)
        combined = self.n + len(idx)
        if combined > self.cfg.total_cap:
            keep = self.cfg.total_cap - len(idx)
            if keep <= 0:
                self.pos = new_pos[: self.cfg.total_cap]
                self.vel = new_vel[: self.cfg.total_cap]
                self.age = new_age[: self.cfg.total_cap]
                self.kind = new_kind[: self.cfg.total_cap]
                return
            self.pos = self.pos[-keep:]
            self.vel = self.vel[-keep:]
            self.age = self.age[-keep:]
            self.kind = self.kind[-keep:]
        self.pos = np.concatenate([self.pos, new_pos], axis=0)
        self.vel = np.concatenate([self.vel, new_vel], axis=0)
        self.age = np.concatenate([self.age, new_age], axis=0)
        self.kind = np.concatenate([self.kind, new_kind], axis=0)

    # ----------------------------- W7.5 — per-class advect
    @block("W7.5", "Per-class advect: foam/spray/bubble with distinct rules")
    def step(self, dt: float, dom: Tuple[float, float, float],
             density: Optional[np.ndarray] = None,
             dx: Optional[float] = None):
        """Per-class dynamics. If ``density`` + ``dx`` provided, bubbles pop
        when their host cell density drops below ``cfg.pop_threshold``
        (i.e. they reached / breached the surface)."""
        if self.n == 0:
            return
        # Per-class gravity & drag vectors lined up to particle order.
        grav = np.choose(self.kind, [
            self.cfg.gravity_foam, self.cfg.gravity_spray, self.cfg.gravity_bubble,
        ]).astype(np.float32)
        drag = np.choose(self.kind, [
            self.cfg.drag_foam, self.cfg.drag_spray, self.cfg.drag_bubble,
        ]).astype(np.float32)
        lifetime = np.choose(self.kind, [
            self.cfg.lifetime_foam, self.cfg.lifetime_spray, self.cfg.lifetime_bubble,
        ]).astype(np.float32)
        # exponential drag per-axis (uniform per particle)
        damp = np.maximum(0.0, 1.0 - drag * dt)[:, None]
        self.vel *= damp
        # Round-23: gravity acts on Z (vertical), not Y. See spawn-offset
        # rationale above. This is THE bug that made the entire whitewater
        # system look broken: bubbles couldn't rise to the surface to pop,
        # spray flew sideways, foam drifted in a horizontal "river".
        self.vel[:, 2] += grav * dt
        self.pos += self.vel * dt
        self.age += dt
        # Domain clamp / kill
        in_dom = (
            (self.pos[:, 0] > 0.0) & (self.pos[:, 0] < dom[0]) &
            (self.pos[:, 1] > 0.0) & (self.pos[:, 1] < dom[1]) &
            (self.pos[:, 2] > 0.0) & (self.pos[:, 2] < dom[2])
        )
        keep = (self.age < lifetime) & in_dom
        # Runtime re-classification + bubble pop, using the current density.
        # Particles change class as they cross the surface — spray that lands
        # in water becomes foam, foam that gets pushed deep becomes a bubble,
        # bubbles that rise into air pop. Without this, classes are frozen at
        # emit and the scene shows almost only one class (whichever the
        # emit-projection happened to favour).
        if density is not None and dx is not None:
            i, j, k = _world_to_cell(self.pos, dx, density.shape)
            d_at = density[i, j, k]
            # spray landed in water → foam (density crossed above_surface)
            became_foam = (self.kind == KIND_SPRAY) & (d_at > self.cfg.density_above_surface)
            self.kind = np.where(became_foam, KIND_FOAM, self.kind).astype(np.int32)
            # foam pushed deeper → bubble (density crossed below_surface)
            became_bubble = (self.kind == KIND_FOAM) & (d_at > self.cfg.density_below_surface)
            self.kind = np.where(became_bubble, KIND_BUBBLE, self.kind).astype(np.int32)
            # bubble reached the surface → pop (kill)
            bubble_pop = (self.kind == KIND_BUBBLE) & (d_at < self.cfg.pop_threshold)
            keep &= ~bubble_pop
        if not keep.all():
            self.pos = self.pos[keep]
            self.vel = self.vel[keep]
            self.age = self.age[keep]
            self.kind = self.kind[keep]
