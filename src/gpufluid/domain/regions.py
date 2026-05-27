"""[Layer D4 / BLK D4.7] Inflow / outflow regions.

Inflows continuously emit particles inside a box region at a given rate.
Outflows delete particles inside their box.

Implementation note (v0.5):
    Particle compaction happens host-side once per frame in
    ``FlipSolver3D.prepare_frame``. Positions/velocities are read, filtered
    against outflow boxes, then new emitted particles are appended, then
    the combined arrays are uploaded back to GPU as fresh Warp arrays.
    Transfer cost: ~12 B/particle round-trip; ~1 ms per 100k particles.
    For >1 M particles, switch to GPU stream compaction (planned).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

from ..blocks import block


# [BLK D4.7]
@dataclass
class InflowBox:
    """A box region that emits particles every frame."""
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rate_per_sec: float = 5000.0   # particles per second
    frame_start: int = 0
    frame_end: int = 10_000

    def emit(self, frame_idx: int, frame_dt: float, rng: np.random.Generator) -> Optional[np.ndarray]:
        """Return (N,3) positions or None if not emitting this frame."""
        if frame_idx < self.frame_start or frame_idx > self.frame_end:
            return None
        n = int(round(self.rate_per_sec * frame_dt))
        if n <= 0:
            return None
        lo = np.asarray(self.lo, dtype=np.float32)
        hi = np.asarray(self.hi, dtype=np.float32)
        return lo + (hi - lo) * rng.random((n, 3)).astype(np.float32)

    def publish_for_frame(self, queue, frame_idx: int, frame_dt: float,
                          rng: np.random.Generator) -> None:
        """F3.6.C1 dual-path: instead of returning (pos, vel), push a
        FluidEmitEvent onto `queue`.

        Semantically identical to a single iteration of the legacy
        `apply_inflows` loop. When this box is not emitting (frame
        outside [start, end] or rate=0), nothing is pushed.
        """
        from ..primitives.frame_events import FluidEmitEvent
        pos = self.emit(frame_idx, frame_dt, rng)
        if pos is None or len(pos) == 0:
            return
        vel = np.broadcast_to(np.asarray(self.velocity, dtype=np.float32),
                              (len(pos), 3)).copy()
        queue.push_emit(FluidEmitEvent(positions=pos, velocities=vel))


# [BLK D4.7]
@dataclass
class OutflowBox:
    """A box region that deletes any particle inside it."""
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    frame_start: int = 0
    frame_end: int = 10_000

    def __post_init__(self):
        # Round-44: mirror of round-33 MPM-inflow-keyframe `lo > hi`
        # fix. Pre-round-44 a reversed AABB silently produced a
        # zero-volume box → `apply_outflows` containment test failed
        # for every particle → outflow became silent no-op. User saw
        # particles accumulating where they expected drain. Now: loud
        # raise upfront with the offending axis named.
        for axis, (lo_v, hi_v) in enumerate(zip(self.lo, self.hi)):
            if hi_v < lo_v:
                axis_name = "xyz"[axis]
                raise ValueError(
                    f"OutflowBox: hi[{axis_name}]={hi_v} < lo[{axis_name}]"
                    f"={lo_v}. Outflow box AABB must have hi >= lo on "
                    f"every axis (got lo={self.lo}, hi={self.hi}).")

    def publish_for_frame(self, queue, frame_idx: int) -> None:
        """F3.6.C1 dual-path: push a FluidOutflowEvent describing the
        cull bbox if this outflow is active for `frame_idx`. The solver
        applies the cull on drain; this method does NOT touch particle
        arrays itself.
        """
        if frame_idx < self.frame_start or frame_idx > self.frame_end:
            return
        from ..primitives.frame_events import FluidOutflowEvent
        queue.push_outflow(FluidOutflowEvent(
            lo=tuple(float(x) for x in self.lo),
            hi=tuple(float(x) for x in self.hi),
            inside=True,
        ))


# [BLK D4.7]
@block("D4.7", "Filter particles: drop those inside any outflow box")
def apply_outflows(pos: np.ndarray, vel: np.ndarray,
                   outflows: List[OutflowBox], frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    if not outflows or len(pos) == 0:
        return pos, vel
    keep = np.ones(len(pos), dtype=bool)
    for o in outflows:
        if frame_idx < o.frame_start or frame_idx > o.frame_end:
            continue
        lo = np.asarray(o.lo, dtype=np.float32)
        hi = np.asarray(o.hi, dtype=np.float32)
        inside = np.all((pos >= lo) & (pos <= hi), axis=1)
        keep &= ~inside
    return pos[keep], vel[keep]


# [BLK D4.7]
@block("D4.7", "Emit particles from all inflow boxes for this frame")
def apply_inflows(inflows: List[InflowBox], frame_idx: int, frame_dt: float,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if not inflows:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))
    parts_pos: list = []
    parts_vel: list = []
    for inf in inflows:
        p = inf.emit(frame_idx, frame_dt, rng)
        if p is None or len(p) == 0:
            continue
        parts_pos.append(p)
        v = np.broadcast_to(np.asarray(inf.velocity, dtype=np.float32),
                            (len(p), 3)).copy()
        parts_vel.append(v)
    if not parts_pos:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))
    return np.concatenate(parts_pos, axis=0), np.concatenate(parts_vel, axis=0)
