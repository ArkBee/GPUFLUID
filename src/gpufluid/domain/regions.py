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


# [BLK D4.7]
@dataclass
class OutflowBox:
    """A box region that deletes any particle inside it."""
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    frame_start: int = 0
    frame_end: int = 10_000


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
