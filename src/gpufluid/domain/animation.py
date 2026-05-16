"""[Layer D4 / BLK D4.6] Animated obstacles.

An animated obstacle is an analytic SDF primitive whose parameters change
per frame. v0.5 supports two motion specs:

* **linear** — constant world-space velocity vector applied to ``center``.
* **keyframes** — list of ``(frame, center)`` pairs, linearly interpolated.

Mesh-based animated obstacles are supported via per-frame ``mesh_to_sdf``
(D4.3) calls. This is CPU-bound (trimesh + rtree) and runs once per
frame at solver scale — fine for 64³ grids, slow at 256³+. Future:
GPU triangle SDF kernel (D4.6.GPU) for sub-millisecond per-frame builds.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Optional, Literal, Union

from ..blocks import block


@dataclass
class LinearMotion:
    kind: Literal["linear"] = "linear"
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fps: int = 24  # for converting frame index to time


@dataclass
class KeyframeMotion:
    kind: Literal["keyframes"] = "keyframes"
    keyframes: List[Tuple[int, Tuple[float, float, float]]] = field(default_factory=list)
    # (frame_idx, world_position) — linearly interpolated; held outside range


Motion = Union[LinearMotion, KeyframeMotion]


# [BLK D4.6]
@block("D4.6", "Resolve animated centre at a given frame")
def evaluate_center(base_center: Sequence[float], motion: Optional[Motion],
                    frame_idx: int) -> np.ndarray:
    """Return world-space centre of an animated obstacle at the given frame."""
    base = np.asarray(base_center, dtype=np.float32)
    if motion is None:
        return base
    if isinstance(motion, LinearMotion):
        t = frame_idx / float(motion.fps)
        return base + t * np.asarray(motion.velocity, dtype=np.float32)
    if isinstance(motion, KeyframeMotion):
        kfs = sorted(motion.keyframes, key=lambda k: k[0])
        if not kfs:
            return base
        if frame_idx <= kfs[0][0]:
            return np.asarray(kfs[0][1], dtype=np.float32)
        if frame_idx >= kfs[-1][0]:
            return np.asarray(kfs[-1][1], dtype=np.float32)
        for (fa, pa), (fb, pb) in zip(kfs, kfs[1:]):
            if fa <= frame_idx <= fb:
                if fb == fa:
                    return np.asarray(pa, dtype=np.float32)
                t = (frame_idx - fa) / float(fb - fa)
                return np.asarray(pa, dtype=np.float32) * (1 - t) + np.asarray(pb, dtype=np.float32) * t
        return base
    raise ValueError(f"unknown motion kind: {motion!r}")
