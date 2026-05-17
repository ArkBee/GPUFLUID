"""[Layer G1] Motion spec + per-frame centre evaluator (pure utility).

Relocated 2026-05-17 from ``domain/animation.py`` as part of F3.6.B
(see ``docs/DESIGN.md §3.2.4.2``). These types are stateless data
transformations — given a base position and a motion spec, return a
world-space position at a frame index. No coupling to FlipSolver3D
or scene config, so they belong in G1.

Block IDs (after F3.6.B):

  G1.15  LinearMotion + KeyframeMotion + Motion union  (was D4.6 part)
  G1.16  evaluate_center                                (was D4.6 part)

Pickle/checkpoint safety: these types are constructed FRESH per bake
from ``MotionCfg`` in ``cli/config.py``; they are not serialised into
``.npz`` checkpoints (F3.5 stores only numpy arrays + scalars). So
the relocation has no on-disk migration impact.
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

block("G1.15", "Motion spec dataclasses (LinearMotion + KeyframeMotion union)")(LinearMotion)


@block("G1.16", "Resolve animated centre at a given frame (pure host transform)")
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
