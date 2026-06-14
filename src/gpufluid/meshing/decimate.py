"""[Layer M5 / BLK M5.6] Mesh decimation (quadric-error simplification).

Reduces triangle count of marching-cubes output. Smaller cache files,
faster Blender playback, smoother appearance from topology cleanup.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

from ..blocks import block


# [BLK M5.6]
@block("M5.6", "Quadric-error mesh decimation via pyfqmr")
def decimate_mesh(verts: np.ndarray, faces: np.ndarray,
                  target_ratio: float = 0.5,
                  aggressiveness: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Return decimated (verts, faces). ``target_ratio`` in (0, 1] keeps
    that fraction of the original faces."""
    # audit-2026-06-15r7 #4: a non-positive ratio is a no-op, NOT "decimate to
    # nothing". Without this, target_ratio<=0 fell through and
    # target_count=max(4, int(len*ratio)) collapsed the WHOLE frame to a 4-face
    # degenerate blob (the call site only gates on `< 1.0`, so 0.0/-x reach
    # here). Validation also lives at the config parse layer; this guard keeps
    # the library function safe for any direct caller.
    if (verts is None or faces is None or len(faces) == 0
            or target_ratio >= 1.0 or target_ratio <= 0.0):
        return verts, faces
    target_count = max(4, int(len(faces) * float(target_ratio)))
    import pyfqmr
    s = pyfqmr.Simplify()
    s.setMesh(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32))
    s.simplify_mesh(target_count=target_count,
                    aggressiveness=aggressiveness,
                    preserve_border=True, verbose=False)
    v2, f2, _ = s.getMesh()
    return v2.astype(np.float32), f2.astype(np.int32)
