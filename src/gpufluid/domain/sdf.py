"""[Layer D4] Domain marker injection from a precomputed SDF field.

The analytic SDF primitives + cell-centre grid moved to
``primitives/sdf.py`` (layer G1) on 2026-05-17 as part of F3.6.A1 —
see ``docs/DESIGN.md §3.2.4.2``. Only the solver-coupled
``mark_solid_from_sdf`` host helper remains here: it mutates a
solver marker array, which is genuine D4 responsibility.

For convenience and to keep callers' diffs minimal, the math
primitives are re-exported from this module — but new code should
import them directly from ``gpufluid.primitives.sdf``.
"""
from __future__ import annotations
import numpy as np
from ..blocks import block

# Re-exports (transitional — new imports should go to primitives.sdf).
from ..primitives.sdf import (  # noqa: F401
    cell_centers,
    sdf_sphere,
    sdf_box,
    sdf_cylinder_y,
    sdf_plane,
    sdf_union,
)


@block("D4.4", "Apply SDF as solid marker (marker=2 where sdf<=padding)")
def mark_solid_from_sdf(marker: np.ndarray, sdf: np.ndarray, padding: float = 0.0) -> np.ndarray:
    """In-place set marker=2 where sdf <= padding. Returns marker."""
    marker[sdf <= padding] = 2
    return marker
