"""[Round-17] Pure-Python warnings for scene sources outside the unit cube.

Extracted from `bake.collect_scene` per senior code smell #2. Pure,
bpy-free, unit-testable. Caller (collect_scene) accumulates the
warning strings and reports them via `self.report({"WARNING"}, ...)`.

The original inline `_out_of_domain` produced two flavours of warning
depending on severity:

1. **fully-outside** — any axis has `hi < 0` or `lo > 1`. Particles
   will be wall-clamped to the nearest face; the source effectively
   doesn't seed.
2. **wall-clip** — any axis has `lo < eps_norm` or `hi > 1 - eps_norm`.
   The seeded region partly extends past the eps margin; visible
   artefacts (sticking, leaking) on first frame.

Returns `None` when no warning needed; a string otherwise. Caller
appends to a list.
"""
from __future__ import annotations

from typing import Optional, Sequence


def out_of_domain_warning(
    name: str,
    lo: Sequence[float],
    hi: Sequence[float],
    kind: str,
    eps_norm: float,
) -> Optional[str]:
    """Return a warning string if `[lo, hi]` (already in normalised
    [0,1]³ coords) is outside or touching the domain wall.

    `name` and `kind` are interpolated into the message — typically the
    source's object name and "Fluid source" / "Inflow" / etc.

    Returns the first-triggered warning (fully-outside ranks above
    wall-clip per axis order x → y → z). Caller collects multiple
    by re-calling per source.
    """
    for axis, axis_name in enumerate("xyz"):
        if hi[axis] < 0.0 or lo[axis] > 1.0:
            return (
                f"{kind} '{name}' is fully outside the Domain on "
                f"{axis_name} (normalised range "
                f"[{lo[axis]:.2f},{hi[axis]:.2f}] vs [0,1]); "
                "particles will be wall-clamped to the nearest face. "
                "Move it inside the Domain bounds.")
        if lo[axis] < eps_norm or hi[axis] > 1.0 - eps_norm:
            return (
                f"{kind} '{name}' touches/clips the Domain wall on "
                f"{axis_name} (normalised "
                f"[{lo[axis]:.2f},{hi[axis]:.2f}]). "
                "Solver will wall-clamp those particles — move the "
                "source deeper inside to avoid visual artefacts.")
    return None
