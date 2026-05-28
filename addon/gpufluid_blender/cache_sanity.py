"""[Round-61] Pure-Python post-bake / post-render output sanity.

bpy-free and unit-testable. This module is the SINGLE SOURCE of the
"how many frames did we actually get vs. how many did we ask for?"
decision, shared by both the sync and modal paths of OT_bake and
OT_render so the two can never drift again (lesson §9.6 — mirror-
operator drift).

Root cause it closes (live-found 2026-05-28): a bake could finish,
auto-attach, and report "gpufluid bake complete" while producing ZERO
usable mesh frames — the user saw an empty viewport with a green
success message. Three silent spots converged on the same disease:

  1. The sync truncation check used ``0 < actual < expected`` — the
     ``0 < actual`` clause EXCLUDED the actual==0 case, i.e. the single
     worst outcome was the one it ignored.
  2. The modal path had no frame-count check at all.
  3. attach_cache reported ``n`` frames as plain INFO regardless of
     ``n == 0``.

The classifier escalates the zero case to ERROR (nothing baked / cache
torn) and keeps the existing WARNING for partial truncation.
"""
from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

# (level, message). level is "ERROR" / "WARNING" / None. message is "" when
# level is None. level maps directly to the set passed to operator.report().
Verdict = Tuple[Optional[str], str]


def count_mesh_frames(cache_dir: str) -> int:
    """Number of baked surface frames in ``<cache_dir>/mesh/frame_*.ply``.

    Counts the (legitimately empty) ``frame_0000.ply`` too — the MPM
    initial state has 0 vertices before particles settle into a surface.
    That is fine for shortfall detection: in a healthy bake the file
    count is >= the requested ``frames``, so the empty frame 0 can never
    trip a false "truncated" verdict.
    """
    mesh_dir = os.path.join(cache_dir, "mesh")
    if not os.path.isdir(mesh_dir):
        return 0
    return len(glob.glob(os.path.join(mesh_dir, "frame_*.ply")))


def count_pngs(out_dir: str) -> int:
    """Number of ``*.png`` files written into ``out_dir`` (render output)."""
    if not os.path.isdir(out_dir):
        return 0
    return len(glob.glob(os.path.join(out_dir, "*.png")))


def _level(actual: int, expected: int) -> Optional[str]:
    """ERROR when nothing was produced, WARNING when fewer than asked,
    None when complete (or when ``expected`` is unknown and we got >0)."""
    if actual == 0:
        return "ERROR"
    if expected > 0 and actual < expected:
        return "WARNING"
    return None


def bake_frame_sanity(actual: int, expected: int) -> Verdict:
    """Classify a bake's mesh-frame output. ``expected`` is the requested
    ``simulation.frames`` (0 when unknown)."""
    level = _level(actual, expected)
    if level == "ERROR":
        exp = f" (requested {expected})" if expected > 0 else ""
        return ("ERROR",
                f"gpufluid bake produced 0 mesh frames{exp} — the solver "
                f"wrote no output. The CLI most likely errored (check the "
                f"system console) or the cache was cleared mid-bake. "
                f"Nothing usable was attached.")
    if level == "WARNING":
        return ("WARNING",
                f"gpufluid bake produced {actual}/{expected} frames — the "
                f"CLI exited cleanly but truncated (likely solver "
                f"divergence/OOM or early-stop). Cache attached as-is.")
    return (None, "")


def render_output_sanity(actual: int, expected: int) -> Verdict:
    """Classify a render's PNG output. ``expected`` is the frame-range
    length (0 when the caller doesn't know it — only the 0-output case is
    flagged then)."""
    level = _level(actual, expected)
    if level == "ERROR":
        exp = f" (expected {expected})" if expected > 0 else ""
        return ("ERROR",
                f"gpufluid render produced 0 PNG frames{exp} — Blender "
                f"wrote no images. Check the system console: the output "
                f"path may be unwritable, the frame range empty, or the "
                f"scene camera/engine misconfigured.")
    if level == "WARNING":
        return ("WARNING",
                f"gpufluid render produced {actual}/{expected} PNG frames "
                f"— fewer than the frame range. The render may have been "
                f"interrupted; check the system console.")
    return (None, "")
