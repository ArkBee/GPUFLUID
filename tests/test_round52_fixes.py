"""Round-52 — live-test finding: BBOX obstacles silently drop rotation.

Pre-round-52, `obstacle_type=BBOX` + non-zero `rotation_euler` produced
an AABB of the rotated corners (BIGGER than the original cube) and
silently dropped the orientation — MPM saw no tilt. User observed:
"system doesn't take object tilt into account" after assembling a
flow-around-tilted-cubes scene and seeing fluid behave like cubes
were axis-aligned.

Fix: emit a `warnings.append(...)` from `collect_scene` when this
combination is detected, recommending `obstacle_type=MESH` (which
exports the OBJ with rotation baked into world-space vertices and is
handled correctly by the SDF mesh-marker D4.3.GPU.BVH path).

Source-grep contract test (lesson §9.12).
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


def test_bake_warns_on_bbox_with_rotation():
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "bake.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )
    # The detect-and-warn block must exist near the BBOX branch.
    bbox_pos = code.find('obstacle_type == "BBOX"')
    assert bbox_pos > 0, "BBOX branch missing — collect_scene refactored?"
    # Round-57 supersedes R52's "warn user that rotation is dropped"
    # contract — MPM now handles OBB natively, so rotation is no longer
    # silently lost. We still require the BBOX branch to INSPECT
    # rotation_euler (that's the entry point to the OBB path), but the
    # MESH-fallback warning is gone for the MPM case.
    window = code[bbox_pos:bbox_pos + 3000]  # R57 widened the branch
    assert "rotation_euler" in window, (
        "round-52 regressed: BBOX branch must inspect rotation_euler")
    # Some warnings.append still exists (Apply-Rotation case + non-
    # cubic domain warning) — just looser assertion.
    assert "warnings.append" in window, (
        "BBOX branch must still warn on the edge cases (Apply-Rotation "
        "+ non-cubic domain) — R57 only removed the rotation-lost warning")
