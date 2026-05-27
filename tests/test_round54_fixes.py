"""Round-54 — broaden BBOX→MESH auto-promote to also catch
'Apply Rotation' result (rotation_euler=0 but mesh data rotated).

Round-53 only checked `rotation_euler != 0`. Live test: user pressed
Ctrl+A → R to apply rotation. rotation_euler became (0,0,0), round-53
auto-promote disabled, fell back to BBOX axis-aligned (fat box). Same
silent failure as pre-round-52.

Round-54 adds a SECOND detection path: walk `mesh.loop_triangles`
and check if all face normals are axis-aligned (within 5°). If any
face is tilted, promote regardless of rotation_euler. Catches both
"tilted via object transform" AND "tilted via mesh data".

Source-grep contract (lesson §9.12) — the loop_triangles scan + the
non_axis_aligned flag + the combined trigger condition must all stay
inside the BBOX branch.
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


def test_bake_auto_promotes_when_mesh_non_axis_aligned():
    """Round-54 contract: BBOX branch must run a face-normal check
    so post-Apply-Rotation meshes still get the MESH treatment."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "bake.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )
    bbox_pos = code.find('obstacle_type == "BBOX"')
    next_branch = code.find('elif oprops.obstacle_type', bbox_pos + 1)
    body = code[bbox_pos:next_branch]
    # Face-normal scan must be present.
    assert "loop_triangles" in body, (
        "round-54 regressed: BBOX branch must walk loop_triangles "
        "to catch post-Apply-Rotation tilted meshes")
    assert "non_axis_aligned" in body, (
        "round-54 regressed: non_axis_aligned flag missing")
    # The promote-trigger must be the disjunction.
    assert ("rotation_nonzero or non_axis_aligned" in body
            or "non_axis_aligned or rotation_nonzero" in body), (
        "round-54 regressed: promote condition must combine both "
        "trigger paths with OR")


def test_axis_aligned_threshold_uses_safe_margin():
    """5° = cos ≈ 0.996. We use 0.99 as a safe margin so floating-
    point noise in face normals doesn't trip false positives on
    truly axis-aligned cubes."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "bake.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )
    bbox_pos = code.find('obstacle_type == "BBOX"')
    next_branch = code.find('elif oprops.obstacle_type', bbox_pos + 1)
    body = code[bbox_pos:next_branch]
    # Margin literal must be present (allow either 0.99 or stricter).
    assert "0.99" in body, (
        "round-54: axis-aligned threshold (0.99 ~= cos(5°)) missing")
