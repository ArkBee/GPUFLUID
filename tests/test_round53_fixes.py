"""Round-53 — auto-promote BBOX → MESH on tilted obstacles.

Round-52 emitted a warning when a BBOX obstacle had non-zero rotation
(rotation_euler) — but the user still had to manually flip the
obstacle_type to MESH. Round-53 closes that gap: the warning fires
AND the obstacle is silently promoted to MESH type, so the bake
"just works" for tilted geometry.

Source-grep contract (lesson §9.12) — the MESH-emission body must
live inside the BBOX branch's non-zero-rotation arm.
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


def test_bake_auto_promotes_tilted_bbox_to_mesh():
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "bake.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )
    bbox_pos = code.find('obstacle_type == "BBOX"')
    assert bbox_pos > 0
    # Look ahead until the next obstacle_type branch (`elif` block).
    next_branch = code.find('elif oprops.obstacle_type', bbox_pos + 1)
    body = code[bbox_pos:next_branch] if next_branch > 0 else code[bbox_pos:]
    # Non-zero rotation branch must call _export_obj AND emit type="mesh".
    assert "_export_obj(o, mesh_path)" in body, (
        "round-53 regressed: BBOX+rotation must trigger OBJ export "
        "(auto-promote to MESH)")
    assert '"type": "mesh"' in body, (
        "round-53 regressed: promoted obstacle must be emitted as "
        "type=mesh so the SDF mesh-marker path picks it up")


def test_cache_dir_hoisted_outside_obstacle_loop():
    """Round-53 contract: `cache_dir` must be resolved once before
    the obstacles loop so both the MESH branch and the new BBOX-promo
    path can use it without duplicating the `bpy.path.abspath` call."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "bake.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )
    # `cache_dir = bpy.path.abspath(...)` must appear ONCE in
    # collect_scene (the hoisted definition), not twice (the old
    # duplicate inside the MESH branch).
    n = code.count("cache_dir = bpy.path.abspath(dprops.cache_dir)")
    assert n == 1, (
        f"round-53: expected exactly one cache_dir resolution in "
        f"collect_scene (hoisted); found {n}")
