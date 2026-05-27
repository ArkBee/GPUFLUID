"""Round-56 — MPM doesn't support mesh obstacles; auto-promote must
be FLIP-only.

Diagnostic chain (live-test, post-restart):
  1. round-53 added auto-promote BBOX→MESH for tilted obstacles.
  2. round-55 fixed the key-name typo (`obj_path` → `path`).
  3. Live bake "succeeded" — auto-promote warnings visible — but the
     fluid still ignored the tilt. Visual: fluid coats axis-aligned
     bounding boxes, the visible cube meshes are just rendered on top.
  4. Inspection: `cli/commands.py:178` MPM path handles ONLY
     `isinstance(ob, ObstacleBoxCfg)` — anything else is silently
     dropped. Mesh-type obstacle = dead-letter for MPM.

Fix: auto-promote stays for FLIP (where the SDF mesh-marker
D4.3.GPU.BVH path is real), but for solver=MPM we emit a DIFFERENT
warning explaining the limitation + workarounds, and fall back to the
axis-aligned BBOX so the bake at least HAS an obstacle (even if
mis-sized) rather than silently going through it.

Source-grep contract (lesson §9.12).
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


def test_mpm_skips_auto_promote_warns_about_limitation():
    """R57 supersedes R56: MPM now has a real OBB collider, so the
    "limitation" warning of R56 is gone and rotated boxes are handled
    natively. We just verify the solver-aware branching is still there
    (now to route MPM through OBB vs FLIP through SDF-mesh)."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "bake.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )
    bbox_pos = code.find('obstacle_type == "BBOX"')
    next_branch = code.find('elif oprops.obstacle_type', bbox_pos + 1)
    body = code[bbox_pos:next_branch]
    # Solver-aware branching must exist.
    assert 'dprops.solver == "mpm"' in body, (
        "round-56 regressed: BBOX branch must check solver type")
    assert "is_mpm" in body, (
        "round-56 regressed: solver routing flag missing")
    # Both branches end up emitting a usable obstacle.
    assert '"type": "box"' in body, (
        "BBOX branch must still produce type=box entries")
