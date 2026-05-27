"""Round-55 — fix the key-name typo I introduced in round-53.

Round-53's BBOX→MESH auto-promote emitted `"obj_path"` but
`config_builder.py:160` reads `o["path"]` → KeyError mid-bake. Live
test caught this on the first real bake-with-tilted-cubes after the
addon install path was synced from repo (post-restart).

Source-grep contract (lesson §9.12) — both MESH paths (original at
line ~218 and round-53 auto-promote at line ~178) must use the same
key name `"path"`.
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


def test_auto_promote_uses_path_not_obj_path():
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "bake.py").read_text()
    code = "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )
    # The pre-round-55 buggy form must be gone from live code.
    assert '"obj_path"' not in code, (
        "round-55 regressed: auto-promote MESH dict still uses "
        "`obj_path` key — config_builder reads `path`, KeyError")
    # And both MESH-emitting blocks must use `"path"`.
    # Count occurrences: original MESH branch + round-53 auto-promote.
    n_path_keys = code.count('"path": mesh_path')
    assert n_path_keys >= 2, (
        f"round-55: expected >= 2 `\"path\": mesh_path` (original MESH "
        f"branch + round-53 auto-promote); found {n_path_keys}")
