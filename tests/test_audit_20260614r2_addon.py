"""Audit 2026-06-14 round 2 — addon render fps + parented-inflow matrix (#9, #10).

Both files import bpy at module top, so these are source contracts (§9.12)
on the comment/docstring-stripped source.
"""
from __future__ import annotations

import ast
from pathlib import Path

ADDON = Path(__file__).resolve().parents[1] / "addon" / "gpufluid_blender"


def _code(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    doc = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc.update(range(body[0].lineno, body[0].end_lineno + 1))
    return "\n".join(l for i, l in enumerate(src.splitlines(), 1)
                     if i not in doc and not l.lstrip().startswith("#"))


def test_render_forwards_fps_from_scene_toml():
    code = _code(ADDON / "operators" / "render.py")
    assert '"--fps"' in code, \
        "audit #9: OT_render must forward --fps (else the CLI defaults to 60)"
    assert '["simulation"]["fps"]' in code, \
        "audit #9: fps must be read from the cache's scene.toml [simulation]"


def test_animation_world_matrix_includes_parent_inverse():
    code = _code(ADDON / "operators" / "_animation.py")
    assert "o.matrix_parent_inverse" in code, \
        "audit #10: world-matrix composition must include matrix_parent_inverse"
