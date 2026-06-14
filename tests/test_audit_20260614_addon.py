"""Audit 2026-06-14 — whitewater cache not scaled by dom_size (#3).

Surface mesh verts are mapped to world as `verts*dom_size + origin`, but the
whitewater path only did `positions + origin` (no dom_size key existed), so
foam/spray collapsed into a 1 m cube on any non-unit domain (the default
`Add Domain` is a 2 m cube). Confirmed 3/3.
"""
from __future__ import annotations

import ast
from pathlib import Path

from gpufluid_blender import cache_binding as cb  # addon/ on path via conftest

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


# ---- binding round-trip (cache_binding is pure — no bpy) ------------------

def test_ww_binding_round_trips_dom_size():
    obj = {}
    cb.set_ww_binding(obj, "cache/", (0.1, 0.2, 0.3),
                      frame_offset=2, dom_size=(2.0, 3.0, 4.0))
    assert cb.get_ww_cache_dom_size(obj) == (2.0, 3.0, 4.0)


def test_ww_dom_size_key_in_all_ww_keys():
    # detach iterates ALL_WW_KEYS — the new key must be scrubbed too
    assert cb.KEY_WW_CACHE_DOM_SIZE in cb.ALL_WW_KEYS


def test_ww_dom_size_default_is_unit_cube():
    assert cb.get_ww_cache_dom_size({}) == (1.0, 1.0, 1.0)


# ---- source contracts on the bpy-heavy modules ----------------------------

def test_ww_rebuild_scales_by_dom_size():
    code = _code(ADDON / "cache_loader" / "__init__.py")
    assert "np.asarray(dom_size" in code, \
        "audit #3: whitewater rebuild must multiply positions by dom_size"
    # the handler must read + pass dom_size, not leave it at the default
    assert "get_ww_cache_dom_size(obj)" in code, \
        "audit #3: ww handler should read dom_size from the binding"


def test_bake_auto_attach_ww_passes_dom_size():
    code = _code(ADDON / "operators" / "bake.py")
    # the attach_ww_cache auto-attach call must forward dom_size
    assert "attach_ww_cache(" in code and "dom_size=tuple(float(c) for c in dom_size)" in code, \
        "audit #3: post-bake whitewater auto-attach must pass dom_size"
