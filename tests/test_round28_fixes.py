"""Round-28 regression tests for round-27 reviewer findings.

  - PreloadCache.pop frees meshes (was: silent leak per call).
  - _headless_render rejects every missing required key (was: only
    cache/scene/out checked).
  - blocks registry logs WARNING on duplicate qualname (was: silent
    drop — this is what let the round-21 cmd_render dup slip through).
  - render_bridge bounds-checks face indices (was: crashed inside
    Blender polygon validator on corrupt PLY).
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


# ─── PreloadCache.pop frees meshes ──────────────────────────────────────

class _FakeMesh:
    def __init__(self, name, users=0):
        self.name = name
        self.users = users


def _fresh_cache_loader_with_fake_bpy():
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.app = types.SimpleNamespace(
        handlers=types.SimpleNamespace(
            frame_change_pre=[], load_post=[],
            persistent=lambda f: f,
        )
    )

    removed: list[str] = []

    class _Meshes:
        def remove(self, m, do_unlink=True):
            removed.append(m.name)

    class _Objs:
        def __iter__(self): return iter([])
        def get(self, _n, default=None): return default
    fake_bpy.data = types.SimpleNamespace(
        meshes=_Meshes(), objects=_Objs(), scenes=[])
    fake_bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(addons={}),
        scene=None)
    fake_bpy.types = types.SimpleNamespace(
        Operator=type("_O", (), {}),
        PropertyGroup=type("_P", (), {}),
        AddonPreferences=type("_AP", (), {}),
        Panel=type("_Pn", (), {}),
        Mesh=_FakeMesh,
    )
    _p = lambda **k: None
    fake_bpy.props = types.SimpleNamespace(
        StringProperty=_p, FloatProperty=_p,
        FloatVectorProperty=_p, IntProperty=_p,
        BoolProperty=_p, EnumProperty=_p,
        PointerProperty=_p,
    )
    fake_bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    sys.modules["bpy"] = fake_bpy

    sys.modules.pop("gpufluid_blender", None)
    sys.modules.pop("gpufluid_blender.cache_loader", None)
    pkg = types.ModuleType("gpufluid_blender")
    pkg.__path__ = [str(_ADDON_DIR / "gpufluid_blender")]
    pkg.logger = logging.getLogger("gpufluid.addon.test")
    sys.modules["gpufluid_blender"] = pkg

    cl_dir = _ADDON_DIR / "gpufluid_blender" / "cache_loader"
    spec = importlib.util.spec_from_file_location(
        "gpufluid_blender.cache_loader",
        cl_dir / "__init__.py",
        submodule_search_locations=[str(cl_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpufluid_blender.cache_loader"] = mod
    spec.loader.exec_module(mod)
    return mod, removed


def test_preload_cache_pop_frees_meshes():
    """Round-28: pre-round-28 the dict-shim pop() returned the table
    without freeing — Alembic-attach re-attach leaked N meshes per
    call."""
    cl, removed = _fresh_cache_loader_with_fake_bpy()
    table = {0: _FakeMesh("m0", users=0),
             1: _FakeMesh("m1", users=0),
             2: _FakeMesh("m2", users=0)}
    cl._PRELOAD["dom"] = table
    popped = cl._PRELOAD.pop("dom")
    assert popped is table or popped == table
    assert sorted(removed) == ["m0", "m1", "m2"], (
        f"meshes leaked: removed={removed}")


def test_preload_cache_pop_missing_returns_default():
    """Round-28: missing key returns default sentinel without calling
    remove() (no-op contract preserved)."""
    cl, removed = _fresh_cache_loader_with_fake_bpy()
    sentinel = object()
    out = cl._PRELOAD.pop("nope", sentinel)
    assert out is sentinel
    assert removed == []


# ─── _headless_render expanded key check ────────────────────────────────

def _load_headless_render():
    spec = importlib.util.spec_from_file_location(
        "_test_headless2",
        _ADDON_DIR / "gpufluid_blender" / "_headless_render.py",
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("missing", ["cache", "scene", "out", "label", "color", "fps"])
def test_headless_render_rejects_each_missing_key(monkeypatch, missing):
    """Round-28: every key build_renderer_argv unconditionally indexes
    must be checked upfront (pre-round-28 only cache/scene/out)."""
    full = {"cache": "C:/c", "scene": "C:/s.toml", "out": "C:/o",
            "label": "L", "color": [0.5, 0.5, 0.5], "fps": 24,
            "samples": 4, "frames": 24}
    full.pop(missing)
    import json
    mod = _load_headless_render()
    monkeypatch.setattr(mod, "_argv_after_doubledash",
                        lambda: [json.dumps(full)])
    with pytest.raises(SystemExit, match=f"missing required key '{missing}'"):
        mod.main()


# ─── blocks dedup warns ─────────────────────────────────────────────────

def test_blocks_duplicate_qualname_warns(caplog):
    """Round-28: duplicate qualname under one block_id was silently
    dropped pre-round-28 (this is the bug class that let the round-21
    cmd_render duplicate slip through). Now: WARNING via
    'gpufluid.blocks' logger."""
    # Fresh-import blocks module so the registry is clean enough for
    # this test (other tests may have already populated it).
    sys.modules.pop("gpufluid.blocks", None)
    import gpufluid.blocks as bl

    @bl.block("Z99.7", "first")
    def my_callable(x):
        return x + 1

    with caplog.at_level(logging.WARNING, logger="gpufluid.blocks"):
        # Same qualname (`my_callable`) re-registered under same block_id.
        # The decorator factory must log a warning, not silently drop.
        @bl.block("Z99.7", "second")
        def my_callable(x):  # noqa: F811
            return x + 2

    assert any("duplicate qualname" in rec.message for rec in caplog.records), (
        f"expected duplicate-qualname warning, got: {caplog.text}")


# ─── render_bridge face bounds check ────────────────────────────────────

def test_render_bridge_strips_out_of_range_faces(monkeypatch):
    """Round-28: rebuild_surface_mesh must strip face rows referencing
    out-of-range vertex indices (was: crashed inside Blender's polygon
    validator on the foreach_set call)."""
    # bpy stub (render_bridge imports it lazily via FrameMeshLoader,
    # but the module top is stdlib + numpy only).
    sys.modules.setdefault(
        "bpy",
        types.ModuleType("bpy"),
    )
    spec = importlib.util.spec_from_file_location(
        "_test_render_bridge_28",
        _ADDON_DIR / "gpufluid_blender" / "render_bridge.py",
    )
    rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)

    sent_faces_len = {}

    class _Coll:
        def add(self, _n): pass
        def foreach_set(self, key, arr):
            if key == "vertices":
                sent_faces_len["n"] = len(arr)

    class _ColorAttrs:
        def get(self, _name): return None
        def new(self, **k):
            return types.SimpleNamespace(
                data=types.SimpleNamespace(foreach_set=lambda *a, **k: None))

    fake_mesh = types.SimpleNamespace(
        vertices=_Coll(), loops=_Coll(), polygons=_Coll(),
        color_attributes=_ColorAttrs(),
        update=lambda **k: None,
        clear_geometry=lambda: None,
    )

    class _Surf:
        def __init__(self):
            self.data = fake_mesh
            self.modifiers = types.SimpleNamespace(__iter__=lambda s: iter([]))

    verts = np.zeros((5, 3), dtype=np.float32)  # n_v == 5
    faces = np.array([
        [0, 1, 2],   # ok
        [0, 7, 2],   # 7 out of range
        [3, 4, -1],  # negative
        [1, 2, 3],   # ok
    ], dtype=np.int32)
    rb.rebuild_surface_mesh(_Surf(), verts, faces, colors=None)
    # 2 valid faces × 3 indices = 6
    assert sent_faces_len.get("n") == 6, (
        f"expected 6 vertex indices after strip, got: {sent_faces_len}")
