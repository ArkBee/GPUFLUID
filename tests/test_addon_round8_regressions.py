"""Round-8 regression tests — cover the round-6/7 code that shipped
without unit tests (reviewer caught this in the commit-message audit:
the "29/29 green" claim was status-quo, not proof the new code works).

Four areas:
  1. _emit_scalar rejects inf / nan with key context (round-7 minor #1)
  2. _emit_with_key preserves key path on error (round-7)
  3. _prune_stale drops keys whose obj.name isn't in any scene (round-8)
  4. _is_running flag is a class-level shared lock (round-5/6 reentrance)

These tests run without bpy by mocking the small surface each function
needs. They are NOT a substitute for the live Blender tests — they
just prevent the next refactor from silently breaking the contract.
"""
from __future__ import annotations

import math
import sys
import types
import pytest
from pathlib import Path

_ADDON_DIR = Path(__file__).resolve().parent.parent / "addon"


# ─── 1) _emit_scalar finite-float check + 2) _emit_with_key prefix ─────

def _load_config_builder():
    """Import config_builder without booting the full addon (no bpy)."""
    import importlib.util
    # config_builder imports `from typing import Any, Dict, List` only —
    # no bpy needed. Load directly.
    spec = importlib.util.spec_from_file_location(
        "test_config_builder",
        _ADDON_DIR / "gpufluid_blender" / "config_builder.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_emit_scalar_finite_floats_emit_g_format():
    cb = _load_config_builder()
    assert cb._emit_scalar(0.0) == "0"
    assert cb._emit_scalar(1.5) == "1.5"
    assert cb._emit_scalar(-9.81) == "-9.81"


def test_emit_scalar_rejects_inf():
    cb = _load_config_builder()
    with pytest.raises(ValueError, match="non-finite"):
        cb._emit_scalar(float("inf"))


def test_emit_scalar_rejects_negative_inf():
    cb = _load_config_builder()
    with pytest.raises(ValueError, match="non-finite"):
        cb._emit_scalar(float("-inf"))


def test_emit_scalar_rejects_nan():
    cb = _load_config_builder()
    with pytest.raises(ValueError, match="non-finite"):
        cb._emit_scalar(float("nan"))


def test_emit_table_wraps_inf_with_key_context():
    """Round-7 promise: when inf hits scalar emit, error names the key."""
    cb = _load_config_builder()
    out: list = []
    try:
        cb._emit_table(out, "simulation", {"gravity": float("inf")})
    except ValueError as e:
        assert "simulation.gravity" in str(e) or "gravity" in str(e)
        return
    pytest.fail("expected ValueError pointing at simulation.gravity")


def test_emit_table_wraps_array_of_tables_with_index():
    """Round-7: array-of-tables entries cite [i] prefix."""
    cb = _load_config_builder()
    out: list = []
    bad_scene = {"obstacle": [
        {"type": "box", "center": [0.5, 0.5, 0.5]},
        {"type": "box", "center": float("nan")},   # broken second entry
    ]}
    try:
        cb._emit_table(out, "", bad_scene)
    except ValueError as e:
        assert "obstacle[1].center" in str(e)
        return
    pytest.fail("expected ValueError pointing at obstacle[1].center")


# ─── 3) _prune_stale drops dead keys ────────────────────────────────────

def _make_minimal_bpy_for_cache_loader():
    """Stub enough bpy.* to import cache_loader/__init__.py."""
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.app = types.SimpleNamespace(
        handlers=types.SimpleNamespace(
            frame_change_pre=[], load_post=[],
            persistent=lambda f: f,
        )
    )
    fake_bpy.data = types.SimpleNamespace(
        meshes=types.SimpleNamespace(remove=lambda m, do_unlink=True: None),
        scenes=[],
        objects=[],
    )
    class _Op: pass
    class _PG: pass
    fake_bpy.types = types.SimpleNamespace(
        Operator=_Op, PropertyGroup=_PG, Mesh=type("Mesh", (), {}),
        AddonPreferences=_PG, Panel=type("Panel", (), {}),
    )
    _prop = lambda **k: None
    fake_bpy.props = types.SimpleNamespace(
        StringProperty=_prop, FloatProperty=_prop, FloatVectorProperty=_prop,
        IntProperty=_prop, BoolProperty=_prop, EnumProperty=_prop,
        PointerProperty=_prop,
    )
    fake_bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    fake_bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(addons={}),
        scene=None,
    )
    sys.modules["bpy"] = fake_bpy
    return fake_bpy


def _load_cache_loader():
    fake_bpy = _make_minimal_bpy_for_cache_loader()
    # Stub gpufluid_blender package + logger
    import logging
    pkg = types.ModuleType("gpufluid_blender")
    pkg.__path__ = [str(_ADDON_DIR / "gpufluid_blender")]
    pkg.logger = logging.getLogger("gpufluid.addon.test")
    sys.modules["gpufluid_blender"] = pkg

    import importlib.util
    cl_dir = _ADDON_DIR / "gpufluid_blender" / "cache_loader"
    spec = importlib.util.spec_from_file_location(
        "gpufluid_blender.cache_loader",
        cl_dir / "__init__.py",
        submodule_search_locations=[str(cl_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpufluid_blender.cache_loader"] = mod
    spec.loader.exec_module(mod)
    return mod, fake_bpy


def test_prune_stale_drops_keys_not_in_scene():
    cl, fake_bpy = _load_cache_loader()
    # Plant entries
    cl._PRELOAD.clear()
    cl._PRELOAD["alive"] = {0: types.SimpleNamespace(name="m0", users=0)}
    cl._PRELOAD["dead"] = {0: types.SimpleNamespace(name="m0", users=0)}
    cl._PRELOAD["also_dead"] = {0: types.SimpleNamespace(name="m0", users=0)}

    # Scene has only "alive"
    scene = types.SimpleNamespace(
        objects=[types.SimpleNamespace(name="alive")])
    cl._prune_stale(scene)
    assert "alive" in cl._PRELOAD
    assert "dead" not in cl._PRELOAD
    assert "also_dead" not in cl._PRELOAD


def test_prune_stale_handles_empty_preload():
    cl, _ = _load_cache_loader()
    cl._PRELOAD.clear()
    # Should not raise on empty dict / None scene
    cl._prune_stale(None)
    cl._prune_stale(types.SimpleNamespace(objects=[]))


def test_prune_stale_no_scene_unions_all_scenes():
    cl, fake_bpy = _load_cache_loader()
    cl._PRELOAD.clear()
    cl._PRELOAD["in_scene_a"] = {0: types.SimpleNamespace(name="m0", users=0)}
    cl._PRELOAD["in_scene_b"] = {0: types.SimpleNamespace(name="m0", users=0)}
    cl._PRELOAD["nowhere"] = {0: types.SimpleNamespace(name="m0", users=0)}

    fake_bpy.data.scenes = [
        types.SimpleNamespace(objects=[types.SimpleNamespace(name="in_scene_a")]),
        types.SimpleNamespace(objects=[types.SimpleNamespace(name="in_scene_b")]),
    ]
    cl._prune_stale(scene=None)   # union mode
    assert "in_scene_a" in cl._PRELOAD
    assert "in_scene_b" in cl._PRELOAD
    assert "nowhere" not in cl._PRELOAD


# ─── 4) _is_running contract (class-level shared lock) ──────────────────

def test_is_running_is_class_level_attribute():
    """The reentrance guard MUST be a class attribute, not an instance
    attribute — otherwise two OT_bake instances each have their own
    flag and the guard doesn't work across button clicks."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "bake.py").read_text(encoding="utf-8")
    # The class declares _is_running with a type annotation at class scope
    assert "_is_running: bool = False" in src, (
        "OT_bake._is_running must be declared at class scope so the lock "
        "is shared across instances. Round-5/6 contract.")
    # And every read/write goes through the class, not self
    assert "GPUFLUID_OT_bake._is_running" in src
    # No bare `self._is_running = True` writes that would shadow the class attr
    assert "self._is_running = True" not in src
    assert "self._is_running = False" not in src


def test_render_has_same_is_running_contract():
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "render.py").read_text(encoding="utf-8")
    assert "_is_running: bool = False" in src
    assert "GPUFLUID_OT_render._is_running" in src
    assert "self._is_running = True" not in src
    assert "self._is_running = False" not in src


def test_bake_modal_popen_wrapped_in_try_oserror():
    """Round-8 reviewer bug #1: modal Popen MUST be in try/except OSError
    that clears _is_running on failure. Without it, OSError between
    `_is_running=True` and Popen success leaks the flag forever."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "bake.py").read_text(encoding="utf-8")
    # Find the MODAL PATH block and verify try/except OSError contains
    # GPUFLUID_OT_bake._is_running = False
    modal_idx = src.find("# ─── MODAL PATH")
    assert modal_idx > -1, "MODAL PATH section disappeared"
    block = src[modal_idx:modal_idx + 1500]
    assert "except OSError" in block, (
        "modal Popen needs except OSError — round-8 bug #1")
    assert "_is_running = False" in block, (
        "OSError branch must clear _is_running — round-8 bug #1")


def test_sync_path_uses_try_finally_for_flag():
    """Sync path must clear _is_running even on exception (try/finally)."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "bake.py").read_text(encoding="utf-8")
    sync_idx = src.find("# ─── SYNC PATH")
    assert sync_idx > -1
    block = src[sync_idx:sync_idx + 2000]
    assert "finally:" in block
    assert "_is_running = False" in block


def test_render_abort_cleans_all_three_instance_attrs():
    """Round-8 reviewer bug #6: render._abort must clear _proc,
    _stdout_q, _stdout_thread — symmetry with round-7 bake fix."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "render.py").read_text(encoding="utf-8")
    abort_idx = src.find("def _abort")
    assert abort_idx > -1
    # Look in the _abort body (first 1500 chars after def)
    block = src[abort_idx:abort_idx + 1500]
    assert "self._proc = None" in block
    assert "self._stdout_q = None" in block
    assert "self._stdout_thread = None" in block
