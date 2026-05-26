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
        # Round-3 reviewer caught weak `or "gravity"` disjunct that
        # would have masked the wrap. Require the qualified path.
        assert "simulation.gravity" in str(e), (
            f"expected 'simulation.gravity' in error, got: {e}")
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


def test_runner_modal_popen_wrapped_in_try_oserror():
    """Round-8 bug #1 + round-9 strengthening: subprocess spawn MUST
    be wrapped to clear the reentrance flag on failure. After round-14
    the bake/render-shared lifecycle moved to ModalSubprocessRunner;
    the contract now lives in `_spawn_and_drain` + `start_modal`
    error path.
    """
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "_runner.py").read_text(encoding="utf-8")
    # start_modal must catch the spawn exception and clear class flag
    start_idx = src.find("def start_modal")
    assert start_idx > -1, "start_modal missing"
    block = src[start_idx:start_idx + 2000]
    assert "_spawn_and_drain" in block
    assert "(OSError, RuntimeError)" in block, (
        "start_modal needs to catch OSError AND RuntimeError "
        "(round-9 extension: Thread.start raises RuntimeError)")
    assert "_is_running = False" in block, (
        "start_modal error branch must clear class flag — round-8 bug #1")
    # _spawn_and_drain itself orphan-kills Popen if Queue/Thread setup fails
    spawn_idx = src.find("def _spawn_and_drain")
    assert spawn_idx > -1
    spawn_block = src[spawn_idx:spawn_idx + 1500]
    assert "self._proc.kill()" in spawn_block, (
        "_spawn_and_drain must kill orphan Popen on Queue/Thread failure "
        "— round-9 reviewer")


def test_runner_sync_path_uses_try_finally_for_flag():
    """Sync path must clear _is_running even on exception. Round-14
    moved the contract into ModalSubprocessRunner.start_sync."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "_runner.py").read_text(encoding="utf-8")
    sync_idx = src.find("def start_sync")
    assert sync_idx > -1
    # start_sync body — until the next def or class boundary
    # Skip the nested `def _drain_lines` (12-space indent) — use 4-space.
    next_def_idx = src.find("\n    def ", sync_idx + 100)
    block = src[sync_idx:next_def_idx if next_def_idx > -1 else sync_idx + 4000]
    assert "finally:" in block, "start_sync lost its try/finally"
    assert "op_class._is_running = False" in block


def test_runner_finish_and_abort_clean_instance_state():
    """Round-14 refactor: _abort + _finish lifecycle for both bake and
    render now lives in ModalSubprocessRunner. The class's
    _clear_instance_state must be called by both paths so re-fired
    modal-on-same-instance never sees stale Queue/Thread/timer/proc.
    """
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "_runner.py").read_text(encoding="utf-8")
    # _clear_instance_state must drop all four
    clear_idx = src.find("def _clear_instance_state")
    assert clear_idx > -1
    block = src[clear_idx:clear_idx + 600]
    for attr in ("self._proc = None", "self._stdout_q = None",
                 "self._stdout_thread = None", "self._timer = None"):
        assert attr in block, f"_clear_instance_state must reset {attr}"
    # abort + _finish call _clear_instance_state
    for method in ("def abort(", "def _finish("):
        idx = src.find(method)
        assert idx > -1, f"{method} missing"
        m_block = src[idx:idx + 1500]
        assert "_clear_instance_state" in m_block, (
            f"{method} must call _clear_instance_state to honour the "
            f"round-7+round-10 contract")


def test_emit_scalar_nested_inline_table_recurses():
    """Round-10 reviewer gap: round-9 test only covered flat inline-table.
    Nested case `{a: {b: inf}}` should produce chained key context."""
    cb = _load_config_builder()
    nested = {"motion": {"kind": "linear", "amp": float("inf")}}
    try:
        cb._emit_scalar(nested)
    except ValueError as e:
        # Outer wraps with 'motion', inner wraps with 'amp'.
        # Either name reaching the message proves recursion fires.
        msg = str(e)
        assert "motion" in msg and "amp" in msg, (
            f"expected both 'motion' and 'amp' in chain, got: {msg}")
        return
    pytest.fail("expected ValueError for nested inline-table")


def test_render_finish_cleans_all_instance_attrs():
    """Round-14: render._finish moved to ModalSubprocessRunner._finish.
    The 3-attr cleanup is asserted by
    test_runner_finish_and_abort_clean_instance_state. This test
    asserts the render-specific contract: render.py no longer has
    a local _finish (runner owns it)."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "render.py").read_text(encoding="utf-8")
    assert "def _finish(self" not in src, (
        "render._finish should be removed in round-14 — runner._finish "
        "owns the modal-end lifecycle")


def test_render_abort_cleans_all_three_instance_attrs():
    """Round-14: render._abort moved to ModalSubprocessRunner.abort.
    The 3-attr cleanup contract now lives in _clear_instance_state
    (asserted by test_runner_finish_and_abort_clean_instance_state).
    This test now verifies render.py STILL delegates to runner.cancel
    instead of doing its own teardown."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "render.py").read_text(encoding="utf-8")
    cancel_idx = src.find("def cancel(")
    assert cancel_idx > -1, "render must keep cancel() callback"
    block = src[cancel_idx:cancel_idx + 400]
    assert "self._runner.cancel" in block, (
        "render.cancel must route through runner.cancel — not "
        "re-implement teardown (round-14 day-1 #2 contract)")
    # And no leftover local _abort that would bypass the runner
    assert "def _abort" not in src, (
        "render._abort should be removed in round-14; runner.abort owns it")


# ─── Round-9 additions ──────────────────────────────────────────────────

def test_emit_scalar_inline_table_propagates_inner_key():
    """Round-9 fix: inline-table values with bad inner key now name
    that key in the error, not just bare repr."""
    cb = _load_config_builder()
    bad = {"kind": "linear", "velocity": float("inf")}
    try:
        cb._emit_scalar(bad)
    except ValueError as e:
        assert "velocity" in str(e), (
            f"expected 'velocity' in error message, got: {e}")
        assert "non-finite" in str(e) or "inf" in str(e)
        return
    pytest.fail("expected ValueError naming 'velocity'")


# NOTE: a sibling test for nested motion inside [[obstacle]] entries
# uncovered a deeper round-1 bug — _emit_table's array-of-tables branch
# drops table-valued fields like `motion` (dict) entirely instead of
# inlining them. Filed as BACKLOG entry; not in round-9 scope to fix.
# The scalar-side inline-table propagation IS verified by
# test_emit_scalar_inline_table_propagates_inner_key above.


def test_sync_has_watchdog_timeout_property():
    """Round-9 watchdog: both OT_bake and OT_render must expose
    sync_timeout_sec on the operator class itself (UI prop). The
    actual TimeoutExpired/kill logic lives in ModalSubprocessRunner
    after round-14."""
    for op_file in ("bake.py", "render.py"):
        src = (_ADDON_DIR / "gpufluid_blender" / "operators" / op_file).read_text(encoding="utf-8")
        assert "sync_timeout_sec" in src, (
            f"{op_file} must declare sync_timeout_sec IntProperty")
    runner_src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "_runner.py").read_text(encoding="utf-8")
    assert "TimeoutExpired" in runner_src, (
        "runner sync path must catch subprocess.TimeoutExpired")
    assert "self._proc.kill()" in runner_src, (
        "runner timeout branch must kill the hung subprocess")


def test_sync_watchdog_clears_is_running_on_timeout():
    """If the watchdog fires, the class-level _is_running MUST be cleared.
    After round-14, the contract lives in ModalSubprocessRunner.start_sync
    — the try/finally clears `op_class._is_running` regardless of which
    branch (success / TimeoutExpired / OSError / rc!=0) returns first."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators" / "_runner.py").read_text(encoding="utf-8")
    sync_idx = src.find("def start_sync")
    assert sync_idx > -1
    # Take everything until the next def at module-or-class level
    # Use \n + 4-space indent to skip nested def_drain_lines (12-space).
    end_idx = src.find("\n    def ", sync_idx + 100)
    block = src[sync_idx:end_idx if end_idx > -1 else sync_idx + 4000]
    timeout_idx = block.find("except subprocess.TimeoutExpired")
    assert timeout_idx > -1, "start_sync missing TimeoutExpired branch"
    # The try/finally at the outer level clears _is_running; check it's
    # in the same block as TimeoutExpired (so timeout return triggers it).
    finally_idx = block.find("finally:", timeout_idx)
    assert finally_idx > -1, "no finally: after TimeoutExpired branch"
    finally_block = block[finally_idx:finally_idx + 400]
    assert "op_class._is_running = False" in finally_block, (
        "start_sync finally: must clear op_class._is_running")
