"""Addon Python-interpreter auto-detection (preferences._detect_interpreter).

[BLK A8.8] guard: covers the detect_interp member of the A8.8 helper-operator
family — these tests exercise `_detect_interpreter`, the entire logic behind
GPUFLUID_OT_detect_interpreter (audit-20260610: coverage ref added when A8.8
gained a real @block registration; the rest of the family — mark_*/clear/
open_cache_dir — still has no behavioural test, see follow-ups).

Blender is normally launched from the Start menu, so its cwd is never the
project — register-time detection kept returning '' and the Bake button
errored until the user set the path by hand. The fix: also search up from
project-adjacent roots (the .blend dir + each Domain's cache_dir). This locks
that the detector finds a `.venv` under a passed-in root.

The preferences module imports bpy at top + defines Operator/AddonPreferences
classes, so we stub bpy enough to import it, then test the pure helpers.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


def _load_prefs():
    bpy = types.ModuleType("bpy")
    # props: each factory just returns a sentinel (class-body annotations).
    props = types.SimpleNamespace(**{
        n: (lambda *a, **k: None) for n in
        ("StringProperty", "IntProperty", "BoolProperty", "EnumProperty",
         "FloatProperty", "FloatVectorProperty", "PointerProperty")})
    bpy.props = props
    bpy.types = types.SimpleNamespace(Operator=object, AddonPreferences=object)
    bpy.utils = types.SimpleNamespace(register_class=lambda c: None,
                                      unregister_class=lambda c: None)
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    bpy.context = types.SimpleNamespace()
    bpy.data = types.SimpleNamespace(filepath="")
    bpy.ops = types.SimpleNamespace()
    sys.modules["bpy"] = bpy
    # preferences.py does `from . import ADDON_PKG` -> load it as a submodule
    # of a stub parent package that exposes ADDON_PKG.
    pkg = types.ModuleType("_gfpkg")
    pkg.ADDON_PKG = "gpufluid_blender"
    pkg.__path__ = []
    sys.modules["_gfpkg"] = pkg
    path = (Path(__file__).resolve().parents[1] / "addon" / "gpufluid_blender"
            / "preferences.py")
    spec = importlib.util.spec_from_file_location("_gfpkg.preferences", path)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "_gfpkg"
    sys.modules["_gfpkg.preferences"] = m
    spec.loader.exec_module(m)
    return m


def test_detect_finds_venv_under_extra_root(tmp_path, monkeypatch):
    m = _load_prefs()
    # fake a project with a .venv/<py> that "has gpufluid"
    proj = tmp_path / "proj"
    vpy = (proj / ".venv" / ("Scripts" if os.name == "nt" else "bin")
           / ("python.exe" if os.name == "nt" else "python"))
    vpy.parent.mkdir(parents=True)
    vpy.write_text("")  # just needs to exist
    monkeypatch.setattr(m, "_has_gpufluid", lambda p: p == str(vpy))
    # cwd is elsewhere (mirrors Blender's Start-menu launch) -> only the
    # extra root (a file deep in the project) lets detection find it.
    monkeypatch.chdir(tmp_path)
    deep = proj / "scenes" / "demo.blend"
    got = m._detect_interpreter([str(deep)])
    assert got == str(vpy), f"should find project .venv via extra root; got {got!r}"


def test_get_prefs_falls_back_loudly_without_addons_entry(caplog):
    """audit-20260610: manual register() (headless batch scripts) leaves
    context.preferences.addons without our key — get_prefs used to crash
    with a bare KeyError (caught by the headless default-scene smoke).
    It must instead return session-local defaults AND log the fallback
    (§9.10), so the operators' interpreter auto-detect can self-heal."""
    m = _load_prefs()

    class _Addons(dict):
        pass  # plain dict: missing key raises KeyError, like bpy's collection

    ctx = types.SimpleNamespace(
        preferences=types.SimpleNamespace(addons=_Addons()))
    with caplog.at_level("WARNING", logger="gpufluid.addon"):
        prefs = m.get_prefs(ctx)
    assert prefs.interpreter_path == ""
    assert any("fallback" in r.message for r in caplog.records), \
        "audit-20260610: the prefs fallback must be LOUD (§9.10)"
    # session persistence: an auto-detected path written by bake must be
    # visible to a later render call (same singleton, not a fresh object)
    prefs.interpreter_path = "X:/venv/python.exe"
    assert m.get_prefs(ctx).interpreter_path == "X:/venv/python.exe"


def test_detect_returns_empty_without_root(tmp_path, monkeypatch):
    """Without the project-adjacent root, detection from a foreign cwd misses
    the venv (the exact pre-fix failure) — proving the extra root is what fixes it."""
    m = _load_prefs()
    proj = tmp_path / "proj"
    vpy = (proj / ".venv" / ("Scripts" if os.name == "nt" else "bin")
           / ("python.exe" if os.name == "nt" else "python"))
    vpy.parent.mkdir(parents=True)
    vpy.write_text("")
    monkeypatch.setattr(m, "_has_gpufluid", lambda p: p == str(vpy))
    monkeypatch.setattr(m.shutil, "which", lambda n: None)  # no system python
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("GPUFLUID_PYTHON", raising=False)
    monkeypatch.chdir(tmp_path)  # foreign cwd, project not above it
    assert m._detect_interpreter([]) == ""
