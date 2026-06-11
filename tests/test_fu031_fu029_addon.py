"""FU-031 + FU-029 (addon UI side).

FU-031: .blend files saved before FU-017 (solver default flip→mpm) where
the user never touched Solver silently re-bake with MPM. The load_post
handler now emits a once-per-file migration notice — but ONLY when there
is evidence of a previous bake (cache_dir exists), since a fresh scene
also has the property unset and MPM is simply correct there.

FU-029: cfl_factor > 1.0 under-substeps MPM stress waves. Honest UI
only: description guidance + a panel alert row; no clamping, no range
change (back-compat).

FU-031 tests are behavioural (cache_loader loaded under the
test_addon_preload_lru bpy-stub pattern); FU-029 tests are §9.12
source contracts (comment/docstring stripped — the asserted strings
live in CODE: the draw condition and the description= literal).
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "addon" / "gpufluid_blender"


# ── §9.12 helper ─────────────────────────────────────────────────────────

def _code_only(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    doc_lines: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return "\n".join(line for i, line in enumerate(src.splitlines(), 1)
                     if i not in doc_lines
                     and not line.lstrip().startswith("#"))


# ── cache_loader loader (test_addon_preload_lru pattern, private names) ──

def _load_cache_loader():
    bpy = types.ModuleType("bpy")
    bpy.app = types.SimpleNamespace(handlers=types.SimpleNamespace(
        frame_change_pre=[], load_post=[], persistent=lambda f: f))
    bpy.data = types.SimpleNamespace(
        meshes=types.SimpleNamespace(remove=lambda m, do_unlink=True: None),
        objects=[], filepath="")
    bpy.types = types.SimpleNamespace(
        Operator=type("Operator", (), {}),
        Mesh=type("Mesh", (), {}),
        PropertyGroup=type("PropertyGroup", (), {}),
        AddonPreferences=type("AddonPreferences", (), {}),
        Panel=type("Panel", (), {}))
    bpy.props = types.SimpleNamespace(**{
        n: (lambda *a, **k: None) for n in
        ("StringProperty", "FloatProperty", "FloatVectorProperty",
         "IntProperty", "BoolProperty", "EnumProperty", "PointerProperty")})
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(addons={}))
    sys.modules["bpy"] = bpy

    pkg = types.ModuleType("gpufluid_blender")
    pkg.__path__ = [str(ADDON)]
    pkg.logger = logging.getLogger("fu031.test")
    pkg.ADDON_PKG = "gpufluid_blender"
    sys.modules["gpufluid_blender"] = pkg

    cl_dir = ADDON / "cache_loader"
    spec = importlib.util.spec_from_file_location(
        "gpufluid_blender.cache_loader", cl_dir / "__init__.py",
        submodule_search_locations=[str(cl_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpufluid_blender.cache_loader"] = mod
    spec.loader.exec_module(mod)
    return mod


def _domain(name, cache_dir, *, solver_set, is_domain=True):
    return types.SimpleNamespace(
        name=name,
        gpufluid_domain=types.SimpleNamespace(
            is_domain=is_domain, cache_dir=str(cache_dir),
            is_property_set=lambda prop, _s=solver_set: _s))


# ── FU-031 behavioural ───────────────────────────────────────────────────

def test_fu031_fires_for_legacy_bake(tmp_path):
    cl = _load_cache_loader()
    cl._FU031_NOTICE_DONE.clear()
    cache = tmp_path / "cache"
    cache.mkdir()
    msg = cl._fu031_migration_notice(
        [_domain("Dom", cache, solver_set=False)], "/proj/old.blend")
    assert msg is not None, (
        "FU-031: legacy domain (unset solver + existing bake cache) must "
        "produce the migration notice")
    assert "Dom" in msg and "Solver=FLIP" in msg and "MPM" in msg


def test_fu031_latch_fires_once_per_file(tmp_path):
    cl = _load_cache_loader()
    cl._FU031_NOTICE_DONE.clear()
    cache = tmp_path / "cache"
    cache.mkdir()
    objs = [_domain("Dom", cache, solver_set=False)]
    assert cl._fu031_migration_notice(objs, "/proj/a.blend") is not None
    assert cl._fu031_migration_notice(objs, "/proj/a.blend") is None, (
        "FU-031: the latch must suppress a repeat for the same file")
    assert cl._fu031_migration_notice(objs, "/proj/b.blend") is not None, (
        "FU-031: a DIFFERENT legacy file must still get its own notice")


def test_fu031_silent_when_solver_explicitly_set(tmp_path):
    cl = _load_cache_loader()
    cl._FU031_NOTICE_DONE.clear()
    cache = tmp_path / "cache"
    cache.mkdir()
    assert cl._fu031_migration_notice(
        [_domain("Dom", cache, solver_set=True)], "x.blend") is None, (
        "FU-031: an explicit Solver choice means nothing changed — silent")


def test_fu031_silent_for_fresh_scene_without_cache(tmp_path):
    cl = _load_cache_loader()
    cl._FU031_NOTICE_DONE.clear()
    # unset solver but NO bake evidence: empty cache_dir / missing dir
    objs = [_domain("DomA", "", solver_set=False),
            _domain("DomB", tmp_path / "never_baked", solver_set=False)]
    assert cl._fu031_migration_notice(objs, "fresh.blend") is None, (
        "FU-031: fresh scenes (no bake cache) must stay silent — there is "
        "no old bake to reproduce, MPM is simply the correct default")


def test_fu031_wired_into_load_post():
    """The notice must actually be CALLED from _on_load_post (a perfect
    helper nobody calls is the round-59 class of bug). Source contract,
    comment/docstring-stripped."""
    code = _code_only(ADDON / "cache_loader" / "__init__.py")
    handler = code[code.find("def _on_load_post"):
                   code.find("def register_handler")]
    assert "_fu031_migration_notice(bpy.data.objects" in handler, (
        "FU-031: _on_load_post no longer invokes the migration notice")


# ── FU-029 source contracts ──────────────────────────────────────────────

def test_fu029_panel_alert_condition_exists():
    code = _code_only(ADDON / "panels.py")
    assert 'd.solver == "mpm" and d.cfl_factor > 1.0' in code, (
        "FU-029 regressed: the panel must alert when solver=mpm and "
        "cfl_factor > 1.0 (with use_cfl on)")
    pos = code.find('d.solver == "mpm" and d.cfl_factor > 1.0')
    assert ".alert = True" in code[pos:pos + 400], (
        "FU-029: the warning row must use the alert style, not a plain label")
    # the alert must live inside the use_cfl branch (only relevant when
    # adaptive substepping is actually on)
    cfl_branch = code.find("if d.use_cfl:")
    assert 0 < cfl_branch < pos, (
        "FU-029: the alert must be drawn inside the use_cfl branch")


def test_fu029_description_carries_mpm_guidance():
    code = _code_only(ADDON / "properties.py")
    pos = code.find("cfl_factor: bpy.props.FloatProperty(")
    block = code[pos:pos + 700]
    assert "MPM" in block and "under-substep" in block, (
        "FU-029 regressed: cfl_factor description must carry the MPM "
        "stress-wave guidance")
    assert "max=2.0" in block, (
        "FU-029: the range must stay max=2.0 (back-compat, no silent "
        "clamping) — guidance only")
