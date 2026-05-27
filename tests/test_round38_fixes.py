"""Round-38 regression tests for round-37 reviewer findings.

  - Keyframe explosion: bake clamps animated-emitter loop to dprops.frames.
  - MPM _patches tolerates read-only install (catches OSError).
  - start_sync timeout flushes drained stdout via logger.
  - _export_obj triangulates via loop_triangles (no fan-on-concave-ngon).
  - _safe_obj_name strips Windows-illegal filename chars.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


# ─── 1. Keyframe loop clamp source-grep ─────────────────────────────────

def test_bake_clamps_keyframe_loop_to_dprops_frames():
    """Round-38 contract (lesson §9.12): keyframe loop in bake.py
    must NOT iterate `range(fs, fe + 1)` raw — it must clamp `fe`
    to `min(fe, fs + dprops.frames)` so a default `frame_end=10000`
    doesn't trigger 10001-row TOML explosion."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "bake.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    # Pre-round-38 form was `for f in range(fs, fe + 1):` directly
    # after the animated-emitter `if`. Now there should be an
    # `fe_eff = min(fe, fs + ...)` BEFORE the range loop.
    assert "fe_eff" in code or "min(fe," in code, (
        "round-38 regressed: bake.py must clamp keyframe loop range")


# ─── 2. MPM _patches tolerates read-only install ────────────────────────

def test_patch_slip_returns_false_on_oserror(monkeypatch, tmp_path):
    """Round-38: pre-round-38 a PermissionError from write_text
    propagated → entire MPM import path crashed with opaque OS error.
    Now: catches OSError, logs warning, returns False."""
    # Import via the regular package path so the relative imports resolve.
    from gpufluid.sim.mpm import _patches as mod

    # Write a fake solver.py that already has the post-fix form but
    # NOT the marker — the fallback branch fires (line 65) and tries
    # to write the marker. That write is what we want to fail.
    fake_solver = tmp_path / "mpm_solver.py"
    fake_solver.write_text(
        "# fake warp-mpm body\n"
        "state.grid_v_out[grid_x, grid_y, grid_z] = v\n",
        encoding="utf-8")
    # Monkeypatch write_text to raise PermissionError.
    def _ro_write(*a, **k):
        raise PermissionError("read-only install simulation")
    monkeypatch.setattr(Path, "write_text", _ro_write)
    result = mod._patch_slip(fake_solver)
    assert result is False, (
        "round-38 contract: read-only write must NOT raise; return False")


# ─── 3. start_sync timeout flushes drained log ──────────────────────────

def test_start_sync_timeout_flushes_drained_log():
    """Source-grep contract: TimeoutExpired branch must reference
    `drain_thread.join` and write `drained` lines via logger.
    Pre-round-38 the kill path returned without flushing — user lost
    all stdout that led up to the hang."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "_runner.py").read_text()
    # Find the TimeoutExpired branch body
    pos = src.find("except subprocess.TimeoutExpired:")
    next_branch = src.find("\n            drain_thread.join", pos)
    # The join must happen BEFORE the kill-path return — so the join
    # must appear inside the except block.
    end_of_except = src.find("return {\"CANCELLED\"}", pos)
    body = src[pos:end_of_except]
    assert "drain_thread.join" in body, (
        "round-38 regressed: TimeoutExpired must join drainer")
    assert "drained" in body, (
        "round-38 regressed: TimeoutExpired must flush drained lines")


# ─── 4. _export_obj triangulates ────────────────────────────────────────

def test_export_obj_uses_loop_triangles_not_polygons():
    """Round-38 contract (source-grep): _export_obj must iterate
    `loop_triangles` not raw `polygons`. Pre-round-38 raw polygons
    were fan-triangulated downstream by trimesh, which is WRONG for
    concave n-gons (fan crosses outside polygon → solid mask leak)."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "_collect.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    fn_pos = code.find("def _export_obj")
    next_def = code.find("\ndef ", fn_pos + 1)
    body = code[fn_pos:next_def if next_def > 0 else None]
    assert "loop_triangles" in body, (
        "round-38 regressed: _export_obj must triangulate via "
        "loop_triangles, not iterate raw mesh.polygons")
    assert "calc_loop_triangles" in body


# ─── 5. _safe_obj_name strips dangerous chars ───────────────────────────

def test_safe_obj_name_strips_windows_illegal_chars():
    spec = importlib.util.spec_from_file_location(
        "_test_collect",
        _ADDON_DIR / "gpufluid_blender" / "operators" / "_collect.py",
    )
    # Stub bpy for the import.
    fake_bpy = types.ModuleType("bpy")
    sys.modules.setdefault("bpy", fake_bpy)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._safe_obj_name("ok_name") == "ok_name"
    assert mod._safe_obj_name("Obstacle.001") == "Obstacle.001"
    assert mod._safe_obj_name("bad/name") == "bad_name"
    assert mod._safe_obj_name("foo\\bar") == "foo_bar"
    assert mod._safe_obj_name("a:b*c?d<e>f|g\"h") == "a_b_c_d_e_f_g_h"


def test_safe_obj_name_preserves_unicode_letters_drops_them():
    """Unicode letters get dropped by the conservative ASCII regex —
    this is intentional (Windows is fine with Unicode but the safest
    path-component set is [A-Za-z0-9._-])."""
    spec = importlib.util.spec_from_file_location(
        "_test_collect2",
        _ADDON_DIR / "gpufluid_blender" / "operators" / "_collect.py",
    )
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Cyrillic name becomes underscores (acceptable trade — Blender
    # also keeps the original name on the object, only the file name
    # is sanitised).
    assert "_" in mod._safe_obj_name("Куб")
