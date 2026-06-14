"""Audit 2026-06-15 round 7 — under-covered surfaces + the new presets feature.

4 confirmed findings (>=2/3 adversarial skeptics):
  #1 the bake operator's rc=2 recovery message recognised only
     'mpm_divergence' — a diverged FLIP bake lost its actionable message
     (§9.6 drift; CLI/manifest/cache_loader all handle both reasons).
  #2 the bake pre-run stale-frame strip used a bare shutil.rmtree, not the
     round-60 _robust_rmtree — a swallowed WinError 145 leaves stale tail
     frames that inflate frame_count (§9.6 vs OT_clear_cache).
  #3 the slow-path surface rebuild (_rebuild_mesh) ignored dom_size, so a
     non-unit domain collapsed the water into a unit cube in the corner
     (§9.6; the fast + whitewater rebuilds already scale by dom_size).
  #4 decimate_ratio had no lower bound — <=0 collapsed the mesh to 4 faces.

#4 is a pure-function + config functional test; #1/#2/#3 live in bpy files
so they are §9.12 source contracts (comment/docstring-stripped).
"""
from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "addon" / "gpufluid_blender"


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


# ── #4 decimate lower-bound (pure function + config) ───────────────────────

def test_r7_4_decimate_nonpositive_is_noop():
    from gpufluid.meshing.decimate import decimate_mesh
    v = np.random.rand(20, 3).astype(np.float32)
    f = np.arange(15, dtype=np.int32).reshape(5, 3)
    for ratio in (0.0, -0.5, -1e9):
        v2, f2 = decimate_mesh(v, f, ratio)
        assert f2 is f, f"ratio {ratio}: must be a no-op, not collapse to 4 faces"


def test_r7_4_config_rejects_bad_decimate_ratio():
    from gpufluid.cli.config import load_scene
    from gpufluid.blocks import BlockError
    for bad in (0.0, -0.3, 1.5):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "s.toml")
        Path(p).write_text(
            f'[domain]\nresolution=[16,16,16]\n[output]\ncache_dir="o"\n'
            f'decimate_ratio={bad}\n', encoding="utf-8")
        with pytest.raises(BlockError):
            load_scene(p)


def test_r7_4_valid_decimate_ratio_still_parses():
    from gpufluid.cli.config import load_scene
    d = tempfile.mkdtemp()
    p = os.path.join(d, "s.toml")
    Path(p).write_text('[domain]\nresolution=[16,16,16]\n[output]\n'
                       'cache_dir="o"\ndecimate_ratio=0.5\n', encoding="utf-8")
    scene = load_scene(p)
    assert scene.output.decimate_ratio == 0.5


# ── #1 FLIP divergence gets a recovery message too ─────────────────────────

def test_r7_1_friendly_error_handles_both_divergence_reasons():
    code = _code(ADDON / "operators" / "bake.py")
    assert 'reason == "mpm_divergence"' in code
    assert 'reason == "flip_divergence"' in code, \
        "r7 #1: a diverged FLIP bake must get its own recovery message"
    assert '"FLIP solver"' in code
    # the early-return must now be on the UNKNOWN reason, not on '!= mpm'
    assert '!= "mpm_divergence"' not in code, \
        "r7 #1: the MPM-only early-return must be gone"


# ── #2 stale-strip uses the hardened rmtree ────────────────────────────────

def test_r7_2_bake_stale_strip_uses_robust_rmtree():
    code = _code(ADDON / "operators" / "bake.py")
    assert "from .helpers import _robust_rmtree" in code, \
        "r7 #2: bake must import the hardened _robust_rmtree (lazily)"
    assert "_robust_rmtree(str(d))" in code, \
        "r7 #2: the stale-frame strip must use _robust_rmtree, not bare rmtree"
    assert "_shutil.rmtree(d)" not in code, \
        "r7 #2: the bare shutil.rmtree in the strip must be gone"


# ── #3 slow-path rebuild scales by dom_size ────────────────────────────────

def test_r7_3_rebuild_mesh_scales_by_dom_size():
    code = _code(ADDON / "cache_loader" / "__init__.py")
    assert "def _rebuild_mesh(obj, verts, faces, origin, dom_size=(1.0, 1.0, 1.0)):" in code, \
        "r7 #3: _rebuild_mesh must take a dom_size parameter"
    # the body must scale by dom_size before the origin offset (mirror the
    # fast path), not just `verts + origin`.
    assert "verts * ds + np.asarray(origin" in code, \
        "r7 #3: _rebuild_mesh must scale verts by dom_size"
    assert "verts + np.asarray(origin, dtype=np.float32)).astype" not in code, \
        "r7 #3: the unscaled `verts + origin` form must be gone"
    # and the call site must pass it
    assert "_rebuild_mesh(obj, verts, faces, origin, dom_size)" in code, \
        "r7 #3: the handler call site must forward dom_size"
    assert "dom_size = list(_cb.get_cache_dom_size(obj))" in code
