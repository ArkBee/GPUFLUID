"""Audit 2026-06-14 round 6 — under-covered surfaces (config/CLI/mesher/MPM).

5 confirmed findings (>=2/3 adversarial skeptics):
  #1 FLIP inflow-only scene silently seeded a phantom default [fluid] box
     (§9.6 drift vs the MPM source-selector's guard).
  #2 MPM resolved a relative cache_dir against CWD; FLIP against the scene
     dir — same string, two destinations (§9.6 drift).
  #3 GPU marching-cubes retry capped at range(2): a frame needing >2x the
     budget resized but never re-ran surface(), emitting the PREVIOUS
     frame's stale mesh.
  #4 a static coloured re-bake kept a prior bake's colour sidecar (the
     demo30 grey-water hazard, for the static branch).
  #5 color_mix_mode="off" was parsed but ignored (consumers only checked
     =="mixbox"), so colour was still written — opposite of the docstring.

#1/#2/#5 have functional tests; #3/#4 (GPU / re-bake disk state) are §9.12
source contracts on the comment/docstring-stripped source.
"""
from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _code(rel: str) -> str:
    src = (REPO / rel).read_text(encoding="utf-8")
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


def _scene(toml: str):
    from gpufluid.cli.config import load_scene
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scene.toml")
    Path(p).write_text(toml, encoding="utf-8")
    return load_scene(p), d


_INFLOW_ONLY = """
[domain]
resolution = [16,16,16]
[simulation]
solver = "flip"
[output]
cache_dir = "out"
[[inflow]]
lo = [0.2,0.2,0.8]
hi = [0.4,0.4,0.9]
"""


# ── #1 absent [fluid] -> None; authored [fluid] -> kept ────────────────────

def test_r6_1_inflow_only_has_no_static_fluid():
    scene, _ = _scene(_INFLOW_ONLY)
    assert scene.fluid is None, \
        "r6 #1: an absent [fluid] must parse to None, not a default box"
    assert scene.fluids == [] and len(scene.inflow) == 1


def test_r6_1_authored_fluid_is_preserved():
    from gpufluid.cli.config import FluidBoxCfg
    scene, _ = _scene(_INFLOW_ONLY + "\n[fluid]\nlo=[0.1,0.1,0.1]\nhi=[0.3,0.3,0.3]\n")
    assert isinstance(scene.fluid, FluidBoxCfg), \
        "r6 #1: an explicitly authored [fluid] must still parse to a box"


def test_r6_1_flip_seed_branch_guards_none():
    # The FLIP source-selector must not blindly seed scene.fluid.
    code = _code("src/gpufluid/cli/commands.py")
    assert "elif scene.fluid is not None:" in code, \
        "r6 #1: FLIP must only seed the single [fluid] when one was authored"


# ── #2 cache_dir resolves against the scene dir for BOTH solvers ───────────

def test_r6_2_resolve_cache_dir_uses_scene_dir():
    from gpufluid.cli.commands import _resolve_cache_dir
    scene, d = _scene(_INFLOW_ONLY)  # cache_dir = "out" (relative)
    got = _resolve_cache_dir(scene)
    assert got == (Path(d) / "out").resolve(), \
        "r6 #2: a relative cache_dir must resolve against the scene file's dir"
    assert got.is_absolute()


def test_r6_2_both_solver_paths_share_the_helper():
    code = _code("src/gpufluid/cli/commands.py")
    # The bare CWD-relative `.resolve()` on cache_dir is gone; both paths route
    # through the shared helper so they can't drift again.
    assert "Path(scene.output.cache_dir).resolve()" not in code, \
        "r6 #2: the MPM path must not resolve cache_dir against CWD"
    assert code.count("= _resolve_cache_dir(scene)") == 2, \
        "r6 #2: both the MPM and FLIP simulate paths must use the shared helper"


# ── #3 marching-cubes retry re-runs surface() and raises on exhaustion ─────

def test_r6_3_mc_retry_reruns_and_raises():
    code = _code("src/gpufluid/meshing/surface.py")
    assert "for _attempt in range(2):" not in code, \
        "r6 #3: the 2-attempt cap that left a stale mesh must be gone"
    assert "for _attempt in range(6):" in code
    # for/else: exhausting every retry must RAISE, not fall through to a
    # stale self._mc.verts read.
    assert "raise RuntimeError(" in code and "marching cubes buffer exhausted" in code, \
        "r6 #3: buffer exhaustion must raise instead of emitting a stale mesh"


# ── #4 static colour sidecar is written unconditionally (re-bake safe) ─────

def test_r6_4_static_sidecar_write_is_unconditional():
    code = _code("src/gpufluid/sim/mpm/solver.py")
    assert "if not once.exists():" not in code, \
        "r6 #4: the existence skip that kept a stale colour sidecar must be gone"
    # The static branch still uses the single shared frame_0000.npy, just
    # always overwriting it.
    assert "np.save(once, cur)" in code


# ── #5 color_mix_mode='off' validated + honoured ───────────────────────────

def test_r6_5_unknown_color_mix_mode_rejected():
    from gpufluid.blocks import BlockError
    bad = "[domain]\nresolution=[16,16,16]\n[output]\ncache_dir=\"out\"\ncolor_mix_mode=\"weird\"\n"
    with pytest.raises(BlockError):
        _scene(bad)


def test_r6_5_off_is_accepted_and_honoured():
    scene, _ = _scene("[domain]\nresolution=[16,16,16]\n[output]\ncache_dir=\"o\"\ncolor_mix_mode=\"off\"\n")
    assert scene.output.color_mix_mode == "off"
    code = _code("src/gpufluid/cli/commands.py")
    # Both colour-writing consumers (MPM + FLIP) must skip when "off".
    assert 'cfg.color_mix_mode != "off"' in code, \
        "r6 #5: the MPM colour pass must skip when color_mix_mode=='off'"
    assert 'str(scene.output.color_mix_mode) != "off"' in code, \
        "r6 #5: the FLIP colour pass must skip when color_mix_mode=='off'"
