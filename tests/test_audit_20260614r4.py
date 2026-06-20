"""Audit 2026-06-14 round 4 — USD / CLI-render / addon-obstacle fixes.

  #1 USD writer authored no subdivisionScheme → importers smoothed the
     mesh as catmullClark. (Functional test lives in test_i6_5_usd.py,
     where the pxr round-trip already runs.)
  #2 the FLIP BBOX→MESH auto-promotion emitted no scale/translate, so the
     collider landed at raw world-metre coords (outside [0,1]³) and
     silently vanished. §9.6 mirror of the explicit MESH branch.
  #3 cmd_render let a missing/typo'd Blender executable escape as a raw
     FileNotFoundError traceback instead of a one-line BlockError.
  #5 cmd_render returned Blender's rc verbatim — rc=0 over an empty render
     (no PNG written) reported success blindly.

#2 and #5 are addon/CLI text contracts (§9.12, comment/docstring-stripped
source); #3 is a real functional call into cmd_render with a bogus binary.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


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


# ── #2 both MESH-obstacle branches carry the world→[0,1]³ map ───────────────

def test_r4_2_both_mesh_obstacle_branches_emit_scale_and_translate():
    code = _code(REPO / "addon" / "gpufluid_blender" / "operators" / "bake.py")
    # Explicit MESH branch + auto-promoted BBOX→MESH branch must BOTH set the
    # unit-cube scale and the origin translate. If either drops them,
    # config_builder defaults to scale=1/translate=0 and the collider is lost.
    assert code.count('"scale": avg_inv,') == 2, \
        "r4 #2: both MESH-obstacle branches must set scale=avg_inv"
    assert code.count("-origin.x * avg_inv, -origin.y * avg_inv,") == 2, \
        "r4 #2: both MESH-obstacle branches must set the origin translate"
    # audit-2026-06-15r14 (§9.6): both mesh-emitting branches share the uniform-
    # scale defect, so both must emit the non-cubic anisotropy warning — the
    # explicit-MESH branch had it; the auto-promote branch did not.
    assert code.count("squashed on the short domain axis") == 2, \
        "r14: both MESH-obstacle branches must warn on a non-cubic domain"


# ── #3 a bad Blender executable surfaces as a clean BlockError ──────────────

def _render_args(cache: Path, scene: Path, out: Path, blender: str):
    return argparse.Namespace(
        cache=str(cache), scene=str(scene), out=str(out),
        label="Water", color=[0.2, 0.5, 0.7], samples=16, fps=24,
        frames=0, blender=blender, material=None, cam_angle=None)


def test_r4_3_missing_blender_raises_blockerror(tmp_path):
    from gpufluid.blocks import BlockError
    from gpufluid.cli import commands

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache.json").write_text("{}", encoding="utf-8")  # passes the existence gate
    (cache / "scene.toml").write_text("", encoding="utf-8")
    out = tmp_path / "render_out"

    args = _render_args(cache, cache / "scene.toml", out,
                        blender="gpufluid_no_such_blender_binary_xyz")
    with pytest.raises(BlockError) as ei:
        commands.cmd_render(args)
    msg = str(ei.value)
    assert "Blender" in msg and "PATH" in msg, \
        "r4 #3: error must name the missing Blender exe and how to fix it"


# ── #5 an empty render (rc=0, no PNG) is no longer reported as success ──────

def test_r4_5_cmd_render_warns_on_zero_png():
    code = _code(REPO / "src" / "gpufluid" / "cli" / "commands.py")
    assert 'out_dir.glob("**/*.png")' in code, \
        "r4 #5: cmd_render must count PNGs written before trusting rc==0"
    assert "if n_png == 0:" in code, \
        "r4 #5: the zero-PNG case must be flagged, not silently reported OK"
    assert "file=sys.stderr" in code, \
        "r4 #5: the empty-render warning must go to stderr"
