"""Audit 2026-06-14 round 2 — whitewater density-field misuse (#2, #5, #12).

extractor.dens is only the calibrated [0,1] DENSITY field the classifier expects
when mesh_method=='trilinear'. For 'sdf' it is a signed distance (inverts
spray/bubble), for 'cubic' it is inflated (>1 -> almost all bubble), and on an
empty-particle frame it is STALE. The fix passes None in those cases, and the
classifier's documented no-grid fallback is "everything is foam".
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from gpufluid.sim.whitewater import WhitewaterSystem, KIND_FOAM

CMDS = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli" / "commands.py"


def test_classify_kinds_none_density_is_all_foam():
    """The fallback the fix relies on: no density grid -> all foam, never a
    misclassification against the wrong field."""
    ws = WhitewaterSystem()
    pos = np.random.default_rng(0).uniform(0, 1, size=(20, 3)).astype(np.float32)
    kind = ws.classify_kinds(pos, None, dx=1.0 / 32.0)
    assert (kind == KIND_FOAM).all(), "no-density classify must be all foam"


def _code():
    src = CMDS.read_text(encoding="utf-8")
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


def test_ww_density_gated_on_trilinear_and_nonempty_frame():
    code = _code()
    # the density must NOT be handed to whitewater unconditionally any more
    assert 'extractor.dens.numpy() if extractor is not None else None' not in code, \
        "audit #2/#5/#12 regressed: ww_density passed for any mesh_method"
    # it must be gated on trilinear + a non-empty extracted frame
    assert 'scene.output.mesh_method == "trilinear"' in code, \
        "audit #2/#5: ww_density must be gated on mesh_method=='trilinear'"
    assert "verts is not None and len(verts) > 0" in code, \
        "audit #12: ww_density must require a non-empty current frame (no stale dens)"
