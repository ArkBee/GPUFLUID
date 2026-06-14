"""Audit 2026-06-14 round 2 — meshing colour path (#3, #7, #13)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from gpufluid.blocks import BlockError

CMDS = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli" / "commands.py"


# ---- #3: the missing meshing.colormap module --------------------------------

def test_get_colormap_returns_rgb_in_unit_range():
    from gpufluid.meshing.colormap import get_colormap
    f = get_colormap("viridis")
    rgb = f(np.array([0.0, 5.0, 10.0], np.float32), 0.0, 10.0)
    assert rgb.shape == (3, 3) and rgb.dtype == np.float32
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_get_colormap_unknown_name_raises_blockerror():
    from gpufluid.meshing.colormap import get_colormap
    with pytest.raises(BlockError):
        get_colormap("definitely-not-a-real-colormap-xyz")


def test_get_colormap_flat_range_is_not_nan():
    from gpufluid.meshing.colormap import get_colormap
    f = get_colormap("inferno")
    rgb = f(np.array([5.0, 5.0], np.float32), 5.0, 5.0)  # t_min == t_max
    assert not np.isnan(rgb).any()


def test_meshing_colormap_module_importable():
    # #3 was a bare ModuleNotFoundError taking the whole mesh pass down.
    import importlib
    importlib.import_module("gpufluid.meshing.colormap")


# ---- #7: mixbox fallback must keep the NEAREST K, not arbitrary K -----------

def test_mixbox_fallback_keeps_nearest_neighbours(monkeypatch):
    pytest.importorskip("scipy")
    from gpufluid.meshing import mixbox as _mb
    # Force the no-mixbox FALLBACK branch (the buggy one) regardless of whether
    # pymixbox happens to be installed in this env.
    monkeypatch.setattr(_mb, "HAVE_MIXBOX", False)
    remix_vertices_mixbox = _mb.remix_vertices_mixbox
    rng = np.random.default_rng(0)
    far_blue = rng.uniform(0.03, 0.09, size=(60, 3)).astype(np.float32)
    # the single RED particle is the NEAREST (at the vertex) but the LAST index,
    # so a bare [:16] slice would exclude it; nearest-K must include it.
    pos = np.vstack([far_blue, [[0.0, 0.0, 0.0]]]).astype(np.float32)
    cols = np.vstack([np.tile([0.0, 0.0, 1.0], (60, 1)),
                      [[1.0, 0.0, 0.0]]]).astype(np.float32)
    out = remix_vertices_mixbox(np.zeros((1, 3), np.float32), pos, cols,
                                search_radius_world=0.2, max_neighbours=16)
    r, g, b = out[0]
    assert r > 200 and b < 60, f"nearest RED must dominate, got {out[0].tolist()}"


# ---- #13: sidecar shape-mismatch must warn, not silently white --------------

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


def test_sidecar_shape_mismatch_emits_warning():
    code = _code()
    assert "WARNING: temperatures sidecar" in code, \
        "audit #13: temperature-sidecar mismatch must warn"
    assert "WARNING: colours sidecar" in code, \
        "audit #13: colour-sidecar mismatch must warn"
