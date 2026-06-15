"""Audit 2026-06-15 round 12 — targeted §9.6 mirror-drift sweep (FLIP↔MPM).

  #1 FLIP seed_mesh hard-cast float(scale) and crashed on a per-axis 3-tuple
     that the MPM seed_mesh twin handles (FU-030 landed on MPM only).
  #2 the FLIP per-frame mesh write ignored [output] temperature_colormap that
     the MPM mesh pass honours (B11.3) — FLIP lava/blackbody bakes lost colour.
  #3 the MPM silent-source colour pad was WHITE while every FLIP/reseed pad is
     BLACK (the r2/r3-standardized convention); the same multi-source scene
     flipped colour between the two solvers.

#1 has a GPU functional test (in conftest-gated form) + a contract; #2/#3 are
§9.12 source contracts.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_CUDA

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


# ── #1 FLIP seed_mesh accepts a per-axis scale (no TypeError) ──────────────

def test_r12_1_flip_seed_mesh_scale_dispatch_contract():
    code = _code("src/gpufluid/solvers/solver3d.py")
    seed = code[code.find("def seed_mesh(self, mesh_path"):]
    seed = seed[:seed.find("def ", 10)]
    assert "if float(scale) != 1.0:" not in seed, \
        "r12 #1: FLIP seed_mesh must not hard-cast float(scale) (crashes on 3-tuple)"
    assert "sc = np.asarray(scale, dtype=np.float64).ravel()" in seed, \
        "r12 #1: FLIP seed_mesh must use the scalar/3-vector dispatch (mirror MPM)"


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_r12_1_flip_seed_mesh_per_axis_no_crash(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from gpufluid.solvers.solver3d import FlipSolver3D
    obj = tmp_path / "cube.obj"
    trimesh.creation.box(extents=(0.3, 0.3, 0.3)).export(obj)
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    # a per-axis scale (the FU-030 form the MPM twin + the schema accept) must
    # NOT raise the old `float((sx,sy,sz))` TypeError.
    s.seed_mesh(str(obj), ppc=4, scale=(0.5, 1.0, 1.0),
                translate=(0.4, 0.4, 0.4))
    assert s.n_particles > 0


# ── #2 FLIP mesh write honours temperature_colormap ────────────────────────

def test_r12_2_flip_mesh_honours_temperature_colormap():
    code = _code("src/gpufluid/cli/commands.py")
    # The FLIP block uses the underscore-prefixed `_cmap_name` locals (the MPM
    # pass uses `cmap_name`/`out_cfg`), so these tokens pin the FLIP path
    # specifically. The whole derivation must be present.
    assert "_cmap_name = scene.output.temperature_colormap" in code, \
        "r12 #2: the FLIP mesh block must read temperature_colormap"
    assert "_cmap = get_colormap(_cmap_name)" in code, \
        "r12 #2: FLIP must build the colormap fn like the MPM pass"
    assert "cols_np = _cmap(temps, t_min, t_max).astype(np.float32)" in code, \
        "r12 #2: FLIP must derive vertex RGB from the temperature ramp"


# ── #3 MPM silent-source colour pad is BLACK (matches FLIP/reseed) ─────────

def test_r12_3_mpm_silent_colour_pad_is_black():
    code = _code("src/gpufluid/sim/mpm/solver.py")
    assert "colors_np = np.zeros((n_particles, 3), dtype=np.float32)" in code, \
        "r12 #3: MPM silent-source colour pad must be black (np.zeros), like FLIP/reseed"
    assert "np.full((n_particles, 3), 1.0" not in code, \
        "r12 #3: the white (np.full 1.0) silent-source pad must be gone"
