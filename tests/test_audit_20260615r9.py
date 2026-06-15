"""Audit 2026-06-15 round 9 — numerical-kernel correctness.

Confirmed findings fixed here:
  #1 compute_vertex_colors reused a STALE HashGrid for trilinear/cubic FLIP —
     vertex colours were blended against frame-0 particle bins forever (§9.8).
  #3 the whitewater density classifier indexed the cell-centred density grid
     with a node-centred floor(p/dx) (½-cell off-by-one) — the MC cell-centring
     fix (audit-2026-06-14 #12) was not mirrored here (§9.6).

#2 (layer-gate ranks W7 by numeric suffix, hiding a real W7->I6 import) is a
REAL finding but its fix is architectural (refactor MpmSolver's direct io.ply
import, or formally document the exception against the empty-whitelist gate) —
deferred to follow-ups, not fixed here.

#3 is a functional test (pure numpy); #1 is a §9.12 source contract (GPU path).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

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


# ── #1 vertex-colour hashgrid rebuilt every frame ──────────────────────────

def test_r9_1_vertex_colours_rebuild_hashgrid_each_frame():
    code = _code(REPO / "src" / "gpufluid" / "meshing" / "surface.py")
    cvc = code[code.find("def compute_vertex_colors"):code.find("def _build_sdf_hashgrid")]
    assert "self._build_sdf_hashgrid(pos, search_radius_world)" in cvc, \
        "r9 #1: compute_vertex_colors must rebuild the hashgrid for this frame"
    # the stale `if self._sdf_grid is None:` cache-guard around the build is gone
    assert "if self._sdf_grid is None:" not in cvc, \
        "r9 #1: the build must be unconditional, not cached from frame 0"


# ── #3 whitewater classifier uses the cell-centred convention ──────────────

def test_r9_3_world_to_cell_is_cell_centred():
    from gpufluid.sim.whitewater import _world_to_cell
    dx = 0.1
    # a particle at the CENTRE of cell c sits at world (c+0.5)*dx -> reads cell c
    pos = np.array([[0.35, 0.35, 0.35]], dtype=np.float32)
    i, j, k = _world_to_cell(pos, dx, (10, 10, 10))
    assert (i[0], j[0], k[0]) == (3, 3, 3)
    # the discriminating case: frac(p/dx) < 0.5 must read the LOWER cell
    # (cell-centred), not the node-centred floor(p/dx).
    p2 = np.array([[0.32, 0.0, 0.0]], dtype=np.float32)  # 0.32/0.1 = 3.2
    ic, _, _ = _world_to_cell(p2, dx, (10, 10, 10))
    assert ic[0] == 2, "r9 #3: frac<0.5 must map to the cell-centred (lower) cell"
    # clamped at the bounds
    edge, _, _ = _world_to_cell(np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                                dx, (10, 10, 10))
    assert edge[0] == 0


def test_r9_3_classifier_uses_the_shared_helper():
    # both read sites must route through _world_to_cell (so they can't drift
    # back to node-centred floor(p/dx)).
    code = _code(REPO / "src" / "gpufluid" / "sim" / "whitewater.py")
    assert code.count("_world_to_cell(") >= 3, \
        "r9 #3: classify_kinds + step + the def must all use _world_to_cell"
    assert "np.floor(pos[:, 0] / dx - 0.5)" in code, \
        "r9 #3: the helper must use the cell-centred (- 0.5) convention"
    assert "(self.pos[:, 0] / dx).astype(np.int32)" not in code, \
        "r9 #3: the old node-centred indexing must be gone"
