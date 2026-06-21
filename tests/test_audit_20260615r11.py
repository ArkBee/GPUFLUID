"""Audit 2026-06-15 round 11 — two §9.6 CPU/GPU & floor/obstacle mirror-drifts.

  #1 CPU reseed left particles trapped inside solid obstacle cells; the GPU
     sibling (k_mark_keep_by_rank) drops them. Below RESEED_GPU_THRESHOLD the
     CPU path accumulated solid-trapped particles (iso-surface bleeds into the
     obstacle).
  #2 the MPM floor slip-plane collider inherited warp-mpm's default
     end_time=999.0 while every obstacle collider set end_time=1.0e9, so a bake
     over 999 simulated seconds silently lost the floor.

#1 is a functional reseed test; #2 is a §9.12 source contract.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

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


# ── #1 CPU reseed removes solid-trapped particles (parity with GPU) ────────

def test_r11_reseed_drops_solid_trapped_particle():
    from gpufluid.sim.reseed import reseed_particles, ReseedConfig
    nx = ny = nz = 4
    dx = 0.25
    marker = np.ones((nx, ny, nz), dtype=np.int32)  # all fluid
    marker[1, 1, 1] = 2                              # one solid cell
    pos = np.array([
        [0.375, 0.375, 0.375],   # -> cell (1,1,1) SOLID, must be dropped
        [0.125, 0.125, 0.125],   # -> cell (0,0,0) fluid, must survive
    ], dtype=np.float32)
    vel = np.zeros((2, 3), dtype=np.float32)
    cfg = ReseedConfig(min_per_cell=0, max_per_cell=1000)  # no emit, no cull
    rng = np.random.default_rng(0)
    new_pos, new_vel, n_emitted, n_culled = reseed_particles(
        pos, vel, marker, dx, cfg, rng)
    assert len(new_pos) == 1, \
        f"solid-trapped particle must be dropped (got {len(new_pos)} survivors)"
    assert np.allclose(new_pos[0], [0.125, 0.125, 0.125]), \
        "the fluid-cell particle must survive"
    # closing-review: the solid drop MUST show up in the returned cull count —
    # this is no emit + no capacity cull, so without it the CLI commit guard
    # (`n_emit>0 or n_cull>0`) would discard the result and the fix is inert.
    assert n_emitted == 0 and n_culled == 1, \
        f"solid drop must be counted in n_culled (got emit={n_emitted}, cull={n_culled})"


def test_r11_reseed_solid_drop_contract():
    code = _code("src/gpufluid/sim/reseed.py")
    assert "keep_mask &= ~solid_mask[ix, iy, iz]" in code, \
        "r11 #1: the CPU reseed must drop solid-trapped particles (mirror GPU)"


# ── #2 floor collider shares the obstacle colliders' unbounded lifetime ────

def test_r11_floor_collider_has_unbounded_end_time():
    code = _code("src/gpufluid/sim/mpm/solver.py")
    # the floor add_surface_collider call must pass end_time (else it expires at
    # warp-mpm's 999.0 default while obstacles persist at 1.0e9). S2.17.10 made
    # surface/friction variables (sticky-floor support), so assert the invariant
    # that survives: the floor collider call carries end_time=1.0e9.
    assert "surface=_floor_surface, friction=_floor_fric, end_time=1.0e9" in code, \
        "r11 #2: the floor collider must still set end_time=1.0e9 like obstacles"
