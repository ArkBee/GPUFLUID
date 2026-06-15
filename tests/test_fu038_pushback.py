"""FU-038 — sphere/cylinder obstacles get the particle-pushback second-layer.

Covers [BLK S2.17.2.SPH] (k_sphere_pushback) and [BLK S2.17.2.CYL]
(k_cylinder_pushback).

Box + mesh obstacles had a two-layer defense: a grid-velocity collider PLUS a
per-particle pushback (k_cube_pushback / k_mesh_sdf_pushback) that ejects a
particle that tunnels inside between substeps. Sphere/cylinder had only the grid
collider (audit-2026-06-15 r12 #4). This adds k_sphere_pushback /
k_cylinder_pushback, mirroring the cube path, and wires them into BOTH pushback
launch sites (_post_step + _apply_pushback).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_CUDA

REPO = Path(__file__).resolve().parents[1]
MPM = REPO / "src" / "gpufluid" / "sim" / "mpm"


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


# ── source contracts: kernels exist + launched in BOTH pushback sites ──────

def test_fu038_pushback_kernels_exist_and_wired():
    pb = _code(MPM / "pushback.py")
    assert "def k_sphere_pushback" in pb, "FU-038: sphere pushback kernel missing"
    assert "def k_cylinder_pushback" in pb, "FU-038: cylinder pushback kernel missing"
    solver = _code(MPM / "solver.py")
    # must be launched in BOTH _post_step and _apply_pushback (the §9.6 pair the
    # cube/mesh pushback already covers) — so check the launch appears twice.
    assert solver.count("wp.launch(k_sphere_pushback") == 2, \
        "FU-038: sphere pushback must launch in both _post_step and _apply_pushback"
    assert solver.count("wp.launch(k_cylinder_pushback") == 2, \
        "FU-038: cylinder pushback must launch in both pushback sites"


# ── GPU behaviour: interior particles are ejected fast (the pushback, not just
#    gravity/grid drift — 4 steps is too few for gravity to clear them) ──────

def _ejected_fraction(cfg_kw, centre, radius):
    from gpufluid.sim.mpm.solver import MpmConfig, MpmSolver, make_column
    # seed a small column entirely INSIDE the obstacle.
    col = make_column((centre[0], centre[1]), 0.05,
                      centre[2] - 0.05, centre[2] + 0.05, n_xy=6, n_z=4)
    seeded_inside = np.mean(
        np.linalg.norm(col - np.asarray(centre), axis=1) < 0.9 * radius)
    cfg = MpmConfig(n_grid=48, dt=0.0015, initial_column=col, **cfg_kw)
    solver = MpmSolver(cfg)
    for i in range(4):   # few steps: gravity barely moves them, pushback ejects
        solver.step(i)
    P = solver.positions()
    deep = np.mean(np.linalg.norm(P - np.asarray(centre), axis=1) < 0.6 * radius)
    return seeded_inside, deep


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_fu038_sphere_pushback_ejects_interior_particles():
    from gpufluid.sim.mpm.solver import MpmSphereCollider
    centre = (0.5, 0.5, 0.5); radius = 0.15
    seeded, deep = _ejected_fraction(
        dict(spheres=(MpmSphereCollider(centre=centre, radius=radius),)),
        centre, radius)
    assert seeded > 0.9, "test setup: particles must start inside the sphere"
    assert deep < 0.05, (
        f"FU-038: sphere pushback failed to eject interior particles "
        f"({deep:.0%} still deep inside after 4 steps)")


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_fu038_cylinder_pushback_ejects_interior_particles():
    from gpufluid.sim.mpm.solver import MpmCylinderCollider
    centre = (0.5, 0.5, 0.5); radius = 0.15
    seeded, deep = _ejected_fraction(
        dict(cylinders=(MpmCylinderCollider(
            centre=centre, radius=radius, half_height=0.2),)),
        centre, radius)
    assert seeded > 0.9, "test setup: particles must start inside the cylinder"
    assert deep < 0.05, (
        f"FU-038: cylinder pushback failed to eject interior particles "
        f"({deep:.0%} still deep inside after 4 steps)")
