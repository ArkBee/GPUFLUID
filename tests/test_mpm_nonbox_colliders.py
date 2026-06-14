"""Non-box obstacles must COLLIDE, not be ghosts.

Goal 2026-06-14: a 4-side behaviour validation of a "tipped bucket" pour found
that sphere/cylinder obstacles were RENDERED (FU-033) but never simulated —
water fell straight through them (settled penetration 4-12%). This adds proper
SDF grid colliders for spheres ([BLK S2.17.1.SPH]) and cylinders
([BLK S2.17.1.CYL]); after the fix the same scene has 0% penetration.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_CUDA

REPO = Path(__file__).resolve().parents[1]
MPM = REPO / "src" / "gpufluid" / "sim" / "mpm"


# ---------------------------------------------------------------- CPU contracts

def test_collider_dataclasses_carry_geometry():
    from gpufluid.sim.mpm.solver import (
        MpmConfig, MpmSphereCollider, MpmCylinderCollider)
    s = MpmSphereCollider(centre=(0.5, 0.5, 0.3), radius=0.12)
    c = MpmCylinderCollider(centre=(0.7, 0.5, 0.1), radius=0.09, half_height=0.2)
    assert s.radius == 0.12 and s.centre == (0.5, 0.5, 0.3)
    assert c.half_height == 0.2
    # MpmConfig accepts them (was box-only before the goal-2026-06-14 fix).
    cfg = MpmConfig(spheres=(s,), cylinders=(c,))
    assert cfg.spheres and cfg.cylinders


def _code(path: Path) -> str:
    """Source minus comments + docstrings (§9.12)."""
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


def test_sphere_and_cylinder_kernels_exist_and_are_registered():
    coll = _code(MPM / "colliders.py")
    assert "def k_sdf_sphere_collide" in coll, "S2.17.1.SPH kernel missing"
    assert "def k_sdf_cylinder_collide" in coll, "S2.17.1.CYL kernel missing"
    solver = _code(MPM / "solver.py")
    # the solver must LAUNCH them (append to grid_postprocess), not just import.
    assert "grid_postprocess.append(k_sdf_sphere_collide)" in solver, \
        "sphere collider not registered in the MPM step"
    assert "grid_postprocess.append(k_sdf_cylinder_collide)" in solver, \
        "cylinder collider not registered in the MPM step"


def test_cli_wires_nonbox_obstacles_to_colliders():
    cmds = _code(REPO / "src" / "gpufluid" / "cli" / "commands.py")
    # both non-box obstacle types must map to a collider — regression guard
    # against reverting to the box-only loop where they were ghosts.
    assert re.search(r"ObstacleSphereCfg[\s\S]{0,120}MpmSphereCollider", cmds), \
        "sphere obstacle not wired to a collider"
    assert re.search(r"ObstacleCylinderYCfg[\s\S]{0,160}MpmCylinderCollider", cmds), \
        "cylinder obstacle not wired to a collider"


# ---------------------------------------------------------------- GPU behaviour

@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_water_does_not_pass_through_a_sphere_collider():
    """Drop a column of water onto a sphere collider and step it; afterwards
    almost no particle may sit DEEP inside the sphere (the ghost-obstacle bug
    left 4-12% there)."""
    pytest.importorskip("warp")
    from gpufluid.sim.mpm.solver import (
        MpmConfig, MpmSolver, MpmSphereCollider, make_column)

    centre = (0.5, 0.5, 0.32)
    radius = 0.13
    col = make_column((0.5, 0.5), 0.10, 0.55, 0.75, n_xy=12, n_z=8)
    cfg = MpmConfig(
        n_grid=64, dt=0.0015, initial_column=col,
        spheres=(MpmSphereCollider(centre=centre, radius=radius),),
    )
    try:
        solver = MpmSolver(cfg)
    except Exception:
        if HAS_CUDA:
            raise
        pytest.skip("no CUDA device for MpmSolver construction")
    for i in range(160):  # let the water fall onto + flow around the sphere
        solver.step(i)
    P = solver.positions()
    c = np.asarray(centre)
    deep = np.mean(np.linalg.norm(P - c, axis=1) < 0.6 * radius)
    assert deep < 0.02, (
        f"water penetrated the sphere collider: {deep:.1%} of particles deep "
        f"inside (>2%) — non-box obstacle is acting as a ghost again")


def test_mesh_pushback_kernel_exists_and_is_wired():
    """[BLK S2.17.2.MESH] — mesh obstacles collide via a precomputed-SDF
    particle pushback (they used to be ghosts)."""
    pb = _code(MPM / "pushback.py")
    assert "def k_mesh_sdf_pushback" in pb, "S2.17.2.MESH kernel missing"
    solver = _code(MPM / "solver.py")
    assert "wp.launch(k_mesh_sdf_pushback" in solver, \
        "mesh collider pushback not launched in the step loop"
    cmds = _code(REPO / "src" / "gpufluid" / "cli" / "commands.py")
    # the collider loop must build a mesh collider (unambiguous — the OTHER
    # ObstacleMeshCfg branch, in _build_obstacle_sdf, is the mesher path).
    assert "meshes.append(MpmMeshCollider(" in cmds, \
        "mesh obstacle not wired to a collider in the obstacle loop"


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_water_does_not_pass_through_a_mesh_collider():
    """A synthetic sphere-shaped SDF used as a MESH collider must block water
    [BLK S2.17.2.MESH] — drop a column on it, assert <2% ends deep inside."""
    pytest.importorskip("warp")
    from gpufluid.sim.mpm.solver import (
        MpmConfig, MpmSolver, MpmMeshCollider, make_column)

    N = 64
    dx = 1.0 / N
    centre = np.array([0.5, 0.5, 0.32])
    radius = 0.13
    ax = np.arange(N) * dx
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    sdf = (np.sqrt((gx - centre[0]) ** 2 + (gy - centre[1]) ** 2
                   + (gz - centre[2]) ** 2) - radius).astype(np.float32)
    col = make_column((0.5, 0.5), 0.10, 0.55, 0.75, n_xy=12, n_z=8)
    cfg = MpmConfig(
        n_grid=N, dt=0.0015, initial_column=col,
        meshes=(MpmMeshCollider(sdf=sdf, nx=N, ny=N, nz=N, dx=dx),))
    try:
        solver = MpmSolver(cfg)
    except Exception:
        if HAS_CUDA:
            raise
        pytest.skip("no CUDA device for MpmSolver construction")
    for i in range(160):
        solver.step(i)
    P = solver.positions()
    deep = np.mean(np.linalg.norm(P - centre, axis=1) < 0.6 * radius)
    assert deep < 0.02, (
        f"water passed through the mesh-SDF collider: {deep:.1%} deep inside")
