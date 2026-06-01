"""S2.17.2 — MPM OBB (rotated box) collider drives downhill sliding.

round-57 added a native OBB box collider (rotation matrix in k_cube_pushback),
and round-59 tests its TOML emission/round-trip — but nothing tested the actual
PHYSICS: does a rotated ramp make water slide downhill in the right direction?

This locks the convention (verified on GPU 2026-06-02): the rotation matrix is
stored row-major and its COLUMNS are the box-local axes in world. A Y-axis
rotation R_y(+θ) — rows [c,0,s],[0,1,0],[-s,0,c] — tilts the top face so its
downhill direction is +x; R_y(-θ) -> downhill -x. (An earlier hand-TOML test
wrongly concluded OBB was broken because it dropped fluid PAST the raised high
edge instead of onto the surface.)

GPU-only (real MPM bake); auto-skips without CUDA via the `gpu` marker.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("warp")

pytestmark = pytest.mark.gpu


def _slide_mean_x(theta_deg: float) -> tuple[float, float, float]:
    """Drop a small fluid block onto a 30°-class ramp centred at (0.5,0.5,0.5)
    and return (start_mean_x, end_mean_x, end_mean_z) after a short bake."""
    from gpufluid.sim.mpm.solver import MpmConfig, MpmCubeCollider, MpmSolver
    from gpufluid.sim.mpm.seeding import seed_sphere
    c = math.cos(math.radians(theta_deg))
    s = math.sin(math.radians(theta_deg))
    # R_y(theta), row-major (columns = box-local axes in world).
    R = ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    # Replicates the decisive CLI bake: a sphere dropped onto the ramp centre.
    col = seed_sphere((0.5, 0.5, 0.62), 0.06, 1.0 / 64)
    cfg = MpmConfig(
        initial_column=col.astype(np.float32),
        n_grid=64, grid_lim=1.0, dt=0.0015, n_frames=60, dump_every=60,
        gravity=(0.0, 0.0, -9.81 / 2.0),  # 2 m domain (world_size_z=2)
        cubes=(MpmCubeCollider(centre=(0.5, 0.5, 0.5),
                               half_size=(0.24, 0.25, 0.025),
                               tangential_friction=0.0, rotation=R),),
        tap=None, anti_splash=None,
        adaptive_substep=True, adaptive_cfl=0.6, adaptive_max_substeps=16,
    )
    cfg.fluid.bulk_modulus = 1000.0
    try:
        solver = MpmSolver(cfg)
    except Exception:
        pytest.skip("no CUDA device for MpmSolver")
    start = float(solver.positions()[:, 0].mean())
    for k in range(1, 701):
        solver.step(k)
    pos = solver.positions()
    return start, float(pos[:, 0].mean()), float(pos[:, 2].mean())


def test_obb_positive_theta_slides_plus_x():
    start, end_x, end_z = _slide_mean_x(+30.0)
    assert end_z > 0.35, ("fluid should rest ON the tilted ramp (~z 0.5), not "
                          f"fall past it; got mean_z={end_z:.3f}")
    assert end_x > start + 0.02, (
        f"R_y(+30) must slide fluid downhill +x; start={start:.3f} end={end_x:.3f}")


def test_obb_negative_theta_slides_minus_x():
    start, end_x, end_z = _slide_mean_x(-30.0)
    assert end_z > 0.35, f"fluid should rest on the ramp; mean_z={end_z:.3f}"
    assert end_x < start - 0.02, (
        f"R_y(-30) must slide fluid downhill -x; start={start:.3f} end={end_x:.3f}")
