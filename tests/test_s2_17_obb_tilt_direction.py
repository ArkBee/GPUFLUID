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

The two slide tests are GPU-only (real MPM bake; `gpu` marker auto-skips
without CUDA). audit-20260610: the convention's matrix math itself is now
also locked by a CPU test so a warp-less CI run doesn't report this file as
fully green-with-skip while asserting nothing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import HAS_CUDA


def _rotation_y(theta_deg: float) -> np.ndarray:
    """R_y(theta), row-major — same layout the collider config stores."""
    c = math.cos(math.radians(theta_deg))
    s = math.sin(math.radians(theta_deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_rotation_convention_downhill_direction_cpu():
    """CPU lock of the documented convention — no warp/CUDA needed.

    COLUMNS of the row-major matrix are the box-local axes in world, so the
    ramp's top-face normal (box-local +z) in world is the THIRD COLUMN:
    (sinθ, 0, cosθ). Projecting gravity onto the tilted plane must give a
    downhill direction whose x-component has the sign of θ — exactly what
    the GPU slide tests below verify dynamically. If someone re-derives the
    matrix as ROWS-are-axes (the transpose), the normal flips to (-s, 0, c)
    and this test fails with the opposite downhill sign."""
    g = np.array([0.0, 0.0, -9.81])
    for theta, sign in ((+30.0, +1.0), (-30.0, -1.0)):
        R = _rotation_y(theta)
        n = R[:, 2]  # box-local +z (top-face normal) in world = third COLUMN
        assert n[2] > 0.5, "ramp must still face mostly up at 30°"
        downhill = g - np.dot(g, n) * n  # gravity component in the plane
        assert sign * downhill[0] > 0.1, (
            f"R_y({theta:+.0f}°) must give a downhill x of sign {sign:+.0f} "
            f"(columns-are-axes convention); got downhill={downhill}")
        assert abs(downhill[1]) < 1e-9, "Y-rotation must not tilt along y"


def _slide_mean_x(theta_deg: float) -> tuple[float, float, float]:
    """Drop a small fluid block onto a 30°-class ramp centred at (0.5,0.5,0.5)
    and return (start_mean_x, end_mean_x, end_mean_z) after a short bake."""
    from gpufluid.sim.mpm.solver import MpmConfig, MpmCubeCollider, MpmSolver
    from gpufluid.sim.mpm.seeding import seed_sphere
    R = tuple(tuple(row) for row in _rotation_y(theta_deg))
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
        # audit-20260610: skip ONLY when CUDA is genuinely absent — on a GPU
        # box a constructor exception is a regression and must FAIL, not
        # report a green skip.
        if HAS_CUDA:
            raise
        pytest.skip("no CUDA device for MpmSolver")
    start = float(solver.positions()[:, 0].mean())
    for k in range(1, 701):
        solver.step(k)
    pos = solver.positions()
    return start, float(pos[:, 0].mean()), float(pos[:, 2].mean())


@pytest.mark.gpu
def test_obb_positive_theta_slides_plus_x():
    pytest.importorskip("warp")
    start, end_x, end_z = _slide_mean_x(+30.0)
    assert end_z > 0.35, ("fluid should rest ON the tilted ramp (~z 0.5), not "
                          f"fall past it; got mean_z={end_z:.3f}")
    assert end_x > start + 0.02, (
        f"R_y(+30) must slide fluid downhill +x; start={start:.3f} end={end_x:.3f}")


@pytest.mark.gpu
def test_obb_negative_theta_slides_minus_x():
    pytest.importorskip("warp")
    start, end_x, end_z = _slide_mean_x(-30.0)
    assert end_z > 0.35, f"fluid should rest on the ramp; mean_z={end_z:.3f}"
    assert end_x < start - 0.02, (
        f"R_y(-30) must slide fluid downhill -x; start={start:.3f} end={end_x:.3f}")
