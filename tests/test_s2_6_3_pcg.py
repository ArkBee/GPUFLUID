"""Tests for S2.6.3 PCG pressure solve + S2.10 CFL + F3.4 step_cfl."""
import numpy as np
import pytest

from gpufluid import FlipSolver3D
from gpufluid.solvers.solver3d import cfl_substep_count

pytestmark = pytest.mark.gpu


def test_s2_10_cfl_picks_one_when_static():
    vel = np.zeros((100, 3), dtype=np.float32)
    assert cfl_substep_count(vel, dx=0.01, target_dt=0.01) == 1


def test_s2_10_cfl_grows_with_velocity():
    vel = np.array([[0.0, 100.0, 0.0]], dtype=np.float32)  # fast particle
    n = cfl_substep_count(vel, dx=0.01, target_dt=0.04, cfl=0.5, max_substeps=32)
    # CFL*dx/v = 0.5*0.01/100 = 5e-5 → 0.04 / 5e-5 = 800 → clamped to 32
    assert n == 32


def test_s2_6_3_pcg_matches_jacobi_within_tolerance():
    """For the same initial state, PCG and many-iter Jacobi should drive the
    particles to a very similar y-centroid after a few steps."""
    def run(pressure_solver, iters):
        s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
        s.seed_box(lo=(0.20, 0.30, 0.20), hi=(0.80, 0.90, 0.80), ppc=8)
        for _ in range(30):
            s.step(0.005, pressure_iters=iters, pressure_solver=pressure_solver)
        return s.get_particles()[0]

    pj = run("jacobi", 200)
    pp = run("pcg", 40)
    # centroids should match closely (same physics)
    cj = pj.mean(axis=0); cp = pp.mean(axis=0)
    assert np.linalg.norm(cj - cp) < 0.05, f"centroids diverged: {cj} vs {cp}"


def test_s2_6_3_pcg_finishes_in_fewer_iters_than_max():
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    s.seed_box(lo=(0.20, 0.30, 0.20), hi=(0.80, 0.90, 0.80), ppc=8)
    # one warmup
    s.step(0.005, pressure_iters=40, pressure_solver="pcg")
    s.step(0.005, pressure_iters=40, pressure_solver="pcg")
    # 16^3 with diagonal preconditioner converges far inside 40 iters
    assert s.last_pressure_iters < 40, f"PCG used full budget: {s.last_pressure_iters}"


def test_f3_4_step_cfl_returns_at_least_one():
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    s.seed_box(lo=(0.20, 0.30, 0.20), hi=(0.80, 0.90, 0.80), ppc=8)
    n = s.step_cfl(target_dt=0.005, pressure_iters=20)
    assert n >= 1
    # particles still valid
    pos, vel = s.get_particles()
    assert np.isfinite(pos).all()
    assert np.isfinite(vel).all()
