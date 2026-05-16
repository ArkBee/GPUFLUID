"""S2.6.2 Gauss-Seidel red-black regression."""
import numpy as np
import pytest
from gpufluid import FlipSolver3D

pytestmark = pytest.mark.gpu


def test_s2_6_2_gsrb_matches_jacobi_with_fewer_iters():
    """GS-RB at half Jacobi iter count should reach similar particle positions."""
    def run(solver, iters):
        s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1/16)
        s.seed_box(lo=(0.20, 0.30, 0.20), hi=(0.50, 0.70, 0.50), ppc=8)
        for _ in range(30):
            s.step(0.005, pressure_iters=iters, pressure_solver=solver)
        return s.get_particles()[0]

    pj = run("jacobi", 200)
    pg = run("gsrb", 100)   # GS-RB roughly 2x convergence rate
    cj = pj.mean(0); cg = pg.mean(0)
    assert np.linalg.norm(cj - cg) < 0.05, f"GSRB centroid diverges from Jacobi: {cj} vs {cg}"


def test_s2_6_2_gsrb_runs():
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1/16)
    s.seed_box(lo=(0.20, 0.30, 0.20), hi=(0.50, 0.70, 0.50), ppc=8)
    s.step(0.005, pressure_iters=30, pressure_solver="gsrb")
    pos, _ = s.get_particles()
    assert np.isfinite(pos).all()
