"""Layer F3 tests — solver smoke tests (S2.3 gravity, F3.3 step pipeline)."""
import numpy as np
import pytest
import warp as wp

from gpufluid import FlipSolver3D

pytestmark = pytest.mark.gpu


def test_f3_2_solver_init_grid_shapes():
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    assert s.u.shape == (17, 16, 16)
    assert s.v.shape == (16, 17, 16)
    assert s.w.shape == (16, 16, 17)
    assert s.marker.shape == (16, 16, 16)


def test_f3_3_step_no_nan_after_50_steps():
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    s.seed_box(lo=(0.10, 0.10, 0.10), hi=(0.50, 0.90, 0.50), ppc=8)
    n0 = s.n_particles
    for _ in range(50):
        s.step(0.005, pressure_iters=20)
    pos, vel = s.get_particles()
    assert s.n_particles == n0
    assert np.isfinite(pos).all()
    assert np.isfinite(vel).all()
    # Particles should stay in the open box (1-cell wall buffer).
    assert pos[:, 0].min() >= 0.0 and pos[:, 0].max() <= s.dom[0]
    assert pos[:, 1].min() >= 0.0 and pos[:, 1].max() <= s.dom[1]
    assert pos[:, 2].min() >= 0.0 and pos[:, 2].max() <= s.dom[2]


def test_s2_3_gravity_lowers_centre_of_mass():
    """After running with gravity, the y-centroid of fluid should drop."""
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16, flip_blend=0.95)
    s.seed_box(lo=(0.20, 0.55, 0.20), hi=(0.80, 0.90, 0.80), ppc=8)
    y0 = s.get_particles()[0][:, 1].mean()
    for _ in range(60):
        s.step(0.005, pressure_iters=30)
    y1 = s.get_particles()[0][:, 1].mean()
    assert y1 < y0 - 0.05, f"centroid did not fall: {y0=:.3f} {y1=:.3f}"


def test_d4_4_obstacle_marker_set_correct_count():
    from gpufluid import sdf_sphere
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    sdf = sdf_sphere(s.cell_centers_np(), center=(0.5, 0.5, 0.5), radius=0.1)
    s.add_solid_from_sdf(sdf)
    m = s.marker.numpy()
    # exterior shell (14³ - 12³ outer surface omitted; just count interior solid cells)
    interior_solid = ((m == 2) & np.pad(np.zeros((14, 14, 14), dtype=bool),
                                        1, constant_values=False)).sum()
    # original wall shell is all i/j/k=0 or last. Interior solid should be > 0.
    assert (m == 2).sum() > (16 * 16 * 16 - 14 * 14 * 14)
