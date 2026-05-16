"""Tests for F3.6 prepare_frame integration."""
import numpy as np
import pytest

from gpufluid import FlipSolver3D
from gpufluid.domain.regions import InflowBox, OutflowBox

pytestmark = pytest.mark.gpu


def test_f3_6_inflow_grows_particle_count():
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    s.seed_box(lo=(0.30, 0.30, 0.30), hi=(0.40, 0.40, 0.40), ppc=8)
    n0 = s.n_particles
    s.add_inflow(InflowBox(lo=(0.1, 0.8, 0.1), hi=(0.2, 0.9, 0.2),
                           velocity=(0, -2, 0), rate_per_sec=2400))
    for f in range(5):
        s.prepare_frame(f, 1.0 / 24)
        # one solver step so particles stick around in arrays
        s.step(0.005, pressure_iters=15)
    assert s.n_particles > n0


def test_f3_6_outflow_removes_particles_in_region():
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    s.seed_box(lo=(0.10, 0.10, 0.10), hi=(0.30, 0.30, 0.30), ppc=8)
    n0 = s.n_particles
    # an outflow that exactly covers the seeded region: everything gets dropped
    s.add_outflow(OutflowBox(lo=(0.05, 0.05, 0.05), hi=(0.35, 0.35, 0.35)))
    s.prepare_frame(0, 1.0 / 24)
    assert s.n_particles == 0, f"expected 0 after outflow, got {s.n_particles} (was {n0})"


def test_f3_6_outflow_keeps_outside_particles():
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    s.seed_box(lo=(0.50, 0.50, 0.50), hi=(0.70, 0.70, 0.70), ppc=8)
    n0 = s.n_particles
    s.add_outflow(OutflowBox(lo=(0.0, 0.0, 0.0), hi=(0.1, 0.1, 0.1)))
    s.prepare_frame(0, 1.0 / 24)
    assert s.n_particles == n0
