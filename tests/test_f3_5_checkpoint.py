"""Regression test for F3.5 restart/checkpoint."""
import numpy as np
import pytest
from pathlib import Path

from gpufluid import FlipSolver3D

pytestmark = pytest.mark.gpu


def test_f3_5_checkpoint_resume_matches_uninterrupted(tmp_path: Path):
    """Bake N steps continuously; bake N/2 then checkpoint, resume, bake N/2.
    Final particle positions must match within numerical tolerance."""
    def make_solver():
        s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1/16)
        s.seed_box(lo=(0.20, 0.30, 0.20), hi=(0.50, 0.70, 0.50), ppc=8)
        return s

    # continuous bake
    s_cont = make_solver()
    for _ in range(20):
        s_cont.step(0.005, pressure_iters=20)
    pos_cont, vel_cont = s_cont.get_particles()

    # checkpoint at 10, resume to 20
    s_a = make_solver()
    for _ in range(10):
        s_a.step(0.005, pressure_iters=20)
    ckpt = tmp_path / "ck.npz"
    s_a.save_checkpoint(ckpt)

    s_b = FlipSolver3D(nx=16, ny=16, nz=16, dx=1/16)
    s_b.load_checkpoint(ckpt)
    for _ in range(10):
        s_b.step(0.005, pressure_iters=20)
    pos_resumed, vel_resumed = s_b.get_particles()

    # exact-match modulo numerical noise from re-initialised RNG / Warp scheduling
    assert pos_resumed.shape == pos_cont.shape
    rms = float(np.sqrt(((pos_resumed - pos_cont) ** 2).sum(1).mean()))
    assert rms < 1e-3, f"resume mismatch: rms={rms:.5f}"
