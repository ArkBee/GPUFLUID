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


def test_checkpoint_persists_rng_state(tmp_path: Path):
    """2026-06-21 (reviewer-checkpoint): the RNG drives inflow particle
    placement (regions.py). Pre-fix it reset to seed-0 on resume -> inflow
    diverged. save/load_checkpoint must now persist + restore the RNG state."""
    s = FlipSolver3D(nx=8, ny=8, nz=8, dx=1 / 8)
    s.seed_box(lo=(0.2, 0.2, 0.2), hi=(0.5, 0.5, 0.5), ppc=4)
    s._rng.random(17)                       # advance off the seed
    saved = s._rng.bit_generator.state
    ck = tmp_path / "rng.npz"
    s.save_checkpoint(ck)

    s2 = FlipSolver3D(nx=8, ny=8, nz=8, dx=1 / 8)   # fresh: _rng is seed-0
    s2.load_checkpoint(ck)
    assert s2._rng.bit_generator.state == saved, "RNG state not restored on resume"
    ref = np.random.default_rng(); ref.bit_generator.state = saved
    assert np.allclose(s2._rng.random(5), ref.random(5)), (
        "resumed RNG draws diverge from the saved stream")
