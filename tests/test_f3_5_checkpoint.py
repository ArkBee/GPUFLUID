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


def test_resume_adds_no_divergence_beyond_atomic_noise_floor():
    """2026-06-21 (FLIP-resume determinism): the resume-vs-continuous test
    above tolerates rms<1e-3, but the solver's actual run-to-run divergence
    is ~1e-9 (GPU atomic-add ordering in P2G scatter is non-deterministic).
    A 1e-3 bound is ~6 orders too loose to catch a *dropped-state* regression
    (e.g. a future edit that stops persisting vel/affine_C/rng): such a bug
    would land at ~1e-4 and still pass <1e-3 (§9.2 — green ≠ correct).

    Measure the inherent noise floor (continuous vs continuous, identical
    setup) and assert resume divergence stays within a small multiple of it.
    Proven empirically: resume adds ~1.2x the floor — i.e. resume is as
    deterministic as the solver itself, no carried state is dropped."""
    def make():
        s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1 / 16)
        s.seed_box(lo=(0.20, 0.30, 0.20), hi=(0.50, 0.70, 0.50), ppc=8)
        return s

    def bake(s, n):
        for _ in range(n):
            s.step(0.005, pressure_iters=20)
        return s.get_particles()[0]

    a = bake(make(), 20)                 # continuous reference
    b = bake(make(), 20)                 # identical continuous -> noise floor
    rms_floor = float(np.sqrt(((a - b) ** 2).sum(1).mean()))

    s = make(); bake(s, 10)
    import tempfile, os
    ck = os.path.join(tempfile.gettempdir(), "det_floor_ck.npz")
    s.save_checkpoint(ck)
    s2 = FlipSolver3D(nx=16, ny=16, nz=16, dx=1 / 16); s2.load_checkpoint(ck)
    c = bake(s2, 10)
    rms_resume = float(np.sqrt(((a - c) ** 2).sum(1).mean()))

    # the floor is genuine atomic noise, not a real divergence
    assert rms_floor < 1e-5, f"noise floor unexpectedly large: {rms_floor:.2e}"
    # resume must not add SYSTEMATIC divergence on top of that floor
    assert rms_resume < 20 * rms_floor + 1e-9, (
        f"resume divergence {rms_resume:.2e} >> atomic noise floor "
        f"{rms_floor:.2e} — checkpoint is dropping carried solver state")


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
