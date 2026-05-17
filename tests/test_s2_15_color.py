"""[BLK S2.15] Per-particle color attribute regression tests.

Exercises the P2G→normalize→G2P color pipeline kernels:
[BLK S2.15.1] P2G scatter, [BLK S2.15.2] grid normalize, [BLK S2.15.3]
G2P gather.


Two red+blue cubes seeded next to each other must:
  1. Each carry the seed color initially (no green leakage).
  2. Develop a blended (purple-ish) population after a few steps.
  3. Conserve the bulk mean color (mass-weighted RGB integral stays constant).
  4. Leave attr_color *untouched* when no color is configured (no extra memory).
"""
from __future__ import annotations

import numpy as np
import pytest

from gpufluid.solvers.solver3d import FlipSolver3D


def _seed_two_color(gravity: float = -2.0):
    s = FlipSolver3D(nx=24, ny=24, nz=24, dx=1.0 / 24.0,
                     gravity=gravity, flip_blend=0.9)
    s.seed_box((0.20, 0.30, 0.30), (0.50, 0.50, 0.70), color=(1.0, 0.0, 0.0))
    s.seed_box((0.50, 0.30, 0.30), (0.80, 0.50, 0.70), color=(0.0, 0.0, 1.0))
    return s


def test_color_initial_seed_is_pure():
    s = _seed_two_color(gravity=0.0)
    col = s.attr_color.numpy()
    # No "green" channel should appear from a red+blue seed.
    assert col[:, 1].max() == 0.0
    # Each particle is either pure red or pure blue (no blending yet).
    pure_red = ((col[:, 0] == 1.0) & (col[:, 2] == 0.0)).sum()
    pure_blue = ((col[:, 0] == 0.0) & (col[:, 2] == 1.0)).sum()
    assert pure_red + pure_blue == s.n_particles


def test_color_blends_at_interface_after_steps():
    """After running a few steps with two adjacent colored boxes, some
    particles must have intermediate RGB (R and B both nonzero)."""
    s = _seed_two_color(gravity=-2.0)
    for _ in range(20):
        s.step(0.005, pressure_iters=40)
    col = s.attr_color.numpy()
    blended = ((col[:, 0] > 0.15) & (col[:, 2] > 0.15)).sum()
    # Demand at least 10% of particles show blending.
    assert blended / s.n_particles > 0.10, (
        f"only {100*blended/s.n_particles:.1f}% blended — color transfer not active?"
    )


def test_color_conserves_bulk_mean():
    """Sum of RGB over all particles should drift only marginally (no source).
    Pure RGB transfer is a *reversible* trilinear scheme — small diffusion is
    OK, large drift means the kernel is buggy."""
    s = _seed_two_color(gravity=-2.0)
    mean0 = s.attr_color.numpy().mean(axis=0)
    for _ in range(20):
        s.step(0.005, pressure_iters=40)
    mean1 = s.attr_color.numpy().mean(axis=0)
    drift = float(np.linalg.norm(mean1 - mean0))
    # 5% drift tolerance — trilinear-trilinear round-trip is not perfectly
    # conservative under particle motion + boundary clamping.
    assert drift < 0.05, f"bulk color drifted {drift:.3f} (>5%)"


def test_no_color_means_no_alloc():
    """If no fluid source carries `color`, attr_color stays None — no memory
    spent for scenes that don't use the feature."""
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16.0)
    s.seed_box((0.2, 0.2, 0.2), (0.6, 0.6, 0.6))
    assert s.attr_color is None
    for _ in range(3):
        s.step(0.005, pressure_iters=20)
    assert s.attr_color is None
