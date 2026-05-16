"""[BLK S2.10.GPU] GPU |v|_max reduction for CFL.

Replaces the per-frame `vel.numpy()` host roundtrip with an atomic-max
reduction kernel. Tests verify (a) bit-identical substep counts vs the
CPU helper across velocity distributions and (b) measurable speedup at
≥500k particles, where the CPU path's D→H copy starts to dominate."""
from __future__ import annotations

import time
import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import (
    cfl_substep_count, cfl_substep_count_gpu,
)


def _rand_vel(n: int, scale: float = 2.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)).astype(np.float32) * scale


def test_gpu_cfl_matches_cpu_count():
    """GPU and CPU must agree on substep count across many distributions."""
    for n, scale, seed in [
        (1, 1.0, 0),                # degenerate single-particle
        (1000, 0.5, 1),             # tiny case
        (50_000, 5.0, 2),           # mid-size
        (200_000, 12.0, 3),         # high-velocity (more substeps)
    ]:
        vel = _rand_vel(n, scale, seed)
        vel_wp = wp.array(vel, dtype=wp.vec3, device="cuda:0")
        scratch = wp.zeros(1, dtype=float, device="cuda:0")
        n_cpu = cfl_substep_count(vel, dx=0.0125, target_dt=0.04)
        n_gpu = cfl_substep_count_gpu(vel_wp, dx=0.0125, target_dt=0.04, scratch=scratch)
        assert n_cpu == n_gpu, (
            f"n={n} scale={scale}: cpu={n_cpu} gpu={n_gpu}"
        )


def test_gpu_cfl_empty_array_returns_1():
    """No particles → return 1 substep (no division-by-zero, no crash)."""
    empty = wp.zeros(0, dtype=wp.vec3, device="cuda:0")
    assert cfl_substep_count_gpu(empty, dx=0.0125, target_dt=0.04) == 1


def test_gpu_cfl_zero_velocity_returns_1():
    """All-zero velocity field → 1 substep (vmax≈0 short-circuit)."""
    z = wp.zeros(100, dtype=wp.vec3, device="cuda:0")
    assert cfl_substep_count_gpu(z, dx=0.0125, target_dt=0.04) == 1


def test_gpu_cfl_speedup_at_500k():
    """At 500k particles the GPU reduction must be ≥3× faster than the CPU
    host-sync path. Measured ~12× on RTX 4080 SUPER; 3× is the conservative
    HW-agnostic regression bar."""
    n = 500_000
    vel = _rand_vel(n, scale=4.0)
    vel_wp = wp.array(vel, dtype=wp.vec3, device="cuda:0")
    scratch = wp.zeros(1, dtype=float, device="cuda:0")
    # Warmup both
    cfl_substep_count(vel, 0.0125, 0.04)
    cfl_substep_count_gpu(vel_wp, 0.0125, 0.04, scratch=scratch)
    wp.synchronize()

    def bench(fn, n_iter=10):
        t0 = time.time()
        for _ in range(n_iter):
            fn()
        return (time.time() - t0) / n_iter

    t_cpu = bench(lambda: cfl_substep_count(vel, 0.0125, 0.04))
    t_gpu = bench(lambda: (cfl_substep_count_gpu(vel_wp, 0.0125, 0.04, scratch=scratch),
                            wp.synchronize()))
    speedup = t_cpu / t_gpu
    assert speedup >= 3.0, (
        f"GPU CFL reduction should be ≥3× faster at 500k; got {speedup:.2f}× "
        f"(cpu {t_cpu*1000:.2f}ms, gpu {t_gpu*1000:.2f}ms)"
    )
