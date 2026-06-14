"""[BLK S2.11.GPU] GPU particle reseed regression + perf.

Drives [BLK S2.11.GPU.COUNT] (atomic per-cell count) and
[BLK S2.11.GPU.RANK] (per-particle rank + alive mask) through the
GPU reseed orchestrator.


Verifies that:
  1. GPU reseed and CPU reseed agree on emit / cull counts for the same
     particle distribution.
  2. Post-reseed per-cell counts are bounded by the configured min/max.
  3. The GPU path is faster than the CPU path at ≥500k particles.
  4. Per-particle colour (S2.15) is compacted in lockstep — kept particles
     keep their colour, emitted particles get white.
"""
from __future__ import annotations

import time
import numpy as np
import pytest
import warp as wp

from gpufluid.sim.reseed import (
    ReseedConfig, RESEED_GPU_THRESHOLD,
    reseed_particles, reseed_particles_gpu,
)


def _random_particles(n: int, N_grid: int, dx: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3)).astype(np.float32) * (N_grid * dx * 0.8) + (N_grid * dx * 0.1)
    vel = rng.standard_normal((n, 3)).astype(np.float32) * 0.1
    marker = np.zeros((N_grid, N_grid, N_grid), dtype=np.int32)
    # walls
    marker[0, :, :] = 2; marker[-1, :, :] = 2
    marker[:, 0, :] = 2; marker[:, -1, :] = 2
    marker[:, :, 0] = 2; marker[:, :, -1] = 2
    return pos, vel, marker


def test_gpu_reseed_matches_cpu_counts():
    """Same input → same n_emitted and n_culled (the *random* selection of
    which particles to drop differs, but the counts are deterministic)."""
    pos, vel, marker = _random_particles(80_000, 32, 1.0 / 32.0)
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16)
    dx = 1.0 / 32.0

    pos_wp = wp.array(pos, dtype=wp.vec3, device="cuda:0")
    vel_wp = wp.array(vel, dtype=wp.vec3, device="cuda:0")
    marker_wp = wp.array(marker, dtype=int, device="cuda:0")
    _, _, _, _, ne_gpu, nc_gpu = reseed_particles_gpu(
        pos_wp, vel_wp, marker_wp, dx, cfg, np.random.default_rng(0))
    _, _, ne_cpu, nc_cpu = reseed_particles(
        pos, vel, marker, dx, cfg, np.random.default_rng(0))
    assert ne_gpu == ne_cpu, f"emit counts differ: gpu={ne_gpu} cpu={ne_cpu}"
    assert nc_gpu == nc_cpu, f"cull counts differ: gpu={nc_gpu} cpu={nc_cpu}"


def test_gpu_reseed_bounds_per_cell_counts():
    """After GPU reseed, no fluid cell should have > max_per_cell particles."""
    pos, vel, marker = _random_particles(120_000, 32, 1.0 / 32.0)
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=12)
    dx = 1.0 / 32.0

    pos_wp = wp.array(pos, dtype=wp.vec3, device="cuda:0")
    vel_wp = wp.array(vel, dtype=wp.vec3, device="cuda:0")
    marker_wp = wp.array(marker, dtype=int, device="cuda:0")
    new_pos, _, _, _, _, _ = reseed_particles_gpu(
        pos_wp, vel_wp, marker_wp, dx, cfg, np.random.default_rng(0))
    p = new_pos.numpy()
    # Bin into 32³ cells and check max count
    idx = np.clip((p / dx).astype(np.int32), 0, 31)
    flat = idx[:, 0] * 32 * 32 + idx[:, 1] * 32 + idx[:, 2]
    counts = np.bincount(flat, minlength=32 ** 3).reshape(32, 32, 32)
    fluid_mask = marker != 2
    fluid_max = int(counts[fluid_mask].max())
    assert fluid_max <= cfg.max_per_cell, (
        f"GPU reseed violated max: {fluid_max} > {cfg.max_per_cell}"
    )


def test_gpu_reseed_threshold_constant():
    """Threshold should stay reasonable; if you change it, update DESIGN.md §5.4."""
    assert 10_000 <= RESEED_GPU_THRESHOLD <= 1_000_000


def test_gpu_reseed_speedup_at_500k():
    """At 500k particles the GPU path must be ≥3× faster than CPU.

    Measured ~6–10× on RTX 4080 SUPER; 3× is the conservative regression bar
    that holds across HW. CPU path scales with `cells_with_excess`, GPU is
    near-linear in n_particles via parallel count + scan + scatter."""
    pos, vel, marker = _random_particles(500_000, 64, 1.0 / 64.0)
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16)
    dx = 1.0 / 64.0

    # GPU side: warmup
    pos_wp = wp.array(pos, dtype=wp.vec3, device="cuda:0")
    vel_wp = wp.array(vel, dtype=wp.vec3, device="cuda:0")
    marker_wp = wp.array(marker, dtype=int, device="cuda:0")
    reseed_particles_gpu(pos_wp, vel_wp, marker_wp, dx, cfg, np.random.default_rng(0))
    wp.synchronize()

    n_runs = 3
    t0 = time.time()
    for _ in range(n_runs):
        # Re-create arrays so we measure the steady-state cost (the input would
        # be different each invocation in a real pipeline).
        pos_wp = wp.array(pos, dtype=wp.vec3, device="cuda:0")
        vel_wp = wp.array(vel, dtype=wp.vec3, device="cuda:0")
        reseed_particles_gpu(pos_wp, vel_wp, marker_wp, dx, cfg,
                             np.random.default_rng(0))
        wp.synchronize()
    t_gpu = (time.time() - t0) / n_runs

    t0 = time.time()
    for _ in range(n_runs):
        reseed_particles(pos, vel, marker, dx, cfg, np.random.default_rng(0))
    t_cpu = (time.time() - t0) / n_runs
    speedup = t_cpu / t_gpu
    assert speedup >= 3.0, (
        f"GPU reseed should be ≥3× faster at 500k particles; "
        f"got {speedup:.2f}× (cpu {t_cpu*1000:.0f}ms, gpu {t_gpu*1000:.0f}ms)"
    )


def test_gpu_reseed_color_lockstep():
    """When attr_color is passed, kept particles must keep their original
    colour and emitted particles get ZERO (0,0,0) — matching the CPU reseed
    path and prepare_frame (audit-2026-06-14r2 #6; was white)."""
    pos, vel, marker = _random_particles(60_000, 24, 1.0 / 24.0)
    # Colour everything red so we can identify emitted particles by their
    # black (zero) colour after reseed.
    col = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(pos), 1))
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16)
    dx = 1.0 / 24.0

    pos_wp = wp.array(pos, dtype=wp.vec3, device="cuda:0")
    vel_wp = wp.array(vel, dtype=wp.vec3, device="cuda:0")
    col_wp = wp.array(col, dtype=wp.vec3, device="cuda:0")
    marker_wp = wp.array(marker, dtype=int, device="cuda:0")
    new_pos, _, new_col, _, n_emit, _ = reseed_particles_gpu(
        pos_wp, vel_wp, marker_wp, dx, cfg, np.random.default_rng(0),
        attr_color_wp=col_wp)
    c = new_col.numpy()
    # The last `n_emit` rows must be black (0); everything before must be red.
    assert n_emit > 0, "no particles emitted — pick a sparser scene"
    assert np.allclose(c[-n_emit:], 0.0), \
        "emitted particles should be zero/black (matches CPU reseed + prepare_frame)"
    # Kept rows: red (allow tiny float tolerance)
    kept = c[:-n_emit]
    assert (kept[:, 0] > 0.99).all() and (kept[:, 1] < 0.01).all() \
        and (kept[:, 2] < 0.01).all(), "kept particles lost their red colour"


def test_gpu_reseed_temperature_lockstep():
    """B11.3 follow-up: scalar attribute (temperature) compacts in lockstep
    with positions/colours, mirroring the S2.15 `attr_color` path. Without
    this, scenes with `reseed=true` + per-source temperature lost the
    scalar on the first reseed pass.

    Pre-fix the function didn't have `attr_temperature_wp` kwarg at all;
    the keyword would have raised TypeError. So this test is also the
    surface-area pin for the new API.
    """
    pos, vel, marker = _random_particles(60_000, 24, 1.0 / 24.0)
    # Every input particle is hot (T=80). Emitted particles should land at
    # T=20.0 — the CPU reseed / prepare_frame default (audit r2 #6; was 0.0).
    temps = np.full(len(pos), 80.0, dtype=np.float32)
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16)
    dx = 1.0 / 24.0

    pos_wp = wp.array(pos, dtype=wp.vec3, device="cuda:0")
    vel_wp = wp.array(vel, dtype=wp.vec3, device="cuda:0")
    temp_wp = wp.array(temps, dtype=float, device="cuda:0")
    marker_wp = wp.array(marker, dtype=int, device="cuda:0")
    new_pos, _, _, new_temp, n_emit, _ = reseed_particles_gpu(
        pos_wp, vel_wp, marker_wp, dx, cfg, np.random.default_rng(0),
        attr_temperature_wp=temp_wp)

    assert new_temp is not None, \
        "reseed must return new_temperature when attr_temperature_wp is set"
    assert new_temp.shape[0] == new_pos.shape[0], \
        "temperature array length must match positions in lockstep"
    t = new_temp.numpy()
    assert n_emit > 0, "no particles emitted — pick a sparser scene"
    # Last `n_emit` rows: the prepare_frame default emit value (20.0).
    assert np.allclose(t[-n_emit:], 20.0), \
        "emitted particles should get T=20.0 (matches CPU reseed + prepare_frame)"
    # Kept rows: T=80 preserved.
    kept = t[:-n_emit]
    assert np.allclose(kept, 80.0, atol=1e-4), \
        "kept particles lost their original temperature"
