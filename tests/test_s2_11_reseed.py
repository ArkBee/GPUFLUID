"""Regression tests for S2.11 particle reseeding."""
import numpy as np
import pytest

from gpufluid.sim.reseed import reseed_particles, ReseedConfig


def test_s2_11_emits_into_underdense_fluid_cells():
    """A 4x4x4 marker with one fluid cell, zero particles → must emit min_per_cell into it."""
    nx = ny = nz = 4
    marker = np.zeros((nx, ny, nz), dtype=np.int32)
    marker[2, 2, 2] = 1   # one fluid cell
    pos = np.zeros((0, 3), dtype=np.float32)
    vel = np.zeros((0, 3), dtype=np.float32)
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16)
    new_pos, new_vel, n_emit, n_cull = reseed_particles(
        pos, vel, marker, dx=0.25, cfg=cfg, rng=np.random.default_rng(0))
    assert n_emit == 4
    assert n_cull == 0
    # all new particles must be in the (2,2,2) cell: x,y,z in [0.5, 0.75]
    assert np.all((new_pos >= 0.5) & (new_pos <= 0.75)), f"emitted outside: {new_pos}"


def test_s2_11_culls_overdense_cell():
    nx = ny = nz = 4
    marker = np.zeros((nx, ny, nz), dtype=np.int32)
    marker[1, 1, 1] = 1
    # 30 particles all in cell (1,1,1)
    pos = (1.0 + np.random.RandomState(0).rand(30, 3) * 0.5).astype(np.float32) * 0.25
    vel = np.zeros_like(pos)
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16)
    new_pos, new_vel, n_emit, n_cull = reseed_particles(
        pos, vel, marker, dx=0.25, cfg=cfg, rng=np.random.default_rng(1))
    assert n_emit == 0
    assert n_cull == 14   # 30 - 16
    assert len(new_pos) == 16


def test_s2_11_skips_solid_cells():
    """Particles inside solid cells should not trigger emission and should be
    eligible for culling like any other cell — but the marker test must not
    cause emission INTO solid cells (which would push particles inside obstacles)."""
    nx = ny = nz = 4
    marker = np.zeros((nx, ny, nz), dtype=np.int32)
    marker[1, 1, 1] = 2   # solid — must NOT be emitted into
    pos = np.zeros((0, 3), dtype=np.float32)
    vel = np.zeros((0, 3), dtype=np.float32)
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16)
    new_pos, _, n_emit, _ = reseed_particles(
        pos, vel, marker, dx=0.25, cfg=cfg, rng=np.random.default_rng(2))
    assert n_emit == 0
    assert len(new_pos) == 0


def test_s2_11_idempotent_when_already_in_range():
    nx = ny = nz = 4
    marker = np.ones((nx, ny, nz), dtype=np.int32)
    # exactly min_per_cell particles in each of 64 cells = 256
    rng = np.random.default_rng(3)
    pos_per_cell = 4
    positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for _ in range(pos_per_cell):
                    positions.append([(i + 0.5) * 0.25, (j + 0.5) * 0.25, (k + 0.5) * 0.25])
    pos = np.array(positions, dtype=np.float32)
    vel = np.zeros_like(pos)
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16)
    new_pos, _, n_emit, n_cull = reseed_particles(pos, vel, marker, dx=0.25, cfg=cfg, rng=rng)
    assert n_emit == 0
    assert n_cull == 0
    assert len(new_pos) == len(pos)
