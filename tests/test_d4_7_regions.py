"""Tests for D4.7 inflow / outflow regions."""
import numpy as np
import pytest

from gpufluid.domain.regions import InflowBox, OutflowBox, apply_inflows, apply_outflows


def test_d4_7_inflow_emits_within_box_and_count():
    inf = InflowBox(lo=(0.0, 0.5, 0.0), hi=(0.2, 0.6, 0.2),
                    velocity=(0, -3, 0), rate_per_sec=2400)
    rng = np.random.default_rng(1)
    # 1/24 second → expect ~100 particles
    pos, vel = apply_inflows([inf], frame_idx=5, frame_dt=1.0 / 24, rng=rng)
    assert 80 <= len(pos) <= 120
    assert pos.shape[1] == 3
    assert (pos >= [0.0, 0.5, 0.0]).all() and (pos <= [0.2, 0.6, 0.2]).all()
    assert np.allclose(vel, [0, -3, 0])


def test_d4_7_inflow_respects_frame_range():
    inf = InflowBox(lo=(0, 0, 0), hi=(1, 1, 1), rate_per_sec=1000,
                    frame_start=10, frame_end=20)
    rng = np.random.default_rng(0)
    p_early, _ = apply_inflows([inf], frame_idx=5, frame_dt=0.04, rng=rng)
    p_in, _ = apply_inflows([inf], frame_idx=15, frame_dt=0.04, rng=rng)
    p_late, _ = apply_inflows([inf], frame_idx=25, frame_dt=0.04, rng=rng)
    assert len(p_early) == 0
    assert len(p_in) > 0
    assert len(p_late) == 0


def test_d4_7_outflow_removes_particles_inside_only():
    pos = np.array([
        [0.1, 0.1, 0.1],  # inside  → remove
        [0.5, 0.5, 0.5],  # outside → keep
        [0.15, 0.15, 0.15],  # inside  → remove
    ], dtype=np.float32)
    vel = np.zeros_like(pos)
    o = OutflowBox(lo=(0, 0, 0), hi=(0.2, 0.2, 0.2))
    new_pos, new_vel = apply_outflows(pos, vel, [o], frame_idx=0)
    assert len(new_pos) == 1
    assert np.allclose(new_pos[0], [0.5, 0.5, 0.5])


def test_d4_7_outflow_empty_passes_through():
    pos = np.random.RandomState(0).rand(10, 3).astype(np.float32)
    vel = np.zeros_like(pos)
    new_pos, new_vel = apply_outflows(pos, vel, [], frame_idx=0)
    assert len(new_pos) == 10
    assert (new_pos == pos).all()


def test_d4_7_inflow_no_regions_empty_output():
    rng = np.random.default_rng(0)
    pos, vel = apply_inflows([], 0, 1.0 / 24, rng)
    assert len(pos) == 0 and len(vel) == 0
