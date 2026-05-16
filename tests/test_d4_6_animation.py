"""Tests for D4.6 animated obstacles (motion descriptors + solver integration)."""
import numpy as np
import pytest

from gpufluid.domain.animation import LinearMotion, KeyframeMotion, evaluate_center


def test_d4_6_linear_motion_advances_centre():
    m = LinearMotion(velocity=(1.0, 0.0, 0.0), fps=24)
    c0 = evaluate_center((0.1, 0.5, 0.5), m, frame_idx=0)
    c1 = evaluate_center((0.1, 0.5, 0.5), m, frame_idx=24)  # 1 second later
    assert np.allclose(c0, [0.1, 0.5, 0.5])
    assert np.allclose(c1, [1.1, 0.5, 0.5])


def test_d4_6_keyframes_interpolate_linearly():
    m = KeyframeMotion(keyframes=[(0, (0, 0, 0)), (10, (10, 0, 0))])
    assert np.allclose(evaluate_center((9, 9, 9), m, 0), [0, 0, 0])
    assert np.allclose(evaluate_center((9, 9, 9), m, 5), [5, 0, 0])
    assert np.allclose(evaluate_center((9, 9, 9), m, 10), [10, 0, 0])
    # before / after the range — clamp to endpoints
    assert np.allclose(evaluate_center((9, 9, 9), m, -1), [0, 0, 0])
    assert np.allclose(evaluate_center((9, 9, 9), m, 99), [10, 0, 0])


def test_d4_6_no_motion_returns_base():
    assert np.allclose(evaluate_center((1, 2, 3), None, 99), [1, 2, 3])


@pytest.mark.gpu
def test_d4_6_solver_animated_obstacle_changes_marker():
    """Moving sphere should mark different cells solid at different frames."""
    from gpufluid import FlipSolver3D
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16)
    s.seed_box(lo=(0.30, 0.30, 0.30), hi=(0.50, 0.50, 0.50), ppc=8)
    s.add_animated_obstacle("sphere", base_center=(0.10, 0.50, 0.50),
                            motion=LinearMotion(velocity=(1.0, 0, 0), fps=24),
                            radius=0.10)
    s.prepare_frame(0, 1.0 / 24)
    m0 = (s.marker.numpy() == 2).sum()
    s.prepare_frame(12, 1.0 / 24)  # 0.5s in → sphere at x≈0.6
    m1 = (s.marker.numpy() == 2).sum()
    # both must have solid cells; their *positions* should differ
    assert m0 > 0 and m1 > 0
    # cellwise diff: cells solid at one but not the other
    m_a = s.marker.numpy()
    s.prepare_frame(0, 1.0 / 24)
    m_b = s.marker.numpy()
    diff = ((m_a == 2) != (m_b == 2)).sum()
    assert diff > 5, f"obstacle didn't move: diff cells = {diff}"
