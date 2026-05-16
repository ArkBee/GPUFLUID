"""Regression tests for D4.2.4 SDF plane."""
import numpy as np
import pytest

from gpufluid.domain.sdf import sdf_plane, cell_centers


def test_d4_2_4_plane_horizontal_ground():
    # ground at y=0, normal pointing up
    grid = cell_centers(4, 4, 4, 0.25)
    s = sdf_plane(grid, point=(0, 0, 0), normal=(0, 1, 0))
    # cells at y<0 (none — grid is [0.125, 0.875]) — all positive
    assert (s > 0).all()
    # the bottom row (j=0, y=0.125) is closest to plane
    assert abs(s[0, 0, 0] - 0.125) < 1e-5


def test_d4_2_4_plane_45deg_slope():
    grid = cell_centers(4, 4, 4, 0.25)
    s = sdf_plane(grid, point=(0.5, 0.5, 0), normal=(1, 1, 0))
    # at the point on the plane SDF ≈ 0
    s_at = sdf_plane(np.array([[[[0.5, 0.5, 0]]]], dtype=np.float32),
                    point=(0.5, 0.5, 0), normal=(1, 1, 0))[0, 0, 0]
    assert abs(s_at) < 1e-5
    # in the +normal direction → positive
    s_up = sdf_plane(np.array([[[[0.7, 0.7, 0]]]], dtype=np.float32),
                    point=(0.5, 0.5, 0), normal=(1, 1, 0))[0, 0, 0]
    assert s_up > 0
