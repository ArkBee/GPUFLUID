"""SDF primitives + marker injection tests.

After F3.6.A1 (2026-05-17) the analytic SDFs live in G1:
[BLK G1.8] cell_centers, [BLK G1.10] sdf_sphere, [BLK G1.11] sdf_box,
[BLK G1.12] sdf_cylinder_y, [BLK G1.13] sdf_plane, [BLK G1.14] sdf_union.
Only [BLK D4.4] mark_solid_from_sdf remains under D4 (it mutates
solver marker state).
"""
import numpy as np
import pytest

from gpufluid.primitives.sdf import (
    cell_centers, sdf_sphere, sdf_box, sdf_cylinder_y, sdf_union,
)
from gpufluid.domain.sdf import mark_solid_from_sdf

EPS = 1e-5


def test_g1_8_cell_centers_shape_and_values():
    nx, ny, nz, dx = 4, 3, 2, 0.5
    cc = cell_centers(nx, ny, nz, dx)
    assert cc.shape == (nx, ny, nz, 3)
    # first cell centre = (0.5,0.5,0.5)*dx
    assert np.allclose(cc[0, 0, 0], [0.25, 0.25, 0.25])
    # last cell centre = ((n-0.5)*dx)
    assert np.allclose(cc[-1, -1, -1], [(nx - 0.5) * dx, (ny - 0.5) * dx, (nz - 0.5) * dx])


def test_d4_2_1_sphere_sdf_signs_and_centre_value():
    grid = cell_centers(20, 20, 20, 0.1)
    sdf = sdf_sphere(grid, center=(1.0, 1.0, 1.0), radius=0.4)
    # at the center cell: distance is small (we land on a cell-centre near 1.0)
    # but the very near-centre point should be approximately -radius
    centre_val = sdf_sphere(np.array([[[[1.0, 1.0, 1.0]]]], dtype=np.float32),
                            center=(1.0, 1.0, 1.0), radius=0.4)[0, 0, 0]
    assert abs(centre_val - (-0.4)) < EPS
    # on-surface point
    on_surf = sdf_sphere(np.array([[[[1.4, 1.0, 1.0]]]], dtype=np.float32),
                         center=(1.0, 1.0, 1.0), radius=0.4)[0, 0, 0]
    assert abs(on_surf) < EPS
    # far point
    far = sdf_sphere(np.array([[[[3.0, 1.0, 1.0]]]], dtype=np.float32),
                     center=(1.0, 1.0, 1.0), radius=0.4)[0, 0, 0]
    assert abs(far - (2.0 - 0.4)) < EPS


def test_d4_2_2_box_sdf_axis_aligned():
    centre_val = sdf_box(np.array([[[[0.0, 0.0, 0.0]]]], dtype=np.float32),
                         center=(0.0, 0.0, 0.0), half_size=(0.5, 0.5, 0.5))[0, 0, 0]
    # at centre of a unit cube, SDF = -0.5 (distance to nearest face)
    assert abs(centre_val - (-0.5)) < EPS
    face_val = sdf_box(np.array([[[[0.5, 0.0, 0.0]]]], dtype=np.float32),
                       center=(0.0, 0.0, 0.0), half_size=(0.5, 0.5, 0.5))[0, 0, 0]
    assert abs(face_val) < EPS
    far = sdf_box(np.array([[[[2.0, 0.0, 0.0]]]], dtype=np.float32),
                  center=(0.0, 0.0, 0.0), half_size=(0.5, 0.5, 0.5))[0, 0, 0]
    assert abs(far - 1.5) < EPS


def test_d4_2_3_cylinder_y_sdf():
    # vertical cylinder radius 1, half-height 2, centred at origin
    # at (0,0,0): inside, distance = -min(radius, half_h) = -1
    centre = sdf_cylinder_y(np.array([[[[0.0, 0.0, 0.0]]]], dtype=np.float32),
                            center=(0.0, 0.0, 0.0), radius=1.0, half_height=2.0)[0, 0, 0]
    assert abs(centre - (-1.0)) < EPS
    # on radial surface (1,0,0): SDF=0
    on_radial = sdf_cylinder_y(np.array([[[[1.0, 0.0, 0.0]]]], dtype=np.float32),
                               center=(0.0, 0.0, 0.0), radius=1.0, half_height=2.0)[0, 0, 0]
    assert abs(on_radial) < EPS
    # above top cap by 1 unit (0,3,0): SDF = 1
    above = sdf_cylinder_y(np.array([[[[0.0, 3.0, 0.0]]]], dtype=np.float32),
                           center=(0.0, 0.0, 0.0), radius=1.0, half_height=2.0)[0, 0, 0]
    assert abs(above - 1.0) < EPS


def test_d4_2_5_union_is_min():
    grid = cell_centers(10, 10, 10, 0.1)
    s1 = sdf_sphere(grid, center=(0.3, 0.3, 0.3), radius=0.2)
    s2 = sdf_sphere(grid, center=(0.7, 0.7, 0.7), radius=0.2)
    u = sdf_union(s1, s2)
    assert np.allclose(u, np.minimum(s1, s2))


def test_d4_4_mark_solid_from_sdf_sets_two():
    marker = np.zeros((5, 5, 5), dtype=np.int32)
    sdf = np.ones((5, 5, 5), dtype=np.float32)
    sdf[2, 2, 2] = -0.5  # inside
    mark_solid_from_sdf(marker, sdf, padding=0.0)
    assert marker[2, 2, 2] == 2
    assert marker.sum() == 2  # only one cell flipped to solid
