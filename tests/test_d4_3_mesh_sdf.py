"""Tests for D4.3 mesh → SDF."""
import numpy as np
import pytest
import trimesh

from gpufluid.domain.mesh_sdf import mesh_to_sdf, mesh_to_sdf_grid
from gpufluid.primitives.sdf import cell_centers
from gpufluid.blocks import BlockError


def test_d4_3_icosphere_signs():
    """SDF of a unit-radius icosphere at origin: negative at origin, ~0 on surface."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # 3 query points: centre, on-surface, far outside
    pts = np.array([
        [[[0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [3.0, 0.0, 0.0]]]
    ], dtype=np.float32)  # shape (1,1,3,3)
    sdf = mesh_to_sdf(mesh, pts)
    assert sdf.shape == (1, 1, 3)
    centre, on_surf, far = sdf[0, 0]
    assert centre < 0.0, f"centre should be inside (negative), got {centre}"
    assert abs(centre + 1.0) < 0.15  # icosphere is faceted, tolerance
    assert abs(on_surf) < 0.15
    assert far > 1.5
    assert abs(far - 2.0) < 0.15


def test_d4_3_grid_helper_shape():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
    sdf = mesh_to_sdf_grid(mesh, nx=8, ny=8, nz=8, dx=0.125,
                           translate=(0.5, 0.5, 0.5))
    assert sdf.shape == (8, 8, 8)
    # cell containing centre should be negative.
    cc = cell_centers(8, 8, 8, 0.125)
    nearest = np.unravel_index(np.argmin(np.linalg.norm(cc - 0.5, axis=-1)), cc.shape[:3])
    assert sdf[nearest] < 0.0


def test_d4_3_missing_file_raises():
    with pytest.raises(BlockError) as ei:
        mesh_to_sdf("nonexistent_mesh_file.obj", cell_centers(4, 4, 4, 0.25))
    assert ei.value.block_id == "D4.3"
