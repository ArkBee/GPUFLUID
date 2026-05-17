"""B7-alt.4 — particle ↔ sub-dense mapping + out-of-bbox warning.

Exercises the `_pos_to_sub_cell` helper and the `_check_particles_in_sub_bbox`
guard added to FlipSolver3D. The guard fires a one-shot stderr warning
when particles drift past the dilation buffer between sub-dense rebuilds
— the most likely user-visible failure mode if `sub_dilation` is set too
small or `sub_rebuild_every` too generous.

These tests don't run step(): they manipulate `_sub_offset` / `_sub_shape`
+ `self.pos` directly to keep the unit-test scope tight.
"""
import io
import sys

import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import FlipSolver3D


def test_pos_to_sub_cell_default_mode_is_identity():
    """With sub_offset=(0,0,0), local cell == global cell. `inside` is
    True iff the cell lies inside the full domain."""
    s = FlipSolver3D(nx=32, ny=32, nz=32)
    # Particle at the centre of cell (10, 14, 21).
    p = np.array([10.5, 14.5, 21.5], dtype=np.float32) * s.dx
    (li, lj, lk), inside = s._pos_to_sub_cell(p)
    assert (li, lj, lk) == (10, 14, 21)
    assert inside is True
    # Particle past the right wall: outside.
    p_out = np.array([35.0, 16.0, 16.0], dtype=np.float32) * s.dx
    _, inside_out = s._pos_to_sub_cell(p_out)
    assert inside_out is False


def test_pos_to_sub_cell_with_offset_translates_correctly():
    """A particle at global cell (16, 16, 16) sits at local (2, 2, 2) in
    a bbox with offset (14, 14, 14)."""
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=0)
    # Fake a rebuild without going through prepare_frame.
    s._rebuild_sub_dense((14, 14, 14), (26, 26, 26))
    assert s._sub_offset == (14, 14, 14)
    assert s._sub_shape == (12, 12, 12)
    p = np.array([16.5, 16.5, 16.5], dtype=np.float32) * s.dx
    (li, lj, lk), inside = s._pos_to_sub_cell(p)
    assert (li, lj, lk) == (2, 2, 2)
    assert inside is True
    # Particle at cell (28, 16, 16) is past the bbox upper edge → outside.
    p_far = np.array([28.5, 16.5, 16.5], dtype=np.float32) * s.dx
    (li, lj, lk), inside = s._pos_to_sub_cell(p_far)
    assert (li, lj, lk) == (14, 2, 2)
    assert inside is False


def test_check_particles_skips_in_default_mode():
    """No sub-dense mode → no warning, count == 0 always."""
    s = FlipSolver3D(nx=16, ny=16, nz=16)
    s.pos = wp.array(np.random.rand(100, 3).astype(np.float32) * s.dx * 8,
                     dtype=wp.vec3, device=s.device)
    s.n_particles = 100
    assert s._check_particles_in_sub_bbox(0) == 0


@pytest.mark.gpu
def test_check_particles_in_sub_bbox_counts_and_warns_once(capsys):
    """Set up a deliberately under-sized bbox, populate particles partly
    outside it, and verify the count + one-shot warning."""
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=0, sub_rebuild_every=8)
    # Force a tight bbox covering cells (14..18)³ — 4 cells per side.
    s._rebuild_sub_dense((14, 14, 14), (18, 18, 18))
    assert s._sub_shape == (4, 4, 4)
    # Particles: half inside, half outside.
    inside = np.array([
        [14.5, 14.5, 14.5],
        [15.5, 16.5, 17.5],
        [17.5, 14.5, 14.5],
    ], dtype=np.float32) * s.dx
    outside = np.array([
        [10.5, 14.5, 14.5],
        [19.5, 14.5, 14.5],
        [14.5, 20.5, 14.5],
        [14.5, 14.5, 22.5],
    ], dtype=np.float32) * s.dx
    pts = np.concatenate([inside, outside])
    s.pos = wp.array(pts, dtype=wp.vec3, device=s.device)
    s.n_particles = len(pts)
    n_out = s._check_particles_in_sub_bbox(frame_idx=5)
    assert n_out == 4
    err = capsys.readouterr().err
    assert "outside the sub-dense bbox" in err
    assert "frame 5" in err
    assert "sub_dilation" in err
    # Second call must NOT emit a second warning (one-shot guard).
    n_out_again = s._check_particles_in_sub_bbox(frame_idx=6)
    assert n_out_again == 4
    err_again = capsys.readouterr().err
    assert err_again == ""


@pytest.mark.gpu
def test_check_particles_no_warning_when_all_inside():
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=2)
    s._rebuild_sub_dense((10, 10, 10), (22, 22, 22))
    pts = np.array([
        [15.5, 15.5, 15.5],
        [12.5, 18.5, 14.5],
        [18.5, 14.5, 19.5],
    ], dtype=np.float32) * s.dx
    s.pos = wp.array(pts, dtype=wp.vec3, device=s.device)
    s.n_particles = len(pts)
    n_out = s._check_particles_in_sub_bbox(frame_idx=0)
    assert n_out == 0
