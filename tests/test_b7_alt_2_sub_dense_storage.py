"""B7-alt.2 — sub-dense field storage + rebuild trigger.

Storage-layer micro for the v1.0 macro that B7-alt.1 greenlit. Adds the
infrastructure that B7-alt.3 will later thread through ~20 kernels:

  * `_sub_offset = (ox, oy, oz)` — global origin of the sub-dense bbox.
  * `_sub_shape = (sx, sy, sz)` — cell-shape of the sub-dense bbox.
  * `_compute_active_bbox(marker_host)` — 8³-tile bbox of fluid cells,
    padded by `_sub_dilation`, clamped to grid.
  * `_should_rebuild_sub_dense(frame_idx)` — periodic OR proximity-driven.
  * `_rebuild_sub_dense(lo, hi)` — reallocates u/v/w/p/p_tmp/div at the
    new bbox and copies the overlapping region from the old buffers
    (CPU round-trip in B7-alt.2; on-device kernel arrives in B7-alt.8).

These tests exercise the storage layer in isolation — they do NOT call
`solver.step()`. Once `enable_sub_dense=True` flips fields to sub-dense
shape, the existing dense kernels would crash on a shape mismatch; that
gap closes in B7-alt.3 when each kernel learns `off_x/y/z`. Until then,
step() with sub_offset != (0,0,0) is guarded by an explicit
NotImplementedError that points future readers at B7-alt.3.
"""
import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import FlipSolver3D, BLOCK_SIZE


# ---------------------------------------------------------------------------
# Default-off invariants — production path must be unchanged.
# ---------------------------------------------------------------------------

def test_default_solver_keeps_full_dense_storage():
    """Vanilla FlipSolver3D — no new behavior. _sub_offset is (0,0,0),
    _sub_shape mirrors (nx,ny,nz), the rebuild flag is False, and the
    field shapes are unchanged from the v0.9 layout."""
    s = FlipSolver3D(nx=32, ny=24, nz=16)
    assert s._enable_sub_dense is False
    assert s._sub_offset == (0, 0, 0)
    assert s._sub_shape == (32, 24, 16)
    # Sanity: face/cell-centered shapes match the documented layout.
    assert s.u.shape == (33, 24, 16)
    assert s.v.shape == (32, 25, 16)
    assert s.w.shape == (32, 24, 17)
    assert s.p.shape == (32, 24, 16)
    assert s.p_tmp.shape == (32, 24, 16)
    assert s.div.shape == (32, 24, 16)


def test_default_solver_does_not_trigger_rebuild():
    """With enable_sub_dense=False, _should_rebuild_sub_dense is always
    False — production code path never touches the new helpers."""
    s = FlipSolver3D(nx=32, ny=32, nz=32)
    assert s._should_rebuild_sub_dense(0) is False
    assert s._should_rebuild_sub_dense(10_000) is False


# ---------------------------------------------------------------------------
# _compute_active_bbox — pure-numpy helper, no GPU.
# ---------------------------------------------------------------------------

def test_compute_active_bbox_empty_returns_none():
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=0)
    marker = np.zeros((32, 32, 32), dtype=np.int32)
    marker[0, :, :] = 2; marker[-1, :, :] = 2  # walls
    marker[:, 0, :] = 2; marker[:, -1, :] = 2
    marker[:, :, 0] = 2; marker[:, :, -1] = 2
    # No fluid cells anywhere.
    assert s._compute_active_bbox(marker) is None


def test_compute_active_bbox_snaps_to_8_tile_boundaries():
    """A single fluid cell at (10, 18, 27) lives in the 8³ tile spanning
    cells (8..16, 16..24, 24..32). Bbox (no dilation) should equal that
    tile exactly."""
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=0)
    marker = np.zeros((32, 32, 32), dtype=np.int32)
    marker[10, 18, 27] = 1
    lo, hi = s._compute_active_bbox(marker)
    bs = BLOCK_SIZE
    assert lo == (8, 16, 24)
    assert hi == (16, 24, 32)
    sub = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
    assert sub == (bs, bs, bs)


def test_compute_active_bbox_applies_dilation_clamped_to_grid():
    """Single fluid tile at cells (16..24)³ inside a 32³ grid; dilation=4
    pads the bbox by 4 cells each side. Clamps cleanly against the grid
    boundary when padding overruns."""
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=4)
    marker = np.zeros((32, 32, 32), dtype=np.int32)
    marker[16:24, 16:24, 16:24] = 1
    lo, hi = s._compute_active_bbox(marker)
    assert lo == (12, 12, 12)
    assert hi == (28, 28, 28)
    # Now place a fluid tile in the corner — dilation should clamp at 0.
    marker[:] = 0
    marker[0:8, 0:8, 0:8] = 1
    lo, hi = s._compute_active_bbox(marker)
    assert lo == (0, 0, 0)        # clamp
    assert hi == (12, 12, 12)     # 8 + dilation=4


def test_compute_active_bbox_spans_multiple_tiles():
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=0)
    marker = np.zeros((32, 32, 32), dtype=np.int32)
    marker[5, 5, 5] = 1     # tile (0..8)³
    marker[20, 20, 20] = 1  # tile (16..24)³
    lo, hi = s._compute_active_bbox(marker)
    assert lo == (0, 0, 0)
    assert hi == (24, 24, 24)


# ---------------------------------------------------------------------------
# Rebuild logic — first rebuild, periodic rebuild, edge-proximity rebuild.
# ---------------------------------------------------------------------------

def test_first_rebuild_shrinks_fields_to_bbox_and_records_offset():
    """Enable sub-dense, paint a single 8³ fluid tile, rebuild. The new
    u/v/w/p/p_tmp/div shapes should match the bbox+dilation; _sub_offset
    records the global origin."""
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=2, sub_rebuild_every=8)
    marker = s._marker_host.copy()
    marker[16:24, 16:24, 16:24] = 1
    lo, hi = s._compute_active_bbox(marker)
    assert lo == (14, 14, 14)
    assert hi == (26, 26, 26)
    s._rebuild_sub_dense(lo, hi)
    assert s._sub_offset == (14, 14, 14)
    assert s._sub_shape == (12, 12, 12)
    # u is face-centered on x, so its shape on x = sx+1; y/z = sy/sz.
    assert s.u.shape == (13, 12, 12)
    assert s.v.shape == (12, 13, 12)
    assert s.w.shape == (12, 12, 13)
    # Cell-centered fields shrink to the cell bbox.
    assert s.p.shape == (12, 12, 12)
    assert s.p_tmp.shape == (12, 12, 12)
    assert s.div.shape == (12, 12, 12)


def test_rebuild_preserves_overlap_in_global_coords():
    """Write a recognisable pattern into the sub-dense p field at known
    global coords; rebuild at a SHIFTED bbox that still overlaps the
    original; verify the pattern lands at the correct local coords in
    the new sub-dense buffer.

    This is the load-bearing invariant for B7-alt.8 to replace later:
    'rebuild copies old → new at the offset delta'.
    """
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_dilation=0)
    # First rebuild: bbox = tile (16..24)³.
    s._rebuild_sub_dense((16, 16, 16), (24, 24, 24))
    assert s._sub_offset == (16, 16, 16)
    assert s.p.shape == (8, 8, 8)
    # Stamp a unique value at local (2, 3, 4) → global (18, 19, 20).
    p_np = s.p.numpy()
    p_np[2, 3, 4] = 42.0
    s.p = wp.array(p_np, dtype=float, device=s.device)
    # Second rebuild: shifted bbox tile (20..28)³ — overlap is (20..24)³.
    s._rebuild_sub_dense((20, 20, 20), (28, 28, 28))
    assert s._sub_offset == (20, 20, 20)
    assert s.p.shape == (8, 8, 8)
    # Global (18, 19, 20) is OUTSIDE the new bbox → not copied.
    # Stamp at global (22, 21, 23) (local (6,5,7) in OLD, local (2,1,3) in NEW)
    # — but we already shifted; instead verify the global-22 etc. is not lost
    # by re-doing the experiment differently:
    s2 = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                      sub_dilation=0)
    s2._rebuild_sub_dense((16, 16, 16), (24, 24, 24))
    p_np = s2.p.numpy()
    # Global (22, 21, 23) lives at local (6, 5, 7) in the old bbox.
    p_np[6, 5, 7] = 99.0
    s2.p = wp.array(p_np, dtype=float, device=s2.device)
    s2._rebuild_sub_dense((20, 20, 20), (28, 28, 28))
    # Same global cell now sits at local (2, 1, 3) in the new bbox.
    new_p = s2.p.numpy()
    assert new_p[2, 1, 3] == pytest.approx(99.0), (
        f"overlap copy lost the value at global (22,21,23); "
        f"new_p[2,1,3]={new_p[2,1,3]}"
    )
    # Cells outside the overlap zero-initialised.
    assert new_p[0, 0, 0] == 0.0
    assert new_p[7, 7, 7] == 0.0


def test_should_rebuild_periodic_trigger():
    """With sub_rebuild_every=4, frame_idx==4 forces a rebuild even when
    the active region hasn't moved."""
    s = FlipSolver3D(nx=32, ny=32, nz=32, enable_sub_dense=True,
                     sub_rebuild_every=4, sub_dilation=8)
    # Seed a marker so _compute_active_bbox returns something.
    s._marker_host = s._wall_marker.copy()
    s._marker_host[12:20, 12:20, 12:20] = 1
    # First call always triggers (initial rebuild).
    assert s._should_rebuild_sub_dense(0) is True
    # Pretend we just rebuilt at frame 0.
    s._rebuild_sub_dense(*s._compute_active_bbox(s._marker_host))
    s._last_sub_rebuild_frame = 0
    # Same marker on frame 1, dilation comfortably exceeds proximity:
    # only the periodic trigger can fire it.
    assert s._should_rebuild_sub_dense(1) is False
    assert s._should_rebuild_sub_dense(3) is False
    assert s._should_rebuild_sub_dense(4) is True   # period hit
    assert s._should_rebuild_sub_dense(100) is True


def test_should_rebuild_proximity_trigger():
    """Active fluid encroaches within sub_dilation cells of the sub-dense
    edge — rebuild fires immediately, before the periodic timer."""
    s = FlipSolver3D(nx=64, ny=64, nz=64, enable_sub_dense=True,
                     sub_rebuild_every=1000, sub_dilation=4)
    # Initial rebuild at (16..24)³ + dilation=4 → bbox (12..28)³.
    s._marker_host = s._wall_marker.copy()
    s._marker_host[16:24, 16:24, 16:24] = 1
    s._rebuild_sub_dense(*s._compute_active_bbox(s._marker_host))
    s._last_sub_rebuild_frame = 0
    # Now fluid drifted by 2 cells — raw bbox sits at (18..26), only
    # (28-26)=2 cells from the upper edge. dilation=4 → proximity trigger.
    s._marker_host = s._wall_marker.copy()
    s._marker_host[18:26, 18:26, 18:26] = 1
    assert s._should_rebuild_sub_dense(1) is True
    # Reset to a comfortably-interior bbox — no proximity trigger.
    s._marker_host = s._wall_marker.copy()
    s._marker_host[16:24, 16:24, 16:24] = 1
    assert s._should_rebuild_sub_dense(1) is False


# ---------------------------------------------------------------------------
# Safety: step() must refuse to run while sub_offset != (0,0,0).
# B7-alt.3 will lift this once kernels learn `off_x/y/z`.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_step_raises_with_sub_offset_active_for_unrecognised_solver():
    """B7-alt.3 follow-up landed: every shipped solver knob is now
    offset-aware (pressure × transfer × CSF × viscosity × colour ×
    scalar × block-sparse). The step() guard remains only as a
    belt-and-suspenders check against unrecognised solver options.
    Verify it still fires for an unknown pressure_solver string.

    The covered configs are tested separately by
    tests/test_b7_alt_3_jacobi_dense_flip.py.
    """
    s = FlipSolver3D(nx=16, ny=16, nz=16, enable_sub_dense=True,
                     sub_dilation=0)
    s._rebuild_sub_dense((0, 0, 8), (8, 8, 16))
    assert s._sub_offset == (0, 0, 8)
    s.pos = wp.array(np.zeros((0, 3), dtype=np.float32),
                     dtype=wp.vec3, device=s.device)
    s.vel = wp.array(np.zeros((0, 3), dtype=np.float32),
                     dtype=wp.vec3, device=s.device)
    s.n_particles = 0
    with pytest.raises(NotImplementedError, match=r"pressure_solver"):
        s.step(dt=0.01, pressure_iters=1, pressure_solver="multigrid_v_cycle")
