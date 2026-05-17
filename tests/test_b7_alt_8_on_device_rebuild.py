"""B7-alt.8 — on-device sub-dense rebuild copy.

Replaces the CPU round-trip in `_rebuild_sub_dense` with a per-field
GPU kernel that maps NEW (li,lj,lk) -> GLOBAL -> OLD (si,sj,sk) and
copies. The overlap-preservation invariant (B7-alt.2 test_rebuild_
preserves_overlap_in_global_coords) is the load-bearing correctness
check; this file documents the perf side of the change.
"""
import time

import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import FlipSolver3D, k3_copy_subdense_at_offset


@pytest.mark.gpu
def test_b7_alt_8_kernel_copies_at_offset_delta():
    """Direct kernel exercise: pre-fill src with a recognisable pattern,
    launch the copy at a known offset delta, verify dst picks up exactly
    the overlap region in the right place."""
    dev = "cuda:0"
    # src lives at offset (10, 0, 0) with shape (4, 4, 4); cells global
    # coords (10..14, 0..4, 0..4). Fill with global linear index for ID.
    src_np = np.zeros((4, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                gi = i + 10
                src_np[i, j, k] = gi * 1000 + j * 100 + k
    src = wp.array(src_np, dtype=float, device=dev)
    # dst lives at offset (12, 0, 0) with shape (4, 4, 4); overlap is
    # global cells (12..14, 0..4, 0..4) -> local dst (0..2) ↔ src local (2..4).
    dst = wp.zeros((4, 4, 4), dtype=float, device=dev)
    wp.launch(
        k3_copy_subdense_at_offset, dim=(4, 4, 4),
        inputs=[src, dst, 12, 0, 0, 10, 0, 0],
        device=dev,
    )
    dst_np = dst.numpy()
    # Local dst (0, 0, 0) -> global (12, 0, 0) -> src local (2, 0, 0)
    #   = 12 * 1000 + 0 + 0 = 12000
    assert dst_np[0, 0, 0] == pytest.approx(12000.0)
    # Local dst (2, 3, 1) -> global (14, 3, 1) -> src local (4, 3, 1) OOB
    #   on x — should stay 0 (kernel bounds-checks).
    assert dst_np[2, 3, 1] == pytest.approx(0.0)
    # Local dst (1, 2, 3) -> global (13, 2, 3) -> src local (3, 2, 3)
    #   = 13 * 1000 + 200 + 3 = 13203
    assert dst_np[1, 2, 3] == pytest.approx(13203.0)


@pytest.mark.gpu
def test_b7_alt_8_rebuild_faster_than_cpu_roundtrip_at_256():
    """At 256^3 the previous CPU round-trip per-field cost ~50-100 ms
    (read sub-shape numpy, slice, re-allocate). On-device should be a
    handful of milliseconds total for 12 fields. Sanity bar: <100 ms
    total for the whole rebuild including allocs."""
    N = 256
    s = FlipSolver3D(nx=N, ny=N, nz=N, transfer_mode="flip",
                     enable_sub_dense=True,
                     sub_rebuild_every=4, sub_dilation=6)
    s.seed_box((25/N, 25/N, 25/N), (150/N, 150/N, 150/N), ppc=2)
    s.step(dt=0.005, pressure_iters=10, pressure_solver="jacobi")
    # First rebuild — sets up the sub-dense bbox.
    s.prepare_frame(0, 1.0 / 24)
    wp.synchronize_device("cuda:0")
    bbox0 = (s._sub_offset, s._sub_shape)
    # Force a SECOND rebuild at a shifted bbox so the copy path runs
    # (the kernel only fires when there's overlap; the first rebuild
    # has no old data).
    new_lo = tuple(min(N - 8, c + 8) for c in s._sub_offset)
    new_hi = tuple(min(N, l + s._sub_shape[a]) for a, l in enumerate(new_lo))
    wp.synchronize_device("cuda:0")
    t0 = time.time()
    s._rebuild_sub_dense(new_lo, new_hi)
    wp.synchronize_device("cuda:0")
    rebuild_ms = (time.time() - t0) * 1000.0
    print(f"\n[B7-alt.8] second rebuild at 256^3, "
          f"old_bbox={bbox0[1]} -> new_bbox={s._sub_shape}, "
          f"on-device rebuild={rebuild_ms:.1f} ms")
    assert rebuild_ms < 100.0, (
        f"on-device rebuild took {rebuild_ms:.1f} ms — slower than the "
        f"100 ms bar; CPU round-trip may have crept back"
    )
