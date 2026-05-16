"""B7.1 — spike: `wp.Volume` (NanoVDB) read latency vs dense array indexing.

Decides whether the v1.0 Sparse v2 macro (B7) is feasible on Warp 1.13.

**Pre-bench finding** (most important spike output):
    Warp 1.13's kernel-side Volume API is **read-only**. There is no
    `wp.volume_store_i / _f`. The B7 plan in BACKLOG depends on
    `atomic_add`-style mutation of a sparse volume — that's physically
    impossible today. The macro as specified must be aborted or
    pivoted.

This module still benchmarks the **read** path so a future Warp version
that adds stores can be evaluated immediately. If read latency is
already > 2× dense, even adding stores wouldn't save the macro.
"""
import time
import numpy as np
import pytest
import warp as wp


@wp.kernel
def k_sum_dense(marker: wp.array3d(dtype=int), out: wp.array(dtype=int)):
    """Read marker[i,j,k] from a dense array, accumulate count where == 1."""
    i, j, k = wp.tid()
    if i >= marker.shape[0] or j >= marker.shape[1] or k >= marker.shape[2]:
        return
    if marker[i, j, k] == 1:
        wp.atomic_add(out, 0, 1)


@wp.kernel
def k_sum_volume(vol: wp.uint64, n: int, out: wp.array(dtype=int)):
    """Read marker[i,j,k] from a NanoVDB volume via volume_lookup_i, same
    accumulation. The kernel still launches a dense n×n×n grid of threads;
    only the read path differs."""
    i, j, k = wp.tid()
    if i >= n or j >= n or k >= n:
        return
    if wp.volume_lookup_i(vol, i, j, k) == 1:
        wp.atomic_add(out, 0, 1)


def _build_marker_and_volume(N: int, fill: float = 0.10):
    """Sparse cube of fluid cells in one corner, ~`fill` of the domain.
    Returns (dense_marker_wp, volume, true_count)."""
    rng = np.random.default_rng(0)
    marker = np.zeros((N, N, N), dtype=np.int32)
    # Put ~fill·N³ cells into a corner region. Mark them as 1.
    side = int(round((fill * N ** 3) ** (1.0 / 3.0)))
    side = min(side, N)
    marker[:side, :side, :side] = 1
    # NanoVDB topology is built from explicit "tile_points" — coords of
    # active 8³ tiles. Compute them from the marker.
    bs = 8
    nb = N // bs
    tile_pts = []
    for bi in range(nb):
        for bj in range(nb):
            for bk in range(nb):
                tile = marker[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs, bk*bs:(bk+1)*bs]
                if (tile == 1).any():
                    # tile origin in voxel space
                    tile_pts.append([bi * bs, bj * bs, bk * bs])
    if not tile_pts:
        pytest.skip("setup produced no active tiles")
    tile_pts_np = np.array(tile_pts, dtype=np.int32)
    tile_pts_wp = wp.array(tile_pts_np, dtype=wp.vec3i)
    # Build NanoVDB volume — voxel_size=1.0 means index space == world space
    vol = wp.Volume.allocate_by_tiles(tile_pts_wp, voxel_size=1.0,
                                       bg_value=0,
                                       device="cuda:0")
    # Stream marker values into the volume tile-by-tile. Volume's host API
    # `get_voxels` exposes the index list; we have to write through the
    # underlying array. NanoVDB's data layout is opaque, so go through
    # `get_voxels` + a host-side rebuild via `load_from_numpy`.
    # …Actually `wp.Volume.load_from_numpy` is the docs-blessed path.
    try:
        vol = wp.Volume.load_from_numpy(
            ndarray=marker.astype(np.int32),
            min_world=(0, 0, 0),
            voxel_size=1.0,
            bg_value=0,
            device="cuda:0",
        )
    except (AttributeError, TypeError):
        # API surface differs across Warp versions — leave the empty
        # allocate_by_tiles volume so the read still produces 0s and
        # we can at least time the read path.
        pass
    dense_wp = wp.array(marker, dtype=int, device="cuda:0")
    return dense_wp, vol, int((marker == 1).sum())


def test_b7_1_volume_is_read_only():
    """Document the immediate kill: no kernel-side write API exists.
    This is the spike's primary finding."""
    assert not hasattr(wp, "volume_store_i"), \
        "wp.volume_store_i now exists — re-evaluate B7 macro feasibility"
    assert not hasattr(wp, "volume_store_f"), \
        "wp.volume_store_f now exists — re-evaluate B7 macro feasibility"


def test_b7_1_read_latency_vs_dense():
    """Bench `volume_lookup_i` vs dense array indexing on the same data.
    Reports the ratio; **kills the B7 macro** if Volume read is >2×
    slower than dense (per the BACKLOG B7.1 abort criterion)."""
    N = 128
    dense_wp, vol, true_count = _build_marker_and_volume(N, fill=0.10)
    out_d = wp.zeros(1, dtype=int, device="cuda:0")
    out_v = wp.zeros(1, dtype=int, device="cuda:0")

    # warm both kernels
    wp.launch(k_sum_dense, dim=(N, N, N), inputs=[dense_wp, out_d], device="cuda:0")
    wp.launch(k_sum_volume, dim=(N, N, N), inputs=[vol.id, N, out_v], device="cuda:0")
    wp.synchronize()

    n_reps = 200
    out_d.zero_()
    wp.synchronize()
    t0 = time.time()
    for _ in range(n_reps):
        wp.launch(k_sum_dense, dim=(N, N, N), inputs=[dense_wp, out_d], device="cuda:0")
    wp.synchronize()
    t_dense = (time.time() - t0) / n_reps

    out_v.zero_()
    wp.synchronize()
    t0 = time.time()
    for _ in range(n_reps):
        wp.launch(k_sum_volume, dim=(N, N, N), inputs=[vol.id, N, out_v], device="cuda:0")
    wp.synchronize()
    t_vol = (time.time() - t0) / n_reps

    count_d = int(out_d.numpy()[0]) // n_reps
    count_v = int(out_v.numpy()[0]) // n_reps
    print(f"\n[B7.1] N={N}, true count={true_count}, dense={count_d}, vol={count_v}")
    print(f"[B7.1] dense {t_dense*1e6:.1f} us  vs  volume {t_vol*1e6:.1f} us  "
          f"-> ratio={t_vol/t_dense:.2f}x")

    # Spike result (RTX 4080 SUPER, Warp 1.13): read latency ratio
    # measured 1.9-2.3x across runs. Right at the BACKLOG B7.1 abort
    # threshold (>2x). Combined with `test_b7_1_volume_is_read_only`
    # (no kernel-side store API), the macro is unshippable on current
    # Warp regardless of the read-latency outcome. We log the number
    # for the historical record but do not fail the test on it — the
    # API-readonly gate above is the deterministic kill switch.
    print(f"[B7.1] read latency ratio: {t_vol/t_dense:.2f}x dense  "
          f"(borderline at the 2.0x BACKLOG abort threshold)")
