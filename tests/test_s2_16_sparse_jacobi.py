"""[BLK S2.16 + S2.6.4] Active-block tracker + block-sparse Jacobi pressure.

Tests:
  1. The bitmask agrees with the marker: a block is marked active iff at
     least one of its 512 cells has marker==1.
  2. Block-sparse Jacobi produces a numerically equivalent pressure field
     to dense Jacobi (same divergence, same boundary conditions, same
     iteration count).
  3. At 128³ with ≤10% fluid fill, pressure-only timing is faster than
     the dense path (loose ≥1.3× regression bar; measured 2.4× on RTX
     4080 SUPER).

Limitation note (documented in DESIGN.md §5.5): this is *iteration sparsity
only* — dense memory is still allocated for every cell. True sparse storage
(NanoVDB / wp.Volume) is a separate sprint.
"""
from __future__ import annotations

import time
import numpy as np
import pytest
import warp as wp
import warp.utils as wputils

from gpufluid.solvers.solver3d import (
    FlipSolver3D, k3_jacobi_pressure, k3_jacobi_pressure_per_tile,
    k_mark_active_blocks, k_compact_active_blocks, BLOCK_SIZE,
)


def _settled_solver(N: int, lo, hi):
    s = FlipSolver3D(nx=N, ny=N, nz=N, dx=1.0 / N, gravity=-9.81,
                     flip_blend=0.95)
    s.seed_box(lo, hi)
    s.step(0.005, pressure_iters=10)
    return s


def _build_active(s: FlipSolver3D):
    N = s.nx
    nbx = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_active = wp.zeros((nbx, nbx, nbx), dtype=int, device=s.device)
    wp.launch(k_mark_active_blocks, dim=(N, N, N),
              inputs=[s.marker, block_active, BLOCK_SIZE], device=s.device)
    return block_active


def test_bitmask_matches_marker():
    """For each 8³ block, bitmask==1 iff any cell has marker==1."""
    s = _settled_solver(32, (0.30, 0.30, 0.30), (0.70, 0.70, 0.70))
    active = _build_active(s).numpy()
    marker = s.marker.numpy()
    nbx = active.shape[0]
    for bi in range(nbx):
        for bj in range(nbx):
            for bk in range(nbx):
                tile = marker[
                    bi * BLOCK_SIZE : (bi + 1) * BLOCK_SIZE,
                    bj * BLOCK_SIZE : (bj + 1) * BLOCK_SIZE,
                    bk * BLOCK_SIZE : (bk + 1) * BLOCK_SIZE,
                ]
                expected = int((tile == 1).any())
                assert active[bi, bj, bk] == expected, (
                    f"block ({bi},{bj},{bk}): bitmask={active[bi,bj,bk]} "
                    f"expected={expected}"
                )


def test_sparse_jacobi_matches_dense_numerically():
    """Same solver state → dense Jacobi and per-tile sparse Jacobi must
    produce the same pressure field (within fp32 noise)."""
    s = _settled_solver(48, (0.30, 0.30, 0.30), (0.70, 0.70, 0.70))
    nx = s.nx
    # Build active list
    block_active = _build_active(s)
    n_blocks = block_active.shape[0] ** 3
    prefix = wp.zeros(n_blocks, dtype=int, device=s.device)
    coords = wp.zeros(n_blocks, dtype=wp.vec3i, device=s.device)
    flat = block_active.flatten()
    wputils.array_scan(flat, prefix, inclusive=True)
    n_active = int(prefix[n_blocks - 1 : n_blocks].numpy()[0])
    nbx = block_active.shape[0]
    wp.launch(k_compact_active_blocks, dim=(nbx, nbx, nbx),
              inputs=[block_active, prefix, coords], device=s.device)
    # B (post Option B): per-tile kernels require an `n_active_dev` cap.
    # Build a 1-element device array holding the host n_active value.
    n_active_dev = wp.array([n_active], dtype=int, device=s.device)
    # Dense pressure
    s.p.zero_(); s.p_tmp.zero_()
    for _ in range(80):
        wp.launch(k3_jacobi_pressure, dim=(nx, nx, nx),
                  inputs=[s.p, s.p_tmp, s.div, s.marker, 0, 0, 0],
                  device=s.device)
        s.p, s.p_tmp = s.p_tmp, s.p
    p_dense = s.p.numpy().copy()
    # Sparse pressure (reset, same div). Worst-case launch dim is the same
    # value as n_active * 512 here because the test already pre-computed
    # n_active — but kernels would also work with `nbx**3 * 512` since the
    # in-kernel cap fires.
    s.p.zero_(); s.p_tmp.zero_()
    for _ in range(80):
        wp.launch(k3_jacobi_pressure_per_tile, dim=n_active * 512,
                  inputs=[s.p, s.p_tmp, s.div, s.marker, coords, BLOCK_SIZE,
                          n_active_dev, 0, 0, 0],
                  device=s.device)
        s.p, s.p_tmp = s.p_tmp, s.p
    p_sparse = s.p.numpy()
    max_abs_err = float(np.abs(p_dense - p_sparse).max())
    # fp32 + 80 Jacobi iters in a 48³ field — sub-1e-3 absolute error.
    assert max_abs_err < 1e-3, (
        f"sparse Jacobi diverged from dense: max_abs_err={max_abs_err:.3e}"
    )


def test_sparse_jacobi_speedup_at_128():
    """At 128³ with ~9% fluid fill the per-tile sparse Jacobi must beat the
    dense launch over 60 iterations. Measured 2.4× on RTX 4080 SUPER; the
    regression bar is set at 1.3× to absorb HW variability."""
    N = 128
    s = _settled_solver(N, (0.05, 0.05, 0.05), (0.50, 0.50, 0.50))
    fill = float((s.marker.numpy() == 1).sum() / (N ** 3))
    assert 0.05 < fill < 0.15, f"setup landed at fill {fill:.2f}, expected ~10%"
    # Build active list
    block_active = _build_active(s)
    nbx = block_active.shape[0]; n_blocks = nbx ** 3
    prefix = wp.zeros(n_blocks, dtype=int, device=s.device)
    coords = wp.zeros(n_blocks, dtype=wp.vec3i, device=s.device)
    flat = block_active.flatten()
    wputils.array_scan(flat, prefix, inclusive=True)
    n_active = int(prefix[n_blocks - 1 : n_blocks].numpy()[0])
    wp.launch(k_compact_active_blocks, dim=(nbx, nbx, nbx),
              inputs=[block_active, prefix, coords], device=s.device)
    n_active_dev = wp.array([n_active], dtype=int, device=s.device)
    iters = 80

    def bench_dense():
        s.p.zero_(); s.p_tmp.zero_()
        for _ in range(iters):
            wp.launch(k3_jacobi_pressure, dim=(N, N, N),
                      inputs=[s.p, s.p_tmp, s.div, s.marker, 0, 0, 0],
                  device=s.device)
            s.p, s.p_tmp = s.p_tmp, s.p

    def bench_sparse():
        s.p.zero_(); s.p_tmp.zero_()
        for _ in range(iters):
            wp.launch(k3_jacobi_pressure_per_tile, dim=n_active * 512,
                      inputs=[s.p, s.p_tmp, s.div, s.marker, coords, BLOCK_SIZE,
                              n_active_dev, 0, 0, 0],
                      device=s.device)
            s.p, s.p_tmp = s.p_tmp, s.p

    bench_dense(); bench_sparse(); wp.synchronize()  # warm
    n = 3
    t0 = time.time()
    for _ in range(n):
        bench_dense(); wp.synchronize()
    td = (time.time() - t0) / n
    t0 = time.time()
    for _ in range(n):
        bench_sparse(); wp.synchronize()
    ts = (time.time() - t0) / n
    speedup = td / ts
    assert speedup >= 1.3, (
        f"sparse Jacobi speedup at 128³ low fill should be ≥1.3×; "
        f"got {speedup:.2f}× (dense {td*1000:.1f}ms, sparse {ts*1000:.1f}ms)"
    )
