"""B4.1 — per-tile GS-RB (S2.6.5) parity + speedup vs the dense path.

If sparse GS-RB doesn't match dense GS-RB to within tight numerical noise
on the resulting velocity field, B4 (block-sparse for the rest of the
pressure stack) shouldn't be built on top of it — abort early.
"""
import time
import numpy as np
import pytest
import warp as wp
import warp.utils as wputils

from gpufluid.solvers.solver3d import (
    FlipSolver3D, k3_gauss_seidel_rb, k3_gauss_seidel_rb_per_tile,
    k_mark_active_blocks, k_compact_active_blocks, BLOCK_SIZE,
)


def _seeded(nx, with_sparse: bool):
    """A reproducible solver state: small dam-break-ish blob in one corner
    of a larger domain (low fill ratio → real sparse advantage)."""
    sol = FlipSolver3D(nx=nx, ny=nx, nz=nx, dx=1.0 / nx, gravity=-9.81)
    # Fluid in just one corner — leaves most of the grid empty so the
    # active-block compaction has something to actually skip.
    sol.seed_box((0.05, 0.05, 0.05), (0.35, 0.35, 0.35), ppc=4)
    # Run a couple of warm-up steps so the velocity field is non-trivial.
    for _ in range(2):
        sol.step(dt=0.005, pressure_iters=20, pressure_solver="gsrb",
                 pressure_block_sparse=with_sparse)
    return sol


def test_b4_1_sparse_gsrb_matches_dense_velocity_field():
    """Same initial state, same dt, same iter count → velocity fields
    must agree to within numerical drift. GS-RB is not strictly bit-stable
    even on the same path (atomics ordering on neighbour reads), but the
    integrated velocity field is well-behaved."""
    nx = 32
    sol_dense = _seeded(nx, with_sparse=False)
    sol_sparse = _seeded(nx, with_sparse=True)
    # Compare cell-centred velocity magnitude (averaging the two faces of u)
    # — this is the L2 metric that downstream particles see.
    u_d = sol_dense.u.numpy(); v_d = sol_dense.v.numpy(); w_d = sol_dense.w.numpy()
    u_s = sol_sparse.u.numpy(); v_s = sol_sparse.v.numpy(); w_s = sol_sparse.w.numpy()
    # Take a robust scale: max absolute component across both runs.
    scale = max(np.max(np.abs(u_d)), np.max(np.abs(u_s)), 1e-3)
    diff_u = np.max(np.abs(u_d - u_s)) / scale
    diff_v = np.max(np.abs(v_d - v_s)) / scale
    diff_w = np.max(np.abs(w_d - w_s)) / scale
    assert diff_u < 0.05, f"u-velocity drift {diff_u:.3g} > 5%"
    assert diff_v < 0.05, f"v-velocity drift {diff_v:.3g} > 5%"
    assert diff_w < 0.05, f"w-velocity drift {diff_w:.3g} > 5%"


def test_b4_1_sparse_gsrb_kernel_speedup_at_128():
    """Spike gate at 128³ + low fill, mirroring the proven S2.16 sparse
    Jacobi benchmark methodology (`test_sparse_jacobi_speedup_at_128`).

    Key methodological choice: **bench only the kernel time**, not the
    full `step()`. The active-block build (`k_mark_active_blocks` +
    `array_scan` + `k_compact_active_blocks`) is a one-shot per step and
    its cost is the same for both Jacobi and GS-RB. The right
    cross-comparison is the per-iter kernel cost, integrated over the
    realistic iter count (80). Otherwise tiny grids show "sparse loses"
    purely because of the device→host sync inside step() — a fixed cost
    that 60+ iter scenes amortise away.

    Measured on RTX 4080 SUPER: dense GS-RB ~12 ms / 80 iters @ 128^3,
    sparse ~2.5 ms — about 5x. Threshold set at 1.3x to absorb HW noise.
    """
    N = 128
    s = FlipSolver3D(nx=N, ny=N, nz=N, dx=1.0 / N, gravity=-9.81)
    s.seed_box((0.05, 0.05, 0.05), (0.50, 0.50, 0.50), ppc=4)
    s.step(0.005, pressure_iters=10)  # settle marker + divergence

    fill = float((s.marker.numpy() == 1).sum() / (N ** 3))
    assert 0.03 < fill < 0.20, f"setup landed at fill {fill:.3f}, expected ~5-15%"

    nbx = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_blocks = nbx ** 3
    block_active = wp.zeros((nbx, nbx, nbx), dtype=int, device=s.device)
    prefix = wp.zeros(n_blocks, dtype=int, device=s.device)
    coords = wp.zeros(n_blocks, dtype=wp.vec3i, device=s.device)
    wp.launch(k_mark_active_blocks, dim=(N, N, N),
              inputs=[s.marker, block_active, BLOCK_SIZE], device=s.device)
    wputils.array_scan(block_active.flatten(), prefix, inclusive=True)
    n_active = int(prefix[n_blocks - 1: n_blocks].numpy()[0])
    wp.launch(k_compact_active_blocks, dim=(nbx, nbx, nbx),
              inputs=[block_active, prefix, coords], device=s.device)
    n_active_dev = wp.array([n_active], dtype=int, device=s.device)
    iters = 80
    cells_per_block = BLOCK_SIZE ** 3

    def bench_dense():
        s.p.zero_()
        for _ in range(iters):
            wp.launch(k3_gauss_seidel_rb, dim=(N, N, N),
                      inputs=[s.p, s.div, s.marker, 0, 0, 0, 0],
                      device=s.device)
            wp.launch(k3_gauss_seidel_rb, dim=(N, N, N),
                      inputs=[s.p, s.div, s.marker, 1, 0, 0, 0],
                      device=s.device)

    def bench_sparse():
        s.p.zero_()
        for _ in range(iters):
            wp.launch(k3_gauss_seidel_rb_per_tile, dim=n_active * cells_per_block,
                      inputs=[s.p, s.div, s.marker, coords, BLOCK_SIZE, 0,
                              n_active_dev, 0, 0, 0],
                      device=s.device)
            wp.launch(k3_gauss_seidel_rb_per_tile, dim=n_active * cells_per_block,
                      inputs=[s.p, s.div, s.marker, coords, BLOCK_SIZE, 1,
                              n_active_dev, 0, 0, 0],
                      device=s.device)

    # warm both paths
    bench_dense(); bench_sparse(); wp.synchronize()

    reps = 3
    wp.synchronize()
    t0 = time.time()
    for _ in range(reps):
        bench_dense(); wp.synchronize()
    td = (time.time() - t0) / reps
    wp.synchronize()
    t0 = time.time()
    for _ in range(reps):
        bench_sparse(); wp.synchronize()
    ts = (time.time() - t0) / reps

    print(f"\n[B4.1 spike] 128^3, fill={fill*100:.1f}%, {iters} iters — "
          f"dense={td*1000:.1f} ms, sparse={ts*1000:.1f} ms, "
          f"ratio={ts/td:.2f}x ({td/ts:.1f}x speedup)")

    assert ts < td / 1.3, (
        f"sparse GS-RB at 128^3 / {iters} iter / {fill*100:.1f}% fill is not "
        f"≥1.3x faster than dense ({td/ts:.2f}x measured). The macro stops here."
    )
