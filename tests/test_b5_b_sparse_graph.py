"""B5 Option B — block-sparse pressure becomes CUDA-graph-eligible via
`n_active_dev` device buffer + worst-case launch dim.

Pre-B, `_build_active_blocks` did `prefix[n_blocks-1].numpy()[0]` to get
a host-int `n_active`, which aborted graph capture. After B:
  * `k_store_n_active(prefix, n_blocks, n_active_dev)` writes that value
    to a 1-element device buffer instead.
  * Six per-tile kernels (k3_jacobi_pressure_per_tile,
    k3_gauss_seidel_rb_per_tile, k3_apply_A_per_tile, k3_apply_invM_per_tile,
    k3_dot_fluid_per_tile, k3_axpy_devscalar_per_tile) gain an
    `n_active_dev: wp.array(dtype=int)` parameter and an early-return
    `if blk >= n_active_dev[0]: return` after computing `blk`.
  * Per-tile launches use `dim = n_blocks * cells_per_block` (constant
    for a given resolution) instead of `n_active * cells_per_block`.

Sparse PCG additionally inherits Option A's on-device stop flag — its
per-tile kernels take BOTH `n_active_dev` (B cap) AND `done` (A flag).

What we assert:

  1. Sparse jacobi/gs-rb/PCG all capture a graph on first step.
  2. End state of sparse-graph PCG matches sparse-direct PCG to ~1e-4
     relative — the worst-case launch + in-kernel cap must NOT alter
     numerics.
  3. The `_n_active_dev` device buffer is reachable on the solver
     (surface-area pin against silent refactors).
"""
import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import FlipSolver3D


def _seeded(N=32, **kwargs):
    s = FlipSolver3D(nx=N, ny=N, nz=N, dx=1.0 / N, gravity=-9.81, **kwargs)
    s.seed_box((0.30, 0.30, 0.30), (0.60, 0.60, 0.60), ppc=4)
    return s


@pytest.mark.gpu
def test_b5_b_sparse_jacobi_captures():
    s = _seeded(enable_cuda_graphs=True)
    s.step(0.005, pressure_iters=10, pressure_solver="jacobi",
           pressure_block_sparse=True)
    assert s._cuda_graph is not None
    assert s._cuda_graph_misses == 1
    s.step(0.005, pressure_iters=10, pressure_solver="jacobi",
           pressure_block_sparse=True)
    assert s._cuda_graph_hits == 1


@pytest.mark.gpu
def test_b5_b_sparse_gsrb_captures():
    s = _seeded(enable_cuda_graphs=True)
    s.step(0.005, pressure_iters=10, pressure_solver="gsrb",
           pressure_block_sparse=True)
    assert s._cuda_graph is not None


@pytest.mark.gpu
def test_b5_b_sparse_pcg_captures():
    """Sparse PCG inherits BOTH options. This is the v0.9 9/9 closure."""
    s = _seeded(enable_cuda_graphs=True)
    s.step(0.005, pressure_iters=10, pressure_solver="pcg",
           pressure_block_sparse=True)
    assert s._cuda_graph is not None, \
        "sparse PCG needs Options A + B together; both should be active"


@pytest.mark.gpu
def test_b5_b_sparse_pcg_graph_matches_direct():
    """Two solvers from the same seed: one runs sparse PCG steps directly,
    the other through the graph cache. End state must match — the
    worst-case launch dim + in-kernel cap + device-side convergence
    detection must not introduce silent numerical drift."""
    s_direct = _seeded(enable_cuda_graphs=False)
    s_graph = _seeded(enable_cuda_graphs=True)
    for _ in range(4):
        s_direct.step(0.005, pressure_iters=30, pressure_solver="pcg",
                       pressure_block_sparse=True)
        s_graph.step(0.005, pressure_iters=30, pressure_solver="pcg",
                      pressure_block_sparse=True)
    assert s_graph._cuda_graph_hits == 3
    u_d = s_direct.u.numpy(); u_g = s_graph.u.numpy()
    scale = max(np.abs(u_d).max(), 1e-3)
    diff = np.abs(u_d - u_g).max() / scale
    assert diff < 1e-4, (
        f"sparse-graph PCG drifted from sparse-direct PCG: "
        f"|Δu|/|u| = {diff:.3g}"
    )


@pytest.mark.gpu
def test_b5_b_n_active_dev_present():
    """Surface-area pin: `_n_active_dev` (int, shape=(1,), device) MUST
    exist after the first sparse-pressure step. Future refactors that
    silently delete it will revert the v0.9 9/9 closure — fail loudly."""
    s = _seeded()
    s.step(0.005, pressure_iters=5, pressure_solver="jacobi",
           pressure_block_sparse=True)
    assert hasattr(s, "_n_active_dev")
    assert s._n_active_dev is not None
    assert s._n_active_dev.shape == (1,)
    assert s._n_active_dev.dtype == wp.int32
