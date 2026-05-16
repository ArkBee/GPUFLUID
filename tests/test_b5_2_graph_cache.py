"""B5.2 — CUDA-graph cache + invalidation on prepare_frame.

The B5.1 spike proved capture/replay works manually with a 2.17x
speedup. This module wires it into FlipSolver3D as an automatic
cache: solver._cuda_graph captures the first eligible step() in a
given (config, n_particles, dt, …) combination and replays it until
the topology changes.

What this tests:
  * Hit/miss accounting reflects what actually happened.
  * `prepare_frame` invalidates the cache (otherwise frame 2 would
    silently replay frame 1's stale topology).
  * Direct steps and graph-replay steps land on the same velocity
    field (the replay isn't drifting).
  * Disabled-by-default — `enable_cuda_graphs=False` (the default) is
    bit-identical to the original solver behaviour.
  * Ineligible configs (PCG, CSF, block-sparse) fall through to the
    direct path without errors.
"""
import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import FlipSolver3D


def _seeded(N=32, **kwargs):
    s = FlipSolver3D(nx=N, ny=N, nz=N, dx=1.0 / N, gravity=-9.81, **kwargs)
    s.seed_box((0.30, 0.30, 0.30), (0.60, 0.60, 0.60), ppc=4)
    return s


def test_b5_2_disabled_by_default_no_capture():
    """The flag is opt-in. Default behaviour: no graph created."""
    s = _seeded()
    for _ in range(3):
        s.step(0.005, pressure_iters=10)
    assert s._cuda_graph is None
    assert s._cuda_graph_hits == 0
    assert s._cuda_graph_misses == 0


def test_b5_2_first_step_misses_then_hits():
    """First eligible step in a fresh topology captures (miss). Subsequent
    steps with the same args replay (hits)."""
    s = _seeded(enable_cuda_graphs=True)
    s.step(0.005, pressure_iters=10, pressure_solver="jacobi")
    assert s._cuda_graph is not None
    assert s._cuda_graph_misses == 1
    assert s._cuda_graph_hits == 0
    for _ in range(4):
        s.step(0.005, pressure_iters=10, pressure_solver="jacobi")
    assert s._cuda_graph_hits == 4
    assert s._cuda_graph_misses == 1


def test_b5_2_prepare_frame_invalidates_graph():
    """prepare_frame can change n_particles or marker. Cache MUST drop."""
    s = _seeded(enable_cuda_graphs=True)
    s.step(0.005, pressure_iters=10, pressure_solver="jacobi")
    assert s._cuda_graph is not None
    s.prepare_frame(frame_idx=1, frame_dt=1.0 / 24.0)
    assert s._cuda_graph is None, "prepare_frame did not invalidate the cache"


def test_b5_2_changed_args_force_recapture():
    """Different pressure_iters → different launch count → cache key must
    differ → recapture, not silent reuse of the stale graph."""
    s = _seeded(enable_cuda_graphs=True)
    s.step(0.005, pressure_iters=10, pressure_solver="jacobi")
    s.step(0.005, pressure_iters=20, pressure_solver="jacobi")  # different
    s.step(0.005, pressure_iters=20, pressure_solver="jacobi")  # same as above
    assert s._cuda_graph_misses == 2  # initial capture + 10→20 recapture
    assert s._cuda_graph_hits == 1    # only the 3rd step matched its predecessor


def test_b5_2_ineligible_config_falls_through():
    """Jacobi/GS-RB are graph-eligible, with or without CSF. PCG dense
    is now ALSO eligible (Option A — on-device stop-flag pattern).
    `pressure_block_sparse=True` is still excluded — `_build_active_blocks`
    reads `n_active.numpy()`; fixing this is Option B."""
    # PCG (dense): eligible after the on-device convergence-check landed.
    # The `_no_host_sync` path now sets `_pcg_done` on device when
    # `r·r < tol_sq·r0_norm²`; subsequent iter kernels early-return.
    s_pcg = _seeded(enable_cuda_graphs=True)
    s_pcg.step(0.005, pressure_iters=5, pressure_solver="pcg")
    assert s_pcg._cuda_graph is not None, \
        "PCG dense should be graph-eligible after the stop-flag refactor"

    # CSF: eligible after k3_csf_subtract_bias_*_dev refactor.
    s_csf = _seeded(enable_cuda_graphs=True, surface_tension=0.1)
    s_csf.step(0.001, pressure_iters=10, pressure_solver="jacobi")
    assert s_csf._cuda_graph is not None, \
        "CSF should be graph-eligible after the S2.14.6 device-side refactor"

    # Block-sparse: still INELIGIBLE — `_build_active_blocks` reads
    # `n_active.numpy()` inside step(). Fixing this needs a deeper change
    # (launch the per-tile kernels at worst-case dim + cap in-kernel).
    # Tracked as Option B in §12 of HANDOFF.
    s_sparse = _seeded(enable_cuda_graphs=True)
    s_sparse.step(0.005, pressure_iters=10, pressure_solver="jacobi",
                   pressure_block_sparse=True)
    assert s_sparse._cuda_graph is None, \
        "block-sparse pressure should still fall through to the direct path"


def test_b5_2_replay_matches_direct_to_fp_precision():
    """Two solvers from the same seed: one runs steps directly, the other
    through the graph cache. End state must match — otherwise the cache
    is corrupting state."""
    s_direct = _seeded(enable_cuda_graphs=False)
    s_graph = _seeded(enable_cuda_graphs=True)
    for _ in range(5):
        s_direct.step(0.005, pressure_iters=10, pressure_solver="jacobi")
        s_graph.step(0.005, pressure_iters=10, pressure_solver="jacobi")
    # The second solver had 1 capture + 4 hits.
    assert s_graph._cuda_graph_hits == 4
    u_d = s_direct.u.numpy(); u_g = s_graph.u.numpy()
    scale = max(np.abs(u_d).max(), 1e-3)
    diff = np.abs(u_d - u_g).max() / scale
    assert diff < 1e-4, f"graph path drifted: {diff:.3g}"
