"""B5 Option A — PCG dense becomes CUDA-graph-eligible via on-device stop-flag.

Pre-A, `_no_host_sync=True` in `_pressure_pcg` ran exactly `max_iter`
iterations regardless of convergence — the launch-overhead saving was
eaten by redundant iter work on scenes that converge early
(`big_pcg.toml` measured 4.8s → 9.3s net REGRESSION).

A flips this: after each per-iter `r·r`, `k3_check_converged` writes
`_pcg_done = 1` on device when `r_norm² < tol² × r0_norm²`. Every iter-
body kernel reads `done[0]` in its first line; once set, the rest of the
loop early-returns. Kernel SEQUENCE stays constant (good for graph
capture) while behaviour still respects `tol`.

What we assert here (rule from HANDOFF §2.3: tests must demonstrate the
feature *changes the answer*, not just "doesn't crash"):

  1. PCG with `enable_cuda_graphs=True` actually captures a graph the
     FIRST time we step (post-A; pre-A it would refuse).
  2. Graph-path PCG end-state matches direct-path PCG end-state to
     `~1e-4` relative on the velocity field. The stop-flag must NOT
     cause silent divergence vs. the host-sync path.
  3. The `_pcg_done` device flag is reachable on the solver — pin the
     surface area so future refactors don't silently remove it.
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
def test_b5_a_pcg_captures_on_first_step():
    """Pre-A this assertion held only for jacobi/gsrb. The whole point of
    A is to flip this for pcg too."""
    s = _seeded(enable_cuda_graphs=True)
    s.step(0.005, pressure_iters=10, pressure_solver="pcg")
    assert s._cuda_graph is not None, \
        "PCG dense should capture a graph on first step (Option A)"
    assert s._cuda_graph_misses == 1
    assert s._cuda_graph_hits == 0
    # Second step with same args ⇒ replay.
    s.step(0.005, pressure_iters=10, pressure_solver="pcg")
    assert s._cuda_graph_hits == 1, \
        "second PCG step should replay the cached graph"


@pytest.mark.gpu
def test_b5_a_pcg_graph_matches_direct_path():
    """Two solvers from the same seed: one runs PCG steps directly with
    host-sync convergence-check, the other through the CUDA-graph cache
    with on-device stop-flag. End state must match.

    Tolerance: the host-sync path may early-exit at iter N where N is
    < max_iter, while the graph path runs all max_iter iterations with
    the stop-flag blocking work past convergence. The result on the `p`
    field should still be the SAME pressure, because (a) PCG is
    deterministic and (b) the gated kernels do zero work post-convergence
    (they don't perturb state). Allow 1e-4 relative on |v| for any
    floating-point ordering noise.
    """
    s_direct = _seeded(enable_cuda_graphs=False)
    s_graph = _seeded(enable_cuda_graphs=True)
    for _ in range(4):
        s_direct.step(0.005, pressure_iters=30, pressure_solver="pcg")
        s_graph.step(0.005, pressure_iters=30, pressure_solver="pcg")
    assert s_graph._cuda_graph_hits == 3, \
        "expected 1 miss + 3 hits across 4 PCG steps"
    u_d = s_direct.u.numpy(); u_g = s_graph.u.numpy()
    scale = max(np.abs(u_d).max(), 1e-3)
    diff = np.abs(u_d - u_g).max() / scale
    assert diff < 1e-4, (
        f"PCG graph path drifted from direct path: |Δu|/|u| = {diff:.3g}"
    )


@pytest.mark.gpu
def test_b5_a_pcg_done_flag_present():
    """Surface area pin: `_pcg_done` (int, 1-element, device) MUST exist
    after the first PCG solve. Future refactors that silently delete it
    will revert the eligibility win — fail loudly here."""
    s = _seeded()
    s.step(0.005, pressure_iters=5, pressure_solver="pcg")
    assert hasattr(s, "_pcg_done"), \
        "Option A requires `_pcg_done` device buffer on the solver"
    assert s._pcg_done.shape == (1,)
    assert s._pcg_done.dtype == wp.int32
    # And the tol_sq sibling.
    assert hasattr(s, "_pcg_tol_sq")
    assert s._pcg_tol_sq.shape == (1,)
    assert s._pcg_tol_sq.dtype == wp.float32


@pytest.mark.gpu
def test_b5_a_pcg_with_csf_is_graph_eligible():
    """PCG + σ>0 should also be graph-eligible — the B5 follow-up made
    CSF device-side, and Option A makes PCG device-side, so both
    together should fly. Pin it so a future test for sparse-PCG-with-CSF
    can be added by analogy."""
    s = _seeded(enable_cuda_graphs=True, surface_tension=0.1)
    # σ>0 forces step_cfl; we use step() directly to keep this test
    # narrow on graph eligibility — step_cfl wraps step() internally.
    s.step(0.001, pressure_iters=8, pressure_solver="pcg")
    assert s._cuda_graph is not None, \
        "PCG + CSF should be graph-eligible after Option A"
