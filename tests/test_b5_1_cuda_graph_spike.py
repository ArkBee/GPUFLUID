"""B5.1 — CUDA-graphs spike: capture step() once, replay it.

The hot path issues ~20-30 `wp.launch` calls per step. Each launch has a
~5-10 µs CPU overhead on Windows. A CUDA graph captures all of them
into a single replayable object, so subsequent "steps" are one launch
call instead of N.

Spike scope (per BACKLOG B5.1-B5.5):
  1. Audit step() for host-syncs that would invalidate the graph.
  2. Pick a configuration that has zero syncs.
  3. Capture the step.
  4. Replay it N times, compare to N direct-step calls — same answer.
  5. Bench: any speedup at the spike scale?

Sync-bearing branches identified in step():
  PCG     — r_norm² check every 10 iters (-> .numpy())
  CSF     — S2.14.6 force-balance reads 3 sums per step
  Sparse  — every variant reads n_active via .numpy()

So the only currently-graph-safe combination is:
  pressure_solver = "jacobi" or "gsrb"
  pressure_block_sparse = False
  surface_tension = 0
  transfer_mode = "flip" / "pic" / "apic" (all safe)

This is the spike's first finding: of the 9 solver configurations we
ship, **3 are graph-eligible** as of today. The rest need their syncs
moved out of step() before they can use graphs. That's the macro-sized
B5 work.
"""
import time
import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import FlipSolver3D


def _graph_eligible_solver(N: int):
    """The narrowest config that has no syncs inside step():
    Jacobi pressure, no CSF, no block-sparse."""
    s = FlipSolver3D(nx=N, ny=N, nz=N, dx=1.0 / N, gravity=-9.81,
                     surface_tension=0.0)
    s.seed_box((0.30, 0.30, 0.30), (0.60, 0.60, 0.60), ppc=4)
    # one warm step to populate the marker + load the kernel modules.
    s.step(0.005, pressure_iters=20, pressure_solver="jacobi")
    wp.synchronize()
    return s


def test_b5_1_graph_capture_completes():
    """The graph can be captured at all — proves no hidden host-sync
    sneaked into step() under the Jacobi/no-CSF/no-sparse config.

    If this fails, the audit missed a sync. Stack trace will name the
    .numpy()/.synchronize() call site → fix it before continuing."""
    s = _graph_eligible_solver(32)
    wp.capture_begin(device="cuda:0", force_module_load=False)
    try:
        s.step(0.005, pressure_iters=20, pressure_solver="jacobi")
    except Exception:
        # Make sure we end the capture even on error, otherwise Warp's
        # global capture state stays stuck and breaks unrelated tests.
        try:
            wp.capture_end(device="cuda:0")
        except Exception:
            pass
        raise
    graph = wp.capture_end(device="cuda:0")
    assert graph is not None


def test_b5_1_graph_replay_matches_direct_step():
    """Two solvers starting from the same state — one calls step()
    directly N times, the other captures and replays a graph N times.
    Final velocity / pressure / particle positions must agree."""
    N = 32
    rng_seed_state = np.random.get_state()
    s_direct = _graph_eligible_solver(N)
    np.random.set_state(rng_seed_state)
    s_graph = _graph_eligible_solver(N)

    # capture on s_graph from a state that matches s_direct after warmup
    wp.capture_begin(device="cuda:0", force_module_load=False)
    try:
        s_graph.step(0.005, pressure_iters=20, pressure_solver="jacobi")
    except Exception:
        try: wp.capture_end(device="cuda:0")
        except Exception: pass
        raise
    graph = wp.capture_end(device="cuda:0")

    n_steps = 5
    for _ in range(n_steps):
        s_direct.step(0.005, pressure_iters=20, pressure_solver="jacobi")
    for _ in range(n_steps):
        wp.capture_launch(graph)
    wp.synchronize()

    # Compare end-state velocity fields. Identical kernel sequence → bit-
    # identical results (no scheduler nondeterminism in pure Jacobi).
    u_d = s_direct.u.numpy(); u_g = s_graph.u.numpy()
    scale = max(np.abs(u_d).max(), 1e-3)
    diff = np.abs(u_d - u_g).max() / scale
    assert diff < 1e-4, f"graph replay diverged from direct: {diff:.3g}"


def test_b5_1_graph_replay_speedup():
    """Bench direct vs graph replay. Target ≥1.15x at 64³ per BACKLOG."""
    N = 64
    s = _graph_eligible_solver(N)

    # capture
    wp.capture_begin(device="cuda:0", force_module_load=False)
    try:
        s.step(0.005, pressure_iters=20, pressure_solver="jacobi")
    except Exception:
        try: wp.capture_end(device="cuda:0")
        except Exception: pass
        raise
    graph = wp.capture_end(device="cuda:0")

    # warm both paths
    for _ in range(3):
        s.step(0.005, pressure_iters=20, pressure_solver="jacobi")
    wp.synchronize()
    for _ in range(3):
        wp.capture_launch(graph)
    wp.synchronize()

    n = 30
    t0 = time.time()
    for _ in range(n):
        s.step(0.005, pressure_iters=20, pressure_solver="jacobi")
    wp.synchronize()
    t_direct = (time.time() - t0) / n

    t0 = time.time()
    for _ in range(n):
        wp.capture_launch(graph)
    wp.synchronize()
    t_graph = (time.time() - t0) / n

    print(f"\n[B5.1] N={N}, 20 jacobi iters — "
          f"direct={t_direct*1000:.2f} ms, graph={t_graph*1000:.2f} ms, "
          f"speedup={t_direct/t_graph:.2f}x")
    assert t_graph < t_direct / 1.15, (
        f"CUDA graph replay only {t_direct/t_graph:.2f}x faster than direct "
        f"at 64^3 / 20 iters. BACKLOG B5.5 target was >=1.15x. Macro "
        f"B5 may not be worth shipping at this scale."
    )
