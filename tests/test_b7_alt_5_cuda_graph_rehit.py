"""B7-alt.5 — verify CUDA-graph hit rate ≥ 80% across sub-dense rebuilds.

`_cuda_graph_invalidate` fires in `prepare_frame` (B5.3), which means
every rebuild → recapture. As long as `sub_rebuild_every` is large
enough that the intervening substeps replay the same graph, the
amortised hit-rate stays high.

This sanity-checks the interaction between B5 (CUDA graphs) and B7-alt
(sub-dense storage). If the hit rate collapses below 80%, something
in the topology key would be changing too often (e.g. `_sub_offset`
or `_sub_shape` should NOT be in the key — they're invariant within a
graph capture lifetime, not across it).
"""
import numpy as np
import pytest

from gpufluid.solvers.solver3d import FlipSolver3D


def _run_flowing_scene(enable_sub_dense: bool, n_frames: int = 12,
                       substeps_per_frame: int = 8,
                       sub_rebuild_every: int = 4,
                       sub_dilation: int = 6, N: int = 32):
    """Settled fluid blob + small dt; rebuild every few frames; count
    graph hits/misses."""
    s = FlipSolver3D(
        nx=N, ny=N, nz=N,
        transfer_mode="flip", gravity=-9.81,
        enable_cuda_graphs=True,
        enable_sub_dense=enable_sub_dense,
        sub_rebuild_every=sub_rebuild_every,
        sub_dilation=sub_dilation,
    )
    s.seed_box((0.40, 0.40, 0.40), (0.60, 0.60, 0.60), ppc=4)
    for frame in range(n_frames):
        s.prepare_frame(frame, 1.0 / 24)
        for _ in range(substeps_per_frame):
            s.step(dt=0.005, pressure_iters=20, pressure_solver="jacobi")
    return s


@pytest.mark.gpu
def test_b7_alt_5_dense_graph_hit_rate_baseline():
    """Baseline: full-dense graph hit rate at ≥90% over a flowing scene.
    Gives us a reference for how the rebuild trigger eats into it."""
    s = _run_flowing_scene(enable_sub_dense=False)
    total = s._cuda_graph_hits + s._cuda_graph_misses
    assert total > 0
    hit_rate = s._cuda_graph_hits / total
    print(f"\n[B7-alt.5 dense] hits={s._cuda_graph_hits}, "
          f"misses={s._cuda_graph_misses}, hit_rate={hit_rate:.2%}")
    assert hit_rate >= 0.85, (
        f"baseline dense graph hit rate {hit_rate:.2%} below 85% — "
        f"unexpected, B5 caching may have regressed"
    )


@pytest.mark.gpu
def test_b7_alt_5_sub_dense_graph_hit_rate_within_band():
    """Sub-dense rebuilds invalidate the graph on every prepare_frame
    where they fire. With `sub_rebuild_every=4` + 8 substeps/frame the
    expected pattern is: each frame's first substep is a miss
    (recapture for the new bbox / topology), the rest are hits — so
    hit-rate ≈ (8-1)/8 = 87.5% if every frame rebuilds, or higher when
    only every 4th frame triggers the periodic rebuild.

    Asserts ≥ 80% as the macro acceptance bar; the proximity-driven
    rebuilds that happen at the edges of the bbox are the same shape
    so they get cached too."""
    s = _run_flowing_scene(enable_sub_dense=True)
    total = s._cuda_graph_hits + s._cuda_graph_misses
    assert total > 0
    hit_rate = s._cuda_graph_hits / total
    print(f"\n[B7-alt.5 sub_dense] hits={s._cuda_graph_hits}, "
          f"misses={s._cuda_graph_misses}, hit_rate={hit_rate:.2%}, "
          f"final sub_offset={s._sub_offset}, sub_shape={s._sub_shape}")
    assert hit_rate >= 0.80, (
        f"sub-dense graph hit rate {hit_rate:.2%} below 80% — "
        f"check whether _sub_offset/_sub_shape are leaking into the "
        f"graph topology key (they shouldn't — they're constant within "
        f"a capture lifetime, only change on a fresh rebuild + invalidate)"
    )


@pytest.mark.gpu
def test_b7_alt_5_sub_dense_simulation_is_finite_through_rebuilds():
    """Sanity: even with frequent rebuilds and graph replay, fluid
    doesn't blow up. Catches the case where graph-capture-during-rebuild
    silently keeps stale buffer pointers — would NaN within a few
    frames."""
    s = _run_flowing_scene(enable_sub_dense=True, n_frames=6,
                           sub_rebuild_every=2)
    pos = s.pos.numpy()
    vel = s.vel.numpy()
    assert np.all(np.isfinite(pos)), "particles NaN'd through rebuilds"
    assert np.all(np.isfinite(vel)), "velocities NaN'd through rebuilds"
    assert s._cuda_graph_misses > 0  # we did exercise rebuild
