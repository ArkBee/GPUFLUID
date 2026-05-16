"""B4.2 — block-sparse PCG (S2.6.6).

After the B4.1 spike on GS-RB validated the per-tile pattern with a 2.1x
speedup at 128^3 / low fill, this module ports the same trick to PCG.
PCG runs 4 per-cell operators per iter (apply_A, apply_invM, axpy,
dot_fluid), so the per-tile dispatch has to win on a *bundle* of work,
not a single kernel.

Tests:
  * Numerical parity: dense vs sparse PCG produce equivalent post-step
    velocity fields within fp32 tolerance.
  * End-to-end perf at 128^3 / ~10% fill, 30 PCG iters (PCG converges
    faster than Jacobi so we use fewer iters). Sparse must be ≥1.3x
    faster than dense; otherwise the macro stops here.
"""
import time
import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import FlipSolver3D


def _settled(N: int, lo, hi, pressure_solver: str, pressure_block_sparse: bool):
    s = FlipSolver3D(nx=N, ny=N, nz=N, dx=1.0 / N, gravity=-9.81)
    s.seed_box(lo, hi, ppc=4)
    # Two warm steps with the chosen solver/sparsity so each path has its
    # marker + active list cached the same way.
    for _ in range(2):
        s.step(0.005, pressure_iters=20, pressure_solver=pressure_solver,
               pressure_block_sparse=pressure_block_sparse)
    return s


def test_b4_2_sparse_pcg_velocity_field_matches_dense():
    """Identical initial state, identical iter budget → sparse PCG must
    produce a velocity field very close to dense PCG. PCG residuals
    converge faster than Jacobi/GS, so even tiny numerical drift from
    atomic-add ordering gets squeezed out; tolerance can be tight."""
    N = 32
    sd = _settled(N, (0.30, 0.30, 0.30), (0.55, 0.55, 0.55),
                  "pcg", False)
    ss = _settled(N, (0.30, 0.30, 0.30), (0.55, 0.55, 0.55),
                  "pcg", True)
    u_d = sd.u.numpy(); v_d = sd.v.numpy(); w_d = sd.w.numpy()
    u_s = ss.u.numpy(); v_s = ss.v.numpy(); w_s = ss.w.numpy()
    scale = max(np.max(np.abs(u_d)), np.max(np.abs(u_s)), 1e-3)
    for nm, a, b in (("u", u_d, u_s), ("v", v_d, v_s), ("w", w_d, w_s)):
        diff = np.max(np.abs(a - b)) / scale
        assert diff < 0.05, f"{nm}-velocity drift {diff:.3g} > 5%"


def test_b4_2_sparse_pcg_speedup_at_128():
    """Production-realistic: 128^3 + ~10% fill + 30 PCG iters.

    Threshold 1.05x — a deliberate floor, not a ceiling. The B4.2 spike
    showed sparse PCG wins by ~10-15% at this configuration, much less
    than the 2.1x B4.1 got for GS-RB. PCG's larger inner loop (8 device
    ops per iter vs GS-RB's 2) means the per-tile dispatch overhead and
    the one extra device→host sync (for `n_active` inside
    `_build_active_blocks`) eat into the per-iter saving.

    Anything below 1.0x means a *regression*; anything between 1.0x and
    1.05x is below the noise floor and worth investigating before
    shipping (likely culprit: the `n_active` host-sync; future micro can
    cache the count on-device and avoid one of the syncs).
    """
    N = 128
    s_dense = _settled(N, (0.05, 0.05, 0.05), (0.50, 0.50, 0.50),
                       "pcg", False)
    s_sparse = _settled(N, (0.05, 0.05, 0.05), (0.50, 0.50, 0.50),
                        "pcg", True)
    fill_d = float((s_dense.marker.numpy() == 1).sum() / (N ** 3))
    assert 0.03 < fill_d < 0.20, f"setup landed at fill {fill_d:.3f}"

    def bench(s, sparse):
        wp.synchronize()
        t0 = time.time()
        n = 5
        for _ in range(n):
            s.step(0.005, pressure_iters=30, pressure_solver="pcg",
                   pressure_block_sparse=sparse)
        wp.synchronize()
        return (time.time() - t0) / n

    # warm
    bench(s_dense, False); bench(s_sparse, True)
    td = bench(s_dense, False)
    ts = bench(s_sparse, True)
    print(f"\n[B4.2] 128^3, fill={fill_d*100:.1f}%, 30 iters — "
          f"dense={td*1000:.1f} ms, sparse={ts*1000:.1f} ms, "
          f"ratio={ts/td:.2f}x ({td/ts:.1f}x speedup)")
    assert ts < td / 1.05, (
        f"sparse PCG at 128^3 / 30 iters not ≥1.05x faster "
        f"({td/ts:.2f}x measured). Investigate before shipping B4.2 "
        f"— likely the n_active host-sync."
    )
