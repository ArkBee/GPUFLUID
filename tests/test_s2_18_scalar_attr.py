"""B11 / [BLK S2.18] — per-particle scalar attribute transfer (temperature).

Exercises every kernel of the scalar-attr pipeline: [BLK S2.18.1] P2G
scatter, [BLK S2.18.2] grid normalize, [BLK S2.18.3] G2P gather.

Proves the new pipeline *changes the answer*, mirroring the rule in
HANDOFF §2.3 (tests must demonstrate the feature does something, not
just that it doesn't crash).
"""
import numpy as np
import pytest

from gpufluid.solvers.solver3d import FlipSolver3D


def test_s2_18_temperature_p2g_g2p_round_trip_smooths_gradient():
    """Seed particles with a sharp temperature step (cold half + hot half)
    that share interior cells. After one P2G→normalize→G2P round-trip,
    particles in the interface cells must take on a value *between* the
    cold and hot endpoints — that's the per-cell mean from S2.18.

    Pre-fix the temperature array doesn't exist; this test wouldn't even
    compile against the old solver, so it's the regression for the whole
    feature."""
    sol = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16.0, gravity=0.0)
    # Two boxes sharing the cells around x = 0.5 — the interior overlap is
    # where P2G mixes the two temperatures into a per-cell mean.
    sol.seed_box((0.30, 0.30, 0.30), (0.55, 0.55, 0.55),
                 ppc=8, temperature=20.0)
    sol.seed_box((0.45, 0.30, 0.30), (0.70, 0.55, 0.55),
                 ppc=8, temperature=80.0)
    assert sol.attr_temperature is not None
    t0 = sol.attr_temperature.numpy().copy()
    # Initial state: every particle is exactly 20 or 80 (no mixing yet).
    assert set(np.unique(t0).round(4).tolist()) == {20.0, 80.0}

    # Run a single step with dt=0 — but step() doesn't accept dt=0 because
    # of the inner CFL math; do dt very small so positions barely move and
    # the change comes entirely from the P2G→G2P pass.
    sol.step(dt=1e-6, pressure_iters=1)
    t1 = sol.attr_temperature.numpy()

    # Particles in the interface region (x ∈ [0.45, 0.55]) now share a cell
    # with the *other* side's particles. After normalize → G2P they take on
    # values strictly between 20 and 80.
    pos = sol.pos.numpy()
    interface = (pos[:, 0] >= 0.45) & (pos[:, 0] <= 0.55)
    assert interface.any(), "no particles in interface band — adjust the test"
    mids = t1[interface]
    # At least one interface particle must land in the open (20, 80) interval.
    assert ((mids > 20.0 + 1e-3) & (mids < 80.0 - 1e-3)).any(), (
        f"interface should mix; min={mids.min():.2f} max={mids.max():.2f}"
    )
    # Mass-conservation check: the global mean shouldn't drift more than ~5%.
    # Hot/cold counts are equal (both ppc=8, same volume), so target = 50.
    assert abs(t1.mean() - t0.mean()) < 5.0, (
        f"global mean drifted too much: {t0.mean():.2f} → {t1.mean():.2f}"
    )


def test_s2_18_uniform_temperature_field_is_invariant():
    """Constant attribute field stays constant after a round-trip.
    The kernel must not introduce drift in the trivial case."""
    sol = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16.0, gravity=0.0)
    sol.seed_box((0.30, 0.30, 0.30), (0.70, 0.70, 0.70), ppc=8, temperature=42.0)
    t0 = sol.attr_temperature.numpy().copy()
    assert np.allclose(t0, 42.0)
    sol.step(dt=1e-6, pressure_iters=1)
    t1 = sol.attr_temperature.numpy()
    # All-42 in → all-~42 out (G2P only writes when local weight > 1e-6).
    assert np.all(np.abs(t1 - 42.0) < 1e-3), \
        f"uniform field drifted: range {t1.min():.4f}..{t1.max():.4f}"


def test_s2_18_no_attr_means_no_section():
    """Solvers that never seeded a temperature must not allocate the scalar
    grids or fire the section. Costs are paid only on demand."""
    sol = FlipSolver3D(nx=16, ny=16, nz=16, dx=1.0 / 16.0,
                       gravity=0.0, enable_timing=True)
    sol.seed_box((0.3, 0.3, 0.3), (0.6, 0.6, 0.6), ppc=4)  # NO temperature
    sol.step(dt=1e-3)
    assert sol.attr_temperature is None
    assert "temperature" not in sol._prof.summarize(), \
        "the temperature section fired even though no particle had one"
