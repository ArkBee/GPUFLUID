"""B3.1 — W7.7 trapped-air potential (Ihmsen 2012 §3.1) on GPU.

The whole point of the potential is to *distinguish* divergent flow from
laminar flow. Tests prove the answer changes between the two regimes; a
test that only checks "runs without crashing" would have missed the bugs
flagged in HANDOFF.md §2.3 ("tests prove the feature changes the answer").
"""
import numpy as np
import pytest

from gpufluid.sim.whitewater_potentials import trapped_air_potential


RADIUS = 0.05  # neighbour radius for all tests below (sim space is [0, 1]³ish)


def test_w7_7_empty_input_returns_empty_array():
    out = trapped_air_potential(np.empty((0, 3), dtype=np.float32),
                                 np.empty((0, 3), dtype=np.float32),
                                 radius=RADIUS)
    assert out.shape == (0,)


def test_w7_7_laminar_flow_potential_is_zero():
    """Uniform velocity field → cos θ = 1 for every pair → potential is 0."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.2, 0.8, size=(500, 3)).astype(np.float32)
    vel = np.tile(np.array([0.0, -5.0, 0.0], dtype=np.float32), (500, 1))
    I = trapped_air_potential(pos, vel, radius=RADIUS, v_max=10.0)
    assert I.shape == (500,)
    # All identical velocities → ang_factor (1 - cos θ) is exactly 0.
    assert np.allclose(I, 0.0, atol=1e-6), \
        f"laminar flow should yield zero potential, got max={I.max():.4g}"


def test_w7_7_diverging_flow_has_positive_potential():
    """Two anti-parallel groups in close contact → potential >> 0 there."""
    # Group A: 200 particles in slab x ∈ [0.45, 0.50], velocity +x
    # Group B: 200 particles in slab x ∈ [0.50, 0.55], velocity -x
    # Their neighbourhoods overlap across x=0.5 -> anti-parallel pairs.
    rng = np.random.default_rng(1)
    n_per = 200
    posA = rng.uniform(low=[0.46, 0.45, 0.45], high=[0.50, 0.55, 0.55],
                       size=(n_per, 3)).astype(np.float32)
    posB = rng.uniform(low=[0.50, 0.45, 0.45], high=[0.54, 0.55, 0.55],
                       size=(n_per, 3)).astype(np.float32)
    velA = np.tile(np.array([+3.0, 0.0, 0.0], dtype=np.float32), (n_per, 1))
    velB = np.tile(np.array([-3.0, 0.0, 0.0], dtype=np.float32), (n_per, 1))
    pos = np.concatenate([posA, posB], axis=0)
    vel = np.concatenate([velA, velB], axis=0)
    # |v_ij| between A and B = 6.0; v_max = 10.0 → vel_factor = 1 - 0.6 = 0.4 > 0.
    I = trapped_air_potential(pos, vel, radius=RADIUS, v_max=10.0)
    assert I.shape == (2 * n_per,)
    # Near the contact slab (x in [0.48, 0.52]) potential should be much larger
    # than at the outer edges.
    contact_mask = (pos[:, 0] > 0.48) & (pos[:, 0] < 0.52)
    contact_I = I[contact_mask].mean()
    assert contact_I > 0.05, f"contact zone potential too low: {contact_I:.4g}"
    # Compare against the laminar reference (zero) — clear separation.
    assert contact_I > 100 * 1e-6


def test_w7_7_potential_clamped_to_unit_interval():
    """Per spec: capped at 1. Stuff a particle with many anti-parallel
    neighbours and verify the output value never exceeds 1.0."""
    rng = np.random.default_rng(2)
    # 5000 particles tightly packed in a small ball, half +x half -x velocity.
    n = 5000
    pos = (rng.uniform(-1, 1, size=(n, 3)) * 0.02 + 0.5).astype(np.float32)
    sign = np.where(rng.uniform(size=n) > 0.5, 1.0, -1.0).astype(np.float32)
    vel = np.zeros((n, 3), dtype=np.float32); vel[:, 0] = sign * 3.0
    I = trapped_air_potential(pos, vel, radius=RADIUS, v_max=10.0)
    assert I.max() <= 1.0 + 1e-6, f"potential exceeded cap: max={I.max():.4g}"
    assert I.min() >= 0.0


def test_w7_7_v_max_scales_pair_contribution():
    """vel_factor = clamp(1 - |v_ij|/v_max, 0, 1). With a single anti-parallel
    pair (|v_ij|=6) v_max=10 gives 0.4 contribution; v_max=1000 gives ~0.994.
    Uses just 2 particles so the per-particle sum cap of 1 doesn't mask the
    effect."""
    pos = np.array([[0.50, 0.50, 0.50],
                    [0.51, 0.50, 0.50]], dtype=np.float32)
    vel = np.array([[+3.0, 0.0, 0.0],
                    [-3.0, 0.0, 0.0]], dtype=np.float32)
    # ang_factor for anti-parallel = (1 - (-1)) = 2.
    I_small = trapped_air_potential(pos, vel, radius=RADIUS, v_max=10.0)
    I_large = trapped_air_potential(pos, vel, radius=RADIUS, v_max=1000.0)
    # Each pair contribution = 2 * vel_factor. With v_max=10, |v_ij|=6 →
    # vel_factor=0.4 → contribution=0.8 (not yet capped). With v_max=1000,
    # contribution ~2 but capped at 1.0 per spec.
    assert I_small[0] == pytest.approx(0.8, abs=1e-3)
    assert I_large[0] == pytest.approx(1.0, abs=1e-3)
    # By symmetry both particles see one neighbour with the same factors.
    assert I_small[1] == pytest.approx(I_small[0])


# ---- B3.3 — emit_from_fluid weights by potential ---------------------------

def test_w7_7_potential_weighted_emit_prefers_turbulent_particles():
    """Build a scene with two regions of equally fast particles, but only one
    is turbulent (anti-parallel pairs → high I_ta). The legacy selector
    treats both regions identically (speed gate alone); the potential-weighted
    selector concentrates emissions on the turbulent region.
    """
    from gpufluid.sim.whitewater import WhitewaterSystem, WhitewaterConfig

    rng = np.random.default_rng(42)
    n_per = 100
    # Turbulent slab: anti-parallel pairs around x=0.2
    posT_a = rng.uniform([0.18, 0.45, 0.45], [0.20, 0.55, 0.55],
                         size=(n_per, 3)).astype(np.float32)
    posT_b = rng.uniform([0.20, 0.45, 0.45], [0.22, 0.55, 0.55],
                         size=(n_per, 3)).astype(np.float32)
    velT_a = np.tile([+5.0, 0.0, 0.0], (n_per, 1)).astype(np.float32)
    velT_b = np.tile([-5.0, 0.0, 0.0], (n_per, 1)).astype(np.float32)
    # Laminar slab: all moving the same way, around x=0.8
    posL = rng.uniform([0.78, 0.45, 0.45], [0.82, 0.55, 0.55],
                       size=(2 * n_per, 3)).astype(np.float32)
    velL = np.tile([5.0, 0.0, 0.0], (2 * n_per, 1)).astype(np.float32)

    pos = np.concatenate([posT_a, posT_b, posL])
    vel = np.concatenate([velT_a, velT_b, velL])
    # Both regions are above any reasonable speed threshold.

    I = trapped_air_potential(pos, vel, radius=0.05, v_max=20.0)
    # Sanity: turbulent slab has higher potential than laminar.
    turb_mean = I[:2 * n_per].mean()
    lam_mean = I[2 * n_per:].mean()
    assert turb_mean > 10 * (lam_mean + 1e-9), \
        f"turb={turb_mean:.4g} should dominate lam={lam_mean:.4g}"

    cfg = WhitewaterConfig(speed_threshold=1.0, emit_per_frame_max=200, total_cap=10000)
    # Legacy selector: uniform random over both regions.
    ww_legacy = WhitewaterSystem(cfg, seed=1)
    ww_legacy.emit_from_fluid(pos, vel)
    # Potential-weighted selector.
    ww_pot = WhitewaterSystem(cfg, seed=1)
    ww_pot.emit_from_fluid(pos, vel, potential=I)

    # Fraction of emitted particles in the turbulent slab (x < 0.4).
    legacy_turb_frac = (ww_legacy.pos[:, 0] < 0.4).mean()
    pot_turb_frac    = (ww_pot.pos[:, 0] < 0.4).mean()
    # Population split is 2/4 = 50% turbulent → legacy should be near 0.5.
    # Potential-weighted should be >> 0.5 (close to 1.0).
    assert 0.3 < legacy_turb_frac < 0.7, \
        f"legacy selector should split ~50/50, got {legacy_turb_frac:.3f}"
    assert pot_turb_frac > 0.9, \
        f"potential-weighted selector should concentrate >90% in turbulent " \
        f"region, got {pot_turb_frac:.3f}"
    # The 5× ratio benchmark from BACKLOG B3.4, in proxy form: at least
    # this much more emission in the turbulent region than the legacy path.
    assert pot_turb_frac / legacy_turb_frac >= 1.8, \
        f"potential-weighted should beat legacy by ≥1.8x for jet scene: "\
        f"{pot_turb_frac:.3f} / {legacy_turb_frac:.3f}"
