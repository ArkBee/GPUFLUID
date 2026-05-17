"""B3.2 — [BLK W7.8] wave-crest potential I_wc = |∇·n̂|.

Spec: DESIGN.md §11.5.1. The whole point of I_wc is to distinguish
*curving* surfaces (wave crests, breaking ridges) from *flat* surfaces
and from bulk-interior particles. These tests prove the answer is
discriminative — a test that only checks "doesn't crash" would have
missed every interesting bug.

Three numerical contrasts:

  1. Flat slab vs sharp crest — crest must score higher.
  2. Bulk-interior vs surface — interior must NOT light up (the χ̃ plateau
     interior has zero gradient and thus ill-defined n̂; the kernel must
     fall through gracefully, not produce noise spikes).
  3. Symmetry — a crest mirrored across the x-axis produces the same
     I_wc values up to the mirror.

Exercises [BLK W7.8] (GPU kernel) and [BLK W7.8.H] (host wrapper).
"""
from __future__ import annotations
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _slab(nx=20, nz=20, y=0.5, jitter=0.0, rng=None) -> np.ndarray:
    """A flat horizontal slab of particles centred at y."""
    rng = rng or np.random.default_rng(0)
    xs = np.linspace(0.1, 0.9, nx)
    zs = np.linspace(0.1, 0.9, nz)
    pts = np.array([(x, y, z) for x in xs for z in zs], dtype=np.float32)
    if jitter > 0:
        pts += rng.normal(0.0, jitter, pts.shape).astype(np.float32)
    return pts


def _hat(nx=20, nz=20, base=0.4, peak=0.7, sharpness=8.0) -> np.ndarray:
    """A slab with a sharp gaussian crest in y centred at (0.5, _, 0.5)."""
    xs = np.linspace(0.1, 0.9, nx)
    zs = np.linspace(0.1, 0.9, nz)
    pts = []
    for x in xs:
        for z in zs:
            r2 = (x - 0.5) ** 2 + (z - 0.5) ** 2
            y = base + (peak - base) * np.exp(-sharpness * r2)
            pts.append((x, y, z))
    return np.array(pts, dtype=np.float32)


# ----------------------------------------------------------- tests

def test_w7_8_crest_centre_scores_higher_than_flat_centre():
    """At the CENTRE of the domain (x≈0.5, z≈0.5), a sharp crest must
    produce a clearly higher I_wc than a flat slab.

    We compare centre particles (not max) because both shapes have the
    same finite-slab edges, which contribute their own I_wc via edge
    curvature. Subtracting edge effects by location isolates the
    "wave-crest" signal: pure flat at the centre → near-zero;
    bump at the centre → curvature shows up.
    """
    from gpufluid.sim.whitewater_potentials import wave_crest_potential

    flat = _slab()
    hat = _hat()
    iwc_flat = wave_crest_potential(flat, grid_res=32, dx=1.0 / 32, n_blur=2)
    iwc_hat = wave_crest_potential(hat, grid_res=32, dx=1.0 / 32, n_blur=2)

    assert iwc_flat.shape == (flat.shape[0],)
    assert iwc_hat.shape == (hat.shape[0],)
    assert np.all((iwc_flat >= 0.0) & (iwc_flat <= 1.0))
    assert np.all((iwc_hat >= 0.0) & (iwc_hat <= 1.0))

    # Centre particles: those within a small radius of (0.5, _, 0.5)
    # ignoring y because flat is fixed-y and hat varies in y.
    centre_xz_flat = np.hypot(flat[:, 0] - 0.5, flat[:, 2] - 0.5)
    centre_xz_hat = np.hypot(hat[:, 0] - 0.5, hat[:, 2] - 0.5)
    flat_centre_iwc = iwc_flat[centre_xz_flat < 0.15].max(initial=0.0)
    hat_centre_iwc = iwc_hat[centre_xz_hat < 0.15].max(initial=0.0)

    assert hat_centre_iwc > flat_centre_iwc * 2.0 + 0.02, (
        f"crest-centre I_wc {hat_centre_iwc:.3f} should clearly exceed "
        f"flat-centre I_wc {flat_centre_iwc:.3f} (ratio "
        f"{hat_centre_iwc / max(flat_centre_iwc, 1e-6):.2f}, target >2.0×).")


def test_w7_8_interior_particles_low_score():
    """Particles inside a dense cube must NOT spike I_wc.

    Interior cells sit at the χ̃ plateau where ∇χ̃ ≈ 0; the kernel's
    epsilon-guarded normalisation must produce a near-zero divergence
    there, not a numerical artifact. Without this guard, I_wc would
    falsely brand bulk fluid as wave crests.
    """
    from gpufluid.sim.whitewater_potentials import wave_crest_potential

    # 8x8x8 dense cube of particles in the middle of the domain
    coords = np.linspace(0.3, 0.7, 8)
    pts = np.array([(x, y, z) for x in coords for y in coords for z in coords],
                   dtype=np.float32)
    iwc = wave_crest_potential(pts, grid_res=32, dx=1.0 / 32, n_blur=2)

    # Distance from cube centre — particles within the inner 50% by
    # radius are "interior"; their I_wc should be near zero.
    centre = pts.mean(axis=0)
    r = np.linalg.norm(pts - centre, axis=1)
    inner = r < r.max() * 0.5
    assert inner.sum() > 0, "test setup: no inner particles"
    inner_max = float(iwc[inner].max())
    assert inner_max < 0.15, (
        f"interior particle peak I_wc = {inner_max:.3f} too high; the "
        f"χ̃ plateau is being misread as curvature. Check the ε guard "
        f"in normal normalisation.")


def test_w7_8_mirror_symmetry():
    """A crest mirrored across z-axis must produce I_wc mirrored to
    within atomic-add ordering noise. This is a pure invariance check —
    no specific magnitude target, just that the algorithm doesn't
    secretly depend on axis orientation."""
    from gpufluid.sim.whitewater_potentials import wave_crest_potential

    hat = _hat()
    hat_mirror = hat.copy()
    hat_mirror[:, 2] = 1.0 - hat_mirror[:, 2]  # flip z

    iwc_a = wave_crest_potential(hat, grid_res=32, dx=1.0 / 32, n_blur=2)
    iwc_b = wave_crest_potential(hat_mirror, grid_res=32, dx=1.0 / 32, n_blur=2)

    # Re-sort by (x, mirrored z) to align particles between the two runs.
    # hat orders as (x, z); hat_mirror orders as (x, 1-z) — same pattern
    # but z-axis reversed. Compare distributions, not per-particle:
    assert np.allclose(np.sort(iwc_a), np.sort(iwc_b), atol=1e-4), (
        "mirrored input produced a different I_wc distribution — "
        "kernel has a hidden axis bias.")
