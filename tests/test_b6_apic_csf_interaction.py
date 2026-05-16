"""B6 — APIC + Surface Tension (CSF) interaction QA.

S2.12 QA proved APIC near *obstacles* is stable. This module closes the
remaining gap: APIC + CSF. APIC reconstructs velocity from a per-particle
affine matrix `C`, which interacts with the discrete CSF impulses in a
way that could in principle amplify parasitic currents.

The test mirrors `test_csf_step_cfl_centre_of_mass_does_not_drift`
(PIC + σ baseline) and asserts the equivalent tolerance under
`transfer_mode="apic"`. Anything > 2% indicates APIC reconstruction is
amplifying CSF noise — at which point B6.3 (damping `affine_C` near the
surface) would be needed.
"""
import numpy as np
import pytest

from gpufluid.solvers.solver3d import FlipSolver3D


def _apic_csf_drift(viscosity: float = 0.05) -> tuple[float, float, float]:
    """Run the canonical surftens_on.toml setup with transfer_mode='apic'
    and report (COM drift, max parasitic speed, kinetic energy ratio)."""
    solver = FlipSolver3D(
        nx=48, ny=48, nz=48, dx=1.0 / 48.0,
        gravity=0.0, viscosity=viscosity,
        surface_tension=1.0,
        csf_smoothing_passes=3,
        flip_blend=0.0,
        transfer_mode="apic",
    )
    solver.seed_box((0.3, 0.3, 0.3), (0.7, 0.7, 0.7))
    com0 = solver.pos.numpy().mean(axis=0)
    ke0 = float(0.5 * (solver.vel.numpy() ** 2).sum())

    for _ in range(60):
        solver.step_cfl(target_dt=0.04, pressure_iters=50, max_substeps=64)

    pos = solver.pos.numpy(); vel = solver.vel.numpy()
    com1 = pos.mean(axis=0)
    drift = float(np.linalg.norm(com1 - com0))
    vmax = float(np.linalg.norm(vel, axis=1).max())
    ke1 = float(0.5 * (vel ** 2).sum())
    return drift, vmax, (ke1 - ke0)


def test_b6_apic_csf_com_drift_within_tolerance():
    """APIC + CSF: same 2% COM drift tolerance as FLIP + CSF.

    If this fails, options are:
      1. Lower the tolerance and document the regression in HANDOFF.
      2. Implement B6.3 (zero `affine_C` near surface cells).
      3. Disable APIC when σ>0 and warn the user.
    """
    drift, vmax, dke = _apic_csf_drift()
    assert drift < 0.02, (
        f"APIC+CSF COM drift {drift:.4f} > 2% of domain. "
        f"max|v| at end = {vmax:.3g}, ΔKE = {dke:.3g}. "
        f"Trigger B6.3 (damp affine_C near surface) or document the regression."
    )


def test_b6_apic_csf_does_not_blow_up_kinetic_energy():
    """In a zero-G, zero-net-force scenario CSF should *contract* the cube
    (releasing surface area = decreasing potential energy) into a sphere,
    converting potential energy to kinetic energy — but viscosity should
    damp the result. We just need to verify nothing explodes: max|v| < 5x
    the capillary-wave speed estimate."""
    _, vmax, _ = _apic_csf_drift()
    # σ/ρ = 1, dx = 1/48 → capillary-wave speed ~ √(σ k / ρ) at k = 1/dx ≈ 48,
    # gives c ≈ √48 ≈ 7 m/s. Allow 5× headroom for transient peaks.
    assert vmax < 35.0, f"APIC+CSF max|v| exploded to {vmax:.2f} m/s"
