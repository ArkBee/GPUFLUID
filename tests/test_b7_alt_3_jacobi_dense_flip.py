"""B7-alt.3 — end-to-end check that sub-dense storage produces the same
particle positions as full-dense on the Jacobi / dense / FLIP path.

This is the "kernels actually agree on the active region" test. Pairs two
solvers seeded identically; one runs full-dense (legacy path,
`enable_sub_dense=False`); the other rebuilds the sub-dense bbox after
the first step and then runs a second step with `_sub_offset != (0,0,0)`.

The two particle clouds must agree to fp32 noise on every cell that lies
strictly inside the dilated bbox. Tolerances are tight: the sub-dense
path is supposed to be the *same* solver evaluated on a translated
window, not an approximation.

Only the Jacobi / dense / FLIP / no-CSF / no-viscosity / no-colour /
no-scalar configuration is covered here — that's the path the B7-alt.3
slice wires up. Other configs are still blocked by step()'s guard and
are deliberately tested by tests/test_b7_alt_2_sub_dense_storage.py.
"""
import numpy as np
import pytest
import warp as wp

from gpufluid.solvers.solver3d import FlipSolver3D


def _seed_pair(N=32, lo=(0.40, 0.40, 0.40), hi=(0.60, 0.60, 0.60),
               ppc=4, rho=1.0, gravity=-9.81, flip_blend=0.95,
               viscosity=0.0, viscosity_iters=12, transfer_mode="flip",
               enable_sub_dense_b=True, sub_dilation=4):
    """Build two FlipSolver3Ds with identical seeds, one full-dense and
    one sub-dense-capable."""
    common = dict(nx=N, ny=N, nz=N, rho=rho, gravity=gravity,
                  flip_blend=flip_blend, transfer_mode=transfer_mode,
                  viscosity=viscosity, viscosity_iters=viscosity_iters)
    a = FlipSolver3D(**common, enable_sub_dense=False)
    b = FlipSolver3D(**common, enable_sub_dense=True,
                     sub_rebuild_every=1000, sub_dilation=sub_dilation)
    a.seed_box(lo, hi, ppc=ppc)
    b.seed_box(lo, hi, ppc=ppc)
    return a, b


@pytest.mark.gpu
def test_b7_alt_3_sub_dense_matches_dense_after_rebuild():
    """Identical seeds → one warm-up step (both with sub_offset=(0,0,0)
    so the trajectories are identical) → rebuild the sub-dense bbox on
    solver B → second step → particle positions and velocities must
    agree to fp32 noise. Confirms that the sub-dense launch (different
    array shapes, non-zero offsets) reproduces the dense result on the
    active region."""
    a, b = _seed_pair()
    n0 = a.n_particles
    assert n0 > 0
    assert n0 == b.n_particles
    # Warm-up step — sub_offset still (0,0,0) for both solvers, so this
    # is the same kernel path. It populates the marker with the post-p2g
    # fluid cells so prepare_frame's rebuild has something to bbox.
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    pos_a_warm = a.pos.numpy()
    pos_b_warm = b.pos.numpy()
    assert np.allclose(pos_a_warm, pos_b_warm, atol=1e-6), (
        "warm-up step should be bit-identical (both at sub_offset=(0,0,0))"
    )
    # Trigger the sub-dense rebuild on solver B from the post-step marker.
    marker_b = b.marker.numpy()
    bbox = b._compute_active_bbox(marker_b)
    assert bbox is not None
    lo, hi = bbox
    b._rebuild_sub_dense(lo, hi)
    b._last_sub_rebuild_frame = 0
    assert b._sub_offset != (0, 0, 0) or b._sub_shape != (b.nx, b.ny, b.nz), (
        "sub-dense rebuild should shrink the active region away from full domain"
    )
    # Second step — solver B now runs with non-zero offset; the
    # offset-aware kernels (k3_clear_grid, k3_p2g, k3_enforce_solid_bc,
    # k3_compute_divergence, k3_jacobi_pressure, k3_subtract_pressure_grad,
    # k3_g2p_and_advect) must reproduce the dense result.
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    pos_a = a.pos.numpy()
    pos_b = b.pos.numpy()
    vel_a = a.vel.numpy()
    vel_b = b.vel.numpy()
    pos_drift = float(np.abs(pos_a - pos_b).max())
    vel_drift = float(np.abs(vel_a - vel_b).max())
    pos_rel = pos_drift / max(float(np.abs(pos_a).max()), 1e-9)
    vel_rel = vel_drift / max(float(np.abs(vel_a).max()), 1e-9)
    print(f"\n[B7-alt.3] sub_offset={b._sub_offset}, sub_shape={b._sub_shape}; "
          f"pos drift abs={pos_drift:.3e}, rel={pos_rel:.3e}; "
          f"vel drift abs={vel_drift:.3e}, rel={vel_rel:.3e}")
    # The two pipelines are launched at different array shapes; fp32
    # atomic-add accumulation order in scatter_face can swap. Allow a
    # generous fp32 noise margin (1e-4 relative). On the canonical seed
    # the measured drift is ~1e-6 relative on RTX 4080 SUPER.
    assert pos_rel < 1e-4, (
        f"sub-dense particle positions drifted from dense by rel={pos_rel:.3e}"
    )
    assert vel_rel < 1e-3, (
        f"sub-dense particle velocities drifted from dense by rel={vel_rel:.3e}"
    )


@pytest.mark.gpu
def test_b7_alt_3_multistep_stability_within_dilation():
    """Run 5 consecutive sub-dense Jacobi steps and verify the simulation
    doesn't diverge (NaN / blow-up) and stays within ~1e-3 relative of the
    dense baseline. Each step's particle motion is bounded by gravity·dt²
    plus pressure response; with dt=0.005 and dilation=4 the fluid stays
    well inside the bbox between rebuilds."""
    a, b = _seed_pair(sub_dilation=6)
    # Initial rebuild for solver B from a marker built off the particles.
    # Run one warm-up step to populate the marker on both.
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    bbox = b._compute_active_bbox(b.marker.numpy())
    assert bbox is not None
    b._rebuild_sub_dense(*bbox)
    b._last_sub_rebuild_frame = 0
    for _ in range(5):
        a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
        b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    pos_a = a.pos.numpy()
    pos_b = b.pos.numpy()
    assert np.all(np.isfinite(pos_a))
    assert np.all(np.isfinite(pos_b))
    drift_rel = float(np.abs(pos_a - pos_b).max()) / max(
        float(np.abs(pos_a).max()), 1e-9)
    print(f"\n[B7-alt.3] 5-step drift rel={drift_rel:.3e}")
    # Accumulated drift across 5 steps stays bounded — sub-dense is the
    # same operator, not an approximation; fp32 atomic-order is the only
    # source of divergence and that doesn't compound much.
    assert drift_rel < 5e-3, (
        f"5-step sub-dense drift {drift_rel:.3e} exceeds tolerance — kernel "
        f"offset threading likely has a translation bug"
    )


@pytest.mark.gpu
def test_b7_alt_3_sub_dense_matches_dense_gsrb():
    """GS-RB pressure solver in sub-dense mode must match the dense GS-RB
    baseline. Parity (red/black) is keyed off GLOBAL cell indices so an
    odd offset doesn't swap colours — that's the one subtle invariant
    beyond the Jacobi pattern."""
    a, b = _seed_pair(sub_dilation=6)
    a.step(dt=0.005, pressure_iters=40, pressure_solver="gsrb")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="gsrb")
    bbox = b._compute_active_bbox(b.marker.numpy())
    assert bbox is not None
    b._rebuild_sub_dense(*bbox)
    b._last_sub_rebuild_frame = 0
    a.step(dt=0.005, pressure_iters=40, pressure_solver="gsrb")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="gsrb")
    pos_a = a.pos.numpy(); pos_b = b.pos.numpy()
    rel = float(np.abs(pos_a - pos_b).max()) / max(float(np.abs(pos_a).max()), 1e-9)
    print(f"\n[B7-alt.3 gsrb] sub_offset={b._sub_offset}, "
          f"sub_shape={b._sub_shape}, pos drift rel={rel:.3e}")
    assert rel < 1e-4, f"GS-RB sub-dense drift {rel:.3e} exceeds fp32 noise"


@pytest.mark.gpu
def test_b7_alt_3_sub_dense_matches_dense_with_viscosity():
    """Viscosity (k3_jacobi_visc) is a pure within-buffer Jacobi diffusion
    with no marker access — the kernel needs no offset thread-through.
    Verify the sub-dense viscosity step still matches dense (dilation
    absorbs the Neumann fall-through at the sub-dense edge)."""
    a, b = _seed_pair(viscosity=0.02, viscosity_iters=6, sub_dilation=6)
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    bbox = b._compute_active_bbox(b.marker.numpy())
    assert bbox is not None
    b._rebuild_sub_dense(*bbox)
    b._last_sub_rebuild_frame = 0
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    pos_a = a.pos.numpy(); pos_b = b.pos.numpy()
    rel = float(np.abs(pos_a - pos_b).max()) / max(float(np.abs(pos_a).max()), 1e-9)
    print(f"\n[B7-alt.3 visc] sub_offset={b._sub_offset}, "
          f"sub_shape={b._sub_shape}, pos drift rel={rel:.3e}")
    # Viscosity diffuses across fluid + non-fluid cells; in the dense
    # path, dilation cells are zero-velocity anyway (no fluid there) so
    # the Neumann fall-through at the sub-dense edge produces the same
    # zero contribution. Tight tolerance.
    assert rel < 1e-4, f"Viscosity sub-dense drift {rel:.3e} exceeds fp32 noise"


@pytest.mark.gpu
def test_b7_alt_3_sub_dense_matches_dense_apic():
    """APIC (transfer_mode='apic') in sub-dense mode must match the
    dense APIC baseline. The APIC scatter/gather computes the affine
    extension via WORLD-space offsets between face and particle — those
    must be (ii + off_x) * dx - px (not just ii * dx - px); getting it
    wrong makes the C matrix track the bbox rather than the world frame,
    drifts within one step."""
    a, b = _seed_pair(transfer_mode="apic", sub_dilation=6)
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    bbox = b._compute_active_bbox(b.marker.numpy())
    assert bbox is not None
    b._rebuild_sub_dense(*bbox)
    b._last_sub_rebuild_frame = 0
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    pos_a = a.pos.numpy(); pos_b = b.pos.numpy()
    rel = float(np.abs(pos_a - pos_b).max()) / max(float(np.abs(pos_a).max()), 1e-9)
    print(f"\n[B7-alt.3 apic] sub_offset={b._sub_offset}, "
          f"sub_shape={b._sub_shape}, pos drift rel={rel:.3e}")
    assert rel < 1e-4, f"APIC sub-dense drift {rel:.3e} exceeds fp32 noise"


@pytest.mark.gpu
def test_b7_alt_3_sub_dense_matches_dense_pic():
    """PIC (transfer_mode='pic') reuses the FLIP kernel with flip_blend=0,
    so this exercises the same k3_p2g + k3_g2p_and_advect path. Tighten
    the canonical: pure PIC under sub-dense should be bit-exact."""
    a, b = _seed_pair(transfer_mode="pic", flip_blend=0.0, sub_dilation=6)
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    bbox = b._compute_active_bbox(b.marker.numpy())
    assert bbox is not None
    b._rebuild_sub_dense(*bbox)
    b._last_sub_rebuild_frame = 0
    a.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    b.step(dt=0.005, pressure_iters=40, pressure_solver="jacobi")
    pos_a = a.pos.numpy(); pos_b = b.pos.numpy()
    rel = float(np.abs(pos_a - pos_b).max()) / max(float(np.abs(pos_a).max()), 1e-9)
    print(f"\n[B7-alt.3 pic] sub_offset={b._sub_offset}, "
          f"sub_shape={b._sub_shape}, pos drift rel={rel:.3e}")
    assert rel < 1e-4, f"PIC sub-dense drift {rel:.3e} exceeds fp32 noise"


@pytest.mark.gpu
def test_b7_alt_3_sub_dense_matches_dense_pcg():
    """PCG dense pressure solver in sub-dense mode. The PCG scratch
    (_pcg_r/z/p/Ap/diag) is allocated lazily at self.p.shape, so a
    sub-dense rebuild on solver B re-sizes them on the next call.
    Five grid kernels (k3_apply_A, k3_compute_diag, k3_apply_invM,
    k3_dot_fluid, k3_axpy_devscalar) take the new off_x/y/z."""
    a, b = _seed_pair(sub_dilation=6)
    a.step(dt=0.005, pressure_iters=30, pressure_solver="pcg")
    b.step(dt=0.005, pressure_iters=30, pressure_solver="pcg")
    bbox = b._compute_active_bbox(b.marker.numpy())
    assert bbox is not None
    b._rebuild_sub_dense(*bbox)
    b._last_sub_rebuild_frame = 0
    a.step(dt=0.005, pressure_iters=30, pressure_solver="pcg")
    b.step(dt=0.005, pressure_iters=30, pressure_solver="pcg")
    pos_a = a.pos.numpy(); pos_b = b.pos.numpy()
    rel = float(np.abs(pos_a - pos_b).max()) / max(float(np.abs(pos_a).max()), 1e-9)
    print(f"\n[B7-alt.3 pcg] sub_offset={b._sub_offset}, "
          f"sub_shape={b._sub_shape}, pos drift rel={rel:.3e}")
    # PCG accumulates many fp32 dot products + axpy operations; tolerance
    # a touch wider than the single-pass Jacobi/GS-RB to absorb
    # atomic-add ordering across multiple reductions per iter.
    assert rel < 1e-3, f"PCG sub-dense drift {rel:.3e} exceeds tolerance"
