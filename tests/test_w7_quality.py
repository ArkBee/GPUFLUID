"""[BLK W7.1] [BLK W7.2] [BLK W7.4] [BLK W7.5] Whitewater state + emit + classify + dynamics.

Validates that:
  1. With a density grid supplied, emit classifies particles into all three
     classes when conditions are met (above-surface → spray, near surface →
     foam, deep → bubble).
  2. Without a density grid, emit falls back to all-foam (v0.6 back-compat).
  3. Per-class dynamics are *measurably different*: after a few steps from
     identical starts, spray drops fast, foam barely moves, bubbles rise.
  4. Bubbles pop when they leave the dense interior of the fluid.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpufluid.sim.whitewater import (
    WhitewaterConfig, WhitewaterSystem,
    KIND_FOAM, KIND_SPRAY, KIND_BUBBLE,
)


def _synthetic_density_grid(N: int = 16):
    """A 16³ grid where the lower half is fluid (density=1.0), upper half
    is air (density=0.0), and there's a one-cell surface band at z=N/2.

    Round-23: was indexed on axis 1 (Y) to match the whitewater
    Y-up bug; switched to axis 2 (Z) to match the Z-up project."""
    d = np.zeros((N, N, N), dtype=np.float32)
    d[:, :, : N // 2] = 1.0
    d[:, :, N // 2] = 0.5
    d[:, :, N // 2 + 1] = 0.2
    return d


def test_emit_classifies_three_kinds():
    """Fast particles seeded at different y must get distinct kinds.

    Note (audit-2026-06-15 r10 self-review): this is an all-classes-present
    smoke test — the three z-values happen to map to the same cells (1, 8, 12)
    under BOTH the node- and cell-centred conventions, so it does NOT guard the
    r9 #3 off-by-one. The discriminating convention guard lives in
    tests/test_audit_20260615r9.py (test_r9_3_world_to_cell_is_cell_centred).
    """
    N = 16; dx = 1.0 / N
    d = _synthetic_density_grid(N)
    ws = WhitewaterSystem(WhitewaterConfig(speed_threshold=0.1))
    # Three particles: deep (bubble), at surface (foam), above (spray).
    # Round-23: Z-up. audit-2026-06-15r9 #3: the density grid is CELL-CENTRED
    # (k_density_scatter uses i = floor(p/dx - 0.5)), so the classifier reads it
    # the same way; cell index = floor(z/dx - 0.5). The surface particle sits at
    # z=0.55 (→ cell 8, the d=0.5 band) — z=0.50 reads cell 7 (deep fluid, the
    # node-centred convention this test used to assume).
    pos = np.array([
        [0.5, 0.5, 0.10],   # z/dx=1.6 → cell floor(1.1)=1  → density 1.0 → bubble
        [0.5, 0.5, 0.55],   # z/dx=8.8 → cell floor(8.3)=8  → density 0.5 → foam
        [0.5, 0.5, 0.80],   # z/dx=12.8 → cell floor(12.3)=12 → density 0.0 → spray
    ], dtype=np.float32)
    vel = np.ones_like(pos) * np.array([1.0, 0.0, 0.0])  # all moving fast
    ws.emit_from_fluid(pos, vel, density=d, dx=dx)
    assert ws.n == 3
    kinds = set(ws.kind.tolist())
    assert KIND_BUBBLE in kinds, "deep particle should be classified as bubble"
    assert KIND_FOAM in kinds, "surface particle should be classified as foam"
    assert KIND_SPRAY in kinds, "above-surface particle should be classified as spray"


def test_emit_without_density_is_all_foam_backcompat():
    """No density → behave like v0.6 (all foam, single class)."""
    ws = WhitewaterSystem(WhitewaterConfig(speed_threshold=0.1))
    pos = np.random.RandomState(0).rand(20, 3).astype(np.float32)
    vel = np.ones((20, 3), dtype=np.float32) * 5.0
    ws.emit_from_fluid(pos, vel)
    assert ws.n == 20
    assert (ws.kind == KIND_FOAM).all()


def test_per_class_dynamics_diverge():
    """Same start state, identical zero velocity → after a short window the
    three classes must be in distinct y-bands (spray falling, bubble rising,
    foam nearly steady). Short dt + tall domain so no particle exits."""
    ws = WhitewaterSystem(WhitewaterConfig(speed_threshold=0.0))
    ws.pos = np.array([
        [0.5, 0.5, 0.5],  # foam
        [0.5, 0.5, 0.5],  # spray
        [0.5, 0.5, 0.5],  # bubble
    ], dtype=np.float32)
    ws.vel = np.zeros((3, 3), dtype=np.float32)
    ws.age = np.zeros(3, dtype=np.float32)
    ws.kind = np.array([KIND_FOAM, KIND_SPRAY, KIND_BUBBLE], dtype=np.int32)
    # Tall domain (10 m) so nothing exits during the test window
    for _ in range(20):                      # 20 * 0.01 = 0.2 s
        ws.step(0.01, dom=(10.0, 10.0, 10.0))
    assert ws.n == 3, f"all three should survive the short window; have {ws.n}"
    # Round-23: gpufluid is Z-up project-wide. Pre-round-23 these
    # assertions used `ws.pos[:, 1]` (Y) to match a bug where
    # whitewater's `step()` accidentally applied gravity to Y instead
    # of Z. Tests written against the bug were green by coincidence;
    # any addon user running the live pipeline saw whitewater drifting
    # sideways. See whitewater.py round-23 commentary for the full
    # rationale.
    z_foam, z_spray, z_bubble = ws.pos[:, 2]
    assert z_spray < z_foam, (
        f"spray should fall faster than foam: z_spray={z_spray:.3f} "
        f"z_foam={z_foam:.3f}"
    )
    assert z_bubble > z_foam, (
        f"bubble should rise above foam: z_bubble={z_bubble:.3f} "
        f"z_foam={z_foam:.3f}"
    )


def test_bubble_pops_at_surface():
    """A bubble that climbs above the dense region should be killed by the
    pop rule (cell density drops below pop_threshold)."""
    N = 16; dx = 1.0 / N
    d = _synthetic_density_grid(N)
    cfg = WhitewaterConfig(speed_threshold=0.0,
                            gravity_bubble=10.0,   # big rise to hit surface fast
                            lifetime_bubble=10.0,  # long life so only pop kills it
                            pop_threshold=0.5)
    ws = WhitewaterSystem(cfg)
    # Round-23: bubble starts deep on Z (was Y in pre-round-23 setup).
    ws.pos = np.array([[0.5, 0.5, 0.30]], dtype=np.float32)
    ws.vel = np.zeros((1, 3), dtype=np.float32)
    ws.age = np.zeros(1, dtype=np.float32)
    ws.kind = np.array([KIND_BUBBLE], dtype=np.int32)
    # Step until the bubble has risen above the surface (~z=0.5)
    for _ in range(30):
        ws.step(0.05, dom=(1.0, 1.0, 1.0), density=d, dx=dx)
        if ws.n == 0:
            break
    assert ws.n == 0, (
        f"bubble should have popped at surface but {ws.n} still alive"
    )
