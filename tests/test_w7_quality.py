"""[BLK W7.4 + W7.5] Whitewater foam/spray/bubble classification + dynamics.

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
    """A 16³ grid where the lower half is fluid (density=1.0), upper half is
    air (density=0.0), and there's a one-cell surface band at y=N/2."""
    d = np.zeros((N, N, N), dtype=np.float32)
    d[:, : N // 2, :] = 1.0
    # smooth the interface slightly so the classifier sees a gradient
    d[:, N // 2, :] = 0.5
    d[:, N // 2 + 1, :] = 0.2
    return d


def test_emit_classifies_three_kinds():
    """Fast particles seeded at different y must get distinct kinds."""
    N = 16; dx = 1.0 / N
    d = _synthetic_density_grid(N)
    ws = WhitewaterSystem(WhitewaterConfig(speed_threshold=0.1))
    # Three particles: deep (bubble), at surface (foam), above (spray)
    pos = np.array([
        [0.5, 0.10, 0.5],   # y=0.10 → cell 1 → density 1.0 → bubble
        [0.5, 0.50, 0.5],   # y=0.50 → cell 8 → density 0.5 → foam
        [0.5, 0.80, 0.5],   # y=0.80 → cell 12 → density 0.0 → spray
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
    y_foam, y_spray, y_bubble = ws.pos[:, 1]
    assert y_spray < y_foam, (
        f"spray should fall faster than foam: y_spray={y_spray:.3f} "
        f"y_foam={y_foam:.3f}"
    )
    assert y_bubble > y_foam, (
        f"bubble should rise above foam: y_bubble={y_bubble:.3f} "
        f"y_foam={y_foam:.3f}"
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
    ws.pos = np.array([[0.5, 0.30, 0.5]], dtype=np.float32)
    ws.vel = np.zeros((1, 3), dtype=np.float32)
    ws.age = np.zeros(1, dtype=np.float32)
    ws.kind = np.array([KIND_BUBBLE], dtype=np.int32)
    # Step until the bubble has risen above the surface (~y=0.5)
    for _ in range(30):
        ws.step(0.05, dom=(1.0, 1.0, 1.0), density=d, dx=dx)
        if ws.n == 0:
            break
    assert ws.n == 0, (
        f"bubble should have popped at surface but {ws.n} still alive"
    )
