"""F3.6.C1 — [BLK G1.17] FluidEmitEvent + [BLK G1.18] FluidOutflowEvent +
[BLK G1.19] FrameEventQueue.

Spec: DESIGN.md §3.2.4.2. The queue is the new D4→F3 hook: D4 region
helpers publish per-frame events, F3 (eventually, in C2) drains them.
This file pins the queue contract BEFORE the solver-side migration —
so when C2 lands, we know the queue itself can't be the bug.

Tests cover:
  1. Empty queue invariants — drain returns empty list, doesn't crash.
  2. Round-trip: push then drain returns the same data, in order.
  3. Drain-twice: second drain is empty (queue is one-shot per cycle).
  4. Cross-event isolation: pushing an emit doesn't affect outflow drain.
  5. Dataclass shape — explicit ctor + field access guards.
  6. Parity with legacy: InflowBox.publish_for_frame produces the same
     particle (pos, vel) as the existing `apply_inflows` host helper.
  7. Parity with legacy: OutflowBox.publish_for_frame yields a cull
     bbox that, when applied, gives the same survivors as `apply_outflows`.
"""
from __future__ import annotations
import numpy as np
import pytest


def test_g1_17_18_19_imports_resolve():
    """The three new blocks live in primitives/frame_events.py."""
    from gpufluid.primitives.frame_events import (
        FluidEmitEvent, FluidOutflowEvent, FrameEventQueue,
    )
    assert FluidEmitEvent is not None
    assert FluidOutflowEvent is not None
    assert FrameEventQueue is not None


def test_g1_19_empty_queue_invariants():
    """Fresh queue: drain returns empty lists, no exceptions."""
    from gpufluid.primitives.frame_events import FrameEventQueue
    q = FrameEventQueue()
    assert q.drain_emits() == []
    assert q.drain_outflows() == []
    # Draining again is still safe.
    assert q.drain_emits() == []
    assert q.drain_outflows() == []


def test_g1_19_push_emit_round_trip():
    """Push one emit event, drain it, get same arrays back."""
    from gpufluid.primitives.frame_events import (
        FluidEmitEvent, FrameEventQueue,
    )
    q = FrameEventQueue()
    pos = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    vel = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    q.push_emit(FluidEmitEvent(positions=pos, velocities=vel))
    drained = q.drain_emits()
    assert len(drained) == 1
    assert np.array_equal(drained[0].positions, pos)
    assert np.array_equal(drained[0].velocities, vel)


def test_g1_19_drain_is_one_shot():
    """After drain, the queue is empty until the next push."""
    from gpufluid.primitives.frame_events import (
        FluidEmitEvent, FrameEventQueue,
    )
    q = FrameEventQueue()
    q.push_emit(FluidEmitEvent(positions=np.zeros((3, 3), np.float32),
                                velocities=np.zeros((3, 3), np.float32)))
    assert len(q.drain_emits()) == 1
    assert q.drain_emits() == [], "second drain must return empty"


def test_g1_19_emit_and_outflow_are_independent():
    """Pushing an emit doesn't put anything in the outflow channel and
    vice versa. Cross-channel isolation prevents accidental ordering bugs."""
    from gpufluid.primitives.frame_events import (
        FluidEmitEvent, FluidOutflowEvent, FrameEventQueue,
    )
    q = FrameEventQueue()
    q.push_emit(FluidEmitEvent(positions=np.zeros((1, 3), np.float32),
                                velocities=np.zeros((1, 3), np.float32)))
    q.push_outflow(FluidOutflowEvent(
        lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0), inside=True))
    # Drain outflows first — should NOT touch the emit channel.
    outs = q.drain_outflows()
    assert len(outs) == 1
    emits = q.drain_emits()
    assert len(emits) == 1


def test_g1_19_push_order_preserved():
    """Multiple emits in the same cycle drain in push order."""
    from gpufluid.primitives.frame_events import (
        FluidEmitEvent, FrameEventQueue,
    )
    q = FrameEventQueue()
    for i in range(3):
        p = np.full((1, 3), float(i), dtype=np.float32)
        q.push_emit(FluidEmitEvent(positions=p, velocities=p))
    drained = q.drain_emits()
    assert len(drained) == 3
    for i, ev in enumerate(drained):
        assert ev.positions[0, 0] == float(i), f"order broken at {i}"


def test_g1_17_emit_event_dataclass_shape():
    """Explicit dataclass field access — guards against accidental
    field rename downstream."""
    from gpufluid.primitives.frame_events import FluidEmitEvent
    pos = np.zeros((2, 3), np.float32)
    vel = np.zeros((2, 3), np.float32)
    ev = FluidEmitEvent(positions=pos, velocities=vel)
    assert ev.positions is pos
    assert ev.velocities is vel


def test_g1_18_outflow_event_dataclass_shape():
    from gpufluid.primitives.frame_events import FluidOutflowEvent
    ev = FluidOutflowEvent(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0), inside=True)
    assert ev.lo == (0.0, 0.0, 0.0)
    assert ev.hi == (1.0, 1.0, 1.0)
    assert ev.inside is True


# --------------------------------------------------------- parity with legacy

def test_inflow_publish_matches_apply_inflows():
    """InflowBox.publish_for_frame + drain must produce the same (pos, vel)
    as the legacy `apply_inflows` host helper given the same inputs.

    This is the safety net for F3.6.C2 — when the solver switches from
    legacy pull to queue drain, this test guarantees byte-equivalent
    fluid emission (modulo RNG state — both paths take the same rng,
    same seed, in the same order).
    """
    from gpufluid.domain.regions import InflowBox, apply_inflows
    from gpufluid.primitives.frame_events import FrameEventQueue

    inflows = [
        InflowBox(lo=(0.1, 0.1, 0.1), hi=(0.3, 0.3, 0.3),
                  velocity=(1.0, 0.0, 0.0), rate_per_sec=1000.0),
        InflowBox(lo=(0.5, 0.5, 0.5), hi=(0.7, 0.7, 0.7),
                  velocity=(0.0, -1.0, 0.0), rate_per_sec=500.0,
                  frame_start=2),
    ]
    frame_idx = 5
    frame_dt = 1.0 / 24.0

    # Legacy path
    rng_a = np.random.default_rng(42)
    legacy_pos, legacy_vel = apply_inflows(inflows, frame_idx, frame_dt, rng_a)

    # New event path — same rng seed, same call order
    rng_b = np.random.default_rng(42)
    q = FrameEventQueue()
    for inf in inflows:
        inf.publish_for_frame(q, frame_idx, frame_dt, rng_b)
    drained = q.drain_emits()
    if drained:
        new_pos = np.concatenate([e.positions for e in drained], axis=0)
        new_vel = np.concatenate([e.velocities for e in drained], axis=0)
    else:
        new_pos = np.zeros((0, 3), np.float32)
        new_vel = np.zeros((0, 3), np.float32)

    assert np.array_equal(legacy_pos, new_pos), \
        f"position emit differs:\nlegacy:\n{legacy_pos}\nnew:\n{new_pos}"
    assert np.array_equal(legacy_vel, new_vel), \
        f"velocity emit differs"


def test_outflow_publish_matches_apply_outflows():
    """OutflowBox.publish_for_frame + drain + apply must give the same
    surviving particles as the legacy `apply_outflows`."""
    from gpufluid.domain.regions import OutflowBox, apply_outflows
    from gpufluid.primitives.frame_events import FrameEventQueue

    rng = np.random.default_rng(0)
    pos = rng.random((20, 3)).astype(np.float32)
    vel = rng.random((20, 3)).astype(np.float32)
    outflows = [
        OutflowBox(lo=(0.0, 0.0, 0.0), hi=(0.5, 0.5, 0.5)),
        OutflowBox(lo=(0.8, 0.8, 0.8), hi=(1.0, 1.0, 1.0)),
    ]
    frame_idx = 0

    # Legacy
    legacy_pos, legacy_vel = apply_outflows(pos, vel, outflows, frame_idx)

    # Event path: publish cull events, then apply them manually using the
    # documented semantics (inside=True means "delete particles inside this box").
    q = FrameEventQueue()
    for of in outflows:
        of.publish_for_frame(q, frame_idx)
    cull_events = q.drain_outflows()
    keep = np.ones(len(pos), dtype=bool)
    for ev in cull_events:
        lo = np.asarray(ev.lo, dtype=np.float32)
        hi = np.asarray(ev.hi, dtype=np.float32)
        inside = np.all((pos >= lo) & (pos <= hi), axis=1)
        if ev.inside:
            keep &= ~inside
        else:
            keep &= inside
    new_pos = pos[keep]; new_vel = vel[keep]

    assert np.array_equal(legacy_pos, new_pos)
    assert np.array_equal(legacy_vel, new_vel)
