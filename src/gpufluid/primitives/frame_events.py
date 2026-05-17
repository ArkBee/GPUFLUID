"""[Layer G1] Per-frame event queue for D4 -> F3 communication.

Spec: DESIGN.md §3.2.4.2 (F3.6.C1 phase).

The queue is the architectural fix for the F3<->D4 layer inversion.
Today the solver (F3) pulls per-frame work from `domain/regions.py`
(D4) — that's the wrong direction by the §2 layer rule. The fix is
event-driven: D4 region helpers PUBLISH `FluidEmitEvent` /
`FluidOutflowEvent` instances into a queue at `prepare_frame` start,
and the solver DRAINS the queue. F3 stops importing from D4.

This module ships the queue + event dataclasses but does NOT yet
change the solver (that's F3.6.C2). Until then the queue runs
side-by-side with the legacy `apply_inflows` / `apply_outflows` host
helpers, so callers can adopt at their own pace and we can land C1
without behavioural change.

Block IDs:

  G1.17  FluidEmitEvent       — one inflow emission for the current frame
  G1.18  FluidOutflowEvent    — one cull request for the current frame
  G1.19  FrameEventQueue      — push/drain sink, one-shot per frame

Lifecycle (designed but not yet enforced in solver code, see C2):

  prepare_frame_start:
      queue.clear()
      for inflow in inflows:  inflow.publish_for_frame(queue, ...)
      for outflow in outflows: outflow.publish_for_frame(queue, ...)
      emits   = queue.drain_emits()
      outs    = queue.drain_outflows()
      # apply emits + outs to particle arrays
  step loop:
      # queue is empty for the entire substep loop, so CUDA-graph
      # capture sees no per-frame variation here

Why a queue and not a callback list: the solver wants a known-empty
state during graph capture. A queue with explicit drain guarantees
"no D4 work in flight" inside the capture window; callbacks would
need additional plumbing to enforce that.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..blocks import block


@dataclass
class FluidEmitEvent:
    """One inflow's worth of new fluid for the current frame.

    Arrays are host-side numpy (the existing `apply_inflows` contract).
    The solver uploads them to Warp on drain.
    """
    positions: np.ndarray   # (N, 3) float32, sim-space coordinates
    velocities: np.ndarray  # (N, 3) float32, world velocity per particle


block("G1.17", "FluidEmitEvent — per-frame inflow emission record")(FluidEmitEvent)


@dataclass
class FluidOutflowEvent:
    """One cull request for the current frame.

    Describes a bounding box; the solver applies it to its current
    particle pool. `inside=True` means delete particles INSIDE the
    bbox (the OutflowBox default); `inside=False` means delete those
    OUTSIDE (useful for hard domain clipping).
    """
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    inside: bool = True


block("G1.18", "FluidOutflowEvent — per-frame cull request record")(FluidOutflowEvent)


@block("G1.19", "FrameEventQueue — one-shot push/drain sink for D4 -> F3 events")
class FrameEventQueue:
    """Per-frame event sink + drainer.

    Populated by D4 region helpers (`InflowBox.publish_for_frame`,
    `OutflowBox.publish_for_frame`) at `prepare_frame` start; drained
    by F3 inside the same call. Emptied between frames — events do
    not persist.

    Channels are independent: pushing an emit doesn't show up in
    outflow drain, and vice versa. Within a channel, drain returns
    events in push order.
    """

    def __init__(self):
        self._emits: List[FluidEmitEvent] = []
        self._outflows: List[FluidOutflowEvent] = []

    # ------------------------------------------------------------ push
    def push_emit(self, ev: FluidEmitEvent) -> None:
        self._emits.append(ev)

    def push_outflow(self, ev: FluidOutflowEvent) -> None:
        self._outflows.append(ev)

    # ----------------------------------------------------------- drain
    def drain_emits(self) -> List[FluidEmitEvent]:
        out = self._emits
        self._emits = []
        return out

    def drain_outflows(self) -> List[FluidOutflowEvent]:
        out = self._outflows
        self._outflows = []
        return out

    # ----------------------------------------------------------- utility
    def clear(self) -> None:
        """Drop all queued events without returning them. Used at the top
        of `prepare_frame` to guarantee a clean slate even if the previous
        frame errored mid-drain."""
        self._emits = []
        self._outflows = []
