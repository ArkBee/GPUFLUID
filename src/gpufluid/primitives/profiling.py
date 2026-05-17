"""[BLK G1.9 / B12] Per-section timing instrumentation for the step pipeline.

Wraps `warp.ScopedTimer` to (a) be a no-op when disabled and (b) aggregate
section totals across the whole bake. Users opt in via
``FlipSolver3D(enable_timing=True)`` (or ``gpufluid simulate --timings``).

The contract is intentionally tiny so future kernels can be wrapped by a
caller without import circularity:

    with solver._prof.section("pressure"):
        wp.launch(...)

Then at the end of the bake, ``solver._prof.summarize()`` returns a
``{name: total_seconds}`` dict that the CLI prints and optionally embeds
in ``cache.json``. Sections that never fire (e.g. ``viscosity`` when
``viscosity == 0``) are absent from the dict.
"""
from __future__ import annotations
from contextlib import nullcontext
import warp as wp

from ..blocks import block


@block("G1.9", "Step profiler: per-section timing aggregator over wp.ScopedTimer")
class StepProfiler:
    """Cheap wrapper around `warp.ScopedTimer`.

    ``enabled=False`` (the default) returns ``contextlib.nullcontext`` so
    instrumented code paths pay nothing in the off-state — important because
    ``synchronize=True`` would otherwise block on every section.

    When enabled, each section uses ``synchronize=True`` so GPU work
    finishes before the timer stops. Per-call durations get appended to a
    shared dict keyed by section name; ``summarize()`` reduces them.
    """

    def __init__(self, enabled: bool = False, device=None):
        self.enabled = bool(enabled)
        self.device = device
        self._dict: dict[str, list[float]] = {}

    def section(self, name: str):
        if not self.enabled:
            return nullcontext()
        # `synchronize=True` would call cudaStreamSynchronize inside a CUDA
        # graph capture, which CUDA rejects with "operation not permitted
        # while stream is capturing". Per-section wall-clock under graph
        # replay is meaningless anyway (only the whole capture's time is),
        # so no-op the section instead of crashing.
        if self.device is not None:
            try:
                if self.device.is_capturing:
                    return nullcontext()
            except AttributeError:
                pass
        return wp.ScopedTimer(name, dict=self._dict, synchronize=True,
                              print=False, active=True)

    def summarize(self) -> dict[str, float]:
        """Sum the per-call timings into per-section totals (milliseconds).

        Returns an empty dict when the profiler was never enabled. The unit
        is milliseconds because that's what `wp.ScopedTimer` reports — keep
        it consistent so the CLI doesn't have to know about Warp internals.
        """
        return {k: float(sum(v)) for k, v in self._dict.items()}

    def reset(self) -> None:
        self._dict.clear()

    @property
    def call_counts(self) -> dict[str, int]:
        """How many times each section fired — handy to spot accidental
        per-iter wrapping that explodes the call count."""
        return {k: len(v) for k, v in self._dict.items()}
