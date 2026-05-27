"""Round-50 regression test for round-49 reviewer finding.

cmd_bench measures GPU work, not async kernel-launch latency.
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent


def test_cmd_bench_synchronizes_before_clock_read():
    """Round-50 contract (lesson §9.12): cmd_bench must call
    wp.synchronize() before reading the clock AND after the timed
    loop, so the reported steps/s reflects actual GPU work — not
    just the Python-side async dispatch.

    Pre-round-50 the warmup loop AND the timed loop both ran without
    sync; Warp kernels enqueued asynchronously and time.time()
    bracketed only the dispatch latency → reported numbers inflated
    5-50×; anyone comparing perf branches steered on noise."""
    src = (_REPO / "src" / "gpufluid" / "cli"
           / "commands.py").read_text()
    fn_pos = src.find("def cmd_bench")
    next_def = src.find("\ndef ", fn_pos + 1)
    body = src[fn_pos:next_def]
    # Both syncs must be present in the body.
    sync_count = body.count("wp.synchronize()")
    assert sync_count >= 2, (
        f"round-50 regressed: cmd_bench needs 2× wp.synchronize() "
        f"(warmup + post-timed); found {sync_count}")
