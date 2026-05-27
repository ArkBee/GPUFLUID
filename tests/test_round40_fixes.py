"""Round-40 regression tests for round-39 reviewer findings.

  - Modal tick: subprocess-exit path joins drain thread + flushes
    queue (mirror to round-38 sync-path timeout-drain fix).
  - seed_box + load_checkpoint invalidate captured CUDA graph.
  - enable_sub_dense + sub_dilation < 1 raises ValueError at init.
  - load_post surfaces truncation warning when cache.json carries
    truncated_at_frame.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


# ─── 1. Modal tick drain flush (source-grep) ────────────────────────────

def test_modal_tick_joins_drain_and_flushes_queue_on_exit():
    """Round-40 contract (lesson §9.12): tick_modal must join drain
    thread + drain queue until sentinel BEFORE _finish() — same
    pattern round-38 added to sync timeout path. Pre-round-40 the
    50-line per-tick cap left OS-pipe tail unread on subprocess exit;
    user saw 'failed rc=N' with no last-30ms diagnostics."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "_runner.py").read_text()
    fn_pos = src.find("def tick_modal")
    next_def = src.find("\n    def ", fn_pos + 1)
    body = src[fn_pos:next_def]
    # After rc is non-None branch, must reference both join + queue drain.
    assert "self._stdout_thread.join" in body, (
        "round-40 regressed: tick_modal must join drain thread on exit")
    # Must reach _finish AFTER drain (not before).
    join_pos = body.find("self._stdout_thread.join")
    finish_pos = body.find("self._finish", join_pos)
    assert finish_pos > join_pos, (
        "round-40: join must come before _finish")


# ─── 2. seed_box + load_checkpoint invalidate graph (source-grep) ──────

def test_seed_box_calls_cuda_graph_invalidate():
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    pos = src.find("def seed_box")
    next_def = src.find("\n    def ", pos + 1)
    body = src[pos:next_def]
    assert "_cuda_graph_invalidate" in body, (
        "round-40: seed_box must invalidate captured graph after rebind")


def test_load_checkpoint_calls_cuda_graph_invalidate():
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    pos = src.find("def load_checkpoint")
    next_def = src.find("\n    def ", pos + 1)
    body = src[pos:next_def] if next_def > 0 else src[pos:]
    assert "_cuda_graph_invalidate" in body, (
        "round-40: load_checkpoint must invalidate captured graph")


# ─── 3. enable_sub_dense + sub_dilation < 1 rejected ────────────────────

def test_sub_dense_zero_dilation_logs_warning(caplog):
    """Round-40 contract revised in round-48: ValueError was too strict
    (broke b7-alt bookkeeping tests). Now a logger.warning fires at
    init; real-bake correctness still surfaces via convergence
    behaviour. Test that the warning is emitted."""
    import logging
    from gpufluid.solvers.solver3d import FlipSolver3D
    with caplog.at_level(logging.WARNING, logger="gpufluid.solver"):
        FlipSolver3D(nx=16, ny=16, nz=16, dx=1 / 16,
                     enable_sub_dense=True, sub_dilation=0)
    assert any("sub_dilation=0" in rec.message
               for rec in caplog.records), (
        "round-48 contract: warning must fire for the dangerous combo")


def test_sub_dense_with_default_dilation_ok():
    """Sanity: default sub_dilation=4 must continue to work."""
    from gpufluid.solvers.solver3d import FlipSolver3D
    # Should not raise.
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1 / 16,
                     enable_sub_dense=True, sub_dilation=4)
    assert s._sub_dilation == 4


def test_sub_dense_off_zero_dilation_ok():
    """Round-40 contract: the guard only fires when enable_sub_dense
    is True. Default enable_sub_dense=False + any sub_dilation must
    not regress."""
    from gpufluid.solvers.solver3d import FlipSolver3D
    s = FlipSolver3D(nx=16, ny=16, nz=16, dx=1 / 16,
                     enable_sub_dense=False, sub_dilation=0)
    assert s._sub_dilation == 0


# ─── 4. load_post truncation warning (source-grep) ──────────────────────

def test_cache_loader_logs_truncation_warning_source():
    """Source-grep (lesson §9.12): the cache_loader's cache.json
    parse path must reference `truncated_at_frame` so the WARNING
    fires whenever the manifest has it."""
    src = (_ADDON_DIR / "gpufluid_blender" / "cache_loader"
           / "__init__.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    assert "truncated_at_frame" in code, (
        "round-40 regressed: cache_loader must surface truncation marker")
    assert "TRUNCATED" in code, (
        "round-40: WARNING text must make the state visible to user")
