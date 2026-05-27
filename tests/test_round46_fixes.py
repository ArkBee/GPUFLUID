"""Round-46 regression tests for round-45 reviewer findings.

  - InflowBox + FluidEmitEvent carry per-source color + temperature.
  - CLI FLIP path forwards inflow.color → InflowBox.color.
  - prepare_frame builds pad pre-filter + same mask post-filter
    so the round-42 marker-filter stays in lockstep.
  - _cuda_graph_eligible refuses APIC when affine_C is None.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parent.parent


# ─── 1. InflowBox carries color + temperature ───────────────────────────

def test_inflow_box_accepts_color_and_temperature():
    from gpufluid.domain.regions import InflowBox
    inf = InflowBox(
        lo=(0, 0, 0), hi=(1, 1, 1),
        color=(0.3, 0.7, 0.9), temperature=42.0,
    )
    assert inf.color == (0.3, 0.7, 0.9)
    assert inf.temperature == 42.0


def test_inflow_box_color_defaults_to_none():
    """Round-46 back-compat: existing callers that don't pass colour
    must continue to get None (= use solver default zero-pad)."""
    from gpufluid.domain.regions import InflowBox
    inf = InflowBox(lo=(0, 0, 0), hi=(1, 1, 1))
    assert inf.color is None
    assert inf.temperature is None


# ─── 2. FluidEmitEvent carries color + temperature ──────────────────────

def test_fluid_emit_event_accepts_color_and_temperature():
    from gpufluid.primitives.frame_events import FluidEmitEvent
    pos = np.zeros((5, 3), dtype=np.float32)
    vel = np.zeros((5, 3), dtype=np.float32)
    ev = FluidEmitEvent(positions=pos, velocities=vel,
                        color=(0.1, 0.2, 0.3), temperature=18.5)
    assert ev.color == (0.1, 0.2, 0.3)
    assert ev.temperature == 18.5


def test_fluid_emit_event_defaults_none():
    from gpufluid.primitives.frame_events import FluidEmitEvent
    pos = np.zeros((5, 3), dtype=np.float32)
    vel = np.zeros((5, 3), dtype=np.float32)
    ev = FluidEmitEvent(positions=pos, velocities=vel)
    assert ev.color is None
    assert ev.temperature is None


# ─── 3. publish_for_frame propagates color/temperature ──────────────────

def test_publish_propagates_color_to_event():
    from gpufluid.domain.regions import InflowBox
    from gpufluid.primitives.frame_events import FrameEventQueue
    q = FrameEventQueue()
    inf = InflowBox(
        lo=(0, 0, 0), hi=(0.1, 0.1, 0.1),
        rate_per_sec=1000,
        color=(0.5, 0.6, 0.7), temperature=33.0,
    )
    inf.publish_for_frame(q, frame_idx=0, frame_dt=1 / 24,
                          rng=np.random.default_rng(0))
    emits = q.drain_emits()
    assert len(emits) == 1
    assert emits[0].color == (0.5, 0.6, 0.7)
    assert emits[0].temperature == 33.0


# ─── 4. CLI FLIP plumbing — source-grep ─────────────────────────────────

def test_cli_flip_forwards_inflow_color():
    """Source-grep contract (lesson §9.12): the CLI FLIP add_inflow
    call must reference inflow.color (or getattr-defaulted equivalent)
    so the addon-side TOML colour reaches InflowBox."""
    src = (_REPO / "src" / "gpufluid" / "cli"
           / "commands.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    # The construction call must reference color= kwarg.
    pos = code.find("solver.add_inflow(InflowBox(")
    if pos < 0:
        pos = code.find("add_inflow(InflowBox(")
    assert pos > 0, "expected add_inflow(InflowBox( construction"
    body = code[pos:pos + 600]
    assert "color=" in body, (
        "round-46 regressed: CLI must forward inflow.color to InflowBox")
    assert "temperature=" in body, (
        "round-46 regressed: CLI must forward inflow.temperature")


# ─── 5. prepare_frame col_pad source-grep ──────────────────────────────

def test_prepare_frame_builds_per_event_color_pad():
    """Source-grep: prepare_frame must build per-event colour pad
    BEFORE the round-42 marker-filter mask, then apply the same
    mask to pad to stay row-aligned."""
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    fn_pos = src.find("def prepare_frame")
    next_def = src.find("\n    def ", fn_pos + 1)
    body = src[fn_pos:next_def]
    assert "col_pad" in body, (
        "round-46 regressed: prepare_frame must build per-event col_pad")
    assert "temp_pad" in body, (
        "round-46 regressed: prepare_frame must build per-event temp_pad")
    assert "col_pad = col_pad[keep]" in body, (
        "round-46 regressed: col_pad must be filtered by the same mask")


# ─── 6. CUDA graph eligibility refuses APIC + None affine ──────────────

def test_cuda_graph_ineligible_when_apic_affine_none():
    """Round-46: graph capture must skip APIC steps when affine_C is
    None (next step will allocate; allocation forbidden mid-capture)."""
    from gpufluid.solvers.solver3d import FlipSolver3D
    sol = object.__new__(FlipSolver3D)
    sol.transfer_mode = "apic"
    sol.affine_C = None
    assert sol._cuda_graph_eligible(
        pressure_solver="jacobi", pressure_block_sparse=False) is False


def test_cuda_graph_eligible_apic_with_affine():
    from gpufluid.solvers.solver3d import FlipSolver3D
    sol = object.__new__(FlipSolver3D)
    sol.transfer_mode = "apic"
    sol.affine_C = object()  # sentinel non-None
    assert sol._cuda_graph_eligible(
        pressure_solver="jacobi", pressure_block_sparse=False) is True


def test_cuda_graph_eligible_flip_regardless_of_affine():
    """FLIP mode never touches affine_C lazily, so eligibility is
    unaffected by the new APIC guard."""
    from gpufluid.solvers.solver3d import FlipSolver3D
    sol = object.__new__(FlipSolver3D)
    sol.transfer_mode = "flip"
    sol.affine_C = None
    assert sol._cuda_graph_eligible(
        pressure_solver="jacobi", pressure_block_sparse=False) is True
