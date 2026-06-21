"""MPM physics-benchmark self-diagnosis (2026-06-21, benchmark find/fix loop).

The benchmark (examples/benchmarks/mpm_physics_bench.py) exposed that the MPM
bake SILENTLY produced degenerate results. Three iterations made the system
honest about its own physical fidelity (all §9.10 "silent fallback -> logged"):

  iter 1: over-compression — a weakly-compressible column collapses to a
          ~9-particle/cell film on impact; the bake now records
          compression_ratio in cache.json and warns.
  iter 2: CFL sub-step saturation — the adaptive substepper can under-resolve;
          the bake now records substeps_max / substeps_saturated_steps.
  iter 3: those diagnostics were written but never surfaced downstream;
          cmd_render now warns about them before spending render time.

Pure helpers, CPU-unit-testable (no GPU bake).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from gpufluid.cli.commands import _particles_per_cell, _bake_quality_warnings

COMMANDS = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli" / "commands.py"


# ── iter 1: particles-per-cell packing metric ────────────────────────────

def test_particles_per_cell_uniform_one_per_cell():
    # one particle in each of 5 distinct cells (dx=0.1) -> mean 1, max 1.
    pos = np.array([[0.05, 0.05, 0.05], [0.15, 0.05, 0.05], [0.25, 0.05, 0.05],
                    [0.35, 0.05, 0.05], [0.45, 0.05, 0.05]])
    mean, mx, cells = _particles_per_cell(pos, 0.1)
    assert (mean, mx, cells) == (1.0, 1, 5)


def test_particles_per_cell_detects_pile_up():
    # 9 particles all in the SAME cell -> mean 9, max 9, 1 occupied cell
    # (this is exactly the collapsed-film signature the benchmark caught).
    pos = np.full((9, 3), 0.051)
    mean, mx, cells = _particles_per_cell(pos, 0.1)
    assert mean == 9.0 and mx == 9 and cells == 1


def test_particles_per_cell_empty():
    assert _particles_per_cell(np.empty((0, 3)), 0.1) == (0.0, 0, 0)


# ── iter 3: downstream surfacing of bake-quality diagnostics ─────────────

def test_warnings_flag_overcompression():
    w = _bake_quality_warnings({"compression_ratio": 9.02,
                                "settled_particles_per_cell": 9.0})
    assert len(w) == 1 and "OVER-COMPRESSED" in w[0]


def test_warnings_flag_substep_saturation():
    w = _bake_quality_warnings({"substeps_saturated_steps": 12,
                                "substeps_max": 24, "substeps_cap": 24})
    assert len(w) == 1 and "UNDER-RESOLVED" in w[0]


def test_warnings_flag_truncation():
    w = _bake_quality_warnings({"truncation_reason": "mpm_divergence",
                                "truncated_at_frame": 87})
    assert len(w) == 1 and "TRUNCATED" in w[0] and "87" in w[0]


def test_warnings_silent_on_healthy_bake():
    # a clean bake (incompressible-ish, well-resolved, complete) => NO warnings.
    assert _bake_quality_warnings({
        "compression_ratio": 1.1, "settled_particles_per_cell": 1.1,
        "substeps_saturated_steps": 0, "substeps_max": 3, "substeps_cap": 24,
    }) == []


def test_warnings_multiple_at_once():
    w = _bake_quality_warnings({"compression_ratio": 5.0,
                                "substeps_saturated_steps": 3,
                                "truncation_reason": "mpm_divergence"})
    assert len(w) == 3


# ── §9.12 source contracts: the bake records + cmd_render surfaces ────────

def _code() -> str:
    return "\n".join(l for l in COMMANDS.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))


def test_mpm_bake_records_diagnostics():
    code = _code()
    for key in ('"compression_ratio"', '"substeps_max"',
                '"substeps_saturated_steps"'):
        assert key in code, f"MPM manifest must record {key}"


def test_cmd_render_surfaces_bake_quality():
    code = _code()
    assert "_bake_quality_warnings(" in code, (
        "cmd_render must call _bake_quality_warnings so a degenerate bake is "
        "flagged before rendering")
    # ensure the call sits inside cmd_render (after its def).
    tree = ast.parse(COMMANDS.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "cmd_render")
    seg = ast.get_source_segment(COMMANDS.read_text(encoding="utf-8"), fn)
    assert "_bake_quality_warnings(" in seg
