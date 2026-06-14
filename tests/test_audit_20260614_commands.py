"""Audit 2026-06-14 — cli/commands.py manifest/counter bugs (#5, #6, #7).

These live in the GPU bake path (hard to unit-test without CUDA), so they are
source-grep contract tests (§9.12): strip comments+docstrings, then assert the
buggy form is gone and the corrected form is present. Each was confirmed 3/3.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CMDS = REPO / "src" / "gpufluid" / "cli" / "commands.py"


def _code(path: Path) -> str:
    """Source minus comments + docstrings (so the fix-history comments that
    quote the old anti-pattern don't trip the greps)."""
    src = path.read_text(encoding="utf-8")
    doc = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc.update(range(body[0].lineno, body[0].end_lineno + 1))
    return "\n".join(l for i, l in enumerate(src.splitlines(), 1)
                     if i not in doc and not l.lstrip().startswith("#"))


def test_mpm_notes_is_measured_wallclock_not_frames_dt():
    # #5: notes was `sim={frames*dt}` — neither wall-clock nor simulated time,
    # and it fed the renderer overlay (which expects wall-clock) garbage.
    code = _code(CMDS)
    assert "sim={sim.frames * sim.dt" not in code, \
        "audit #5 regressed: MPM notes back to frames*dt"
    assert 'sim={t_sim_total' in code and "mesh={t_mesh_total" in code, \
        "audit #5: MPM notes should emit measured wall-clock 'sim=Xs mesh=Ys'"


def test_mpm_truncated_at_frame_converted_to_output_units():
    # #6: truncated_at is a raw step number; must be // dump_every for the
    # addon message to be in the same units as frame_count.
    code = _code(CMDS)
    assert "truncated_at // max(1, cfg.dump_every)" in code, \
        "audit #6: truncated_at_frame not converted from steps to output frames"


def test_flip_frame_count_not_overcounted_by_one():
    # #7: divergence breaks BEFORE writing frame N, so N files exist, not N+1.
    code = _code(CMDS)
    assert "flip_truncated_at + 1" not in code, \
        "audit #7 regressed: FLIP frame_count overcounts the diverged frame"
