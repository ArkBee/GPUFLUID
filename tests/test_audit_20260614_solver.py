"""Audit 2026-06-14 — MpmSolver.run NaN-on-dump-boundary honesty bug (#1).

A divergence landing exactly on a dump boundary used to write the NaN PLY to
disk (the dump ran BEFORE the NaN check) and report it as the last good frame.
The full path is GPU-only, so these are source-order contracts (§9.12).
"""
from __future__ import annotations

import ast
from pathlib import Path

SOLVER = (Path(__file__).resolve().parents[1] / "src" / "gpufluid"
          / "sim" / "mpm" / "solver.py")


def _run_code() -> str:
    src = SOLVER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "run")
    body = src.splitlines()[node.lineno - 1:node.end_lineno]
    return "\n".join(l for l in body if not l.lstrip().startswith("#"))


def test_nan_check_runs_before_the_in_loop_dump():
    code = _run_code()
    nan = code.index("np.isnan(pos)")
    dump = code.index("save_frame_ply(out_dir, k)")  # in-loop dump (not ", 0)")
    assert nan < dump, \
        "audit #1: the NaN check must run BEFORE the in-loop dump so a diverged " \
        "boundary frame is never written to disk"


def test_last_dumped_excludes_the_diverged_step():
    code = _run_code()
    assert "((k - 1) // cfg.dump_every)" in code, \
        "audit #1: last_dumped must be the PREVIOUS clean boundary (exclude k)"
    assert "last_dumped = (k // cfg.dump_every)" not in code, \
        "audit #1 regressed: last_dumped back to reporting the diverged boundary"
