"""Audit 2026-06-14 round 2 — GPU/CPU reseed emit-default drift (#6).

reseed_particles_gpu emitted white(1,1,1) colour / 0.0 temperature while the CPU
reseed_particles (and FlipSolver3D.prepare_frame) emit black(0,0,0) / 20.0. The
loop auto-switches on particle count (RESEED_GPU_THRESHOLD=100k), so the same
scene flips emit colour/temperature mid-bake. §9.6 mirror-drift; pin both paths
to the SAME constants so they can't diverge again.
"""
from __future__ import annotations

import ast
from pathlib import Path

RESEED = (Path(__file__).resolve().parents[1] / "src" / "gpufluid"
          / "sim" / "reseed.py")


def _code() -> str:
    src = RESEED.read_text(encoding="utf-8")
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


def test_gpu_reseed_emit_colour_is_black_not_white():
    code = _code()
    assert "emit_col = np.ones((n_emit" not in code, \
        "audit #6 regressed: GPU reseed emits white(1,1,1) again"
    assert "emit_col = np.zeros((n_emit" in code, \
        "GPU reseed emit colour must be zero/black (matches CPU + prepare_frame)"


def test_gpu_reseed_emit_temperature_is_twenty_not_zero():
    code = _code()
    assert "emit_t = np.zeros(n_emit" not in code, \
        "audit #6 regressed: GPU reseed emits T=0.0 again"
    assert "emit_t = np.full(n_emit, 20.0" in code, \
        "GPU reseed emit temperature must be 20.0 (matches CPU + prepare_frame)"


def test_cpu_reseed_emit_defaults_unchanged():
    code = _code()
    assert "emit_col = np.zeros((n_emitted" in code, \
        "CPU reseed emit colour must stay zero/black"
    assert "emit_t = np.full((n_emitted,), 20.0" in code, \
        "CPU reseed emit temperature must stay 20.0"
