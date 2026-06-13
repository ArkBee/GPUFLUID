"""demo30 root cause (2026-06-14): the headless renderer was hardcoded Y-up,
but the MPM solver is Z-up (gravity (0,0,-9.81)) while the legacy FLIP solver
is Y-up (gravity on y). So every MPM scene rendered rotated 90° onto its side —
a flat pool became a vertical wall, which is why MPM water never read as water.

The renderer is now solver-aware: it reads ``[simulation].solver`` and orients
the pivot + obstacle counter-rotation to match. These tests pin that contract
without launching Blender (the renderer module imports bpy at top, so we test
the detection logic on real scene files + a source contract on the wiring).
"""
from __future__ import annotations
import ast
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCENES = REPO / "examples" / "scenes"
RENDERER = REPO / "examples" / "render_fluid_on_cube_eevee.py"


def _solver_is_mpm(toml_path: Path) -> bool:
    """Mirror of the renderer's _scene_is_zup logic, run on real files."""
    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return str(data.get("simulation", {}).get("solver", "flip")).lower() == "mpm"


def _code_only(path: Path) -> str:
    """Source minus full-line comments and all docstrings (§9.12)."""
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
    keep = [ln for i, ln in enumerate(src.splitlines(), 1)
            if i not in doc and not ln.lstrip().startswith("#")]
    return "\n".join(keep)


def test_mpm_scene_detected_zup_flip_detected_yup():
    # The MPM hero/pool scenes must read as Z-up; the FLIP bench as Y-up.
    assert _solver_is_mpm(SCENES / "demo30_pool.toml") is True
    assert _solver_is_mpm(SCENES / "demo30_hero_waterfall.toml") is True
    assert _solver_is_mpm(SCENES / "bench_dam_break.toml") is False
    assert _solver_is_mpm(SCENES / "bench_waterfall.toml") is False


def test_missing_or_absent_solver_defaults_to_yup():
    import tempfile, os
    fd, p = tempfile.mkstemp(suffix=".toml")
    try:
        os.write(fd, b"[simulation]\ndt = 0.005\n")  # no solver key
        os.close(fd)
        assert _solver_is_mpm(Path(p)) is False
    finally:
        os.unlink(p)


def test_renderer_orientation_is_solver_aware_not_hardcoded():
    code = _code_only(RENDERER)
    # _scene_is_zup keys on solver == 'mpm'
    assert "_scene_is_zup" in code, "demo30-zup: solver-detection helper gone"
    assert re.search(r"solver\s*\)?\s*\.lower\(\)\s*==\s*[\"']mpm[\"']", code), \
        "demo30-zup: _scene_is_zup must key on solver == 'mpm'"
    # the pivot rotation must be CONDITIONAL on up_z, not a hardcoded +90°X
    assert re.search(r"pivot\.rotation_euler\s*=\s*\(0,\s*0,\s*0\)\s*if\s*up_z",
                     code), \
        "demo30-zup: pivot must be identity (Z-up) when up_z, else +90X (Y-up)"
    # the obstacle counter-rotation must also depend on up_z
    assert re.search(r"_counter\s*=\s*\(?\s*mathutils\.Quaternion\(\)\s*if\s*up_z",
                     code), \
        "demo30-zup: obstacle counter-rotation must be identity when up_z"
    # up_z must actually be computed from the scene and threaded to build_scene
    assert "up_z = _scene_is_zup(" in code, "demo30-zup: up_z not derived in main"
    assert "up_z=up_z" in code, "demo30-zup: up_z not passed to build_scene"
