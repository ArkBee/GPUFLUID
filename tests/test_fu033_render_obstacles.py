"""FU-033 — renderer draws ALL obstacles of ANY type, not just one box.

Pre-fix examples/render_fluid_on_cube_eevee.py read only ``obstacle[0]``
and RAISED ``"only box obstacles are renderable here"`` on anything
non-box. So after FU-019/FU-030 a sphere/cylinder/mesh obstacle baked
fine but CRASHED the render, and multi-obstacle scenes only ever drew
obstacle #0.

The renderer imports ``bpy`` at module top, so we can't import it whole
under CPython. The two FU-033 functions under test are PURE
(``_load_obstacles_from_scene`` = tomllib only; ``_obb_rows_to_quat`` =
mathutils only), so we exec their source in isolation under a faithful
mathutils stub. The build_scene spawner needs Blender — its
no-raise / all-types / missing-mesh-skip behaviour is covered by the
live render in the commit plus source contracts here.
"""
from __future__ import annotations

import ast
import math
import textwrap
import types
from pathlib import Path

import pytest

RENDERER = (Path(__file__).resolve().parents[1] / "examples"
            / "render_fluid_on_cube_eevee.py")


# ── faithful minimal mathutils (Matrix rows → Quaternion, Shepperd) ──────

class _Quat:
    def __init__(self, wxyz=(1.0, 0.0, 0.0, 0.0)):
        self.w, self.x, self.y, self.z = wxyz


class _Matrix:
    def __init__(self, rows):
        self.m = [[float(c) for c in r] for r in rows]

    def to_quaternion(self):
        m = self.m
        tr = m[0][0] + m[1][1] + m[2][2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m[2][1] - m[1][2]) / s
            y = (m[0][2] - m[2][0]) / s
            z = (m[1][0] - m[0][1]) / s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s
        return _Quat((w, x, y, z))


def _extract_func(name: str):
    """Exec the named top-level def from the renderer source in isolation,
    with a `mathutils` stub in scope. Avoids importing bpy."""
    src = RENDERER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    ns: dict = {"mathutils": types.SimpleNamespace(
        Matrix=_Matrix, Quaternion=_Quat),
        "Path": Path, "tomllib": __import__("tomllib")}
    exec(compile(ast.Module([node], []), str(RENDERER), "exec"), ns)
    return ns[name]


# ── §9.12 helper: comment + docstring stripped source ────────────────────

def _code_only() -> str:
    src = RENDERER.read_text(encoding="utf-8")
    doc_lines: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return "\n".join(line for i, line in enumerate(src.splitlines(), 1)
                     if i not in doc_lines
                     and not line.lstrip().startswith("#"))


# ── (a) loader parses one of each type, no raise ─────────────────────────

_FOUR_OBSTACLE_TOML = textwrap.dedent("""
    [domain]
    resolution = [32, 32, 32]

    [[obstacle]]
    type = "box"
    center = [0.5, 0.3, 0.5]
    half_size = [0.1, 0.1, 0.1]

    [[obstacle]]
    type = "sphere"
    center = [0.2, 0.5, 0.2]
    radius = 0.08

    [[obstacle]]
    type = "cylinder_y"
    center = [0.8, 0.4, 0.8]
    radius = 0.06
    half_height = 0.2

    [[obstacle]]
    type = "mesh"
    path = "/tmp/obs.obj"
    scale = [0.25, 0.25, 0.5]
    translate = [-0.1, -0.2, -0.3]
""")


def test_loader_returns_all_types_no_raise(tmp_path):
    toml = tmp_path / "scene.toml"
    toml.write_text(_FOUR_OBSTACLE_TOML, encoding="utf-8")
    load = _extract_func("_load_obstacles_from_scene")
    obs = load(toml)
    assert [o["type"] for o in obs] == ["box", "sphere", "cylinder_y", "mesh"], (
        "FU-033: loader must return ALL obstacles in order, of every type")
    assert obs[3]["scale"] == [0.25, 0.25, 0.5], (
        "FU-033: loader must pass raw geometry through untouched")


def test_loader_obstacle_free_scene_returns_empty(tmp_path):
    toml = tmp_path / "s.toml"
    toml.write_text("[domain]\nresolution = [16,16,16]\n", encoding="utf-8")
    load = _extract_func("_load_obstacles_from_scene")
    assert load(toml) == [], (
        "FU-033: an obstacle-free scene must return [] (no raise)")


# ── (b) regression: sphere-only scene that pre-fix raised loads clean ────

def test_sphere_only_scene_no_longer_raises(tmp_path):
    """Pins the exact pre-fix failure: a sphere FIRST obstacle hit
    `raise ValueError("only box obstacles are renderable here")`."""
    toml = tmp_path / "sphere.toml"
    toml.write_text(textwrap.dedent("""
        [[obstacle]]
        type = "sphere"
        center = [0.5, 0.5, 0.5]
        radius = 0.1
    """), encoding="utf-8")
    load = _extract_func("_load_obstacles_from_scene")
    obs = load(toml)  # must NOT raise
    assert len(obs) == 1 and obs[0]["type"] == "sphere"


# ── (c) source contract: the only-box raise is gone ──────────────────────

def test_only_box_raise_is_gone():
    code = _code_only()
    assert "only box obstacles are renderable here" not in code, (
        "FU-033 regressed: the non-box render raise is back — sphere/"
        "cylinder/mesh obstacles would crash the render again")
    assert "_load_obstacles_from_scene" in code and \
        "_load_obstacle_from_scene(" not in code, (
        "FU-033: the singular obstacle loader must be replaced by the plural")


# ── (d) OBB rotation matrix → quaternion ─────────────────────────────────

def test_obb_90deg_about_z_matrix_to_quat():
    """A 90° active rotation about +Z (rows [[0,-1,0],[1,0,0],[0,0,1]])
    must convert to the quaternion (w=z=cos45≈0.7071, x=y=0)."""
    conv = _extract_func("_obb_rows_to_quat")
    q = conv([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert q.w == pytest.approx(math.sqrt(0.5), abs=1e-4)
    assert q.z == pytest.approx(math.sqrt(0.5), abs=1e-4)
    assert q.x == pytest.approx(0.0, abs=1e-4)
    assert q.y == pytest.approx(0.0, abs=1e-4)


def test_obb_none_and_malformed_give_identity():
    conv = _extract_func("_obb_rows_to_quat")
    for bad in (None, [], [[1, 0]], "nope"):
        q = conv(bad)
        assert (q.w, q.x, q.y, q.z) == (1.0, 0.0, 0.0, 0.0), (
            f"FU-033: malformed rotation {bad!r} must yield identity "
            f"(a box with no rotation is a plain AABB)")


# ── (e) missing-mesh skip + build_scene contract ─────────────────────────

def test_missing_mesh_warns_and_skips_in_source():
    """build_scene needs Blender; assert the warn+skip path exists and is a
    `continue`, not a raise (§9.12 source contract)."""
    code = _code_only()
    assert 'if not path or not Path(path).exists():' in code, (
        "FU-033: mesh obstacle must guard a missing OBJ path")
    pos = code.find('if not path or not Path(path).exists():')
    window = code[pos:pos + 400]
    assert "continue" in window and "skipping" in window, (
        "FU-033: a missing mesh OBJ must warn + skip (continue), never "
        "crash the render")


def test_build_scene_signature_takes_obstacle_list():
    """Parse the actual def to read its real parameter names (a string
    slice would stop at the first annotation colon)."""
    tree = ast.parse(RENDERER.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "build_scene")
    params = [a.arg for a in fn.args.args]
    assert "obstacles" in params, (
        "FU-033: build_scene must take the obstacle LIST")
    assert "cube_center" not in params and "cube_half" not in params, (
        "FU-033: the single cube_center/cube_half params must be gone")
