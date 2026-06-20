"""Goal 2026-06-20 audit — a TILTED mesh obstacle must RENDER at the same
orientation the solver collided with.

The bake (`cli/commands.py` obstacle path + `domain/mesh_sdf.py`) rotates the
mesh verts with ``trimesh.euler_matrix(rx, ry, rz, "sxyz")`` before building the
collider SDF. The renderer used to apply only scale + translate to the imported
OBJ, so a CLI scene with an obstacle ``rotate_deg`` drew the mesh axis-aligned
while the collider was tilted — a §9.6 render↔solver mismatch visible from the
4-side validation views (the box-OBB twin already honoured ``rotation``).

The fix adds ``_mesh_rotate_deg_to_quat`` (pure: math + mathutils, reuses
``_obb_rows_to_quat``) and applies it in the mesh branch. The renderer imports
``bpy`` at module top, so we exec the two pure helpers in isolation under a
faithful mathutils stub, exactly like ``test_fu033_render_obstacles.py``.
"""
from __future__ import annotations

import ast
import math
import types
from pathlib import Path

import numpy as np
import pytest
from trimesh.transformations import euler_matrix

RENDERER = (Path(__file__).resolve().parents[1] / "examples"
            / "render_fluid_on_cube_eevee.py")


# ── faithful minimal mathutils (Matrix rows → Quaternion, Shepperd) ──────
# (copied from test_fu033_render_obstacles.py — same contract)

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


def _extract_funcs(*names):
    """Exec the named top-level defs from the renderer source into ONE shared
    namespace (so a helper that calls a sibling helper resolves it), with a
    mathutils stub + math in scope. Avoids importing bpy."""
    src = RENDERER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    ns: dict = {
        "mathutils": types.SimpleNamespace(Matrix=_Matrix, Quaternion=_Quat),
        "math": math,
    }
    exec(compile(ast.Module(nodes, []), str(RENDERER), "exec"), ns)
    return ns


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


# ── (a) render rotation == bake rotation (the core correctness) ──────────

@pytest.mark.parametrize("deg", [
    (0.0, 0.0, 0.0),
    (30.0, 0.0, 0.0),
    (0.0, 25.0, 0.0),
    (0.0, 0.0, 90.0),
    (23.0, 41.0, 67.0),
    (-15.0, 80.0, -120.0),
])
def test_render_rotate_matches_trimesh_bake(deg):
    """The quaternion the renderer applies to a tilted mesh obstacle must
    equal the quaternion of the SAME rotation the bake used
    (trimesh.euler_matrix(...,"sxyz")). If they drift, the drawn mesh and the
    collider point different ways — the exact 4-side mismatch this closes."""
    ns = _extract_funcs("_obb_rows_to_quat", "_mesh_rotate_deg_to_quat")
    got = ns["_mesh_rotate_deg_to_quat"](list(deg))

    rx, ry, rz = np.deg2rad(deg)
    bake_R = euler_matrix(rx, ry, rz, "sxyz")[:3, :3]
    expected = _Matrix([list(bake_R[0]), list(bake_R[1]), list(bake_R[2])
                        ]).to_quaternion()
    # Quaternion sign is double-cover; compare up to global sign.
    a = np.array([got.w, got.x, got.y, got.z])
    b = np.array([expected.w, expected.x, expected.y, expected.z])
    if np.dot(a, b) < 0:
        b = -b
    assert np.allclose(a, b, atol=1e-6), (
        f"render rotate {deg} → quat {a} != bake quat {b}")


def test_none_and_empty_give_identity():
    ns = _extract_funcs("_obb_rows_to_quat", "_mesh_rotate_deg_to_quat")
    fn = ns["_mesh_rotate_deg_to_quat"]
    for empty in (None, (), []):
        q = fn(empty)
        assert (q.w, q.x, q.y, q.z) == (1.0, 0.0, 0.0, 0.0), (
            f"axis-aligned mesh ({empty!r}) must render with identity rotation")


# ── (b) §9.12 source contract: the mesh branch actually applies rotate ───

def test_mesh_branch_applies_rotate_deg():
    code = _code_only()
    assert "_mesh_rotate_deg_to_quat(obs.get(\"rotate_deg\"))" in code, (
        "regressed: the mesh obstacle render branch must convert rotate_deg "
        "to an extra rotation (else a tilted mesh renders axis-aligned)")
    # and it must be threaded into _finish_obstacle as extra_rot.
    pos = code.find("_mesh_rotate_deg_to_quat(obs.get")
    window = code[pos:pos + 160]
    assert "extra_rot=extra" in window, (
        "the mesh rotate_deg quat must be passed to _finish_obstacle as "
        "extra_rot (mirror of the box-OBB path)")


# ── cylinder_y axis: render must follow the collider's sim-Y, not world-Z ──
# Goal 2026-06-20: in a Z-up MPM scene cylinder_y RENDERED as a vertical pillar
# (world-Z) while the collider was a horizontal log along sim-Y. Proven by the
# particle dump: 0 particles inside the log (collider) vs hundreds inside the
# drawn pillar. Fix maps mesh-local +Z → sim +Y via −90° about X, applied
# directly (= the Y-up `_counter`, byte-identical there; fixes Z-up).

def test_minus90_about_x_maps_local_z_to_sim_y():
    """The collider axis is sim-Y; the cylinder primitive's axis is local-Z.
    A −90° rotation about X must carry local +Z onto sim +Y (and leave the
    half-height extent along Y, matching k_sdf_cylinder_collide)."""
    a = math.radians(-90.0)
    # R_x(a) · v ; row-form so we don't need mathutils here.
    c, s = math.cos(a), math.sin(a)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    assert np.allclose(Rx @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0],
                       atol=1e-9), "local +Z must map to sim +Y"


def test_cylinder_branch_aligns_to_sim_y_not_world_z():
    code = _code_only()
    cyl = code.find('elif otype == "cylinder_y":')
    assert cyl != -1
    nxt = code.find('elif otype ==', cyl + 1)
    branch = code[cyl:nxt]
    assert "ob.rotation_quaternion = mathutils.Euler(" in branch \
        and "math.radians(-90.0)" in branch, (
        "regressed: the cylinder_y render branch must override the world-"
        "aligning counter with an explicit −90°X (sim-Y alignment); else a "
        "Z-up cylinder renders a vertical pillar through which water flows")
    # the old buggy generalisation must be gone (it claimed no extra rotation
    # was needed because counter@local-Z=sim-Y — true only for the Y-up pivot).
    assert "So NO" not in branch and "extra rotation is needed" not in branch, (
        "the pre-fix 'no extra rotation needed' reasoning (Y-up-only) is back")
