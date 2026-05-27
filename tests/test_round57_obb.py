"""Round-57 — MPM gets a native OBB (oriented bounding box) collider.

Round-52..56 walked a long diagnostic ladder for tilted obstacles:
  * R52: warn that BBOX+rotation loses orientation.
  * R53..55: auto-promote BBOX→MESH for FLIP (real fix), introduced and
    fixed a key-name bug along the way.
  * R56: discovered the MPM CLI path silently drops `type=mesh` —
    auto-promote is dead-letter for MPM. Demoted MPM to "honest
    warning + AABB fallback".

R57 fixes the actual MPM hole: the SDF box kernel (`k_sdf_box_collide`)
and the particle pushback kernel (`k_cube_pushback`) now accept a 3×3
rotation matrix and do the collision math in box-LOCAL frame, then
transform normals/positions back to world. Axis-aligned cubes use an
identity matrix → math reduces to pre-R57 form bit-exactly.

Contract (lesson §9.12):
  * `MpmCubeCollider.rotation` field exists, defaults to None.
  * `ObstacleBoxCfg.rotation` field exists, parsed from TOML.
  * `commands.py` MPM-obstacle assembly forwards `rotation` from the
    config to the collider.
  * Both kernels reference the rotation argument.
  * Addon emits `rotation` for MPM-rotated obstacles instead of falling
    back to AABB.
  * The pre-R57 MPM "limitation" warning text is GONE.
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src" / "gpufluid"
_ADDON = _REPO / "addon" / "gpufluid_blender"


def _code(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    return "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )


def test_mpm_cube_collider_has_rotation_field():
    code = _code(_SRC / "sim" / "mpm" / "solver.py")
    assert "rotation:" in code and "MpmCubeCollider" in code, (
        "round-57 regressed: MpmCubeCollider must expose `rotation` field")
    # Default must be None so existing AABB callers keep working.
    assert "rotation: " in code, "rotation field missing typed annotation"
    # The solver wires rotation into Dirichlet_collider basis vectors.
    assert "param.x_unit" in code, (
        "round-57 regressed: solver must set Dirichlet_collider.x_unit "
        "from cube.rotation (column 0)")
    assert "param.y_unit" in code, "x_unit/y_unit pair must be wired"
    assert "param.direction" in code, (
        "round-57: Z basis vector goes into Dirichlet_collider.direction")


def test_obstacle_box_cfg_accepts_rotation():
    code = _code(_SRC / "cli" / "config.py")
    assert "obstacle.rotation" in code, (
        "round-57 regressed: TOML parser must read obstacle.rotation")
    assert "rotation=rotation" in code or "rotation =rotation" in code, (
        "round-57: parsed rotation must be passed to ObstacleBoxCfg")


def test_commands_forwards_rotation():
    code = _code(_SRC / "cli" / "commands.py")
    assert "rotation=ob.rotation" in code, (
        "round-57 regressed: MPM cube assembly must forward "
        "rotation from ObstacleBoxCfg to MpmCubeCollider")


def test_grid_kernel_uses_basis_vectors():
    code = _code(_SRC / "sim" / "mpm" / "colliders.py")
    # Local-frame projection via dot products with basis vectors.
    assert "param.x_unit" in code, "round-57: kernel must read x_unit"
    assert "param.y_unit" in code, "round-57: kernel must read y_unit"
    assert "param.direction" in code, "round-57: kernel must read direction (Z axis)"
    # Local→world transform for the normal at the end.
    assert ("nlx * ex" in code or "ex * nlx" in code), (
        "round-57 regressed: normal must be transformed back to world "
        "via the basis vectors (nlx*ex + nly*ey + nlz*ez)")


def test_pushback_kernel_accepts_rotation_matrix():
    code = _code(_SRC / "sim" / "mpm" / "pushback.py")
    assert "R: wp.mat33" in code, (
        "round-57 regressed: k_cube_pushback must accept a mat33 "
        "rotation matrix as the last argument")
    assert "wp.transpose(R)" in code, (
        "round-57: pushback must compute R^T for world→local transform")


def test_solver_passes_mat33_to_pushback():
    code = _code(_SRC / "sim" / "mpm" / "solver.py")
    assert "wp.mat33(" in code, (
        "round-57: solver must build a wp.mat33 per cube (identity for AABB)")


def test_addon_emits_rotation_for_mpm_obb():
    code = _code(_ADDON / "operators" / "bake.py")
    bbox_pos = code.find('obstacle_type == "BBOX"')
    next_branch = code.find('elif oprops.obstacle_type', bbox_pos + 1)
    body = code[bbox_pos:next_branch]
    assert '"rotation":' in body, (
        "round-57 regressed: BBOX branch must emit rotation key for "
        "the MPM OBB path")
    # The R56 MPM-limitation warning text must be GONE — MPM now
    # handles OBB natively. Keeping the old text would mislead users.
    assert "MPM currently supports only axis-aligned" not in body, (
        "round-57 regressed: the pre-R57 MPM limitation warning must "
        "be removed — MPM supports OBB now")
    # decompose / rotation matrix construction must be present.
    assert "matrix_world.decompose" in body, (
        "round-57: addon must extract rotation via "
        "matrix_world.decompose() to separate it from scale")
