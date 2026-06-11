"""audit-20260610 — guard tests for blocks check_6 flagged as having no
test reference (test_blocks_registry::test_check_6_every_block_has_a_test).

Each test here GENUINELY exercises the registered callable; none is a
naming-only ref. Blocks covered:

  [BLK A8.4]    GpufluidObstacleProps — property-group surface contract
  [BLK A8.7]    UI panels — class surface + registration-list contract
  [BLK A8.7.1]  Whitewater sub-panel — same
  [BLK S2.17.4] k_tap_terminal_velocity — downward-only cap (source contract)
  [BLK S2.17.5] k_anti_splash_vz — symmetric clamp (source contract)
  [BLK S2.17.PATCH.EOS] _patch_eos — behavioural (apply + idempotent)
  [BLK I6.1.MESH] read_points_ply — behavioural roundtrip
  [BLK M5.9.1] [BLK M5.9.H]  cubic mesher path — behavioural (GPU)
  [BLK M5.11.1] [BLK M5.11.2] [BLK M5.11.3] [BLK M5.11.H]
                sdf mesher path — behavioural (GPU)

The S2.17.4/S2.17.5 kernels need a live MPMStateStruct to launch, so the
non-GPU guard is a §9.12 source contract on the kernel BODY semantics
(strips comments + docstrings first).
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "addon" / "gpufluid_blender"
SRC = REPO / "src" / "gpufluid"


def _code_only(path: Path) -> str:
    """Source minus full-line comments and docstrings (§9.12)."""
    src = path.read_text(encoding="utf-8")
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


def _load_addon_file(fname: str, mod_name: str):
    """Standalone-load an addon file under a minimal bpy stub (the file's
    own `from ._blocks import block` fallback handles the missing parent
    package)."""
    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(
        Panel=type("Panel", (), {}), Operator=type("Operator", (), {}),
        PropertyGroup=type("PropertyGroup", (), {}),
        AddonPreferences=type("AddonPreferences", (), {}),
        # referenced by PointerProperty(type=...) annotations, which
        # evaluate at class-body time (properties.py has no
        # `from __future__ import annotations`)
        Object=type("Object", (), {}), Scene=type("Scene", (), {}))
    bpy.props = types.SimpleNamespace(**{
        n: (lambda *a, **k: ("PROPDEF", a, k)) for n in
        ("StringProperty", "IntProperty", "BoolProperty", "EnumProperty",
         "FloatProperty", "FloatVectorProperty", "PointerProperty")})
    bpy.utils = types.SimpleNamespace(register_class=lambda c: None,
                                      unregister_class=lambda c: None)
    sys.modules["bpy"] = bpy
    spec = importlib.util.spec_from_file_location(mod_name, ADDON / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# [BLK A8.4] obstacle property group
# ---------------------------------------------------------------------------

def test_a8_4_obstacle_props_surface():
    """The Obstacle property group must keep its role flag + the
    obstacle_type enum the whole round-52..57 OBB pipeline dispatches on."""
    props = _load_addon_file("properties.py", "_cov_props")
    cls = getattr(props, "GpufluidObstacleProps", None)
    assert cls is not None, "audit-20260610: GpufluidObstacleProps missing"
    declared = dict(getattr(cls, "__annotations__", {}))
    for field in ("is_obstacle", "obstacle_type"):
        assert field in declared, (
            f"audit-20260610: GpufluidObstacleProps lost '{field}' — "
            f"collect_scene dispatches obstacle handling on it")


# ---------------------------------------------------------------------------
# [BLK A8.7] / [BLK A8.7.1] sidebar panels
# ---------------------------------------------------------------------------

_EXPECTED_PANELS = (
    "GPUFLUID_PT_main", "GPUFLUID_PT_domain", "GPUFLUID_PT_fluid",
    "GPUFLUID_PT_obstacle", "GPUFLUID_PT_inflow", "GPUFLUID_PT_outflow",
    "GPUFLUID_PT_whitewater", "GPUFLUID_PT_bake",
)


def test_a8_7_panels_surface_and_category():
    """All 8 sidebar panels exist, carry a label, and sit in the GpuFluid
    N-sidebar category (a renamed/lost panel silently vanishes from the UI
    — there is no error path for that in Blender)."""
    panels = _load_addon_file("panels.py", "_cov_panels")
    for name in _EXPECTED_PANELS:
        cls = getattr(panels, name, None)
        assert cls is not None, f"audit-20260610: panel {name} missing"
        assert getattr(cls, "bl_label", ""), f"{name} lost bl_label"
        assert getattr(cls, "bl_space_type", "") == "VIEW_3D", (
            f"{name} not a VIEW_3D sidebar panel")
        assert getattr(cls, "bl_category", panels.PANEL_CATEGORY) \
            == panels.PANEL_CATEGORY or \
            getattr(cls, "bl_parent_id", ""), (
            f"{name} fell out of the '{panels.PANEL_CATEGORY}' category "
            f"and is not a sub-panel")


def test_a8_7_panels_all_in_register_list():
    """Every panel class must be enumerated in __init__._collect_classes —
    a defined-but-unregistered panel never draws (the live failure mode)."""
    init_code = _code_only(ADDON / "__init__.py")
    for name in _EXPECTED_PANELS:
        assert f"panels.{name}" in init_code, (
            f"audit-20260610: {name} is not in _collect_classes() — it "
            f"would silently never register")


def test_a8_7_1_whitewater_panel_draws_ww_group():
    """[BLK A8.7.1]: the whitewater sub-panel must reference the
    whitewater_group props it fronts (B1.5)."""
    code = _code_only(ADDON / "panels.py")
    ww_pos = code.find("class GPUFLUID_PT_whitewater")
    assert ww_pos > 0
    nxt = code.find("\nclass ", ww_pos + 1)
    body = code[ww_pos:nxt if nxt > 0 else len(code)]
    assert "whitewater_group" in body, (
        "audit-20260610: GPUFLUID_PT_whitewater no longer draws "
        "whitewater_group — the B1.5 knobs become unreachable from the UI")


# ---------------------------------------------------------------------------
# [BLK S2.17.4] / [BLK S2.17.5] velocity-cap kernels (source contract)
# ---------------------------------------------------------------------------

def test_s2_17_4_tap_cap_is_downward_only():
    """k_tap_terminal_velocity must cap ONLY faster-than-terminal downward
    motion (v_z < v_terminal → write v_terminal) and must preserve the
    lateral components. A symmetric clamp here would brake upward splashes
    too — that is S2.17.5's separate job with a different bound."""
    code = _code_only(SRC / "sim" / "mpm" / "velcaps.py")
    fn = code[code.find("def k_tap_terminal_velocity"):
              code.find("def k_anti_splash_vz")]
    assert "if v[2] < v_terminal:" in fn, (
        "audit-20260610: tap cap no longer gates on downward-only "
        "(v[2] < v_terminal) — regressed to symmetric/inverted form")
    assert "wp.vec3(v[0], v[1], v_terminal)" in fn, (
        "audit-20260610: tap cap must preserve lateral velocity and write "
        "exactly v_terminal on z")


def test_s2_17_5_anti_splash_is_symmetric_clamp():
    """k_anti_splash_vz must hard-clamp v_z into [vz_min, vz_max] above the
    obstacle and keep x/y untouched."""
    code = _code_only(SRC / "sim" / "mpm" / "velcaps.py")
    fn = code[code.find("def k_anti_splash_vz"):]
    assert "wp.clamp(v[2], vz_min, vz_max)" in fn, (
        "audit-20260610: anti-splash clamp form changed — |v_z| is no "
        "longer bounded by [vz_min, vz_max]")
    assert "wp.vec3(v[0], v[1], vz)" in fn, (
        "audit-20260610: anti-splash must preserve lateral velocity")


# ---------------------------------------------------------------------------
# [BLK S2.17.PATCH.EOS] overlay patch (behavioural — mirrors the round-38
# SLIP-patch test in test_round38_fixes.py)
# ---------------------------------------------------------------------------

def test_s2_17_patch_eos_applies_and_is_idempotent(tmp_path):
    from gpufluid.sim.mpm import _patches as mod

    fake_utils = tmp_path / "mpm_utils.py"
    fake_utils.write_text(
        "def kirchoff_stress_water(J, bulk, gamma):\n"
        "    pressure = -bulk * (wp.pow(J, -gamma) - 1.)\n",
        encoding="utf-8")
    assert mod._patch_eos(fake_utils) is True, (
        "audit-20260610: _patch_eos must report True on first application")
    patched = fake_utils.read_text(encoding="utf-8")
    assert "wp.max(J, 0.5)" in patched and mod._EOS_MARKER in patched, (
        "audit-20260610: EOS patch must insert the J clamp + marker")
    assert mod._patch_eos(fake_utils) is False, (
        "audit-20260610: _patch_eos must be a no-op (False) once the "
        "marker is present — re-patching would corrupt the kernel source")


# ---------------------------------------------------------------------------
# [BLK I6.1.MESH] points-only PLY reader (behavioural roundtrip)
# ---------------------------------------------------------------------------

def test_i6_1_mesh_points_ply_roundtrip(tmp_path):
    from gpufluid.blocks import BlockError
    from gpufluid.io.ply import read_points_ply, write_points_ply

    pts = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]],
                   dtype=np.float32)
    p = tmp_path / "pts.ply"
    n = write_points_ply(p, pts)
    assert n == 3
    back = read_points_ply(p)
    assert back.shape == (3, 3) and back.dtype == np.float32
    assert np.allclose(back, pts), (
        "audit-20260610: points-PLY roundtrip corrupted positions")
    # Truncated header must fail with the block-tagged error, not hang/garble.
    bad = tmp_path / "bad.ply"
    bad.write_bytes(b"ply\nformat binary_little_endian 1.0\n")
    with pytest.raises(BlockError):
        read_points_ply(bad)


# ---------------------------------------------------------------------------
# [BLK M5.9.1] [BLK M5.9.H] [BLK M5.11.1] [BLK M5.11.2] [BLK M5.11.3]
# [BLK M5.11.H] — cubic + sdf mesher paths (behavioural, GPU)
# ---------------------------------------------------------------------------

def _box_particles(lo, hi, n_per_axis):
    xs = np.linspace(lo[0], hi[0], n_per_axis, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], n_per_axis, dtype=np.float32)
    zs = np.linspace(lo[2], hi[2], n_per_axis, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)


@pytest.mark.gpu
@pytest.mark.parametrize("method, iso", [("cubic", 0.3), ("sdf", 0.0)])
def test_m5_alt_mesh_methods_produce_bounded_mesh(method, iso):
    """The cubic (M5.9.1 kernel + M5.9.H host) and sdf (M5.11.1 hashgrid +
    M5.11.2 kernel + M5.11.3 smoothing + M5.11.H host) extract paths had
    ZERO tests — every mesher test used the trilinear default. A dense
    particle box must yield a non-empty mesh whose bbox stays within a few
    cells of the seeded box, on BOTH alternative paths."""
    wp = pytest.importorskip("warp")
    from gpufluid import MeshExtractor

    nx = ny = nz = 32
    dx = 1.0 / 32
    lo = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    hi = np.array([0.75, 0.75, 0.75], dtype=np.float32)
    pos_wp = wp.array(_box_particles(lo, hi, 20), dtype=wp.vec3)
    extractor = MeshExtractor(nx, ny, nz, dx)
    verts, faces = extractor.extract(
        pos_wp, iso_level=iso, smooth_passes=1, mesh_method=method)
    assert verts is not None and len(verts) > 0, (
        f"audit-20260610: mesh_method={method!r} produced no mesh")
    assert faces is not None and len(faces) > 0
    tol = 5 * dx  # cubic kernel is wider than trilinear; allow extra bleed
    assert np.all(verts.min(axis=0) >= lo - tol), (
        f"{method}: mesh bbox leaked below the particle box")
    assert np.all(verts.max(axis=0) <= hi + tol), (
        f"{method}: mesh bbox leaked above the particle box")
