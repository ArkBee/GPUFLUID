"""FU-030 (addon side) — per-axis SPHERE/MESH fluid-source scaling.

Pre-FU-030 collect_scene scaled sphere radius / mesh scale by the SCALAR
avg_inv and warned on non-cubic domains (audit-20260610). FU-030 makes
the emission exact per axis against the FIXED CLI schema:

  [[fluids]] type="sphere" → radii = [rx, ry, rz]   (scalar radius is
                              back-compat for hand-written TOMLs only)
  [[fluids]] type="mesh"   → scale = [sx, sy, sz] OR legacy scalar

These tests are CONTRACT tests against the TOML text/schema (tomllib
round-trip) — they do NOT import the CLI, which is changing in parallel.

bake.collect_scene is loaded behaviourally with the bpy/mathutils stub
harness from tests/test_audit_20260610_addon.py (slimmed).
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import tomllib
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "addon" / "gpufluid_blender"
PKG = "_gfafu030"

# ── config_builder, standalone (pure python — same pattern as
#    tests/test_a8_config_builder.py) ────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "_fu030_config_builder", ADDON / "config_builder.py")
_cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cb)
build_toml = _cb.build_toml


# ── slim bake-module loader (stub bpy/mathutils + bpy-heavy siblings) ───

class _Vec:
    def __init__(self, seq=(0.0, 0.0, 0.0)):
        self.x, self.y, self.z = (float(c) for c in seq)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __getitem__(self, i):
        return (self.x, self.y, self.z)[i]


def _load_bake():
    bpy = types.ModuleType("bpy")
    bpy.props = types.SimpleNamespace(**{
        n: (lambda *a, **k: None) for n in
        ("StringProperty", "IntProperty", "BoolProperty", "EnumProperty",
         "FloatProperty", "FloatVectorProperty", "PointerProperty")})
    bpy.types = types.SimpleNamespace(Operator=object)
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    bpy.data = types.SimpleNamespace(filepath="", objects={})
    bpy.ops = types.SimpleNamespace()
    bpy.context = types.SimpleNamespace()
    sys.modules["bpy"] = bpy
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = _Vec
    sys.modules["mathutils"] = mathutils

    pkg = types.ModuleType(PKG)
    pkg.__path__ = []
    pkg.logger = logging.getLogger("fu030")
    sys.modules[PKG] = pkg
    ops_pkg = types.ModuleType(f"{PKG}.operators")
    ops_pkg.__path__ = []
    sys.modules[f"{PKG}.operators"] = ops_pkg

    stubs = {
        f"{PKG}.config_builder": {"build_toml": lambda d: ""},
        f"{PKG}.preferences": {
            "get_prefs": lambda c: None,
            "_detect_interpreter": lambda r=None: "",
            "_context_roots": lambda: []},
        f"{PKG}.operators._animation": {
            "_bbox_world": lambda o: o._bbox,
            "_bbox_world_at_frame": lambda o, f: o._bbox,
            "_is_animated": lambda o, c: False},
        f"{PKG}.operators._collect": {
            "_export_obj": lambda o, p: None,
            "_output_dict": lambda dp: {"cache_dir": dp.cache_dir},
            "_safe_obj_name": lambda n: n},
        f"{PKG}.operators.helpers": {"subprocess_drain": lambda *a, **k: None},
        f"{PKG}.operators._runner": {"ModalSubprocessRunner": object},
    }
    for name, attrs in stubs.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    for name, rel in (("domain_transform", "domain_transform.py"),
                      ("scene_validator", "scene_validator.py")):
        spec = importlib.util.spec_from_file_location(f"{PKG}.{name}",
                                                      ADDON / rel)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = PKG
        sys.modules[f"{PKG}.{name}"] = mod
        spec.loader.exec_module(mod)

    spec = importlib.util.spec_from_file_location(
        f"{PKG}.operators.bake", ADDON / "operators" / "bake.py")
    bake = importlib.util.module_from_spec(spec)
    bake.__package__ = f"{PKG}.operators"
    sys.modules[f"{PKG}.operators.bake"] = bake
    spec.loader.exec_module(bake)
    return bake


def _fake_obj(name, bbox, *, fluid=None):
    o = types.SimpleNamespace(name=name)
    o._bbox = bbox
    o.gpufluid_fluid = fluid or types.SimpleNamespace(is_fluid=False)
    o.gpufluid_inflow = types.SimpleNamespace(is_inflow=False)
    o.gpufluid_obstacle = types.SimpleNamespace(is_obstacle=False)
    o.gpufluid_outflow = types.SimpleNamespace(is_outflow=False)
    return o


def _domain_props(cache_dir):
    return types.SimpleNamespace(
        is_domain=True, cache_dir=str(cache_dir), resolution=32,
        solver="mpm", dt=0.005, frames=10, fps=24,
        pressure_iters=40, pressure_solver="jacobi",
        use_cfl=False, cfl_factor=1.0, cfl_max_substeps=4,
        flip_blend=0.97, gravity=-9.81,
        reseed=False, reseed_every_n_frames=4,
        reseed_min_per_cell=4, reseed_max_per_cell=16,
        mpm_bulk_modulus=1500.0, mpm_rpic_damping=0.0,
        mpm_grid_v_damping=1.0, mpm_cube_friction=0.3,
        mpm_v_terminal=3.0, mpm_vz_max_splash=1.5,
        mpm_initial_velocity=0.0, toml_overrides="",
    )


def _collect(tmp_path, *, dom_lo=(0.0, 0.0, 0.0), dom_hi=(4.0, 4.0, 2.0),
             src_bbox=((1.0, 1.0, 0.5), (2.0, 2.0, 1.5)),
             source_type="SPHERE"):
    bake = _load_bake()
    dom = _fake_obj("Domain", (dom_lo, dom_hi))
    dom.gpufluid_domain = _domain_props(tmp_path / "cache")
    src = _fake_obj("Src", src_bbox,
                    fluid=types.SimpleNamespace(
                        is_fluid=True, ppc=8, source_type=source_type,
                        fill_mesh=False))
    ctx = types.SimpleNamespace(
        scene=types.SimpleNamespace(objects=[dom, src]))
    return bake.collect_scene(ctx, dom)


# ── collect_scene behavioural ───────────────────────────────────────────

def test_fu030_sphere_radii_per_axis_on_noncubic_domain(tmp_path):
    """4×4×2 m domain → inv_size = (.25, .25, .5). A 1 m sphere (half
    extents 0.5 each axis) must emit radii = (0.125, 0.125, 0.25) — exact
    per axis, NOT the old scalar max*avg_inv."""
    scene_dict, _, _, warnings = _collect(tmp_path)
    entry = scene_dict["fluids"][0]
    assert entry["kind"] == "sphere"
    assert "radius" not in entry, (
        "FU-030: scalar radius must NOT be emitted alongside radii")
    assert entry["radii"] == pytest.approx((0.125, 0.125, 0.25)), (
        "FU-030: radii must be world half-extent per axis × inv_size[i]")
    # the audit-20260610 anisotropy warning is obsolete now — exact fit
    assert not any("non-cubic" in w for w in warnings), (
        "FU-030: the non-cubic SPHERE-source warning must be gone "
        "(per-axis radii make the fit exact)")


def test_fu030_sphere_radii_degenerate_on_cubic_domain(tmp_path):
    """Cubic domain: all three radii equal (consistency with the old
    scalar behaviour in the isotropic case)."""
    scene_dict, _, _, _ = _collect(
        tmp_path, dom_hi=(4.0, 4.0, 4.0),
        src_bbox=((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)))
    radii = scene_dict["fluids"][0]["radii"]
    assert radii[0] == radii[1] == radii[2] == pytest.approx(0.125)


def test_fu030_mesh_scale_translate_per_axis(tmp_path):
    """Non-zero origin + non-cubic domain: scale must equal inv_size per
    axis and translate must be -origin[i]*inv_size[i] so that
    v*scale + translate == (v - origin) * inv_size exactly."""
    scene_dict, _, _, warnings = _collect(
        tmp_path, dom_lo=(1.0, 2.0, 3.0), dom_hi=(5.0, 6.0, 5.0),
        src_bbox=((2.0, 3.0, 3.5), (3.0, 4.0, 4.5)),
        source_type="MESH")
    entry = scene_dict["fluids"][0]
    assert entry["kind"] == "mesh"
    assert entry["scale"] == pytest.approx((0.25, 0.25, 0.5))
    assert entry["translate"] == pytest.approx((-0.25, -0.5, -1.5)), (
        "FU-030: translate must be -origin[i]*inv_size[i] per axis")
    # exactness check at a sample world point (the domain centre):
    w = (3.0, 4.0, 4.0)
    mapped = tuple(w[i] * entry["scale"][i] + entry["translate"][i]
                   for i in range(3))
    assert mapped == pytest.approx((0.5, 0.5, 0.5))
    assert not any("non-cubic" in w_ for w_ in warnings), (
        "FU-030: the non-cubic MESH-source warning must be gone")


# ── TOML writer contract (schema text, not the CLI) ─────────────────────

def _scene_with_fluids(fluids):
    return {
        "domain": {"resolution": (32, 32, 16), "dx": 1 / 32.0,
                   "origin": [0, 0, 0]},
        "fluids": fluids,
        "obstacles": [],
        "simulation": {"dt": 0.005, "frames": 5, "fps": 24,
                       "pressure_iters": 40, "flip_blend": 0.95,
                       "gravity": -9.81},
        "output": {"cache_dir": "/tmp/c", "iso_level": 0.6,
                   "smooth_passes": 2, "particles": False, "preview": False},
    }


def test_fu030_toml_sphere_radii_round_trip():
    toml = build_toml(_scene_with_fluids([
        {"kind": "sphere", "center": (0.5, 0.5, 0.5),
         "radii": (0.125, 0.125, 0.25), "ppc": 8}]))
    parsed = tomllib.loads(toml)
    f = parsed["fluids"][0]
    assert f["type"] == "sphere"
    assert f["radii"] == pytest.approx([0.125, 0.125, 0.25])
    assert "radius" not in f, (
        "FU-030: writer must emit exactly one of radii/radius")


def test_fu030_toml_sphere_scalar_radius_back_compat():
    toml = build_toml(_scene_with_fluids([
        {"kind": "sphere", "center": (0.5, 0.5, 0.5),
         "radius": 0.2, "ppc": 8}]))
    f = tomllib.loads(toml)["fluids"][0]
    assert f["radius"] == pytest.approx(0.2)
    assert "radii" not in f


def test_fu030_toml_mesh_vec3_scale_round_trip():
    toml = build_toml(_scene_with_fluids([
        {"kind": "mesh", "path": "C:\\tmp\\src.obj",
         "scale": (0.25, 0.25, 0.5),
         "translate": (-0.25, -0.5, -1.5), "ppc": 8}]))
    f = tomllib.loads(toml)["fluids"][0]
    assert f["type"] == "mesh"
    assert f["scale"] == pytest.approx([0.25, 0.25, 0.5]), (
        "FU-030: vec3 scale must survive the TOML writer (round-59 class)")
    assert f["translate"] == pytest.approx([-0.25, -0.5, -1.5])


def test_fu030_toml_mesh_scalar_scale_back_compat():
    toml = build_toml(_scene_with_fluids([
        {"kind": "mesh", "path": "/tmp/src.obj", "scale": 0.3,
         "translate": (0, 0, 0), "ppc": 8}]))
    f = tomllib.loads(toml)["fluids"][0]
    assert f["scale"] == pytest.approx(0.3)
