"""§9.13 live-MCP UX-defaults fixes (2026-06-11, default-scene smoke).

Live evidence: add_domain made an invisible 1×1×1 m Empty buried inside
the default 2 m Cube (zero feedback); that Cube marked obstacle covered
100% of the domain with no warning; a UV-sphere marked fluid seeded a
CUBE of water (source_type=BBOX default). Result: frozen nonsense,
every operator reporting plain FINISHED.

Covers (behavioural, established bpy-stub patterns):
  1. add_domain sizes/places the Empty around the selection (+~10%/side
     margin), floor-resting 2 m cube fallback, INFO report, show_in_front.
  2. mark_fluid auto-promotes non-box meshes (verts != 8) to
     Source Shape=MESH; 8-vert boxes keep BBOX; explicit SPHERE untouched.
  3. collect_scene warns when an obstacle covers ≥90% of the domain.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "addon" / "gpufluid_blender"
PKG = "_gfux20260611"


# ── stub loaders ─────────────────────────────────────────────────────────

class _Vec:
    def __init__(self, seq=(0.0, 0.0, 0.0)):
        self.x, self.y, self.z = (float(c) for c in seq)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __getitem__(self, i):
        return (self.x, self.y, self.z)[i]


def _stub_bpy():
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
    return bpy


def _load(modname: str, relpath: str, *, extra_stubs=()):
    """Load an addon operators/* file under the stub package; returns
    (module, bpy_stub)."""
    bpy = _stub_bpy()
    sys.modules["bpy"] = bpy
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = _Vec
    sys.modules["mathutils"] = mathutils

    pkg = types.ModuleType(PKG)
    pkg.__path__ = []
    pkg.logger = logging.getLogger("ux20260611")
    sys.modules[PKG] = pkg
    ops_pkg = types.ModuleType(f"{PKG}.operators")
    ops_pkg.__path__ = []
    sys.modules[f"{PKG}.operators"] = ops_pkg

    base_stubs = {
        f"{PKG}.operators._animation": {
            "_bbox_world": lambda o: o._bbox,
            "_bbox_world_at_frame": lambda o, f: o._bbox,
            "_is_animated": lambda o, c: False},
        f"{PKG}.config_builder": {"build_toml": lambda d: ""},
        f"{PKG}.preferences": {
            "get_prefs": lambda c: None,
            "_detect_interpreter": lambda r=None: "",
            "_context_roots": lambda: []},
        f"{PKG}.operators._collect": {
            "_export_obj": lambda o, p: None,
            "_output_dict": lambda dp: {"cache_dir": dp.cache_dir},
            "_safe_obj_name": lambda n: n},
        f"{PKG}.operators.helpers": {"subprocess_drain": lambda *a, **k: None},
        f"{PKG}.operators._runner": {"ModalSubprocessRunner": object},
    }
    for name, attrs in dict(base_stubs, **dict(extra_stubs)).items():
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

    spec = importlib.util.spec_from_file_location(f"{PKG}.{modname}",
                                                  ADDON / relpath)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = f"{PKG}.operators"
    sys.modules[f"{PKG}.{modname}"] = mod
    spec.loader.exec_module(mod)
    return mod, bpy


def _roles():
    return dict(
        gpufluid_fluid=types.SimpleNamespace(is_fluid=False,
                                             source_type="BBOX"),
        gpufluid_obstacle=types.SimpleNamespace(is_obstacle=False,
                                                obstacle_type="BBOX"),
        gpufluid_inflow=types.SimpleNamespace(is_inflow=False),
        gpufluid_outflow=types.SimpleNamespace(is_outflow=False),
    )


def _fake_obj(name, bbox=((0, 0, 0), (1, 1, 1)), **over):
    o = types.SimpleNamespace(name=name, _bbox=bbox,
                              gpufluid_domain=types.SimpleNamespace(
                                  is_domain=False, cache_dir=""),
                              **_roles())
    for k, v in over.items():
        setattr(o, k, v)
    return o


class _Op:
    """Mix-in: instantiate an operator class from the loaded module with a
    report recorder."""

    @staticmethod
    def make(cls):
        op = cls()
        op.reports = []
        op.report = lambda lv, msg: op.reports.append((set(lv), msg))
        return op


# ── 1. add_domain sizing + feedback ─────────────────────────────────────

def _new_domain_ctx(bpy, selected):
    """Wire bpy.ops.object.empty_add to create a fake Empty and make it
    the active object (what the real operator does)."""
    ctx = types.SimpleNamespace(
        selected_objects=selected,
        scene=types.SimpleNamespace(
            cursor=types.SimpleNamespace(location=(3.0, 4.0, 9.9))),
        active_object=None)

    def _empty_add(type="CUBE", radius=0.5, location=(0, 0, 0)):
        ctx.active_object = _fake_obj(
            "Empty", location=tuple(location), scale=(1.0, 1.0, 1.0),
            empty_display_size=radius, show_in_front=False, type="EMPTY")
        return {"FINISHED"}

    bpy.ops.object = types.SimpleNamespace(empty_add=_empty_add)
    return ctx


def test_add_domain_encloses_selection_with_margin():
    helpers, bpy = _load("operators.helpers", "operators/helpers.py")
    cube = _fake_obj("Cube", bbox=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    ctx = _new_domain_ctx(bpy, [cube])
    op = _Op.make(helpers.GPUFLUID_OT_add_domain)
    assert op.execute(ctx) == {"FINISHED"}
    dom = ctx.active_object
    # 2 m cube + 10% margin per side → 2.4 m extent, centred on the cube.
    assert dom.scale == pytest.approx((2.4, 2.4, 2.4)), (
        "§9.13: domain must enclose the selection bbox with ~10% margin")
    assert tuple(dom.location) == pytest.approx((0.0, 0.0, 0.0))
    assert dom.show_in_front is True, (
        "§9.13: domain wireframe must draw in front (was invisible inside "
        "the default Cube)")
    assert dom.gpufluid_domain.is_domain is True
    infos = [m for lv, m in op.reports if "INFO" in lv]
    assert any("2.40" in m and "must be inside" in m for m in infos), (
        "§9.13: add_domain must report the created size (was silent)")


def test_add_domain_skips_existing_domains_in_selection():
    helpers, bpy = _load("operators.helpers", "operators/helpers.py")
    old_dom = _fake_obj("OldDomain", bbox=((-9, -9, -9), (9, 9, 9)))
    old_dom.gpufluid_domain.is_domain = True
    cube = _fake_obj("Cube", bbox=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    ctx = _new_domain_ctx(bpy, [old_dom, cube])
    op = _Op.make(helpers.GPUFLUID_OT_add_domain)
    op.execute(ctx)
    assert ctx.active_object.scale == pytest.approx((1.2, 1.2, 1.2)), (
        "§9.13: a pre-existing domain Empty in the selection must not "
        "inflate the new domain (frames are not content)")


def test_add_domain_no_selection_floor_cube_at_cursor():
    helpers, bpy = _load("operators.helpers", "operators/helpers.py")
    ctx = _new_domain_ctx(bpy, [])
    op = _Op.make(helpers.GPUFLUID_OT_add_domain)
    op.execute(ctx)
    dom = ctx.active_object
    assert dom.scale == pytest.approx((2.0, 2.0, 2.0))
    # x/y at the 3D cursor; z rests ON the floor: z ∈ [0, 2] → centre 1.0
    # (cursor z is deliberately ignored).
    assert tuple(dom.location) == pytest.approx((3.0, 4.0, 1.0)), (
        "§9.13: default domain must rest on z=0 at the 3D cursor")
    assert any("2.00" in m for lv, m in op.reports if "INFO" in lv)


def test_add_domain_flat_selection_gets_padded_thickness():
    """dodge-guard check: a flat plane selection (zero z extent) must not
    produce a zero-thickness domain (DomainTransform rejects it)."""
    helpers, bpy = _load("operators.helpers", "operators/helpers.py")
    plane = _fake_obj("Plane", bbox=((0.0, 0.0, 0.5), (2.0, 2.0, 0.5)))
    ctx = _new_domain_ctx(bpy, [plane])
    op = _Op.make(helpers.GPUFLUID_OT_add_domain)
    op.execute(ctx)
    assert ctx.active_object.scale[2] > 0.0, (
        "§9.13: degenerate axis must be padded (10% of largest extent)")
    assert ctx.active_object.scale[2] == pytest.approx(0.4)


# ── 2. mark_fluid auto-promote ──────────────────────────────────────────

def _mesh_obj(n_verts):
    return _fake_obj("Src", data=types.SimpleNamespace(
        vertices=[0] * n_verts))


def test_mark_fluid_box_keeps_bbox():
    helpers, _ = _load("operators.helpers", "operators/helpers.py")
    obj = _mesh_obj(8)
    op = _Op.make(helpers.GPUFLUID_OT_mark_fluid)
    op.execute(types.SimpleNamespace(active_object=obj))
    assert obj.gpufluid_fluid.is_fluid is True
    assert obj.gpufluid_fluid.source_type == "BBOX", (
        "§9.13: an 8-vert box must keep the fast BBOX source")
    assert not op.reports


def test_mark_fluid_sphere_auto_promotes_to_mesh():
    helpers, _ = _load("operators.helpers", "operators/helpers.py")
    obj = _mesh_obj(482)  # default UV sphere vert count
    op = _Op.make(helpers.GPUFLUID_OT_mark_fluid)
    op.execute(types.SimpleNamespace(active_object=obj))
    assert obj.gpufluid_fluid.source_type == "MESH", (
        "§9.13 regressed: non-box source must auto-promote to MESH "
        "(live-MCP: UV-sphere source seeded a CUBE of water)")
    assert any("Mesh Volume" in m for lv, m in op.reports if "INFO" in lv), (
        "§9.13: the auto-promote must be reported, not silent")


def test_mark_fluid_explicit_sphere_choice_untouched():
    helpers, _ = _load("operators.helpers", "operators/helpers.py")
    obj = _mesh_obj(482)
    obj.gpufluid_fluid.source_type = "SPHERE"
    op = _Op.make(helpers.GPUFLUID_OT_mark_fluid)
    op.execute(types.SimpleNamespace(active_object=obj))
    assert obj.gpufluid_fluid.source_type == "SPHERE", (
        "§9.13: auto-promote must only upgrade the BBOX default, never "
        "override an explicit user choice")


# ── 3. full-domain obstacle warning (collect_scene behavioural) ─────────

def _domain_props(cache_dir):
    return types.SimpleNamespace(
        is_domain=True, cache_dir=str(cache_dir), resolution=32,
        solver="flip",  # dodge the MPM OBB path (needs full matrix stubs)
        dt=0.005, frames=10, fps=24,
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


def _collect_with_obstacle(tmp_path, obstacle_bbox):
    bake, _ = _load("operators.bake", "operators/bake.py")
    dom = _fake_obj("Domain", bbox=((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)))
    dom.gpufluid_domain = _domain_props(tmp_path / "cache")
    src = _fake_obj("Water", bbox=((0.2, 0.2, 1.0), (0.8, 0.8, 1.6)))
    src.gpufluid_fluid = types.SimpleNamespace(is_fluid=True, ppc=8,
                                               source_type="BBOX",
                                               fill_mesh=False)
    obs = _fake_obj("Cube", bbox=obstacle_bbox,
                    type="EMPTY", rotation_euler=(0.0, 0.0, 0.0))
    obs.gpufluid_obstacle = types.SimpleNamespace(is_obstacle=True,
                                                  obstacle_type="BBOX",
                                                  motion_type="NONE")
    ctx = types.SimpleNamespace(
        scene=types.SimpleNamespace(objects=[dom, src, obs]))
    return bake.collect_scene(ctx, dom)


def test_full_domain_obstacle_warns(tmp_path):
    _, _, _, warnings = _collect_with_obstacle(
        tmp_path, ((-0.1, -0.1, -0.1), (2.1, 2.1, 2.1)))
    assert any("covers ~the whole domain" in w and "Cube" in w
               for w in warnings), (
        "§9.13 regressed: an obstacle engulfing the domain must warn "
        "(live-MCP: default Cube swallowed the domain, bake 'succeeded' "
        "into frozen nonsense with zero feedback)")


def test_floor_obstacle_does_not_warn(tmp_path):
    """False-positive guard: a legitimate floor slab (10% of the domain)
    must not trip the full-domain warning."""
    _, _, _, warnings = _collect_with_obstacle(
        tmp_path, ((0.0, 0.0, 0.0), (2.0, 2.0, 0.2)))
    assert not any("covers ~the whole domain" in w for w in warnings)
