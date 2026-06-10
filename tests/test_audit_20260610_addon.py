"""audit-20260610 — fixes from the independent addon review of branch
fix/round61-zero-frame-honesty.

Covers (one section per fix):
  1. bake.collect_scene cache_dir hoist — UnboundLocalError on any
     source_type=MESH / legacy fill_mesh bake          [behavioural]
  2. render.py strips stale *.png before rendering      [code contract]
  3. render.py mirrors bake's interpreter auto-detect   [code contract]
  4. runner.start_sync suppresses "(sync) finished"
     INFO when on_complete returns False                [behavioural]
  5. SPHERE/MESH fluid sources warn on non-cubic
     domains (mirror of the obstacle warnings)          [behavioural]
  6. inflow temperature default == fluid source default [source contract]
  7. save_userpref guard carries a `# dodge:` comment   [raw source — see note]
  8. render passes use_progress=False (no stuck-0% bar) [code contract]

bpy stub pattern follows tests/test_addon_interpreter_detect.py: stub
`bpy`/`mathutils` plus the bpy-heavy sibling modules, load the REAL
pure modules (domain_transform, scene_validator, cache_sanity, _runner)
and the REAL bake/render files under a private package name.

Contract greps run on comment- and docstring-stripped source (§9.12)
so prose can never satisfy an assert. The ONE exception is fix 7,
which asserts a comment EXISTS — stripping would be self-defeating,
so it greps raw source on purpose.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import re
import sys
import types
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parents[1] / "addon" / "gpufluid_blender"
PKG = "_gfa20260610"   # private stub-package name; no collision with other tests


# ---------------------------------------------------------------------------
# §9.12 helper: source minus comment lines AND docstrings
# ---------------------------------------------------------------------------

def _code_only(path: Path) -> str:
    """Source with full-line comments and all docstrings removed, so a
    contract assert can only match executable code (audit-20260610)."""
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
    kept = [line for i, line in enumerate(src.splitlines(), start=1)
            if i not in doc_lines and not line.lstrip().startswith("#")]
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Stub-loader: real bake/render/_runner with bpy + bpy-heavy siblings stubbed
# ---------------------------------------------------------------------------

def _stub_bpy() -> types.ModuleType:
    bpy = types.ModuleType("bpy")
    bpy.props = types.SimpleNamespace(**{
        n: (lambda *a, **k: None) for n in
        ("StringProperty", "IntProperty", "BoolProperty", "EnumProperty",
         "FloatProperty", "FloatVectorProperty", "PointerProperty")})
    bpy.types = types.SimpleNamespace(
        Operator=object, AddonPreferences=object, PropertyGroup=object,
        Panel=object)
    bpy.utils = types.SimpleNamespace(register_class=lambda c: None,
                                      unregister_class=lambda c: None)
    bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    bpy.context = types.SimpleNamespace()
    bpy.data = types.SimpleNamespace(filepath="", objects={})
    bpy.ops = types.SimpleNamespace()
    return bpy


class _Vec:
    """Minimal mathutils.Vector: .x/.y/.z, iteration, indexing — exactly the
    surface collect_scene touches (§9.7 mock fidelity)."""

    def __init__(self, seq=(0.0, 0.0, 0.0)):
        self.x, self.y, self.z = (float(c) for c in seq)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __getitem__(self, i):
        return (self.x, self.y, self.z)[i]


def _load_real(name: str, relpath: str, package: str):
    spec = importlib.util.spec_from_file_location(f"{PKG}.{name}",
                                                  ADDON / relpath)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    sys.modules[f"{PKG}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_addon() -> dict:
    """Load real bake.py / render.py / _runner.py (+ pure deps) under the
    stub package. Fresh per call so per-test monkeypatching can't leak."""
    sys.modules["bpy"] = _stub_bpy()
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = _Vec
    sys.modules["mathutils"] = mathutils

    pkg = types.ModuleType(PKG)
    pkg.ADDON_PKG = "gpufluid_blender"
    pkg.__path__ = []
    pkg.logger = logging.getLogger("audit20260610")
    sys.modules[PKG] = pkg
    ops_pkg = types.ModuleType(f"{PKG}.operators")
    ops_pkg.__path__ = []
    sys.modules[f"{PKG}.operators"] = ops_pkg

    # bpy-heavy siblings -> stubs (bake/render only need these names)
    cfg = types.ModuleType(f"{PKG}.config_builder")
    cfg.build_toml = lambda d: ""
    prefs = types.ModuleType(f"{PKG}.preferences")
    prefs.get_prefs = lambda ctx: types.SimpleNamespace(interpreter_path="")
    prefs._detect_interpreter = lambda roots=None: ""
    prefs._context_roots = lambda: []
    anim = types.ModuleType(f"{PKG}.operators._animation")
    anim._bbox_world = lambda o: o._bbox
    anim._bbox_world_at_frame = lambda o, f: o._bbox
    anim._is_animated = lambda o, ctx: False
    coll = types.ModuleType(f"{PKG}.operators._collect")
    coll._export_obj = lambda o, p: None
    coll._output_dict = lambda dp: {"cache_dir": dp.cache_dir}
    coll._safe_obj_name = lambda n: n
    helpers = types.ModuleType(f"{PKG}.operators.helpers")
    helpers.subprocess_drain = lambda *a, **k: None
    for m in (cfg, prefs, anim, coll, helpers):
        sys.modules[m.__name__] = m

    # real pure modules
    _load_real("domain_transform", "domain_transform.py", PKG)
    _load_real("scene_validator", "scene_validator.py", PKG)
    _load_real("cache_sanity", "cache_sanity.py", PKG)
    runner = _load_real("operators._runner", "operators/_runner.py",
                        f"{PKG}.operators")
    bake = _load_real("operators.bake", "operators/bake.py",
                      f"{PKG}.operators")
    render = _load_real("operators.render", "operators/render.py",
                        f"{PKG}.operators")
    return {"bake": bake, "render": render, "_runner": runner}


def _fake_obj(name, bbox, *, fluid=None):
    o = types.SimpleNamespace(name=name)
    o._bbox = bbox
    o.gpufluid_fluid = fluid or types.SimpleNamespace(is_fluid=False)
    o.gpufluid_inflow = types.SimpleNamespace(is_inflow=False)
    o.gpufluid_obstacle = types.SimpleNamespace(is_obstacle=False)
    o.gpufluid_outflow = types.SimpleNamespace(is_outflow=False)
    return o


def _domain_props(cache_dir, resolution=32, frames=10):
    return types.SimpleNamespace(
        is_domain=True, cache_dir=str(cache_dir), resolution=resolution,
        solver="mpm", dt=0.005, frames=frames, fps=24,
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


def _collect(mods, tmp_path, *, dom_hi=(1.0, 1.0, 1.0), source_type="BBOX",
             fill_mesh=False, exported=None):
    bake = mods["bake"]
    if exported is not None:
        bake._export_obj = lambda o, p: exported.append((o.name, p))
    dom = _fake_obj("Domain", ((0.0, 0.0, 0.0), dom_hi))
    dom.gpufluid_domain = _domain_props(tmp_path / "cache")
    src = _fake_obj("Suzanne", ((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)),
                    fluid=types.SimpleNamespace(
                        is_fluid=True, ppc=8, source_type=source_type,
                        fill_mesh=fill_mesh))
    ctx = types.SimpleNamespace(
        scene=types.SimpleNamespace(objects=[dom, src]))
    return bake.collect_scene(ctx, dom)


# ---------------------------------------------------------------------------
# Fix 1 — cache_dir hoisted above the fluid-sources loop (HIGH)
# ---------------------------------------------------------------------------

def test_mesh_source_collect_scene_no_unboundlocal(tmp_path):
    mods = _load_addon()
    exported = []
    try:
        scene_dict, _, _, _ = _collect(mods, tmp_path, source_type="MESH",
                                       exported=exported)
    except UnboundLocalError as e:  # pragma: no cover - regression signal
        pytest.fail(
            f"audit-20260610 fix-1 regressed: cache_dir must be assigned "
            f"BEFORE the fluid-sources loop or every source_type=MESH bake "
            f"crashes with UnboundLocalError ({e})")
    entry = scene_dict["fluids"][0]
    assert entry["kind"] == "mesh", \
        "audit-20260610: MESH source must emit kind='mesh'"
    assert str(tmp_path / "cache") in entry["path"], \
        "audit-20260610: mesh .obj must be exported into the cache_dir"
    assert exported == [("Suzanne", entry["path"])], \
        "audit-20260610: _export_obj must be called for the MESH source"


def test_legacy_fill_mesh_promotion_no_crash(tmp_path):
    """Old .blends with fill_mesh=True used to bake as a box; the S2.17.10
    BBOX->MESH promotion made them hit the same UnboundLocalError."""
    mods = _load_addon()
    scene_dict, _, _, _ = _collect(mods, tmp_path, source_type="BBOX",
                                   fill_mesh=True, exported=[])
    assert scene_dict["fluids"][0]["kind"] == "mesh", \
        "audit-20260610: legacy fill_mesh=True must promote to kind='mesh'"


# ---------------------------------------------------------------------------
# Fix 5 — SPHERE/MESH sources warn on non-cubic domains (§9.6 vs obstacles)
# ---------------------------------------------------------------------------

def test_sphere_source_warns_on_noncubic_domain(tmp_path):
    mods = _load_addon()
    _, _, _, warnings = _collect(mods, tmp_path, dom_hi=(2.0, 1.0, 1.0),
                                 source_type="SPHERE")
    assert any("SPHERE" in w and "non-cubic" in w for w in warnings), \
        ("audit-20260610 fix-5 regressed: SPHERE fluid source on a "
         "non-cubic domain must warn (obstacles do; sources drifted §9.6)")


def test_mesh_source_warns_on_noncubic_domain(tmp_path):
    mods = _load_addon()
    _, _, _, warnings = _collect(mods, tmp_path, dom_hi=(2.0, 1.0, 1.0),
                                 source_type="MESH", exported=[])
    assert any("MESH" in w and "non-cubic" in w for w in warnings), \
        ("audit-20260610 fix-5 regressed: MESH fluid source on a non-cubic "
         "domain must warn about uniform single-scalar scaling")


def test_no_anisotropy_warning_on_cubic_domain(tmp_path):
    """False-positive guard: a cubic domain must NOT trigger the new
    source-anisotropy warnings."""
    mods = _load_addon()
    _, _, _, warnings = _collect(mods, tmp_path, dom_hi=(1.0, 1.0, 1.0),
                                 source_type="SPHERE")
    assert not any("non-cubic" in w for w in warnings), \
        "audit-20260610: cubic domain must not warn about anisotropy"


# ---------------------------------------------------------------------------
# Fix 4 — sync "(sync) finished" INFO suppressed after a 0-output ERROR
# ---------------------------------------------------------------------------

class _FakeOp:
    _is_running = False

    def __init__(self):
        self.reports = []

    def report(self, level_set, msg):
        self.reports.append((set(level_set), msg))


def _run_sync(mods, on_complete):
    runner = mods["_runner"].ModalSubprocessRunner("bake")
    op = _FakeOp()
    res = runner.start_sync(op, [sys.executable, "-c", "print('ok')"], 60,
                            log_prefix="bake", on_complete=on_complete)
    return res, op.reports


def test_sync_finished_info_suppressed_when_callback_returns_false():
    mods = _load_addon()
    res, reports = _run_sync(mods, lambda: False)
    assert res == {"FINISHED"}
    assert not any("(sync) finished" in msg for _, msg in reports), \
        ("audit-20260610 fix-4 regressed: on_complete returning False (it "
         "already reported the 0-output ERROR) must veto the unconditional "
         "'(sync) finished' INFO — the log used to end on a success line")


def test_sync_finished_info_kept_for_legacy_none_callback():
    mods = _load_addon()
    res, reports = _run_sync(mods, lambda: None)
    assert res == {"FINISHED"}
    assert any("(sync) finished" in msg for _, msg in reports), \
        ("audit-20260610: legacy on_complete returning None must keep the "
         "'(sync) finished' INFO (only an explicit False suppresses)")


def test_bake_sync_callback_returns_false_on_zero_frames(tmp_path):
    """Mirror check, bake side: _sync_post_bake_then_attach must return
    False (veto) on a 0-frame cache and True when frames exist."""
    mods = _load_addon()
    op = mods["bake"].GPUFLUID_OT_bake()
    op.reports = []
    op.report = lambda lv, msg: op.reports.append((set(lv), msg))
    op._cache_dir_str = str(tmp_path)          # empty -> 0 mesh frames
    op._toml_str_snapshot = ""
    assert op._sync_post_bake_then_attach() is False, \
        "audit-20260610 fix-4: 0-frame bake callback must return False"
    assert any("ERROR" in lv for lv, _ in op.reports), \
        "audit-20260610: 0-frame bake must still report ERROR"
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    (mesh / "frame_0000.ply").write_text("")
    assert op._sync_post_bake_then_attach() is True, \
        "audit-20260610 fix-4: bake callback must return True when frames exist"


def test_render_sync_callback_returns_false_on_zero_pngs(tmp_path):
    """Mirror check, render side: _sync_post_render_report must return
    False (veto) on 0 PNGs and True when PNGs exist."""
    mods = _load_addon()
    op = mods["render"].GPUFLUID_OT_render()
    op.reports = []
    op.report = lambda lv, msg: op.reports.append((set(lv), msg))
    op._out_dir = str(tmp_path)                # empty -> 0 PNGs
    assert op._sync_post_render_report() is False, \
        "audit-20260610 fix-4: 0-PNG render callback must return False"
    assert any("ERROR" in lv for lv, _ in op.reports), \
        "audit-20260610: 0-PNG render must still report ERROR"
    (tmp_path / "f_0001.png").write_text("")
    assert op._sync_post_render_report() is True, \
        "audit-20260610 fix-4: render callback must return True with PNGs"


# ---------------------------------------------------------------------------
# Fix 2 — render strips stale *.png before rendering (code contract)
# ---------------------------------------------------------------------------

def test_render_strips_stale_pngs_before_run():
    code = _code_only(ADDON / "operators" / "render.py")
    assert 'out_dir.glob("*.png")' in code and ".unlink()" in code, \
        ("audit-20260610 fix-2 regressed: render.execute must delete stale "
         "*.png from out_dir before spawning — otherwise the round-61 "
         "post-render honesty count reports a previous run's frames as "
         "fresh success")


# ---------------------------------------------------------------------------
# Fix 3 — render mirrors bake's interpreter auto-detect (code contract, §9.6)
# ---------------------------------------------------------------------------

def test_render_mirrors_interpreter_autodetect():
    for fname in ("bake.py", "render.py"):
        code = _code_only(ADDON / "operators" / fname)
        assert "_detect_interpreter(_context_roots())" in code, \
            (f"audit-20260610 fix-3 regressed in {fname}: both mirror "
             f"operators must run the last-chance interpreter auto-detect "
             f"(commit c5834f1 added it to bake only — §9.6 drift)")


# ---------------------------------------------------------------------------
# Fix 6 — inflow temperature default == fluid source default (source contract)
# ---------------------------------------------------------------------------

def _temperature_default(code: str, cls_name: str) -> float:
    block = re.search(rf"class {cls_name}\b.*?(?=\nclass |\Z)", code,
                      re.DOTALL).group(0)
    i = block.index('name="Temperature"')
    return float(re.search(r"default=([0-9.]+)", block[i:]).group(1))


def test_inflow_temperature_default_matches_fluid_source():
    code = _code_only(ADDON / "properties.py")
    t_fluid = _temperature_default(code, "GpufluidFluidProps")
    t_inflow = _temperature_default(code, "GpufluidInflowProps")
    assert t_inflow == t_fluid == 300.0, \
        (f"audit-20260610 fix-6 regressed: inflow temperature default "
         f"({t_inflow}) must equal the Fluid source default ({t_fluid}) — "
         f"the description claims they 'mix coherently', so the numbers "
         f"must actually agree")


# ---------------------------------------------------------------------------
# Fix 7 — save_userpref guard has a `# dodge:` comment (§9.11)
# NOTE: deliberately greps RAW source — the contract is that a COMMENT
# exists, so stripping comments first would be self-defeating.
# ---------------------------------------------------------------------------

def test_save_userpref_guard_has_dodge_comment():
    for fname in ("bake.py", "render.py"):
        raw = (ADDON / "operators" / fname).read_text(encoding="utf-8")
        i = raw.index("save_userpref()")
        assert "# dodge:" in raw[i:i + 700], \
            (f"audit-20260610 fix-7 regressed in {fname}: the bare "
             f"try/except around save_userpref() must carry a '# dodge:' "
             f"comment (§9.11 — unjustified guards invert intent)")


# ---------------------------------------------------------------------------
# Fix 8 — render opts out of the round-62 progress bar (code contract)
# ---------------------------------------------------------------------------

def test_render_disables_stuck_progress_bar():
    runner_code = _code_only(ADDON / "operators" / "_runner.py")
    assert "use_progress: bool = True" in runner_code, \
        ("audit-20260610 fix-8 regressed: start_modal must accept "
         "use_progress so callers without a parsable N/M frame line can "
         "opt out of the progress bar")
    render_code = _code_only(ADDON / "operators" / "render.py")
    assert "use_progress=False" in render_code, \
        ("audit-20260610 fix-8 regressed: render must pass "
         "use_progress=False — headless Blender prints 'Fra:N' with no "
         "total, so the bar would sit at 0% forever (reads as hung)")
