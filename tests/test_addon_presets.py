"""2026-06-15 — addon usability: one-click physics presets + safe defaults +
soft ranges + bilingual hint language.

The preset *data* (PRESETS / HINT_TEXT) is loaded under a minimal bpy stub and
asserted directly. The property defaults / soft ranges / panel wiring are
§9.12 source contracts (comment/docstring-stripped) since they live in
bpy-dependent files.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "addon" / "gpufluid_blender"


def _code(path: Path) -> str:
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
    return "\n".join(l for i, l in enumerate(src.splitlines(), 1)
                     if i not in doc and not l.lstrip().startswith("#"))


def _load_presets():
    """Load operators/presets.py under a minimal bpy stub to read its data."""
    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(Operator=type("Operator", (), {}))
    bpy.props = types.SimpleNamespace(
        EnumProperty=lambda *a, **k: ("PROP", a, k),
        BoolProperty=lambda *a, **k: ("PROP", a, k))
    sys.modules["bpy"] = bpy
    spec = importlib.util.spec_from_file_location(
        "_gpufluid_presets_under_test", ADDON / "operators" / "presets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRESETS_MOD = _load_presets()


# ── preset data ────────────────────────────────────────────────────────────

_FLUID_PRESETS = {"water", "splash", "honey_slump", "draft"}
_VE_PRESETS = {"honey_coil", "clay", "slime", "chocolate", "caramel", "gel"}


def test_material_library_presets_exist():
    assert set(PRESETS_MOD.PRESETS) == _FLUID_PRESETS | _VE_PRESETS


def test_every_preset_enables_cfl_and_uses_mpm():
    # The whole point: a non-expert picks a preset and the bake doesn't NaN.
    for key, vals in PRESETS_MOD.PRESETS.items():
        assert vals["use_cfl"] is True, f"preset {key} must enable CFL substepping"
        assert vals["solver"] == "mpm", f"preset {key} should target MPM"
        assert 8 <= vals["resolution"] <= 512
        assert 1e-5 <= vals["dt"] <= 0.1
        assert vals["mpm_material"] in ("fluid", "viscoelastic")
        if vals["mpm_material"] == "fluid":
            assert 100.0 <= vals["mpm_bulk_modulus"] <= 10000.0
        else:
            assert 100.0 <= vals["mpm_young_modulus"] <= 2_000_000.0
            assert vals["mpm_yield_stress"] >= 0.0


def test_viscoelastic_presets_coil_setup():
    """Every coiling material must turn the floor sticky AND set a slow,
    wobbling pour on inflows (the geometry coiling needs) — else it won't coil."""
    for key in _VE_PRESETS:
        v = PRESETS_MOD.PRESETS[key]
        assert v["mpm_material"] == "viscoelastic"
        assert v["mpm_floor_friction"] > 0.0, f"{key}: needs a (semi-)sticky floor"
        inflow = v.get("_inflow")
        assert inflow is not None, f"{key}: must set a coiling pour on inflows"
        assert inflow["velocity_z"] < 0.0, f"{key}: pour must be downward"
        assert "velocity_jitter" in inflow, f"{key}: must seed the coil"


def test_fluid_presets_have_no_inflow_override():
    # Plain fluids leave the user's inflow velocity alone.
    for key in _FLUID_PRESETS:
        assert "_inflow" not in PRESETS_MOD.PRESETS[key]


def test_preset_enum_items_match_preset_keys():
    enum_keys = {k for k, _label, _desc in PRESETS_MOD.PRESET_ENUM_ITEMS}
    assert enum_keys == set(PRESETS_MOD.PRESETS), \
        "every dropdown item must have a backing preset table (and vice-versa)"


def test_hint_text_is_bilingual_and_nonempty():
    assert set(PRESETS_MOD.HINT_TEXT) == {"ru", "en"}
    assert PRESETS_MOD.HINT_TEXT["ru"] and PRESETS_MOD.HINT_TEXT["en"]
    # same number of lines so neither language silently drops a hint
    assert len(PRESETS_MOD.HINT_TEXT["ru"]) == len(PRESETS_MOD.HINT_TEXT["en"])


def test_operator_registered_in_init():
    code = _code(ADDON / "__init__.py")
    assert "op_presets.GPUFLUID_OT_apply_preset" in code, \
        "the preset operator must be registered in _collect_classes"
    assert "panels.GPUFLUID_PT_help" in code, \
        "the help panel must be registered"


# ── safe defaults ──────────────────────────────────────────────────────────

def test_use_cfl_defaults_on():
    code = _code(ADDON / "properties.py")
    assert 'BoolProperty(name="CFL Substepping", default=True' in code, \
        "safe default: CFL substepping must be ON by default (no OOTB NaN)"
    assert 'default=False' not in code.split('name="CFL Substepping"')[1][:80], \
        "the CFL default must not have regressed to False"


# ── soft ranges on the stability knobs ─────────────────────────────────────

def test_key_knobs_carry_soft_ranges():
    code = _code(ADDON / "properties.py")
    # Each of these must keep the SLIDER in a sane band while leaving the hard
    # min/max wide for typed over-range values.
    for needle in (
        "soft_min=32, soft_max=192",          # resolution
        "soft_min=0.002, soft_max=0.008",     # dt
        "soft_min=0.2, soft_max=1.0",         # cfl_factor (<=1.0 for MPM)
        "soft_min=500.0, soft_max=2000.0",    # bulk modulus
    ):
        assert needle in code, f"missing soft range: {needle}"


# ── panel + language wiring ────────────────────────────────────────────────

def test_panel_exposes_preset_dropdown():
    code = _code(ADDON / "panels.py")
    assert 'operator_menu_enum("gpufluid.apply_preset", "preset"' in code, \
        "the Domain panel must expose the preset dropdown"


def test_hint_language_preference_exists():
    code = _code(ADDON / "preferences.py")
    assert "hint_language" in code
    assert '("ru", "Русский"' in code and '("en", "English"' in code, \
        "the hint-language preference must offer Russian and English"
