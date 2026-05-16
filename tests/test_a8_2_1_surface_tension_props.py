"""B1.1 — verify A8.2.1 GpufluidSurfaceTensionGroup exists on Domain props.

bpy is unavailable in pytest, so we install a minimal stub before importing
addon.properties. The stub captures every `bpy.props.X(...)` call and the
PropertyGroup class registration, so we can introspect what was declared.

This test guards three things that, if missed, would silently break the
addon UI for surface tension:
  1. The class GpufluidSurfaceTensionGroup is declared with surface_tension
     (float) and csf_smoothing_passes (int) annotations.
  2. GpufluidDomainProps gets a PointerProperty(type=GpufluidSurfaceTensionGroup)
     under the name 'surface_tension_group'.
  3. The inner group is registered BEFORE the domain group in __init__._CLASSES
     (Blender requires referenced PropertyGroups to register first).
"""
import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_bpy_stub():
    """Build a minimal bpy stub that records prop declarations."""
    bpy = types.ModuleType("bpy")
    bpy.types = types.ModuleType("bpy.types")
    bpy.props = types.ModuleType("bpy.props")

    class PropertyGroup:
        pass

    bpy.types.PropertyGroup = PropertyGroup
    bpy.types.Object = type("Object", (), {})

    # Each props factory returns a tagged tuple so tests can introspect it.
    def _factory(kind):
        def make(**kwargs):
            return ("prop", kind, kwargs)
        return make

    for name in ("BoolProperty", "IntProperty", "FloatProperty", "StringProperty",
                 "EnumProperty", "PointerProperty", "FloatVectorProperty"):
        setattr(bpy.props, name, _factory(name))

    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy.types
    sys.modules["bpy.props"] = bpy.props
    return bpy


def _load_properties_module():
    _install_bpy_stub()
    addon_dir = Path(__file__).resolve().parents[1] / "addon" / "gpufluid_blender"
    spec = importlib.util.spec_from_file_location(
        "_gpufluid_addon_properties", addon_dir / "properties.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a8_2_1_surface_tension_group_has_required_fields():
    props = _load_properties_module()
    cls = getattr(props, "GpufluidSurfaceTensionGroup", None)
    assert cls is not None, "GpufluidSurfaceTensionGroup missing from properties.py"

    # Class-level annotations are where bpy props live.
    declared = dict(getattr(cls, "__annotations__", {}))
    assert "surface_tension" in declared, "missing 'surface_tension' float"
    assert "csf_smoothing_passes" in declared, "missing 'csf_smoothing_passes' int"

    # Field kinds — guard against accidental type swap.
    assert declared["surface_tension"][1] == "FloatProperty"
    assert declared["csf_smoothing_passes"][1] == "IntProperty"

    # Default values must match CLI defaults (cli/config.py:175-176).
    assert declared["surface_tension"][2]["default"] == 0.0
    assert declared["csf_smoothing_passes"][2]["default"] == 2
    # σ must be non-negative.
    assert declared["surface_tension"][2].get("min", 0) >= 0


def test_a8_2_1_domain_has_pointer_to_surface_tension_group():
    props = _load_properties_module()
    domain = props.GpufluidDomainProps
    declared = dict(getattr(domain, "__annotations__", {}))
    assert "surface_tension_group" in declared, \
        "GpufluidDomainProps missing 'surface_tension_group' PointerProperty"
    kind = declared["surface_tension_group"]
    assert kind[1] == "PointerProperty"
    assert kind[2]["type"] is props.GpufluidSurfaceTensionGroup


def test_a8_2_1_registered_before_domain_in_init():
    """Inner group must register before Domain (Blender ordering requirement)."""
    _install_bpy_stub()
    src = (Path(__file__).resolve().parents[1] /
           "addon" / "gpufluid_blender" / "__init__.py").read_text(encoding="utf-8")
    i_inner = src.find("GpufluidSurfaceTensionGroup")
    i_domain = src.find("GpufluidDomainProps")
    assert i_inner != -1, "GpufluidSurfaceTensionGroup not registered in __init__._CLASSES"
    assert i_inner < i_domain, (
        "GpufluidSurfaceTensionGroup must appear before GpufluidDomainProps in _CLASSES "
        "— Blender rejects PointerProperty(type=X) if X isn't registered yet"
    )


# ---- B1.2 — per-fluid-source colour fields on GpufluidFluidProps -----------

def test_a8_3_1_fluid_has_color_and_use_color():
    props = _load_properties_module()
    fluid = props.GpufluidFluidProps
    declared = dict(getattr(fluid, "__annotations__", {}))
    assert "use_color" in declared, "FluidProps missing 'use_color' toggle"
    assert "color" in declared, "FluidProps missing 'color' RGB vector"
    assert declared["use_color"][1] == "BoolProperty"
    assert declared["color"][1] == "FloatVectorProperty"
    cargs = declared["color"][2]
    assert cargs.get("size") == 3
    assert cargs.get("subtype") == "COLOR"
    assert cargs.get("min", 0) >= 0.0 and cargs.get("max", 1) <= 1.0
    assert tuple(cargs.get("default")) == (1.0, 1.0, 1.0)
