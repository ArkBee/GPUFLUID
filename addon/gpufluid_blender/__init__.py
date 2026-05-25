"""[BLK A8.1] gpufluid Blender addon — registration root.

Block index for this package:
    A8.1  addon register/unregister (this file)
    A8.2  GpufluidDomain property group       — properties.py
    A8.3  GpufluidFluid property group        — properties.py
    A8.4  GpufluidObstacle property group     — properties.py
    A8.5  Bake operator                       — operators/bake.py
    A8.6  Cache import (PLY/frame handler)    — cache_loader/ (package)
    A8.7  UI panels                           — panels.py
    A8.8  Helper operators                    — operators/helpers.py
"""
import logging

# Module logger. Phase 1: minimal setup with NullHandler so library-style
# imports stay silent unless the host configures handlers. Phase 4 migrated
# all addon-side print() calls to logger.info / logger.warning.
logger = logging.getLogger("gpufluid.addon")
logger.addHandler(logging.NullHandler())


bl_info = {
    "name": "gpufluid",
    "author": "gpufluid contributors",
    "version": (0, 8, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > GpuFluid",
    "description": "GPU FLIP fluid simulator (NVIDIA Warp)",
    "category": "Physics",
}

# When loaded outside Blender (e.g. pytest of bpy-free submodules like
# `render_bridge`), `bpy` is missing. The bpy-dependent submodules are
# only needed when Blender registers the addon, so defer those imports.
try:
    import bpy  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — running headless / under pytest
    bpy = None  # type: ignore[assignment]

if bpy is not None:
    from . import properties
    from . import preferences
    from .operators import bake as op_bake
    from .operators import helpers as op_helpers
    from .operators import render as op_render
    from . import cache_loader
    from . import panels


def _collect_classes():
    """Lazy class enumeration — only evaluated inside register() so it does
    not run during pytest collection (where bpy is missing)."""
    return (
        preferences.GpufluidPreferences,
        preferences.GPUFLUID_OT_detect_interpreter,
        properties.GpufluidSurfaceTensionGroup,
        properties.GpufluidWhitewaterGroup,
        properties.GpufluidDomainProps,
        properties.GpufluidFluidProps,
        properties.GpufluidObstacleProps,
        properties.GpufluidInflowProps,
        properties.GpufluidOutflowProps,
        op_bake.GPUFLUID_OT_bake,
        op_render.GPUFLUID_OT_render,
        op_helpers.GPUFLUID_OT_add_domain,
        op_helpers.GPUFLUID_OT_mark_fluid,
        op_helpers.GPUFLUID_OT_mark_obstacle,
        op_helpers.GPUFLUID_OT_mark_inflow,
        op_helpers.GPUFLUID_OT_mark_outflow,
        op_helpers.GPUFLUID_OT_clear_cache,
        op_helpers.GPUFLUID_OT_open_cache_dir,
        op_helpers.GPUFLUID_OT_apply_eevee_preset,
        cache_loader.GPUFLUID_OT_attach_cache,
        cache_loader.GPUFLUID_OT_attach_ww_cache,
        cache_loader.GPUFLUID_OT_detach_cache,
        panels.GPUFLUID_PT_main,
        panels.GPUFLUID_PT_domain,
        panels.GPUFLUID_PT_fluid,
        panels.GPUFLUID_PT_obstacle,
        panels.GPUFLUID_PT_inflow,
        panels.GPUFLUID_PT_outflow,
        panels.GPUFLUID_PT_whitewater,
        panels.GPUFLUID_PT_bake,
    )


# [BLK A8.1]
def register():
    for cls in _collect_classes():
        bpy.utils.register_class(cls)
    bpy.types.Object.gpufluid_domain = bpy.props.PointerProperty(type=properties.GpufluidDomainProps)
    bpy.types.Object.gpufluid_fluid = bpy.props.PointerProperty(type=properties.GpufluidFluidProps)
    bpy.types.Object.gpufluid_obstacle = bpy.props.PointerProperty(type=properties.GpufluidObstacleProps)
    bpy.types.Object.gpufluid_inflow = bpy.props.PointerProperty(type=properties.GpufluidInflowProps)
    bpy.types.Object.gpufluid_outflow = bpy.props.PointerProperty(type=properties.GpufluidOutflowProps)
    cache_loader.register_handler()
    # Auto-fill interpreter on a fresh install (extension reinstall wipes prefs).
    preferences.auto_fill_interpreter_on_first_use()


# [BLK A8.1]
def unregister():
    cache_loader.unregister_handler()
    del bpy.types.Object.gpufluid_outflow
    del bpy.types.Object.gpufluid_inflow
    del bpy.types.Object.gpufluid_obstacle
    del bpy.types.Object.gpufluid_fluid
    del bpy.types.Object.gpufluid_domain
    for cls in reversed(_collect_classes()):
        bpy.utils.unregister_class(cls)
