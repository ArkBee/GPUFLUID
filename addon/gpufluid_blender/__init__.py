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


# Round-18 (senior code-smell #5, lesson 9.10): single source of truth for
# the addon-root package name. Used as the key into
# `bpy.context.preferences.addons[...]`. THIS file lives at the addon root,
# so its `__package__` IS the addon root, regardless of layout:
#   * Legacy install:        ADDON_PKG == "gpufluid_blender"
#   * 4.2+ extension install: ADDON_PKG == "bl_ext.user_default.gpufluid_blender"
#
# Previously nested modules (cache_loader/__init__.py) did
# `__package__.rsplit('.', 1)[0]` — works ONLY because cache_loader is
# exactly one level deep. A future nested-deeper submodule would silently
# get the wrong key + fall through to default prefs. Centralised here so
# every consumer imports the SAME constant via `from .. import ADDON_PKG`.
ADDON_PKG: str = __package__


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

# audit-20260610: real A8.* registration via the _blocks shim (no-op inside
# Blender — see _blocks.py). Defensive: standalone file-loads in tests have
# no parent package, so the relative import can fail there.
try:
    from ._blocks import block
except ImportError:
    # dodge: spec_from_file_location loads with no parent package —
    # registration is irrelevant in those harnesses. See _blocks.py.
    def block(_bid, _desc=""):
        def _w(fn):
            return fn
        return _w

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
@block("A8.1", "Addon register/unregister")
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
@block("A8.1", "Addon register/unregister (teardown half)")
def unregister():
    cache_loader.unregister_handler()
    # Round-61: guard every teardown step. A half-registered state — e.g.
    # a prior reload that left a stale double-registration so the class
    # object in _collect_classes() no longer matches the registered one —
    # made `unregister_class` raise "missing bl_rna (may not be
    # registered)" MID-LOOP. Pre-R61 that aborted the whole unregister:
    # every remaining class leaked, the Object props were half-deleted,
    # and the addon dropped into a corrupt half-enabled state that needed
    # a full Blender restart to clear (live-hit 2026-05-28 on a reload).
    # Now each step is best-effort so teardown always completes.
    for prop in ("gpufluid_outflow", "gpufluid_inflow", "gpufluid_obstacle",
                 "gpufluid_fluid", "gpufluid_domain"):
        if hasattr(bpy.types.Object, prop):
            try:
                delattr(bpy.types.Object, prop)
            except Exception:
                pass  # already gone / mid-corrupt state — keep tearing down
    for cls in reversed(_collect_classes()):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            # not-registered / stale bl_rna — don't let one bad class
            # leak all the others (the pre-R61 corruption mode).
            pass
