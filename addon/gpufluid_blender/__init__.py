"""[BLK A8.1] gpufluid Blender addon — registration root.

Block index for this package:
    A8.1  addon register/unregister (this file)
    A8.2  GpufluidDomain property group       — properties.py
    A8.3  GpufluidFluid property group        — properties.py
    A8.4  GpufluidObstacle property group     — properties.py
    A8.5  Bake operator                       — operators/bake.py
    A8.6  Cache import (PLY/frame handler)    — cache_loader.py
    A8.7  UI panels                           — panels.py
    A8.8  Helper operators                    — operators/helpers.py
"""
bl_info = {
    "name": "gpufluid",
    "author": "gpufluid contributors",
    "version": (0, 8, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > GpuFluid",
    "description": "GPU FLIP fluid simulator (NVIDIA Warp)",
    "category": "Physics",
}

import bpy

from . import properties
from . import preferences
from .operators import bake as op_bake
from .operators import helpers as op_helpers
from . import cache_loader
from . import panels


_CLASSES = (
    preferences.GpufluidPreferences,
    properties.GpufluidSurfaceTensionGroup,
    properties.GpufluidWhitewaterGroup,
    properties.GpufluidDomainProps,
    properties.GpufluidFluidProps,
    properties.GpufluidObstacleProps,
    properties.GpufluidInflowProps,
    properties.GpufluidOutflowProps,
    op_bake.GPUFLUID_OT_bake,
    op_helpers.GPUFLUID_OT_add_domain,
    op_helpers.GPUFLUID_OT_mark_fluid,
    op_helpers.GPUFLUID_OT_mark_obstacle,
    op_helpers.GPUFLUID_OT_mark_inflow,
    op_helpers.GPUFLUID_OT_mark_outflow,
    op_helpers.GPUFLUID_OT_clear_cache,
    op_helpers.GPUFLUID_OT_open_cache_dir,
    cache_loader.GPUFLUID_OT_attach_cache,
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
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.gpufluid_domain = bpy.props.PointerProperty(type=properties.GpufluidDomainProps)
    bpy.types.Object.gpufluid_fluid = bpy.props.PointerProperty(type=properties.GpufluidFluidProps)
    bpy.types.Object.gpufluid_obstacle = bpy.props.PointerProperty(type=properties.GpufluidObstacleProps)
    bpy.types.Object.gpufluid_inflow = bpy.props.PointerProperty(type=properties.GpufluidInflowProps)
    bpy.types.Object.gpufluid_outflow = bpy.props.PointerProperty(type=properties.GpufluidOutflowProps)
    cache_loader.register_handler()


# [BLK A8.1]
def unregister():
    cache_loader.unregister_handler()
    del bpy.types.Object.gpufluid_outflow
    del bpy.types.Object.gpufluid_inflow
    del bpy.types.Object.gpufluid_obstacle
    del bpy.types.Object.gpufluid_fluid
    del bpy.types.Object.gpufluid_domain
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
