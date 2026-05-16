"""Addon preferences: Python interpreter that has `gpufluid` installed.

The addon spawns the CLI as a subprocess. Blender's own Python does not
have `gpufluid` (heavy deps: Warp/CUDA). User points us at a venv that does.
"""
import bpy
import sys


class GpufluidPreferences(bpy.types.AddonPreferences):
    # Use __package__ so this works both for legacy add-on install
    # (`gpufluid_blender`) and for Blender 5.x extension install
    # (`bl_ext.user_default.gpufluid_blender`).
    bl_idname = __package__

    interpreter_path: bpy.props.StringProperty(
        name="Python interpreter",
        description="Path to python.exe of a venv where `gpufluid` is installed",
        subtype="FILE_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text="Path to a Python that has `gpufluid` installed:")
        col.prop(self, "interpreter_path")
        col.label(text="Example: E:\\projects\\gpu_flip\\gpufluid\\.venv\\Scripts\\python.exe", icon="INFO")
        col.separator()
        col.label(text=f"Blender's own Python: {sys.executable}  (NOT this — see README)", icon="ERROR")


def get_prefs(context) -> "GpufluidPreferences":
    return context.preferences.addons[__package__].preferences
