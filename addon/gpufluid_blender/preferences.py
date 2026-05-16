"""Addon preferences: Python interpreter that has `gpufluid` installed.

The addon spawns the CLI as a subprocess. Blender's own Python does not
have `gpufluid` (heavy deps: Warp/CUDA). User points us at a venv that does.

Reinstall paper-cut (HANDOFF §8.0): Blender's extension manager wipes
`AddonPreferences` on every zip reinstall, so the freshly upgraded addon
starts with `interpreter_path=""` and the first bake fails with WinError
87. To smooth that over we (a) auto-detect a plausible interpreter when
the field is empty and (b) expose a "Detect" button that re-runs the
sniff at any time.
"""
import bpy
import os
import shutil
import sys


def _detect_interpreter() -> str:
    """Best-effort guess at a Python with `gpufluid` installed.

    Search order:
      1. `$VIRTUAL_ENV/Scripts/python.exe` (or `bin/python` on POSIX) —
         the venv Blender was launched from, if any.
      2. `$GPUFLUID_PYTHON` env var — explicit override for advanced
         setups / CI.
      3. `shutil.which("python")` — picks up whatever is first on PATH.
         Last resort; the user almost certainly wants a venv instead, but
         a path that exists beats an empty string.

    Returns "" if nothing plausible is found — UI shows the empty field
    so the user knows to fill it in.
    """
    env_override = os.environ.get("GPUFLUID_PYTHON", "").strip()
    if env_override and os.path.exists(env_override):
        return env_override

    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        cand = (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
                else os.path.join(venv, "bin", "python"))
        if os.path.exists(cand):
            return cand

    found = shutil.which("python")
    if found:
        return found
    return ""


class GPUFLUID_OT_detect_interpreter(bpy.types.Operator):
    bl_idname = "gpufluid.detect_interpreter"
    bl_label = "Detect Python interpreter"
    bl_description = ("Re-run the auto-detect for the Python interpreter "
                      "($VIRTUAL_ENV, $GPUFLUID_PYTHON, or python on PATH)")

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        guess = _detect_interpreter()
        if not guess:
            self.report({"WARNING"},
                        "No Python found in $VIRTUAL_ENV / $GPUFLUID_PYTHON / PATH. "
                        "Set the path manually.")
            return {"CANCELLED"}
        prefs.interpreter_path = guess
        bpy.ops.wm.save_userpref()
        self.report({"INFO"}, f"interpreter set to {guess}")
        return {"FINISHED"}


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
        row = col.row(align=True)
        row.prop(self, "interpreter_path", text="")
        row.operator("gpufluid.detect_interpreter", text="Detect", icon="VIEWZOOM")
        col.label(text="Example: E:\\projects\\gpu_flip\\gpufluid\\.venv\\Scripts\\python.exe",
                  icon="INFO")
        col.separator()
        col.label(text=f"Blender's own Python: {sys.executable}  (NOT this — see README)",
                  icon="ERROR")


def get_prefs(context) -> "GpufluidPreferences":
    return context.preferences.addons[__package__].preferences


def auto_fill_interpreter_on_first_use() -> None:
    """Called from addon `register()` after the preferences class is
    registered. Fills `interpreter_path` if it's empty and we can guess
    something useful. No-op if the user has already set it (so we never
    overwrite a deliberate choice)."""
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
    except (KeyError, AttributeError):
        return
    if prefs.interpreter_path.strip():
        return
    guess = _detect_interpreter()
    if guess:
        prefs.interpreter_path = guess
