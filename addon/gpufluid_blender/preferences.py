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

# Round-18: single-source-of-truth addon-root package name.
# Replaces `__package__` direct use — would silently mis-key on prefs
# lookup if `preferences.py` ever moved into a subpackage.
from . import ADDON_PKG


def _has_gpufluid(python_exe: str) -> bool:
    """Return True if the given Python can import gpufluid (quick subprocess)."""
    import subprocess
    try:
        rc = subprocess.run(
            [python_exe, "-c", "import gpufluid"],
            capture_output=True, timeout=5,
        )
        return rc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _detect_interpreter() -> str:
    """Best-effort guess at a Python with `gpufluid` installed.

    Validates each candidate by spawning ``python -c 'import gpufluid'``;
    only paths that succeed are returned. Without this guard, the system
    Python on PATH gets picked but doesn't have the library installed,
    and every bake fails with ModuleNotFoundError.

    Search order:
      1. ``$GPUFLUID_PYTHON`` env override (always honored if exists,
         no validation — power-user escape hatch)
      2. ``$VIRTUAL_ENV/{Scripts/python.exe,bin/python}`` — current venv
      3. Common project venv layouts near the bake/cwd (`.venv/Scripts/...`,
         `venv/Scripts/...`) walking up from cwd
      4. ``shutil.which("python")`` — only if it can import gpufluid
    """
    env_override = os.environ.get("GPUFLUID_PYTHON", "").strip()
    if env_override and os.path.exists(env_override):
        return env_override

    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        cand = (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
                else os.path.join(venv, "bin", "python"))
        if os.path.exists(cand) and _has_gpufluid(cand):
            return cand

    # Walk up from cwd looking for `.venv/Scripts/python.exe` or `venv/...`
    cwd = os.getcwd()
    for _ in range(6):
        for venv_dir in (".venv", "venv"):
            cand = (os.path.join(cwd, venv_dir, "Scripts", "python.exe")
                    if os.name == "nt"
                    else os.path.join(cwd, venv_dir, "bin", "python"))
            if os.path.exists(cand) and _has_gpufluid(cand):
                return cand
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent

    # Last resort — system Python on PATH, ONLY if it has gpufluid
    found = shutil.which("python")
    if found and _has_gpufluid(found):
        return found
    return ""


class GPUFLUID_OT_detect_interpreter(bpy.types.Operator):
    bl_idname = "gpufluid.detect_interpreter"
    bl_label = "Detect Python interpreter"
    bl_description = ("Re-run the auto-detect for the Python interpreter "
                      "($VIRTUAL_ENV, $GPUFLUID_PYTHON, or python on PATH)")

    def execute(self, context):
        prefs = context.preferences.addons[ADDON_PKG].preferences
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
    # Phase 2 — cache loader bounds. Each preload entry is one full PLY
    # sequence (potentially thousands of mesh datablocks). 8 sequences
    # covers most "compare-a-few-bakes" workflows without unbounded RAM.
    preload_cap: bpy.props.IntProperty(
        name="Preload cache cap",
        description="Maximum number of preloaded mesh sequences kept in "
                    "memory; oldest are evicted (LRU). Each entry can hold "
                    "thousands of mesh datablocks.",
        default=8, min=1, max=64,
    )
    preload_max_frames: bpy.props.IntProperty(
        name="Preload max frames",
        description="Hard cap on frames scanned per preload call. Used to "
                    "stop runaway loops on misconfigured caches.",
        default=10000, min=100, max=1000000,
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.label(text="Path to a Python that has `gpufluid` installed:")
        row = col.row(align=True)
        row.prop(self, "interpreter_path", text="")
        row.operator("gpufluid.detect_interpreter", text="Detect", icon="VIEWZOOM")
        col.label(text="Example (Windows): C:\\path\\to\\gpufluid\\.venv\\Scripts\\python.exe",
                  icon="INFO")
        col.label(text="Example (macOS/Linux): /path/to/gpufluid/.venv/bin/python")
        col.separator()
        col.label(text=f"Blender's own Python: {sys.executable}  (NOT this — see README)",
                  icon="ERROR")
        col.separator()
        col.label(text="Cache loader:")
        col.prop(self, "preload_cap")
        col.prop(self, "preload_max_frames")


def get_prefs(context) -> "GpufluidPreferences":
    return context.preferences.addons[ADDON_PKG].preferences


def auto_fill_interpreter_on_first_use() -> None:
    """Called from addon `register()` after the preferences class is
    registered. Fills `interpreter_path` if it's empty and we can guess
    something useful. No-op if the user has already set it (so we never
    overwrite a deliberate choice)."""
    try:
        prefs = bpy.context.preferences.addons[ADDON_PKG].preferences
    except (KeyError, AttributeError):
        return
    if prefs.interpreter_path.strip():
        return
    guess = _detect_interpreter()
    if guess:
        prefs.interpreter_path = guess
