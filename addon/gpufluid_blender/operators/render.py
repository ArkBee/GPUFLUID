"""[BLK A8.13] Render operator — modal subprocess invocation of `gpufluid render`.

Companion to :mod:`operators.bake` (A8.5). Where Bake spawns the host
Python with ``-m gpufluid.cli simulate <scene.toml>``, Render spawns the
same interpreter with ``-m gpufluid.cli render <cache> <scene>`` — which in
turn launches a headless Blender via :mod:`_headless_render` (A8.12) to
run the Eevee pipeline against the baked cache.

Gate 0.2 (contract audit) explicitly chose subprocess over direct import
so the render path uses the user-configured Python (which has gpufluid
installed) and the user-configured Blender (which has the addon installed
and the right GPU drivers), instead of trying to drive both stacks from
the addon's bpy interpreter.

Concurrency model mirrors A8.5: blocking ``Popen`` + background drain
thread + modal timer ticking a ``queue.Queue``. Phase 4 lifted the drain
helper into ``operators/helpers.subprocess_drain`` so both bake and
render share identical semantics.
"""
from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path
from typing import Optional

import bpy

from .. import logger
from ..preferences import get_prefs
from .helpers import subprocess_drain


def _find_domain(context):
    """Resolve the active Domain object (prefer selection, else first in scene).

    Same picking rule as :class:`GPUFLUID_OT_bake.execute` so users don't
    bake against one domain and render another by accident.
    """
    all_domains = [o for o in context.scene.objects if o.gpufluid_domain.is_domain]
    if not all_domains:
        return None, []
    active = context.active_object
    if active is not None and active.gpufluid_domain.is_domain:
        return active, all_domains
    return all_domains[0], all_domains


def _resolve_cache_dir(domain) -> Optional[Path]:
    """Pull the cache dir off the Domain prop, resolving // and ~."""
    cd = (domain.gpufluid_domain.cache_dir or "").strip()
    if not cd:
        # Bake stores its run dir as a custom prop after a successful bake.
        cd = str(domain.get("gpufluid_cache_dir", "")).strip()
    if not cd:
        return None
    return Path(bpy.path.abspath(cd)).expanduser().resolve()


def _scene_toml_for(cache_dir: Path) -> Path:
    """The bake operator writes <cache_dir>/scene.toml — render reads from there."""
    return cache_dir / "scene.toml"


class GPUFLUID_OT_render(bpy.types.Operator):
    bl_idname = "gpufluid.render"
    bl_label = "Render Cached Sim"
    bl_description = ("Spawn `gpufluid render` to produce an Eevee PNG sequence "
                      "from the active Domain's baked cache")
    bl_options = {"REGISTER"}

    # ─── operator properties (configurable per invocation) ────────────────
    samples: bpy.props.IntProperty(
        name="Samples",
        description="Eevee TAA samples per frame (A8.9 preset)",
        default=16, min=1, soft_max=128,
    )
    color: bpy.props.FloatVectorProperty(
        name="Fluid Color",
        description="Fluid surface RGB (forwarded to --color)",
        size=3, default=(0.20, 0.50, 0.70),
        min=0.0, max=1.0, subtype="COLOR",
    )
    label: bpy.props.StringProperty(
        name="Label",
        description="Overlay text drawn into each frame (also drives material preset)",
        default="Water",
    )
    blender_path: bpy.props.StringProperty(
        name="Blender Executable",
        description="Path to Blender (leave empty to use `blender` on $PATH)",
        default="", subtype="FILE_PATH",
    )
    out_subdir: bpy.props.StringProperty(
        name="Output Subdir",
        description="PNG output dir, relative to cache_dir if not absolute",
        default="render",
    )

    # ─── modal state ──────────────────────────────────────────────────────
    _timer = None
    _proc: Optional[subprocess.Popen] = None
    _stdout_q: Optional[queue.Queue] = None
    _stdout_thread: Optional[threading.Thread] = None
    _out_dir = ""

    @classmethod
    def poll(cls, context):
        return any(o.gpufluid_domain.is_domain for o in context.scene.objects)

    def invoke(self, context, event):
        # Show the operator props dialog so the user can tweak samples/label/color
        # before the render kicks off — mirrors Blender's idiom for "run this
        # job with these args".
        return context.window_manager.invoke_props_dialog(self, width=360)

    def execute(self, context):
        prefs = get_prefs(context)
        interp = bpy.path.abspath(prefs.interpreter_path).strip()
        if not interp or not Path(interp).exists():
            self.report({"ERROR"},
                "Set a valid Python interpreter in Addon Preferences "
                "(must have gpufluid installed).")
            return {"CANCELLED"}

        domain, all_domains = _find_domain(context)
        if domain is None:
            self.report({"ERROR"},
                "No Domain in scene. Bake one first, or mark an Empty as Domain.")
            return {"CANCELLED"}
        if len(all_domains) > 1:
            other = [o.name for o in all_domains if o.name != domain.name]
            self.report({"INFO"},
                f"Rendering domain '{domain.name}'. Other domains "
                f"({', '.join(other)}) are ignored — select one to switch.")

        cache_dir = _resolve_cache_dir(domain)
        if cache_dir is None or not cache_dir.exists():
            self.report({"ERROR"},
                f"Domain '{domain.name}' has no usable cache_dir. Bake first.")
            return {"CANCELLED"}
        cache_json = cache_dir / "cache.json"
        if not cache_json.exists():
            self.report({"ERROR"},
                f"{cache_dir} has no cache.json — bake did not finish?")
            return {"CANCELLED"}
        scene_toml = _scene_toml_for(cache_dir)
        if not scene_toml.exists():
            self.report({"ERROR"},
                f"{scene_toml} missing — re-bake to regenerate the TOML snapshot.")
            return {"CANCELLED"}

        out_dir = Path(self.out_subdir)
        if not out_dir.is_absolute():
            out_dir = cache_dir / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = str(out_dir)

        argv = [
            interp, "-m", "gpufluid.cli", "render",
            str(cache_dir), str(scene_toml),
            "--out", str(out_dir),
            "--label", self.label,
            "--color", f"{self.color[0]:.4f}", f"{self.color[1]:.4f}", f"{self.color[2]:.4f}",
            "--samples", str(int(self.samples)),
        ]
        blender = self.blender_path.strip()
        if blender:
            argv += ["--blender", bpy.path.abspath(blender)]

        try:
            self._proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, encoding="utf-8", errors="replace",
            )
        except OSError as e:
            self.report({"ERROR"}, f"could not spawn render subprocess: {e}")
            return {"CANCELLED"}

        # Background stdout drain — shared helper with OT_bake (Phase 4).
        self._stdout_q = queue.Queue()
        self._stdout_thread = threading.Thread(
            target=subprocess_drain, args=(self._proc, self._stdout_q),
            daemon=True)
        self._stdout_thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(self)
        context.workspace.status_text_set("gpufluid rendering…")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._proc is None:
            return self._finish(context, ok=False)
        rc = self._proc.poll()
        if self._stdout_q is not None:
            for _ in range(50):
                try:
                    line = self._stdout_q.get_nowait()
                except queue.Empty:
                    break
                if line is None:   # sentinel: stdout EOF
                    break
                logger.info("render: %s", line.rstrip())
        if rc is None:
            return {"PASS_THROUGH"}
        return self._finish(context, ok=(rc == 0))

    def _finish(self, context, ok):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        self._proc = None
        if not ok:
            self.report({"ERROR"},
                "gpufluid render failed — see system console")
            return {"CANCELLED"}
        self.report({"INFO"}, f"gpufluid render complete: {self._out_dir}")
        return {"FINISHED"}
