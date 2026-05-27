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

from pathlib import Path
from typing import Optional

import bpy

from .. import logger
from ..preferences import get_prefs
from ._runner import ModalSubprocessRunner


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
        # Bake stores its run dir as a bake-trace prop after a successful
        # bake — round-19 routes through cache_binding (no magic strings).
        from .. import cache_binding as _cb
        trace = _cb.get_bake_trace(domain)
        cd = str(trace["cache_dir"]).strip() if trace else ""
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
    # Same sync-mode contract as OT_bake (round-6): when True, block in
    # Popen.wait() with no modal. Required for scripted/CI/MCP-driven
    # workflows. UI defaults to False so the user keeps the cancellable
    # modal flow with status-bar progress.
    sync: bpy.props.BoolProperty(
        name="Wait for completion",
        description="Block until the render subprocess finishes (no modal). "
                    "Useful for batch scripts. UI defaults to False.",
        default=False,
        options={'SKIP_SAVE', 'HIDDEN'},
    )
    sync_timeout_sec: bpy.props.IntProperty(
        name="Sync timeout (seconds)",
        description="Round-9 hardening: hard cap on sync-mode Popen.wait(). "
                    "Render with Eevee + headless Blender startup can be "
                    "slow but should never be hours — default 1800s = 30min. "
                    "0 = no limit.",
        default=1800, min=0, soft_max=14400,
        options={'SKIP_SAVE', 'HIDDEN'},
    )

    # Round-14: subprocess lifecycle extracted into ModalSubprocessRunner.
    # Mirror of OT_bake — kills duplicate Popen/Queue/Thread/timer/cancel
    # plumbing that 11 rounds kept asymmetrically drifting (lesson 9.6).
    _runner: Optional[ModalSubprocessRunner] = None
    _out_dir = ""
    # Reentrance guard — class-level for cross-instance mutex; runner
    # reads/writes via op.__class__._is_running.
    _is_running: bool = False

    @classmethod
    def poll(cls, context):
        return any(o.gpufluid_domain.is_domain for o in context.scene.objects)

    def invoke(self, context, event):
        # Show the operator props dialog so the user can tweak samples/label/color
        # before the render kicks off — mirrors Blender's idiom for "run this
        # job with these args".
        return context.window_manager.invoke_props_dialog(self, width=360)

    def execute(self, context):
        if GPUFLUID_OT_render._is_running:
            self.report({"WARNING"},
                        "A gpufluid render is already running — wait for "
                        "it to finish before starting a new one.")
            return {"CANCELLED"}
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

        GPUFLUID_OT_render._is_running = True
        self._runner = ModalSubprocessRunner("render")

        # ─── SYNC PATH ────────────────────────────────────────────────
        if self.sync:
            return self._runner.start_sync(
                self, argv, int(self.sync_timeout_sec),
                log_prefix="render", logger=logger,
                on_complete=self._sync_post_render_report,
            )

        # ─── MODAL PATH ───────────────────────────────────────────────
        return self._runner.start_modal(
            self, context, argv, status_msg="gpufluid rendering…")

    def modal(self, context, event):
        result = self._runner.tick_modal(
            self, context, event,
            logger=logger, log_prefix="render",
        )
        if result is None:
            return {"PASS_THROUGH"}
        if "FINISHED" in result:
            # `_out_dir` lives on the operator (not runner state) so it
            # survives `_clear_instance_state` — direct read is safe.
            # Round-16 reviewer flagged earlier comment about "snapshot
            # before runner clears it" as stale.
            self.report({"INFO"},
                        f"gpufluid render complete: {self._out_dir}")
        return result

    def cancel(self, context):
        if self._runner is not None:
            self._runner.cancel(self, context)

    def _sync_post_render_report(self) -> None:
        """Sync on_complete: count PNGs + INFO report."""
        n_png = len(list(Path(self._out_dir).glob("*.png")))
        self.report(
            {"INFO"},
            f"gpufluid render (sync) finished — {n_png} PNG(s) "
            f"in {self._out_dir}")

