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
from ..preferences import get_prefs, _detect_interpreter, _context_roots
from ._runner import ModalSubprocessRunner

try:
    from .._blocks import block
except ImportError:
    # dodge: tests load this file under a stub package with no _blocks
    # (spec_from_file_location harnesses) — registration is irrelevant
    # there; the real registration happens on package import. See _blocks.py.
    def block(_bid, _desc=""):
        def _w(fn):
            return fn
        return _w


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


@block("A8.13", "Render operator (sync + modal subprocess to headless "
                "Blender, ESC abort, reentrance guard, watchdog)")
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
        # FU-032: the label no longer secretly selects the shader — any
        # custom text renders the water preset (legacy: literal
        # water/oil/honey labels still infer the matching preset).
        description="Overlay text drawn into each frame",
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
            # audit-20260610 (§9.6): mirror of bake.py's last-chance
            # auto-detect (commit c5834f1) — Render used to hard-fail here
            # where Bake self-healed, so "attach an existing cache → click
            # Render first" dead-ended on an ERROR the user couldn't act on
            # without leaving the panel. Same project-adjacent .venv search;
            # persist the hit so the next click "just works".
            guess = _detect_interpreter(_context_roots())
            if guess:
                prefs.interpreter_path = guess
                interp = bpy.path.abspath(guess).strip()
                self.report({"INFO"},
                            f"auto-detected Python interpreter: {guess}")
                try:
                    bpy.ops.wm.save_userpref()
                except Exception:
                    # dodge: headless/CI Blender or a locked/read-only
                    # userpref file makes save_userpref raise — persisting
                    # is best-effort and must not abort the render (the
                    # in-session prefs value is already set above). Side
                    # effect when it DOES succeed: saves ALL pending
                    # preference changes, not just interpreter_path.
                    pass
        if not interp or not Path(interp).exists():
            self.report({"ERROR"},
                        "No Python with gpufluid found. Set the interpreter "
                        "in Addon Preferences (or click Detect), or set "
                        "$GPUFLUID_PYTHON. It must be a venv where "
                        "`pip install -e .` was run on the gpufluid repo.")
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
        # audit-20260610 (§9.6 mirror of bake's pre-run stale-artefact
        # strip): remove leftover *.png from a previous render of this
        # cache. Without this, the round-61 post-render honesty check
        # counts STALE frames — a re-render that writes ZERO PNGs still
        # reported "complete: N frame(s)" off the previous run's files,
        # the exact false-success disease round-61 fixed for bake. Only
        # top-level *.png files are removed; subdirs are kept.
        for stale in out_dir.glob("*.png"):
            try:
                stale.unlink()
            except OSError as e:
                # dodge: a PNG held open by an external viewer (Windows
                # file lock) must not abort the render — but a survivor
                # poisons the post-render count, so say so out loud.
                self.report(
                    {"WARNING"},
                    f"could not remove stale render frame "
                    f"{stale.name}: {e} — the post-render frame count "
                    f"may overcount.")
        self._out_dir = str(out_dir)

        # audit-2026-06-14r2 #9: forward the bake's fps. The CLI render
        # subparser defaults --fps to 60 and the headless renderer uses it for
        # sc.render.fps + the sim-time overlay (frame/fps). Omitting it made
        # every addon-driven render run at 60 regardless of the Domain's fps
        # (default 24) — overlay sim-time and any PNG-sequence encode came out
        # ~2.5x off. The correct fps is in the cache's scene.toml.
        render_fps = 24
        try:
            import tomllib
            render_fps = int(tomllib.loads(
                Path(scene_toml).read_text(encoding="utf-8"))["simulation"]["fps"])
        except Exception:
            pass
        argv = [
            interp, "-m", "gpufluid.cli", "render",
            str(cache_dir), str(scene_toml),
            "--out", str(out_dir),
            "--label", self.label,
            "--color", f"{self.color[0]:.4f}", f"{self.color[1]:.4f}", f"{self.color[2]:.4f}",
            "--samples", str(int(self.samples)),
            "--fps", str(render_fps),
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
        # use_progress=False (audit-20260610): the render subprocess is a
        # headless Blender whose per-frame output is `Fra:N` — no total, so
        # no `N/M` line exists for tick_modal's frame_regex to advance the
        # round-62 progress bar. A bar permanently stuck at 0% reads as
        # "hung"; the honest static status text is better until the CLI
        # grows a parsable N/M render-progress line.
        return self._runner.start_modal(
            self, context, argv, status_msg="gpufluid rendering…",
            use_progress=False)

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
            # Round-61: re-arm the reentrance mutex around the post-FINISH
            # work — tick_modal's _finish() already cleared _is_running,
            # opening the same single-tick window bake.py closed in
            # round-28. Mirror it here so a second Render click can't
            # spawn a parallel subprocess into the same out_dir while this
            # branch runs (§9.6 — keep bake/render in step).
            cls = self.__class__
            try:
                cls._is_running = True
                # Don't report success blindly — Blender can exit 0 having
                # written no PNGs (unwritable path, empty frame range,
                # misconfigured camera/engine). Count first; escalate the
                # zero case to ERROR (mirror of the bake 0-frame fix).
                from ..cache_sanity import count_pngs, render_output_sanity
                n_png = count_pngs(self._out_dir)
                level, msg = render_output_sanity(n_png, 0)
                if level is not None:
                    self.report({level}, msg)
                else:
                    self.report({"INFO"},
                                f"gpufluid render complete: {n_png} "
                                f"frame(s) in {self._out_dir}")
            finally:
                cls._is_running = False
        return result

    def cancel(self, context):
        if self._runner is not None:
            self._runner.cancel(self, context)

    def _sync_post_render_report(self) -> bool:
        """Sync on_complete: count PNGs and report. Round-61: escalate
        the 0-PNG case to ERROR via the shared cache_sanity classifier
        instead of reporting an empty render as 'finished'.

        audit-20260610: returns False on the 0-PNG ERROR so the runner
        suppresses its unconditional "(sync) finished" INFO — that line
        used to print AFTER the ERROR, ending the Info log on a success
        message over an empty output dir."""
        from ..cache_sanity import count_pngs, render_output_sanity
        n_png = count_pngs(self._out_dir)
        level, msg = render_output_sanity(n_png, 0)
        if level is not None:
            self.report({level}, msg)
            return level != "ERROR"
        self.report(
            {"INFO"},
            f"gpufluid render (sync) finished — {n_png} PNG(s) "
            f"in {self._out_dir}")
        return True

