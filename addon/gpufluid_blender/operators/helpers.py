"""[BLK A8.8] Helper operators: add domain/fluid/obstacle, clear cache, open dir.

Phase 4 also hosts ``subprocess_drain`` — the background-thread stdout
reader shared by ``operators.bake`` and ``operators.render`` modal ops.
"""
import os
import shutil
import stat
import time
import bpy

from ._animation import _bbox_world

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


def _robust_rmtree(path, attempts=5):
    """Delete a directory tree, tolerating Windows file-lock / async-
    delete races.

    Round-60: `OT_clear_cache` crashed with `OSError [WinError 145]
    "directory not empty"` on `mesh/`. Root cause: bare
    `shutil.rmtree` on Windows can fail mid-tree when (a) a child file
    was just unlinked but the directory listing hasn't flushed before
    `os.rmdir` fires (async-delete race), (b) an AV scanner holds a
    transient handle, or (c) a read-only attribute blocks unlink
    (WinError 5). The old handler only caught `PermissionError`
    (WinError 5/32) — WinError 145 maps to plain `OSError`
    (errno ENOTEMPTY), so it propagated uncaught and crashed the op.

    Strategy: retry the whole `rmtree` with linear backoff; between
    attempts clear read-only bits across the tree so the next pass can
    unlink them. Re-raises the last `OSError` if every attempt fails so
    the caller can report a clear message (rather than silently leaving
    a half-deleted cache).
    """
    last_exc = None
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:  # WinError 5/32/145 all land here
            last_exc = exc
            if not os.path.isdir(path):
                return  # someone/something finished the job — treat as success
            # Clear read-only attrs that block unlink (WinError 5).
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        os.chmod(os.path.join(root, name), stat.S_IWRITE)
                    except OSError:
                        pass  # best-effort; the retry will surface real locks
            time.sleep(0.2 * (i + 1))  # 0.2, 0.4, 0.6, 0.8s backoff
    raise last_exc


def subprocess_drain(proc, q):
    """Read ``proc.stdout`` line-by-line into queue ``q``; push ``None``
    sentinel on EOF/error.

    Designed to run in a daemon ``threading.Thread`` so the operator's
    modal timer never blocks on ``readline()``. Previously duplicated in
    ``operators/bake.py`` and ``operators/render.py``; consolidated here
    in Phase 4.
    """
    try:
        for line in iter(proc.stdout.readline, ""):
            q.put(line)
    except Exception:  # noqa: BLE001 — daemon thread, must not propagate
        pass
    finally:
        q.put(None)  # sentinel


@block("A8.8", "Helper operator family (mark_*/clear/open_cache_dir/detect_interp/single-role)")
class GPUFLUID_OT_add_domain(bpy.types.Operator):
    bl_idname = "gpufluid.add_domain"
    bl_label = "Add gpufluid Domain"
    bl_description = ("Create the simulation domain Empty: sized to enclose "
                      "the selected objects (+10% margin), or a 2 m cube "
                      "resting on the floor when nothing is selected")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # §9.13 (live-MCP 2026-06-11): the old fixed 1×1×1 m Empty at
        # (0.5, 0.5, 0.5) sat invisibly INSIDE Blender's default 2 m Cube
        # with zero feedback — users baked a domain that contained none of
        # their scene. New default: enclose the current selection (the
        # thing the user is looking at) with ~10% margin; with no
        # selection, a 2 m cube resting on z=0 at the 3D cursor.
        # Capture the selection BEFORE empty_add (it replaces selection);
        # exclude pre-existing domain Empties (they're frames, not content).
        sel = [o for o in getattr(context, "selected_objects", []) or []
               if not o.gpufluid_domain.is_domain]
        if sel:
            boxes = [_bbox_world(o) for o in sel]
            lo = [min(b[0][i] for b in boxes) for i in range(3)]
            hi = [max(b[1][i] for b in boxes) for i in range(3)]
            size = [hi[i] - lo[i] for i in range(3)]
            ref = max(max(size), 1e-3)
            # dodge: a flat selection (plane: size_z == 0) would produce a
            # zero-thickness domain that DomainTransform rejects — pad
            # degenerate axes by 10% of the LARGEST extent instead.
            pad = [0.10 * size[i] if size[i] > 1e-6 else 0.10 * ref
                   for i in range(3)]
            dom_lo = [lo[i] - pad[i] for i in range(3)]
            dom_hi = [hi[i] + pad[i] for i in range(3)]
        else:
            cursor = getattr(getattr(context.scene, "cursor", None),
                             "location", (0.0, 0.0, 0.0))
            # 2 m cube RESTING ON the floor (z ∈ [0, 2]) — not centred on
            # it — so default-gravity fluid has somewhere to fall.
            dom_lo = [cursor[0] - 1.0, cursor[1] - 1.0, 0.0]
            dom_hi = [cursor[0] + 1.0, cursor[1] + 1.0, 2.0]
        centre = tuple((dom_lo[i] + dom_hi[i]) * 0.5 for i in range(3))
        extent = tuple(dom_hi[i] - dom_lo[i] for i in range(3))

        # Empty+scale semantics the TOML pipeline expects: _bbox_world
        # reads empty_display_size (half-extent 0.5) × matrix_world scale,
        # so world size == scale. Keep display size at 0.5, size via scale.
        bpy.ops.object.empty_add(type="CUBE", radius=0.5, location=centre)
        obj = context.active_object
        obj.name = "gpufluid_Domain"
        obj.scale = extent
        # Draw the wireframe over scene geometry — the old domain was
        # routinely hidden inside meshes it was supposed to contain.
        obj.show_in_front = True
        obj.gpufluid_domain.is_domain = True
        self.report({"INFO"},
                    f"Domain '{obj.name}' created: {extent[0]:.2f}×"
                    f"{extent[1]:.2f}×{extent[2]:.2f} m — fluids/obstacles "
                    f"must be inside it")
        # default cache folder: next to the .blend if saved, otherwise temp.
        # NB: Blender 5.x extensions reject the "//" relative prefix on
        # StringProperty assignment until the file has been saved.
        blend_path = bpy.data.filepath
        if blend_path:
            obj.gpufluid_domain.cache_dir = bpy.path.abspath(f"//cache/{obj.name}")
        else:
            import tempfile, os
            obj.gpufluid_domain.cache_dir = os.path.join(tempfile.gettempdir(),
                                                        f"gpufluid_cache_{obj.name}")
        return {"FINISHED"}


# Single-role sticky: each mark op activates its own role and EXPLICITLY
# clears the other three. Without this, double-marking (live-found
# 2026-05-25 round-4 #23) leaves the object in an invalid is_obstacle=True
# AND is_inflow=True state — the CLI then emits both `[[obstacle]]` and
# `[[inflow]]` entries for the same mesh, which the solver mis-interprets
# (solid surface that also emits particles into itself).
_ROLE_FLAGS = (
    ("gpufluid_fluid", "is_fluid"),
    ("gpufluid_obstacle", "is_obstacle"),
    ("gpufluid_inflow", "is_inflow"),
    ("gpufluid_outflow", "is_outflow"),
)


def _set_single_role(obj, role_attr: str) -> None:
    for grp, attr in _ROLE_FLAGS:
        setattr(getattr(obj, grp), attr, attr == role_attr)


@block("A8.8", "Helper operator family (mark_*/clear/open_cache_dir/detect_interp/single-role)")
class GPUFLUID_OT_mark_fluid(bpy.types.Operator):
    bl_idname = "gpufluid.mark_fluid"
    bl_label = "Mark as Fluid Source"
    bl_description = "Mark the active object as a gpufluid source (uses its bounding box). Clears obstacle/inflow/outflow roles."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        obj = context.active_object
        _set_single_role(obj, "is_fluid")
        # §9.13 / round-53 precedent (obstacle BBOX→MESH auto-promote):
        # source_type=BBOX on a non-box mesh seeds a CUBE of water where
        # the user sees a sphere/suzanne (live-MCP 2026-06-11: UV-sphere
        # source produced a cube of fluid). Cheap box heuristic: exactly
        # 8 vertices. Anything else auto-promotes to MESH so the visible
        # shape is what gets filled. Only promotes from the BBOX default —
        # an explicit SPHERE/MESH choice on a re-mark is left alone.
        me = getattr(obj, "data", None)
        n_verts = (len(me.vertices)
                   if me is not None and hasattr(me, "vertices") else None)
        if (obj.gpufluid_fluid.source_type == "BBOX"
                and n_verts is not None and n_verts != 8):
            obj.gpufluid_fluid.source_type = "MESH"
            self.report({"INFO"},
                        f"'{obj.name}' is not a box ({n_verts} verts) — "
                        f"non-box source → shape will be filled exactly "
                        f"(Source Shape=Mesh Volume)")
        return {"FINISHED"}


@block("A8.8", "Helper operator family (mark_*/clear/open_cache_dir/detect_interp/single-role)")
class GPUFLUID_OT_mark_obstacle(bpy.types.Operator):
    bl_idname = "gpufluid.mark_obstacle"
    bl_label = "Mark as Obstacle"
    bl_description = "Mark the active object as a gpufluid obstacle. Clears fluid/inflow/outflow roles."
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        _set_single_role(context.active_object, "is_obstacle")
        return {"FINISHED"}


@block("A8.8", "Helper operator family (mark_*/clear/open_cache_dir/detect_interp/single-role)")
class GPUFLUID_OT_mark_inflow(bpy.types.Operator):
    bl_idname = "gpufluid.mark_inflow"
    bl_label = "Mark as Inflow"
    bl_description = "Mark the active object as a continuous fluid emitter (uses its bounding box). Clears fluid/obstacle/outflow roles."
    bl_options = {"REGISTER", "UNDO"}
    @classmethod
    def poll(cls, context): return context.active_object is not None
    def execute(self, context):
        _set_single_role(context.active_object, "is_inflow")
        return {"FINISHED"}


@block("A8.8", "Helper operator family (mark_*/clear/open_cache_dir/detect_interp/single-role)")
class GPUFLUID_OT_mark_outflow(bpy.types.Operator):
    bl_idname = "gpufluid.mark_outflow"
    bl_label = "Mark as Outflow"
    bl_description = "Mark the active object as a fluid drain (particles inside it are removed). Clears fluid/obstacle/inflow roles."
    bl_options = {"REGISTER", "UNDO"}
    @classmethod
    def poll(cls, context): return context.active_object is not None
    def execute(self, context):
        _set_single_role(context.active_object, "is_outflow")
        return {"FINISHED"}


@block("A8.8", "Helper operator family (mark_*/clear/open_cache_dir/detect_interp/single-role)")
class GPUFLUID_OT_clear_cache(bpy.types.Operator):
    bl_idname = "gpufluid.clear_cache"
    bl_label = "Clear Cache"
    bl_description = "Delete the configured cache directory"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        d = next((o for o in context.scene.objects if o.gpufluid_domain.is_domain), None)
        return d is not None

    def execute(self, context):
        import time
        d = next((o for o in context.scene.objects if o.gpufluid_domain.is_domain), None)
        cache = bpy.path.abspath(d.gpufluid_domain.cache_dir)
        # MSC modifiers mmap the .abc file — Windows refuses to delete it
        # until Blender releases the cache_file data-block. Sequence:
        #   1) Unlink cache_file from every MSC modifier
        #   2) Remove the MSC modifiers themselves
        #   3) Drop the now-unused cache_file datablocks via the data API
        #      (releases the underlying file handle). Direct data-API removal
        #      avoids the orphans_purge poll-failure outside an Outliner area.
        released = 0
        for obj in context.scene.objects:
            for m in list(obj.modifiers):
                if m.type == "MESH_SEQUENCE_CACHE":
                    m.cache_file = None
                    obj.modifiers.remove(m)
                    released += 1
        # Round-32: pre-round-32 dropped the ENTIRE _PRELOAD — in a
        # multi-domain scene, clearing one domain's cache silently
        # blanked every OTHER domain's preloaded mesh sequence until
        # re-attach (live-found by round-31 reviewer). Now: invalidate
        # ONLY entries that point at this domain's cache (the mesh
        # named after the domain itself, plus its `target_object` if
        # set). Other domains' preloads stay live.
        try:
            from ..cache_loader import _PRELOAD
            keys_to_drop = {d.name}
            tgt = getattr(d.gpufluid_domain, "target_object", None)
            if tgt is not None:
                keys_to_drop.add(tgt.name)
            for _name in list(_PRELOAD.keys()):
                if _name in keys_to_drop:
                    _PRELOAD.invalidate(_name)
        except Exception as exc:  # noqa: BLE001
            # Round-7 reviewer flag: was bare `pass`. If cache_loader gets
            # renamed/refactored, OT_clear_cache silently becomes no-op
            # on the _PRELOAD side. Log so it shows up in the system
            # console instead of leaking _seq_* meshes invisibly.
            from .. import logger as _addon_logger
            _addon_logger.warning(
                "addon.clear_cache.preload_drop_failed: %s", exc)
        # Drop orphan cache_files via data API (was bpy.ops.outliner.orphans_purge
        # which failed poll outside an Outliner area; try/except masked it,
        # leaving .abc mmap'd → next bake's overwrite silently failed).
        for cf in list(bpy.data.cache_files):
            if cf.users == 0:
                try:
                    bpy.data.cache_files.remove(cf, do_unlink=True)
                except Exception:
                    pass
        # Brief settle for Windows file lock release
        time.sleep(0.1)
        if os.path.isdir(cache):
            # Round-60: _robust_rmtree retries with backoff + chmod and
            # catches OSError broadly (WinError 5/32/145). Pre-R60 only
            # PermissionError was caught, so WinError 145 ("dir not
            # empty", the Windows async-delete race on mesh/) escaped
            # and crashed the operator.
            try:
                _robust_rmtree(cache)
                self.report({"INFO"},
                            f"cleared {cache} (released {released} MSC)")
            except OSError as exc:
                self.report(
                    {"ERROR"},
                    f"could not clear cache after retries — files may be "
                    f"held by an attached cache object (detach it first) "
                    f"or an external process: {exc}")
                return {"CANCELLED"}
        else:
            self.report({"INFO"}, "no cache to clear")
        return {"FINISHED"}


@block("A8.8", "Helper operator family (mark_*/clear/open_cache_dir/detect_interp/single-role)")
class GPUFLUID_OT_apply_eevee_preset(bpy.types.Operator):
    """[BLK A8.9] One-click Eevee perf preset for the active scene.

    Configures TAA samples + disables expensive post-FX (bloom/SSR/GTAO/
    volumetrics). Production default: 16 samples → roughly 3× faster than
    Blender's photo-quality defaults at no visible cost for an opaque
    fluid mesh + cube + ground scene. Idempotent — run as many times as
    you like; the only mutated bits are scene.eevee.* attributes.
    """
    bl_idname = "gpufluid.apply_eevee_preset"
    bl_label = "Apply Eevee Production Preset"
    bl_description = ("Set TAA samples to 16 and disable bloom/SSR/GTAO/"
                      "volumetric lights — production-friendly Eevee config "
                      "for fluid renders. Run once per scene.")
    bl_options = {"REGISTER", "UNDO"}

    samples: bpy.props.IntProperty(
        name="TAA Samples", default=16, min=1, max=256,
        description="Eevee render samples per frame; 16 is the recommended default")

    def execute(self, context):
        # Defer the library import to keep the addon's bpy-dependent code
        # decoupled from the import-time path that pytest exercises.
        from ..render_bridge import apply_eevee_preset
        log = apply_eevee_preset(context.scene, samples=self.samples)
        if not log:
            self.report({"WARNING"}, "scene has no .eevee attribute — set engine to EEVEE first")
            return {"CANCELLED"}
        self.report({"INFO"},
                    f"Eevee preset applied: samples={self.samples}, "
                    f"disabled={sum(1 for v in log.values() if v is False)}")
        return {"FINISHED"}


@block("A8.8", "Helper operator family (mark_*/clear/open_cache_dir/detect_interp/single-role)")
class GPUFLUID_OT_open_cache_dir(bpy.types.Operator):
    bl_idname = "gpufluid.open_cache_dir"
    bl_label = "Open Cache Folder"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        d = next((o for o in context.scene.objects if o.gpufluid_domain.is_domain), None)
        return d is not None

    def execute(self, context):
        d = next((o for o in context.scene.objects if o.gpufluid_domain.is_domain), None)
        cache = bpy.path.abspath(d.gpufluid_domain.cache_dir)
        if os.path.isdir(cache):
            if os.name == "nt":
                os.startfile(cache)  # noqa: S606
            else:
                import subprocess
                # dodge: xdg-open missing / not on PATH (headless Linux, minimal
                # desktops) makes Popen raise FileNotFoundError. Pre-public-test
                # audit (2026-06-01): unguarded, this crashed the operator with a
                # traceback popup instead of a graceful warning.
                try:
                    subprocess.Popen(["xdg-open", cache])
                except OSError as exc:
                    self.report({"WARNING"},
                                f"could not open file manager (xdg-open): "
                                f"{exc}. Cache is at: {cache}")
        else:
            self.report({"WARNING"}, "cache dir does not exist yet — bake first")
        return {"FINISHED"}
