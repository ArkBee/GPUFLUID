"""[BLK A8.8] Helper operators: add domain/fluid/obstacle, clear cache, open dir.

Phase 4 also hosts ``subprocess_drain`` — the background-thread stdout
reader shared by ``operators.bake`` and ``operators.render`` modal ops.
"""
import os
import shutil
import bpy


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


class GPUFLUID_OT_add_domain(bpy.types.Operator):
    bl_idname = "gpufluid.add_domain"
    bl_label = "Add gpufluid Domain"
    bl_description = "Create a 1x1x1 empty cube as the simulation domain"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        bpy.ops.object.empty_add(type="CUBE", radius=0.5, location=(0.5, 0.5, 0.5))
        obj = context.active_object
        obj.name = "gpufluid_Domain"
        obj.gpufluid_domain.is_domain = True
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


class GPUFLUID_OT_mark_fluid(bpy.types.Operator):
    bl_idname = "gpufluid.mark_fluid"
    bl_label = "Mark as Fluid Source"
    bl_description = "Mark the active object as a gpufluid source (uses its bounding box)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        context.active_object.gpufluid_fluid.is_fluid = True
        return {"FINISHED"}


class GPUFLUID_OT_mark_obstacle(bpy.types.Operator):
    bl_idname = "gpufluid.mark_obstacle"
    bl_label = "Mark as Obstacle"
    bl_description = "Mark the active object as a gpufluid obstacle"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        context.active_object.gpufluid_obstacle.is_obstacle = True
        return {"FINISHED"}


class GPUFLUID_OT_mark_inflow(bpy.types.Operator):
    bl_idname = "gpufluid.mark_inflow"
    bl_label = "Mark as Inflow"
    bl_description = "Mark the active object as a continuous fluid emitter (uses its bounding box)"
    bl_options = {"REGISTER", "UNDO"}
    @classmethod
    def poll(cls, context): return context.active_object is not None
    def execute(self, context):
        context.active_object.gpufluid_inflow.is_inflow = True
        return {"FINISHED"}


class GPUFLUID_OT_mark_outflow(bpy.types.Operator):
    bl_idname = "gpufluid.mark_outflow"
    bl_label = "Mark as Outflow"
    bl_description = "Mark the active object as a fluid drain (particles inside it are removed)"
    bl_options = {"REGISTER", "UNDO"}
    @classmethod
    def poll(cls, context): return context.active_object is not None
    def execute(self, context):
        context.active_object.gpufluid_outflow.is_outflow = True
        return {"FINISHED"}


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
        # Also drop any pre-loaded PLY mesh tables — pointing at meshes about
        # to be deleted would raise ReferenceError next frame. Phase 4 bonus
        # fix: route each entry through ``_free_table`` so mesh datablocks
        # with users==0 actually get removed instead of leaking in
        # ``bpy.data.meshes`` after a bare ``_PRELOAD.clear()``.
        try:
            from ..cache_loader import _PRELOAD, _free_table
            for _name, _table in list(_PRELOAD.items()):
                _free_table(_table)
            _PRELOAD.clear()
        except Exception:
            pass
        # Drop orphan cache_files via data API (was bpy.ops.outliner.orphans_purge
        # which failed poll outside an Outliner area; try/except masked it,
        # leaving .abc mmap'd → next bake's overwrite silently failed).
        for cf in list(bpy.data.cache_files):
            if cf.users == 0:
                try:
                    bpy.data.cache_files.remove(cf)
                except Exception:
                    pass
        # Brief settle for Windows file lock release
        time.sleep(0.1)
        if os.path.isdir(cache):
            try:
                shutil.rmtree(cache)
                self.report({"INFO"},
                            f"cleared {cache} (released {released} MSC)")
            except PermissionError as exc:
                # Retry once after a longer wait — sometimes mmap takes longer
                time.sleep(0.5)
                try:
                    shutil.rmtree(cache)
                    self.report({"INFO"}, f"cleared after retry")
                except Exception as exc2:
                    self.report({"ERROR"},
                                f"could not clear cache (file locked): {exc2}")
                    return {"CANCELLED"}
        else:
            self.report({"INFO"}, "no cache to clear")
        return {"FINISHED"}


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
                subprocess.Popen(["xdg-open", cache])
        else:
            self.report({"WARNING"}, "cache dir does not exist yet — bake first")
        return {"FINISHED"}
