"""[BLK A8.8] Helper operators: add domain/fluid/obstacle, clear cache, open dir."""
import os
import shutil
import bpy


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
        d = next((o for o in context.scene.objects if o.gpufluid_domain.is_domain), None)
        cache = bpy.path.abspath(d.gpufluid_domain.cache_dir)
        if os.path.isdir(cache):
            shutil.rmtree(cache, ignore_errors=True)
            self.report({"INFO"}, f"cleared {cache}")
        else:
            self.report({"INFO"}, "no cache to clear")
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
