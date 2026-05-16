"""[BLK A8.7] N-sidebar UI panels for gpufluid."""
import bpy


PANEL_CATEGORY = "GpuFluid"


class GPUFLUID_PT_main(bpy.types.Panel):
    bl_label = "gpufluid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("gpufluid.add_domain", icon="OUTLINER_OB_EMPTY")
        col.separator()
        col.operator("gpufluid.mark_fluid", icon="OUTLINER_OB_FORCE_FIELD")
        col.operator("gpufluid.mark_obstacle", icon="MOD_BUILD")
        col.operator("gpufluid.mark_inflow", icon="TRIA_DOWN_BAR")
        col.operator("gpufluid.mark_outflow", icon="TRIA_UP_BAR")
        col.separator()
        col.operator("gpufluid.detach_cache", icon="UNLINKED")


class GPUFLUID_PT_domain(bpy.types.Panel):
    bl_label = "Domain"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.gpufluid_domain.is_domain

    def draw(self, context):
        layout = self.layout
        d = context.active_object.gpufluid_domain
        col = layout.column()
        col.prop(d, "resolution")
        col.prop(d, "cache_dir")
        col.prop(d, "target_object")
        box = layout.box()
        box.label(text="Simulation")
        box.prop(d, "frames")
        box.prop(d, "fps")
        box.prop(d, "dt")
        box.prop(d, "pressure_solver")
        box.prop(d, "pressure_iters")
        box.prop(d, "flip_blend")
        box.prop(d, "gravity")
        sub = box.box()
        sub.prop(d, "use_cfl")
        if d.use_cfl:
            sub.prop(d, "cfl_factor")
            sub.prop(d, "cfl_max_substeps")
        # Surface tension (B1.1)
        st = box.box()
        st.label(text="Surface Tension (S2.14)")
        stg = d.surface_tension_group
        st.prop(stg, "surface_tension")
        if stg.surface_tension > 0.0:
            st.prop(stg, "csf_smoothing_passes")
            st.label(text="CFL substepping auto-enabled when σ > 0",
                     icon="INFO")
        sub2 = box.box()
        sub2.label(text="Particle Reseeding (S2.11)")
        sub2.prop(d, "reseed")
        if d.reseed:
            sub2.prop(d, "reseed_every_n_frames")
            row = sub2.row(align=True)
            row.prop(d, "reseed_min_per_cell")
            row.prop(d, "reseed_max_per_cell")
        box2 = layout.box()
        box2.label(text="Meshing / Output")
        box2.prop(d, "iso_level")
        box2.prop(d, "smooth_passes")
        box2.prop(d, "mesh_smooth_method")
        box2.prop(d, "mesh_smooth_passes")
        box2.prop(d, "decimate_ratio")
        box2.prop(d, "wall_margin_cells")
        box2.prop(d, "write_particles")
        box2.prop(d, "write_usd")


class GPUFLUID_PT_fluid(bpy.types.Panel):
    bl_label = "Fluid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.gpufluid_fluid.is_fluid

    def draw(self, context):
        layout = self.layout
        f = context.active_object.gpufluid_fluid
        layout.prop(f, "is_fluid")
        layout.prop(f, "ppc")
        layout.prop(f, "fill_mesh")
        # Per-source colour (S2.15 / B1.2)
        cbox = layout.box()
        cbox.label(text="Particle Colour (S2.15)")
        cbox.prop(f, "use_color")
        if f.use_color:
            cbox.prop(f, "color")


class GPUFLUID_PT_obstacle(bpy.types.Panel):
    bl_label = "Obstacle"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.gpufluid_obstacle.is_obstacle

    def draw(self, context):
        layout = self.layout
        o = context.active_object.gpufluid_obstacle
        layout.prop(o, "is_obstacle")
        layout.prop(o, "obstacle_type")
        box = layout.box()
        box.label(text="Motion (D4.6)")
        box.prop(o, "motion_type")
        if o.motion_type == "LINEAR":
            box.prop(o, "motion_velocity")


class GPUFLUID_PT_inflow(bpy.types.Panel):
    bl_label = "Inflow"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.gpufluid_inflow.is_inflow

    def draw(self, context):
        layout = self.layout
        i = context.active_object.gpufluid_inflow
        layout.prop(i, "is_inflow")
        layout.prop(i, "rate_per_sec")
        layout.prop(i, "velocity")
        row = layout.row(align=True)
        row.prop(i, "frame_start"); row.prop(i, "frame_end")


class GPUFLUID_PT_outflow(bpy.types.Panel):
    bl_label = "Outflow"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.gpufluid_outflow.is_outflow

    def draw(self, context):
        layout = self.layout
        o = context.active_object.gpufluid_outflow
        layout.prop(o, "is_outflow")
        row = layout.row(align=True)
        row.prop(o, "frame_start"); row.prop(o, "frame_end")


# [BLK A8.7.1] Whitewater sub-panel (B1.5)
class GPUFLUID_PT_whitewater(bpy.types.Panel):
    bl_label = "Whitewater (W7)"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.gpufluid_domain.is_domain

    def draw(self, context):
        layout = self.layout
        ww = context.active_object.gpufluid_domain.whitewater_group
        layout.prop(ww, "enable")
        col = layout.column()
        col.active = ww.enable
        col.prop(ww, "speed_threshold")
        col.prop(ww, "lifetime_sec")
        col.prop(ww, "emit_per_frame_max")
        col.prop(ww, "total_cap")
        box = col.box()
        box.label(text="Class Visibility (cache reader)")
        row = box.row(align=True)
        row.prop(ww, "show_foam", toggle=True)
        row.prop(ww, "show_spray", toggle=True)
        row.prop(ww, "show_bubble", toggle=True)


class GPUFLUID_PT_bake(bpy.types.Panel):
    bl_label = "Bake"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY

    @classmethod
    def poll(cls, context):
        return any(o.gpufluid_domain.is_domain for o in context.scene.objects)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("gpufluid.bake", icon="PLAY")
        row = layout.row(align=True)
        row.operator("gpufluid.clear_cache", icon="TRASH")
        row.operator("gpufluid.open_cache_dir", icon="FILE_FOLDER")
