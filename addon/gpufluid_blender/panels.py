"""[BLK A8.7] N-sidebar UI panels for gpufluid."""
import bpy

try:
    from ._blocks import block
except ImportError:
    # dodge: tests load this file under a stub package with no _blocks
    # (spec_from_file_location harnesses) — registration is irrelevant
    # there; the real registration happens on package import. See _blocks.py.
    def block(_bid, _desc=""):
        def _w(fn):
            return fn
        return _w


PANEL_CATEGORY = "GpuFluid"


@block("A8.7", "UI panels (8 sections under GpuFluid sidebar category)")
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


@block("A8.7", "UI panels (8 sections under GpuFluid sidebar category)")
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
        # UI mode (lazy import for the stub-test load path). "cook" = 2-button
        # mode (pick a material → bake); "pro" = every knob visible.
        try:
            from . import preferences
            pro = getattr(preferences.get_prefs(context), "ui_mode", "cook") == "pro"
        except Exception:
            pro = False

        # ── Material presets — the Cook path: one dropdown sets EVERYTHING ──
        pbox = layout.box()
        pbox.label(text="Материал / Material", icon="PRESET")
        pbox.operator_menu_enum("gpufluid.apply_preset", "preset",
                                text="Выбрать жижу / Pick a fluid")

        if not pro:
            # Cook mode: only length + cache, then Bake (in the Bake panel).
            col = layout.column()
            col.prop(d, "frames")
            col.prop(d, "fps")
            col.prop(d, "cache_dir")
            tip = layout.box()
            tip.label(text="«Кухарка»: выбрал жижу → Bake.", icon="INFO")
            tip.label(text="Все ручки — режим Pro (настройки аддона).")
            return

        # ── Pro mode: every knob ──
        col = layout.column()
        col.prop(d, "resolution")
        col.prop(d, "cache_dir")
        col.prop(d, "target_object")

        # Solver selection (B17.12 / F3.7)
        sbox = layout.box()
        sbox.label(text="Solver", icon="PHYSICS")
        sbox.prop(d, "solver", expand=True)
        if d.solver == "mpm":
            sbox.label(text="MPM Settings (F3.7 / S2.17.*)", icon="MOD_FLUIDSIM")
            sbox.prop(d, "mpm_material")
            if d.mpm_material == "viscoelastic":
                # Rope-coiling material — stiffness/yield instead of bulk.
                sbox.prop(d, "mpm_young_modulus")
                sbox.prop(d, "mpm_poisson")
                sbox.prop(d, "mpm_yield_stress")
                sbox.prop(d, "mpm_viscosity")
                sbox.prop(d, "mpm_floor_friction")
            else:
                sbox.prop(d, "mpm_bulk_modulus")
                sbox.prop(d, "mpm_viscosity")
                sbox.prop(d, "mpm_floor_friction")
            sbox.prop(d, "mpm_rpic_damping")
            sbox.prop(d, "mpm_grid_v_damping")
            sbox.prop(d, "mpm_cube_friction")
            sbox.prop(d, "mpm_v_terminal")
            sbox.prop(d, "mpm_vz_max_splash")
            sbox.prop(d, "mpm_initial_velocity")

        box = layout.box()
        box.label(text="Simulation")
        box.prop(d, "frames")
        box.prop(d, "fps")
        box.prop(d, "dt")
        # FLIP-only sub-section
        if d.solver == "flip":
            box.prop(d, "pressure_solver")
            box.prop(d, "pressure_iters")
            box.prop(d, "flip_blend")
        box.prop(d, "gravity")
        sub = box.box()
        sub.prop(d, "use_cfl")
        if d.use_cfl:
            sub.prop(d, "cfl_factor")
            sub.prop(d, "cfl_max_substeps")
            # FU-029: MPM stress waves travel faster than the advection CFL
            # accounts for — cfl_factor > 1.0 under-substeps them (possible
            # divergence/aliasing at rigid contacts). Honest UI: alert row
            # only; the value is NOT clamped (back-compat with saved scenes).
            if d.solver == "mpm" and d.cfl_factor > 1.0:
                warn = sub.row()
                warn.alert = True
                warn.label(icon="ERROR",
                           text="CFL > 1.0 under-substeps MPM stress waves "
                                "— keep <= 1.0 for stable bakes")
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
        # B18 — per-vertex colour mixing mode (mesher applies when any
        # source has Tint Particles on)
        cmix = box2.box()
        cmix.label(text="Colour Mixing (B18)", icon="COLOR")
        cmix.prop(d, "color_mix_mode")
        if d.color_mix_mode == "mixbox":
            cmix.label(text="Requires `pip install pymixbox`", icon="INFO")


@block("A8.7", "UI panels (8 sections under GpuFluid sidebar category)")
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
        # S2.17.10 — source shape: any object can be a source of its own shape.
        layout.prop(f, "source_type")
        layout.prop(f, "ppc")
        if f.source_type == "MESH":
            layout.label(text="Mesh must be watertight (closed)", icon="INFO")
        # Per-source colour (S2.15 / B1.2)
        cbox = layout.box()
        cbox.label(text="Particle Colour (S2.15)")
        cbox.prop(f, "use_color")
        if f.use_color:
            cbox.prop(f, "color")
        # Per-source temperature (S2.18 / B11)
        tbox = layout.box()
        tbox.label(text="Particle Temperature (S2.18 / B11)")
        tbox.prop(f, "use_temperature")
        if f.use_temperature:
            tbox.prop(f, "temperature")


@block("A8.7", "UI panels (8 sections under GpuFluid sidebar category)")
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


@block("A8.7", "UI panels (8 sections under GpuFluid sidebar category)")
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
        try:
            from . import preferences
            pro = getattr(preferences.get_prefs(context), "ui_mode", "cook") == "pro"
        except Exception:
            pro = False
        layout.prop(i, "is_inflow")
        layout.prop(i, "rate_per_sec")
        layout.prop(i, "velocity")
        # S2.17.7.JITTER — coil seed; in Cook mode the material preset sets it.
        if pro:
            layout.prop(i, "velocity_jitter")
        row = layout.row(align=True)
        row.prop(i, "frame_start"); row.prop(i, "frame_end")
        # B18 — per-inflow colour + temperature (mirrors Fluid panel)
        cbox = layout.box()
        cbox.label(text="Particle Colour (B18)")
        cbox.prop(i, "use_color")
        if i.use_color:
            cbox.prop(i, "color")
        tbox = layout.box()
        tbox.label(text="Particle Temperature (B18)")
        tbox.prop(i, "use_temperature")
        if i.use_temperature:
            tbox.prop(i, "temperature")


@block("A8.7", "UI panels (8 sections under GpuFluid sidebar category)")
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
@block("A8.7.1", "Whitewater sub-panel (B1.5)")
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
        pbox = col.box()
        pbox.label(text="Trapped-Air Potential (W7.7)")
        pbox.prop(ww, "use_potential")
        if ww.use_potential:
            pbox.prop(ww, "potential_radius")
            pbox.prop(ww, "potential_v_max")
            pbox.prop(ww, "trapped_air_weight")  # audit-r8 #2: now tunable
            pbox.prop(ww, "wave_crest_weight")
        box = col.box()
        box.label(text="Class Visibility (cache reader)")
        row = box.row(align=True)
        row.prop(ww, "show_foam", toggle=True)
        row.prop(ww, "show_spray", toggle=True)
        row.prop(ww, "show_bubble", toggle=True)


@block("A8.7", "UI panels (8 sections under GpuFluid sidebar category)")
class GPUFLUID_PT_help(bpy.types.Panel):
    """Collapsible per-knob help in the user's chosen language (2026-06-15).

    Default-closed so it never clutters the panel; one click expands a
    plain-language cheat-sheet of what the main physics knobs do. Language
    follows the addon preference `hint_language`."""
    bl_label = "Помощь / Help"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PANEL_CATEGORY
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.gpufluid_domain.is_domain

    def draw(self, context):
        layout = self.layout
        # Lazy imports: this file is also loaded standalone under a bpy stub in
        # tests (spec_from_file_location) where relative imports can't resolve.
        try:
            from . import preferences
            from .operators import presets as _presets
            lang = getattr(preferences.get_prefs(context), "hint_language", "ru")
            lines = _presets.HINT_TEXT.get(lang, _presets.HINT_TEXT["en"])
        except Exception:
            lines = []
        col = layout.column(align=True)
        for line in lines:
            col.label(text=line)


@block("A8.7", "UI panels (8 sections under GpuFluid sidebar category)")
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
        col.operator("gpufluid.render", text="Render", icon="RENDER_STILL")
        row = layout.row(align=True)
        row.operator("gpufluid.clear_cache", icon="TRASH")
        row.operator("gpufluid.open_cache_dir", icon="FILE_FOLDER")
        layout.label(text="Cache (manual attach):")
        layout.operator("gpufluid.attach_cache", text="Attach Surface", icon="LINKED")
        layout.operator("gpufluid.attach_ww_cache", text="Attach Whitewater", icon="STICKY_UVS_DISABLE")
        layout.operator("gpufluid.detach_cache", text="Detach", icon="UNLINKED")
        # A8.9 — one-click production Eevee preset
        layout.separator()
        layout.label(text="Render:")
        layout.operator("gpufluid.apply_eevee_preset", icon="OUTPUT")
        # Phase 1 escape-hatch: raw TOML merged on top of the generated scene
        # config. Useful for prototyping settings before they get a UI knob.
        domain_obj = next(
            (o for o in context.scene.objects if o.gpufluid_domain.is_domain),
            None,
        )
        if domain_obj is not None:
            adv = layout.box()
            adv.label(text="Advanced: TOML overrides", icon="TEXT")
            adv.prop(domain_obj.gpufluid_domain, "toml_overrides", text="")
