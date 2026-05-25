"""[BLK A8.2/A8.3/A8.4] Property groups for Domain, Fluid, Obstacle."""
import bpy


# [BLK A8.2.1] Surface tension sub-group attached to Domain (B1.1)
# No @block decorator here: addon is loaded inside Blender's bpy process and
# importing gpufluid.blocks pulls Warp/CUDA. The block is still discoverable via
# this tag + BLOCKS.md A8.2.1 row.
class GpufluidSurfaceTensionGroup(bpy.types.PropertyGroup):
    surface_tension: bpy.props.FloatProperty(
        name="Surface Tension σ",
        default=0.0, min=0.0, max=10.0, precision=4,
        description="S2.14 CSF coefficient (σ/ρ, m³/s²). 0 disables CSF. "
                    "Auto-enables CFL substepping when > 0 (capillary-wave dt limit)",
    )
    csf_smoothing_passes: bpy.props.IntProperty(
        name="CSF Smoothing Passes",
        default=2, min=0, max=10,
        description="Box-blur passes on the indicator χ before curvature (more = smoother κ, "
                    "less surface detail)",
    )


# [BLK A8.2.2] Whitewater sub-group attached to Domain (B1.5)
# Mirrors the TOML `output.whitewater_*` knobs (cli/config.py:199-203) plus
# class-visibility toggles consumed by the cache loader at import time
# (cache_loader honors show_* by filtering whitewater_kinds/*.npy on read).
class GpufluidWhitewaterGroup(bpy.types.PropertyGroup):
    enable: bpy.props.BoolProperty(
        name="Whitewater (W7)", default=False,
        description="Emit foam / spray / bubble particles. Three classes are "
                    "auto-assigned by density (see W7.4 classifier)",
    )
    speed_threshold: bpy.props.FloatProperty(
        name="Speed Threshold (m/s)", default=4.0, min=0.0, max=100.0,
        description="Emit when |particle velocity| exceeds this",
    )
    lifetime_sec: bpy.props.FloatProperty(
        name="Lifetime (s)", default=1.5, min=0.05, max=60.0,
        description="Base whitewater particle lifetime (per-class lifetimes "
                    "in WhitewaterConfig stay at their defaults)",
    )
    emit_per_frame_max: bpy.props.IntProperty(
        name="Emit/Frame Cap", default=4000, min=0, max=200000,
        description="Upper bound on new whitewater particles each frame",
    )
    total_cap: bpy.props.IntProperty(
        name="Total Cap", default=80000, min=0, max=2_000_000,
        description="Absolute whitewater particle cap (oldest get culled)",
    )
    # Class-visibility — consumed at cache-load time, not at sim time.
    show_foam: bpy.props.BoolProperty(name="Foam", default=True)
    show_spray: bpy.props.BoolProperty(name="Spray", default=True)
    show_bubble: bpy.props.BoolProperty(name="Bubble", default=True)
    # W7.7 trapped-air potential (B3.3): weight emission by Ihmsen 2012's
    # divergent-velocity potential instead of the legacy |v|>threshold.
    use_potential: bpy.props.BoolProperty(
        name="Trapped-Air Potential (W7.7)", default=False,
        description="Bias emit toward turbulent flow pockets (Ihmsen 2012); "
                    "fewer but more representative whitewater particles",
    )
    potential_radius: bpy.props.FloatProperty(
        name="Potential Radius", default=0.0, min=0.0, max=2.0,
        description="Neighbour search radius (m). 0 = auto = 2.5·dx",
    )
    potential_v_max: bpy.props.FloatProperty(
        name="Potential v_max", default=10.0, min=0.1, max=100.0,
        description="Velocity normaliser for the (1 - |v_ij|/v_max) factor",
    )
    # B3.2 — wave-crest potential weight. 0 = trapped-air only (current).
    # >0 adds beta * |div n_hat| to the emit weight, so wave crests and
    # breaking ridges emit more spray than the velocity-only heuristic.
    wave_crest_weight: bpy.props.FloatProperty(
        name="Wave-Crest Weight (W7.8)", default=0.0, min=0.0, max=4.0,
        description="Blend in surface-curvature emission (Ihmsen 2012 §3.2). "
                    "0 = trapped-air only; 1 = equal blend; higher = "
                    "crest-dominated. Off by default for back-compat.",
    )


# [BLK A8.2]
class GpufluidDomainProps(bpy.types.PropertyGroup):
    is_domain: bpy.props.BoolProperty(name="Is gpufluid Domain", default=False)
    resolution: bpy.props.IntProperty(name="Resolution", default=64, min=8, max=512, soft_max=256,
                                      description="Cells per side of the longest domain axis")
    # default left empty; the Add Domain helper fills it with an absolute path
    # (Blender 5.x extensions reject "//" relative prefix in defaults).
    cache_dir: bpy.props.StringProperty(name="Cache Directory", default="",
                                        subtype="DIR_PATH")
    target_object: bpy.props.PointerProperty(
        type=bpy.types.Object, name="Target mesh",
        description="Object that will receive the mesh sequence. If empty, a new mesh is created on bake."
    )

    # simulation
    frames: bpy.props.IntProperty(name="Frames", default=120, min=1, max=10000)
    fps: bpy.props.IntProperty(name="FPS", default=24, min=1, max=240)
    dt: bpy.props.FloatProperty(name="dt", default=0.005, min=1e-5, max=0.1, precision=5)
    pressure_iters: bpy.props.IntProperty(name="Pressure Iterations", default=60, min=1, max=2000)
    flip_blend: bpy.props.FloatProperty(name="FLIP/PIC Blend", default=0.95, min=0.0, max=1.0,
                                        description="1.0 = pure FLIP, 0.0 = pure PIC")
    gravity: bpy.props.FloatProperty(name="Gravity Y", default=-9.81)

    # pressure / stability (v0.4)
    pressure_solver: bpy.props.EnumProperty(
        name="Pressure Solver",
        items=[("jacobi", "Jacobi", "Robust, simple, slow (>=60 iters typically)"),
               ("pcg", "PCG", "Diagonal-preconditioned conjugate gradient (5-10x fewer iters)")],
        default="pcg",
    )
    use_cfl: bpy.props.BoolProperty(name="CFL Substepping", default=False,
                                    description="Auto-split dt when particles move too fast")
    cfl_factor: bpy.props.FloatProperty(name="CFL", default=0.5, min=0.05, max=2.0)
    cfl_max_substeps: bpy.props.IntProperty(name="Max Substeps / Frame", default=16, min=1, max=128)

    # meshing
    iso_level: bpy.props.FloatProperty(name="Iso Level", default=0.6, min=0.01, max=10.0)
    smooth_passes: bpy.props.IntProperty(name="Density Smoothing Passes", default=2, min=0, max=20)
    mesh_smooth_passes: bpy.props.IntProperty(name="Mesh Smoothing Passes", default=2, min=0, max=30,
                                              description="Post-MC Taubin/Laplacian passes on the surface mesh")
    mesh_smooth_method: bpy.props.EnumProperty(
        name="Mesh Smooth Method",
        items=[("taubin", "Taubin", "Volume-preserving"),
               ("laplacian", "Laplacian", "Cheaper, shrinks slightly")],
        default="taubin",
    )
    decimate_ratio: bpy.props.FloatProperty(name="Decimation Ratio", default=1.0, min=0.05, max=1.0,
                                            description="Keep this fraction of triangle faces (M5.6)")
    wall_margin_cells: bpy.props.IntProperty(name="Wall Margin (cells)", default=2, min=0, max=8,
                                              description="Strip mesh near domain walls (M5.7)")
    write_particles: bpy.props.BoolProperty(name="Save Particles", default=False)
    write_usd: bpy.props.BoolProperty(name="Write USD cache (I6.5)", default=True,
                                       description="Native MeshSequenceCache modifier on import")

    # surface tension (S2.14) — nested group, see GpufluidSurfaceTensionGroup
    surface_tension_group: bpy.props.PointerProperty(type=GpufluidSurfaceTensionGroup)

    # whitewater (W7) — nested group, see GpufluidWhitewaterGroup
    whitewater_group: bpy.props.PointerProperty(type=GpufluidWhitewaterGroup)

    # reseed (S2.11)
    reseed: bpy.props.BoolProperty(name="Particle Reseeding", default=False,
                                    description="Bound per-cell density (fix voids on long sims)")
    reseed_every_n_frames: bpy.props.IntProperty(name="Reseed Every N Frames", default=5, min=1, max=60)
    reseed_min_per_cell: bpy.props.IntProperty(name="Reseed Min/Cell", default=4, min=1, max=64)
    reseed_max_per_cell: bpy.props.IntProperty(name="Reseed Max/Cell", default=16, min=1, max=64)

    # solver dispatch (B17.12)
    solver: bpy.props.EnumProperty(
        name="Solver",
        items=[
            ("flip", "FLIP (general)", "FLIP/APIC pressure-projection. Good for splashes, dam-break, low-viscosity scenes."),
            ("mpm",  "MPM (water on solids)", "MLS-MPM shell-out (F3.7). Required for water spreading laterally on a flat rigid surface; FLIP fails this case."),
        ],
        default="flip",
        description="Underlying solver algorithm. MPM is the production path for water-on-cube and viscous fluids.",
    )
    # MPM-specific knobs (used only when solver == 'mpm')
    mpm_bulk_modulus: bpy.props.FloatProperty(
        name="Bulk Modulus", default=1500.0, min=100.0, max=10000.0,
        description="Water stiffness (high = closer to real water but less stable). 1500 is the production default; 3000+ needs finer grid.")
    mpm_rpic_damping: bpy.props.FloatProperty(
        name="Viscosity (RPIC)", default=0.15, min=0.0, max=1.0,
        description="APIC rotational-mode damping. 0 = pure APIC (lively swirly water). 0.15 = subtle viscosity. 0.3+ = syrupy.")
    mpm_grid_v_damping: bpy.props.FloatProperty(
        name="Global Damping", default=0.998, min=0.9, max=1.0, precision=4,
        description="Per-step grid velocity damping. 1.0 = none. 0.995 = mild. 0.99 = noticeable.")
    mpm_cube_friction: bpy.props.FloatProperty(
        name="Obstacle Top Friction", default=0.6, min=0.0, max=1.0,
        description="Tangential keep-factor on top face of box obstacles. 1.0 = pure slip (water radiates out instantly). 0.3-0.6 = water accumulates then overflows. 0.0 = fully sticky.")
    mpm_v_terminal: bpy.props.FloatProperty(
        name="Stream Terminal Velocity (m/s)", default=-1.0, min=-5.0, max=0.0,
        description="Cap on |v_z| inside the inflow column. -1.0 mimics air drag on a laminar stream. More negative = faster pour (more splash on impact).")
    mpm_vz_max_splash: bpy.props.FloatProperty(
        name="Anti-Splash Cap (m/s)", default=0.3, min=0.0, max=2.0,
        description="Hard clamp on upward v_z above obstacle top. 0.3 = realistic; 0.0 = no splash possible.")
    mpm_initial_velocity: bpy.props.FloatProperty(
        name="Initial Pour Velocity (m/s)", default=-0.3, min=-3.0, max=0.0,
        description="Starting vertical velocity for inflow column, in m/s. "
                    "Normalised by domain size at bake time. "
                    "-0.3 = gentle pour; 0.0 = drop from rest (more splash).")
    # Phase 1 escape-hatch: raw TOML merged on top of the generated config in
    # config_builder.build_toml. Parsed via stdlib tomllib; deep-merged into
    # the final dict before serialization. Errors propagate to the bake
    # operator which reports them to the user.
    toml_overrides: bpy.props.StringProperty(
        name="TOML overrides",
        description="Advanced: raw TOML merged on top of generated config",
        default="",
    )
    # B18 — per-vertex colour mixing mode (renders ride on this when any
    # source has Tint Particles on)
    color_mix_mode: bpy.props.EnumProperty(
        name="Colour Mix Mode",
        items=[
            ("off", "Off", "No per-vertex colour written; mesh stays neutral"),
            ("linear", "Linear RGB",
             "Inverse-distance² RGB average from nearest particles. "
             "Fast (GPU); blue+yellow blends to grey."),
            ("mixbox", "Mixbox (pigment)",
             "Šochorová & Jamriška 2021 pigment-space mixing via "
             "pymixbox. Slower (CPU); blue+yellow blends to green. "
             "Falls back to Linear if pymixbox isn't installed."),
        ],
        default="linear",
        description="How particle colours blend at mesh vertices during bake",
    )


# [BLK A8.3]
class GpufluidFluidProps(bpy.types.PropertyGroup):
    is_fluid: bpy.props.BoolProperty(name="Is gpufluid Source", default=False)
    ppc: bpy.props.IntProperty(name="Particles per Cell (cubed)", default=8, min=1, max=64,
                               description="N particles per cell ≈ ppc")
    # D4.5.2 — fill the object's actual mesh shape (not just its bbox)
    fill_mesh: bpy.props.BoolProperty(name="Fill Mesh Volume (D4.5.2)", default=False,
                                       description="Seed particles inside the object's triangle mesh "
                                                   "(via GPU ray-cast); if off, fills its bounding box")
    # S2.15 — per-particle colour seeded from this source (B1.2)
    use_color: bpy.props.BoolProperty(
        name="Tint Particles (S2.15)", default=False,
        description="Stamp this source's RGB on emitted particles. Enables the per-particle "
                    "colour attribute pipeline (linear RGB; Mixbox arrives in v0.8 B2)",
    )
    color: bpy.props.FloatVectorProperty(
        name="Particle Color",
        default=(1.0, 1.0, 1.0), size=3, min=0.0, max=1.0, subtype="COLOR",
        description="Linear-RGB triple stamped on particles from this source when Tint is on",
    )
    # S2.18 / B11 — per-particle temperature seeded from this source
    use_temperature: bpy.props.BoolProperty(
        name="Stamp Temperature (S2.18 / B11)", default=False,
        description="Stamp this source's temperature on emitted particles. "
                    "Enables the per-particle scalar attribute pipeline "
                    "(used for lava drop, hot/cold mixing demos)",
    )
    temperature: bpy.props.FloatProperty(
        name="Temperature",
        default=300.0, min=0.0, max=5000.0, soft_max=2000.0,
        description="Scalar temperature stamped on particles from this source "
                    "when 'Stamp Temperature' is on. Mixes via P2G/G2P like "
                    "colour (lava drop: 1500 K hot, basin: 300 K cool)",
    )


# [BLK A8.4]
class GpufluidObstacleProps(bpy.types.PropertyGroup):
    is_obstacle: bpy.props.BoolProperty(name="Is gpufluid Obstacle", default=False)
    obstacle_type: bpy.props.EnumProperty(
        name="Type",
        items=[
            ("BBOX", "Bounding Box", "Use object's world bounding box as an axis-aligned box obstacle"),
            ("SPHERE", "Sphere", "Use object location + max(scale) as sphere obstacle"),
            ("CYLINDER_Y", "Cylinder Y", "Use object location + x-scale (radius) + y-scale (half-height) as Y-cylinder"),
            ("MESH", "Mesh", "Export object's mesh, use it as a mesh obstacle"),
            ("PLANE", "Plane (ramp)", "Use object Z-axis as plane normal, location as point; "
                                       "object bbox limits the slab (D4.2.4)"),
        ],
        default="BBOX",
    )
    # v0.5 — animated obstacle motion (D4.6)
    motion_type: bpy.props.EnumProperty(
        name="Motion",
        items=[("NONE", "Static", "Obstacle does not move"),
               ("LINEAR", "Linear velocity", "Obstacle translates with constant velocity")],
        default="NONE",
    )
    motion_velocity: bpy.props.FloatVectorProperty(
        name="Velocity (world u/s)", default=(0.0, 0.0, 0.0), size=3, subtype="VELOCITY",
    )


# [BLK A8.4 / D4.7] Inflow / outflow region property groups
class GpufluidInflowProps(bpy.types.PropertyGroup):
    is_inflow: bpy.props.BoolProperty(name="Is gpufluid Inflow", default=False)
    velocity: bpy.props.FloatVectorProperty(
        name="Initial velocity", default=(0.0, -3.0, 0.0), size=3, subtype="VELOCITY",
    )
    rate_per_sec: bpy.props.FloatProperty(name="Particles/sec", default=10000.0, min=0.0)
    frame_start: bpy.props.IntProperty(name="Frame Start", default=0, min=0)
    frame_end: bpy.props.IntProperty(name="Frame End", default=10000, min=0)
    # B18 — per-source attributes (mirrors GpufluidFluidProps)
    use_color: bpy.props.BoolProperty(
        name="Tint Particles (B18)", default=False,
        description="Stamp this inflow's RGB on emitted particles. Combined "
                    "with the Domain mix-mode, this drives the per-vertex "
                    "mesh colour in the bake.",
    )
    color: bpy.props.FloatVectorProperty(
        name="Particle Color",
        default=(1.0, 1.0, 1.0), size=3, min=0.0, max=1.0, subtype="COLOR",
        description="Linear-RGB stamped on particles from this inflow "
                    "when Tint is on",
    )
    use_temperature: bpy.props.BoolProperty(
        name="Stamp Temperature (B18)", default=False,
        description="Stamp this inflow's temperature on emitted particles "
                    "(carried through MPM bake as per-particle scalar)",
    )
    temperature: bpy.props.FloatProperty(
        name="Temperature", default=20.0, min=0.0, max=5000.0, soft_max=2000.0,
        description="Scalar temperature stamped on particles from this inflow "
                    "(°C in default convention; only meaningful when 'Stamp "
                    "Temperature' is on)",
    )


class GpufluidOutflowProps(bpy.types.PropertyGroup):
    is_outflow: bpy.props.BoolProperty(name="Is gpufluid Outflow", default=False)
    frame_start: bpy.props.IntProperty(name="Frame Start", default=0, min=0)
    frame_end: bpy.props.IntProperty(name="Frame End", default=10000, min=0)
