"""[Layer C7] CLI commands: simulate, bench, info."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import warp as wp

from .. import __version__
from ..blocks import block, BlockError, by_layer, get_registry
from ..solvers.solver3d import FlipSolver3D
from ..meshing.surface import MeshExtractor
from ..io.ply import write_ply, write_particles_npy
from ..io.cache import CacheManifest, write_cache_manifest
from ..io.usd import write_usd_mesh_sequence
from ..primitives.sdf import (
    sdf_sphere, sdf_box, sdf_cylinder_y, sdf_plane, sdf_union,
)
from ..domain.regions import InflowBox, OutflowBox
from ..primitives.animation import LinearMotion, KeyframeMotion
from .config import (
    SceneCfg, load_scene, MotionCfg,
    ObstacleSphereCfg, ObstacleBoxCfg, ObstacleCylinderYCfg, ObstacleMeshCfg, ObstaclePlaneCfg,
)


def _build_motion(mcfg: MotionCfg, fps: int):
    if mcfg.kind == "linear":
        return LinearMotion(velocity=tuple(mcfg.velocity or (0, 0, 0)), fps=fps)
    if mcfg.kind == "keyframes":
        return KeyframeMotion(keyframes=list(mcfg.keyframes or []))
    raise BlockError("C7.2", f"unknown motion kind: {mcfg.kind!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_obstacle_sdf(scene: SceneCfg, grid_xyz) -> Optional[np.ndarray]:
    """Combine all *static* obstacle SDFs via union. Animated ones are
    registered with the solver and rebuilt per frame, not included here."""
    static_indices = [i for i, m in enumerate(scene.obstacle_motion) if m is None] \
        if scene.obstacle_motion else list(range(len(scene.obstacle)))
    if not static_indices:
        return None
    parts = []
    for i in static_indices:
        ob = scene.obstacle[i]
        if isinstance(ob, ObstacleSphereCfg):
            parts.append(sdf_sphere(grid_xyz, ob.center, ob.radius))
        elif isinstance(ob, ObstacleBoxCfg):
            parts.append(sdf_box(grid_xyz, ob.center, ob.half_size))
        elif isinstance(ob, ObstacleCylinderYCfg):
            parts.append(sdf_cylinder_y(grid_xyz, ob.center, ob.radius, ob.half_height))
        elif isinstance(ob, ObstaclePlaneCfg):
            # finite ramp = max(plane_sdf, box_sdf): solid only where both negative
            p_sdf = sdf_plane(grid_xyz, ob.point, ob.normal)
            half_size = tuple((np.asarray(ob.bbox_hi) - np.asarray(ob.bbox_lo)) * 0.5)
            center = tuple((np.asarray(ob.bbox_hi) + np.asarray(ob.bbox_lo)) * 0.5)
            b_sdf = sdf_box(grid_xyz, center, half_size)
            parts.append(np.maximum(p_sdf, b_sdf))
        elif isinstance(ob, ObstacleMeshCfg):
            mesh_path = ob.path
            if scene.config_dir is not None and not Path(mesh_path).is_absolute():
                mesh_path = str(scene.config_dir / mesh_path)
            import trimesh as _tm
            mesh = _tm.load(mesh_path, force="mesh", process=False)
            if ob.rotate_deg is not None:
                from trimesh.transformations import euler_matrix
                rx, ry, rz = [float(a) * np.pi / 180.0 for a in ob.rotate_deg]
                mesh.apply_transform(euler_matrix(rx, ry, rz, "sxyz"))
            if float(ob.scale) != 1.0:
                mesh.apply_scale(float(ob.scale))
            if any(float(t) != 0.0 for t in ob.translate):
                mesh.apply_translation(np.asarray(ob.translate, dtype=np.float64))
            tris = np.asarray(mesh.triangles, dtype=np.float32)
            n_tris = len(tris)
            if n_tris >= 256:
                # GPU BVH path: indicator-style SDF (only inside/outside used by
                # add_solid_from_sdf). Scales to 100k+ triangles where the CPU
                # `mesh_to_sdf` would OOM (proximity query is O(cells × tris)).
                from ..domain.mesh_sdf_gpu import mesh_indicator_sdf_gpu
                parts.append(mesh_indicator_sdf_gpu(grid_xyz, tris, scene.dx))
            else:
                # Small mesh: CPU true-SDF (gives smooth distance near surface)
                from ..domain.mesh_sdf import mesh_to_sdf
                parts.append(mesh_to_sdf(mesh_path, grid_xyz,
                                        scale=ob.scale,
                                        translate=ob.translate,
                                        rotate_deg=ob.rotate_deg))
        else:
            raise BlockError("C7.2", f"obstacle type not handled: {type(ob).__name__}")
    return sdf_union(*parts)


# ---------------------------------------------------------------------------
# C7.2 — simulate
# ---------------------------------------------------------------------------

# [BLK C7.2]
@block("C7.2", "simulate command: run a scene and write a mesh cache")
def _cmd_simulate_mpm(args: argparse.Namespace, scene) -> int:
    """[BLK F3.7] MPM dispatch path inside `gpufluid simulate`.

    Translates the TOML scene config (FLIP-shaped — first [[fluid]] box, all
    [[obstacle]] boxes, single domain) into an :class:`MpmConfig` and runs
    the bake. Writes points-only PLYs into ``output.cache_dir/mesh/`` plus a
    minimal ``cache.json`` so the addon's Attach Cache (A8.6) and the
    `gpufluid render` command (A8.12) can find the result.
    """
    import json
    from ..sim.mpm import MpmSolver
    from ..sim.mpm.solver import (
        MpmConfig, MpmCubeCollider, MpmTap, MpmAntiSplash, make_column,
    )
    from ..sim.mpm.inflow import MpmInflow
    sim = scene.simulation
    # Build initial column from the single [fluid] box (the multi-source
    # [[fluids]] list seeds get coloured-per-source for FLIP; MPM uses
    # only the first geometric region). Fall back to first [[inflow]].
    # Source selection. scene.fluid is auto-populated with a default
    # FluidBoxCfg whose `lo`/`hi` are set, so a naive `hasattr` check would
    # silently swallow the user's real source. Order of preference:
    #   1) explicit [[fluids]] from TOML (addon emission)
    #   2) explicit [fluid] from TOML — but ONLY if no [[inflow]] is set,
    #      otherwise we assume the bake is inflow-driven and the default
    #      scene.fluid is a leftover artefact.
    f0 = None
    if scene.fluids:
        for cand in scene.fluids:
            if hasattr(cand, "lo") and hasattr(cand, "hi"):
                f0 = cand
                break
    elif not scene.inflow and hasattr(scene.fluid, "lo"):
        f0 = scene.fluid
    if f0 is not None:
        lo, hi = f0.lo, f0.hi
        cx = 0.5 * (lo[0] + hi[0])
        cy = 0.5 * (lo[1] + hi[1])
        half = max(0.005, 0.5 * min(hi[0] - lo[0], hi[1] - lo[1]))
        z_lo, z_hi = lo[2], hi[2]
        n_xy = max(4, int(2 * half / scene.dx))
        n_z = max(20, int((z_hi - z_lo) / scene.dx))
        column = make_column((cx, cy), half, z_lo, z_hi, n_xy, n_z)
    elif scene.inflow:
        # No static fluid box — the bake is driven entirely by continuous
        # [[inflow]] zones (S2.17.7). Start with an empty initial column;
        # all particles are pre-allocated under inflow gates below.
        column = np.empty((0, 3), dtype=np.float32)
    else:
        print("[mpm] no [fluid] or [[inflow]] zone — nothing to seed")
        return 2
    # S2.17.7 — continuous emission. Each [[inflow]] from the TOML becomes
    # a time-gated particle batch pre-allocated inside MpmSolver.
    inflows = tuple(
        MpmInflow(
            lo=tuple(inf.lo), hi=tuple(inf.hi),
            velocity=tuple(inf.velocity) if any(inf.velocity) else (0.0, 0.0, -1.0),
            rate_per_sec=float(inf.rate_per_sec),
            frame_start=int(inf.frame_start),
            frame_end=int(min(inf.frame_end, sim.frames)),
            keyframes=tuple(tuple(row) for row in getattr(inf, "keyframes", ()) or ()),
            # B18 — forward per-source attributes (None ⇒ source contributes
            # no colour/temperature; if any other source carries one, this
            # inflow's particles get the default white / 20 °C).
            color=tuple(inf.color) if inf.color is not None else None,
            temperature=float(inf.temperature) if inf.temperature is not None else None,
        )
        for inf in scene.inflow
    )

    # Each [[obstacle]] of type box becomes a slip-with-top-friction cube
    cubes = []
    for ob in scene.obstacle:
        if isinstance(ob, ObstacleBoxCfg):
            cubes.append(MpmCubeCollider(
                centre=tuple(ob.center),
                half_size=tuple(ob.half_size),
                tangential_friction=sim.mpm_cube_friction,
            ))
    # Place tap zone just above the highest cube top; if no cube, above floor
    if cubes:
        tap_z_min = max(c.centre[2] + c.half_size[2] + 0.005 for c in cubes)
        anti_splash_z = max(c.centre[2] for c in cubes)
    else:
        tap_z_min = 0.61
        anti_splash_z = 0.50
    # If we have no static column, derive tap XY from the first inflow zone.
    if f0 is None and scene.inflow:
        inf0 = scene.inflow[0]
        cx = 0.5 * (inf0.lo[0] + inf0.hi[0])
        cy = 0.5 * (inf0.lo[1] + inf0.hi[1])
        half = max(0.005, 0.5 * min(inf0.hi[0] - inf0.lo[0],
                                     inf0.hi[1] - inf0.lo[1]))
    elif f0 is None:
        # neither — shouldn't reach here (early-return above) but be safe
        cx, cy, half = 0.5, 0.5, 0.05
    # sim.frames is the count of OUTPUT frames at sim.fps. MPM internally counts
    # solver steps at dt, so total steps = frames * dump_every. Without this
    # multiplication a 240-frame request would bake only 0.24s @ dt=0.001.
    dump_every = max(1, int((1.0 / sim.fps) / max(sim.dt, 1e-6)))
    total_steps = sim.frames * dump_every
    # ── Gravity scaling for non-unit domains ───────────────────────────────
    # The MPM solver runs in a unit cube [0,1]³ (see solver.MpmDomainWalls).
    # One normalized unit equals `dom_size_z` real metres. To preserve real
    # falling time, gravity in normalised units = real_g (m/s²) / dom_size_z.
    # Without this, a 2.32 m domain has particles falling 2.32× too fast
    # (looks "cartoonish"); a 0.5 m domain has them falling 2× too slow.
    dom_size = scene.domain_size
    dom_z = float(dom_size[2]) if dom_size[2] > 1e-9 else 1.0
    g_world = float(sim.gravity)   # m/s², scalar (-9.81 default)
    g_norm = (0.0, 0.0, g_world / dom_z)
    # Velocities are in m/s real-world; convert to normalised units/s so the
    # solver (which lives in [0,1]³) sees them in the same space as gravity.
    inv_dom_z = 1.0 / dom_z
    # B18.1 — per-particle attributes for the initial column. Pull from
    # the resolved [fluid]/[[fluids]] entry (`f0`); broadcast to N rows.
    initial_color_arr = None
    initial_temp_arr = None
    n_col = column.shape[0]
    if n_col > 0 and f0 is not None:
        if getattr(f0, "color", None) is not None:
            initial_color_arr = np.tile(
                np.asarray(f0.color, dtype=np.float32), (n_col, 1)
            )
        if getattr(f0, "temperature", None) is not None:
            initial_temp_arr = np.full(
                (n_col,), float(f0.temperature), dtype=np.float32
            )
    cfg = MpmConfig(
        initial_column=column,
        initial_color=initial_color_arr,
        initial_temperature=initial_temp_arr,
        color_mix_mode=str(scene.output.color_mix_mode),
        initial_velocity_z=sim.mpm_initial_velocity * inv_dom_z,
        n_grid=max(scene.domain.resolution),
        n_frames=total_steps,
        dump_every=dump_every,
        inflows=inflows,
        fps=int(sim.fps),
        cubes=tuple(cubes),
        gravity=g_norm,
        tap=MpmTap(
            lo_x=cx - half - 0.005, hi_x=cx + half + 0.005,
            lo_y=cy - half - 0.005, hi_y=cy + half + 0.005,
            z_min=tap_z_min,
            v_terminal=sim.mpm_v_terminal * inv_dom_z,
        ),
        anti_splash=MpmAntiSplash(
            z_threshold=anti_splash_z,
            vz_min=-2.0 * inv_dom_z,
            vz_max=sim.mpm_vz_max_splash * inv_dom_z,
        ),
        dt=sim.dt,
    )
    cfg.fluid.bulk_modulus = sim.mpm_bulk_modulus
    cfg.fluid.rpic_damping = sim.mpm_rpic_damping
    cfg.fluid.grid_v_damping_scale = sim.mpm_grid_v_damping
    cache_dir = Path(scene.output.cache_dir).resolve()
    particles_dir = cache_dir / "particles_raw"  # raw MpmSolver dump location
    mesh_dir = cache_dir / "mesh"
    particles_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    solver = MpmSolver(cfg)
    n_init = int(solver.n_initial)
    n_inflow = int(solver.n_particles - solver.n_initial)
    print(f"[mpm] particles={solver.n_particles} "
          f"(initial={n_init}, inflow_pool={n_inflow})  "
          f"cubes={len(cubes)}  inflows={len(inflows)}  out={cache_dir}")
    # Round-20: catch MpmDivergenceError + write truncated cache.json +
    # exit rc=2. Pre-round-20 path silently swallowed NaN-divergence
    # via `print + break` in solver.run, leaving rc=0 + a misleading
    # cache.json with full frame_count. Addon's round-10 truncation
    # sanity caught it via mesh count vs cache.json mismatch, but the
    # user got no specific signal about WHY.
    from ..sim.mpm.solver import MpmDivergenceError
    truncated_at: int | None = None
    try:
        solver.run(particles_dir, progress=True)
    except MpmDivergenceError as exc:
        truncated_at = exc.last_dumped_frame
        # Stderr so addon's stdout-drain doesn't conflate with progress.
        import sys as _sys
        print(
            f"[mpm] {exc} — recovery: lower bulk_modulus (default 1500), "
            f"lower dt, or shrink resolution. See docs/BACKLOG.md MPM "
            f"stability entry.", file=_sys.stderr)
        # Fall through to mesh pass below: meshes whatever frames were
        # successfully dumped to particles_raw, writes truthful
        # cache.json with truncated_at_frame, then exits rc=2.

    # Mesh pass — turn the per-frame point clouds into triangle meshes
    # the addon's Attach Cache (A8.6) can stream.
    out_cfg = scene.output
    frame_count = 0
    if out_cfg.mesh:
        from ..io.ply import read_points_ply, write_ply
        nx = max(scene.domain.resolution)
        mesher = MeshExtractor(nx, nx, nx, 1.0 / nx, use_gpu_mc=True)
        sdf_search_cells = 5.0
        search_r_world = sdf_search_cells * (1.0 / nx)
        has_color_path = solver.attr_color is not None
        has_temp_path = solver.attr_temperature is not None
        colors_dir = cache_dir / "colors"
        temps_dir = cache_dir / "temperatures"
        # B11.3 — when [output] temperature_colormap is set, derive vertex
        # RGB from the temperature sidecar (preferred over the colour
        # sidecar when both exist). When the knob is unset (default),
        # behaviour is unchanged: the regular colour sidecar drives RGB.
        cmap_name = out_cfg.temperature_colormap
        cmap_fn = None
        t_min_cfg, t_max_cfg = out_cfg.temperature_range
        if cmap_name:
            from ..meshing.colormap import get_colormap
            cmap_fn = get_colormap(cmap_name)
        use_temp_color = bool(cmap_fn) and has_temp_path
        # Round-21: index sidecars by `step // dump_every`, not by a
        # local counter. The solver writes sidecar
        # `colors/frame_NNNN.npy` with NNNN = step // dump_every
        # (solver.py save_frame_ply); a local counter that only
        # increments on success would mis-align by every missing PLY,
        # silently swapping colour data between frames.
        frames_written = 0
        for step in range(0, cfg.n_frames + 1, cfg.dump_every):
            ply = particles_dir / f"sim_{step:010d}.ply"
            if not ply.exists():
                continue
            frame_idx = step // max(1, cfg.dump_every)
            pts = read_points_ply(ply)
            if pts.shape[0] == 0:
                write_ply(mesh_dir / f"frame_{frame_idx:04d}.ply",
                          np.zeros((0, 3), np.float32),
                          np.zeros((0, 3), np.int32))
                frames_written += 1
                continue
            pos_wp = wp.array(pts, dtype=wp.vec3, device=mesher.device)
            v, fc = mesher.extract(
                pos_wp, mesh_method="sdf",
                smooth_passes=2, mesh_smooth_passes=5,
                mesh_smooth_method=out_cfg.mesh_smooth_method,
                sdf_particle_radius_cells=2.2,
                sdf_search_radius_cells=sdf_search_cells,
            )
            if v is None:
                v = np.zeros((0, 3), np.float32)
                fc = np.zeros((0, 3), np.int32)
            # B18.4 / B11.3 — per-vertex colour. Load the matching sidecar
            # (already mask-filtered to active particles → same row order
            # as PLY). When [output] temperature_colormap is set AND a
            # temperatures sidecar exists, derive RGB from temperature
            # (preferred over the colour sidecar); otherwise fall back to
            # the regular colour sidecar.
            vc_u8 = None
            cols_np = None  # (P, 3) float32 in [0,1] for whichever source
            # Round-24: sidecars now de-duplicated — frame 0 written
            # once; subsequent frames only get their own .npy when the
            # particle count drifts (mid-bake spawn/death). Fall back
            # to frame_0000.npy whenever the per-frame file is absent.
            if use_temp_color and v.shape[0] > 0:
                temp_path = temps_dir / f"frame_{frame_idx:04d}.npy"
                if not temp_path.exists():
                    temp_path = temps_dir / "frame_0000.npy"
                if temp_path.exists():
                    temps_np = np.load(temp_path).astype(np.float32)
                    if temps_np.shape[0] == pts.shape[0]:
                        cols_np = cmap_fn(temps_np, t_min_cfg, t_max_cfg)
            if cols_np is None and has_color_path and v.shape[0] > 0:
                col_path = colors_dir / f"frame_{frame_idx:04d}.npy"
                if not col_path.exists():
                    col_path = colors_dir / "frame_0000.npy"
                if col_path.exists():
                    loaded = np.load(col_path).astype(np.float32)
                    if loaded.shape[0] == pts.shape[0]:
                        cols_np = loaded
            if cols_np is not None and v.shape[0] > 0:
                cols_wp = wp.array(
                    cols_np, dtype=wp.vec3, device=mesher.device,
                )
                vc_u8 = mesher.compute_vertex_colors(
                    v, pos_wp, cols_wp, search_r_world,
                )
                # B18.5 — pigment-space remix (Mixbox). The GPU pass already
                # produced a linear-RGB average; when mix_mode="mixbox" is
                # configured at the CLI level (forwarded via
                # cfg.color_mix_mode), re-do the blend on the CPU using the
                # pairwise Mixbox LUT. No-op when the mode is "linear"
                # (default). Reused unchanged for both colour & temperature
                # paths since both produce identically-shaped (P,3) RGB.
                if cfg.color_mix_mode == "mixbox":
                    from ..meshing.mixbox import remix_vertices_mixbox
                    vc_u8 = remix_vertices_mixbox(
                        v, pts, cols_np, search_r_world,
                    )
            write_ply(mesh_dir / f"frame_{frame_idx:04d}.ply", v, fc,
                      vertex_colors=vc_u8)
            frames_written += 1
        frame_count = frames_written
        if use_temp_color:
            cmode = f" (vertex colour from temperature: {cmap_name})"
        elif has_color_path:
            cmode = " (with vertex colour)"
        else:
            cmode = ""
        print(f"[mpm] meshed {frame_count} frames -> {mesh_dir}{cmode}")

        # I6.2.ABC — PLY → Alembic conversion is intentionally disabled.
        # Blender 5.1's wm.alembic_export writes 240-sub-object .abc files
        # via the addon's mesh-preload trick, and MeshSequenceCache modifier
        # then plays them back with a phantom domain-origin offset on the
        # empty `object_path`. The addon's PLY-preload path renders the same
        # cache correctly (~150 MB peak RAM, 14 s one-shot attach), so we
        # default to it. Re-enable once we have a single-object animated-
        # vertex Alembic writer (likely via pyalembic, not wm.alembic_export).
    else:
        # Round-20: count actual particles_raw files on disk, not the
        # requested target. Matters when MPM diverged — pre-round-20
        # this reported the FULL n_frames count even on truncated bakes.
        frame_count = len(list(particles_dir.glob("sim_*.ply")))

    # Cache manifest so A8.6 / A8.12 can locate the bake. If MPM
    # diverged, frame_count reports the ACTUAL written count (so the
    # addon's preload reads exactly what's on disk) and
    # truncated_at_frame is the divergence point.
    manifest = {
        "schema_version": 1,
        "gpufluid_version": __version__,
        "solver": "mpm",
        "fps": sim.fps,
        "frame_count": frame_count,
        "mesh_pattern": "mesh/frame_{:04d}.ply" if out_cfg.mesh else None,
        "particles_pattern": "particles_raw/sim_{:010d}.ply",
        "domain_size": list(scene.domain_size),
        "resolution": list(scene.domain.resolution),
        "dx": scene.dx,
        "notes": f"sim={sim.frames * sim.dt:.2f}s",
    }
    if truncated_at is not None:
        # Round-20: surface the divergence point so the addon's
        # _runner.py can give a specific ERROR message instead of the
        # generic "subprocess failed rc=2".
        manifest["truncated_at_frame"] = int(truncated_at)
        manifest["truncation_reason"] = "mpm_divergence"
    (cache_dir / "cache.json").write_text(json.dumps(manifest, indent=2))
    return 0 if truncated_at is None else 2


def cmd_simulate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    scene = load_scene(config_path)
    nx, ny, nz = scene.domain.resolution
    dx = scene.dx

    print(f"[gpufluid] simulate: {config_path}")
    print(f"           resolution {nx}x{ny}x{nz}   dx={dx:.5f}   domain={scene.domain_size}")
    print(f"           frames={scene.simulation.frames}  fps={scene.simulation.fps}  dt={scene.simulation.dt}")

    if scene.simulation.solver == "mpm":
        print(f"           solver=MPM (F3.7 shell-out)")
        return _cmd_simulate_mpm(args, scene)

    solver = FlipSolver3D(nx=nx, ny=ny, nz=nz, dx=dx,
                          gravity=scene.simulation.gravity,
                          flip_blend=scene.simulation.flip_blend,
                          rho=scene.simulation.rho,
                          viscosity=scene.simulation.viscosity,
                          viscosity_iters=scene.simulation.viscosity_iters,
                          surface_tension=scene.simulation.surface_tension,
                          csf_smoothing_passes=scene.simulation.csf_smoothing_passes,
                          transfer_mode=scene.simulation.transfer_mode,
                          enable_timing=bool(getattr(args, "timings", False)),
                          enable_cuda_graphs=bool(getattr(args, "enable_cuda_graphs", False)),
                          enable_sub_dense=bool(getattr(args, "enable_sub_dense", False)),
                          sub_rebuild_every=int(getattr(args, "sub_rebuild_every", 8)),
                          sub_dilation=int(getattr(args, "sub_dilation", 4)))
    grid_xyz = solver.cell_centers_np()
    obstacle_sdf = _build_obstacle_sdf(scene, grid_xyz)
    if obstacle_sdf is not None:
        solver.add_solid_from_sdf(obstacle_sdf)
        n_solid_added = int((obstacle_sdf <= 0).sum())
        print(f"           obstacles: {len(scene.obstacle)} -> {n_solid_added} solid cells")

    # v0.5 — register animated obstacles
    if scene.obstacle_motion:
        for ob, mcfg in zip(scene.obstacle, scene.obstacle_motion):
            if mcfg is None:
                continue
            motion = _build_motion(mcfg, scene.simulation.fps)
            if isinstance(ob, ObstacleSphereCfg):
                solver.add_animated_obstacle("sphere", ob.center, motion, radius=ob.radius)
            elif isinstance(ob, ObstacleBoxCfg):
                solver.add_animated_obstacle("box", ob.center, motion, half_size=ob.half_size)
            elif isinstance(ob, ObstacleCylinderYCfg):
                solver.add_animated_obstacle("cylinder_y", ob.center, motion,
                                             radius=ob.radius, half_height=ob.half_height)
            elif isinstance(ob, ObstacleMeshCfg):
                mesh_path = ob.path
                if scene.config_dir is not None and not Path(mesh_path).is_absolute():
                    mesh_path = str(scene.config_dir / mesh_path)
                # treat the object's current world position as the "base centre";
                # motion will translate the SDF per frame.
                solver.add_animated_obstacle("mesh", base_center=(0.0, 0.0, 0.0),
                                             motion=motion,
                                             mesh_path=mesh_path,
                                             scale=ob.scale,
                                             translate=ob.translate,
                                             rotate_deg=ob.rotate_deg)
            else:
                raise BlockError("C7.2", f"motion not supported on obstacle: {type(ob).__name__}")
    # v0.5 — register inflows / outflows
    for inf in scene.inflow:
        # Round-46: forward per-inflow color + temperature. Pre-round-46
        # the construction dropped both even though `InflowCfg` exposed
        # them (round-45 reviewer caught: MPM path forwarded, FLIP
        # didn't → mirror drift §9.6, particles spawned default-black).
        solver.add_inflow(InflowBox(
            lo=inf.lo, hi=inf.hi, velocity=inf.velocity,
            rate_per_sec=inf.rate_per_sec,
            frame_start=inf.frame_start, frame_end=inf.frame_end,
            color=getattr(inf, "color", None),
            temperature=getattr(inf, "temperature", None),
        ))
    for out in scene.outflow:
        solver.add_outflow(OutflowBox(
            lo=out.lo, hi=out.hi,
            frame_start=out.frame_start, frame_end=out.frame_end,
        ))
    if scene.inflow or scene.outflow:
        print(f"           inflow/outflow: {len(scene.inflow)} in / {len(scene.outflow)} out")

    from .config import FluidBoxCfg, FluidMeshCfg
    def _seed_one(f):
        if isinstance(f, FluidBoxCfg):
            solver.seed_box(f.lo, f.hi, ppc=f.ppc, color=f.color,
                            temperature=f.temperature)
        elif isinstance(f, FluidMeshCfg):
            mp = f.path
            if scene.config_dir is not None and not Path(mp).is_absolute():
                mp = str(scene.config_dir / mp)
            solver.seed_mesh(mp, ppc=f.ppc, scale=f.scale,
                             translate=f.translate, rotate_deg=f.rotate_deg,
                             color=f.color, temperature=f.temperature)

    if getattr(args, "resume", None):
        solver.load_checkpoint(args.resume)
        print(f"           resumed from {args.resume}: {solver.n_particles} particles")
    elif scene.fluids:
        # S2.15: multi-source [[fluids]] (each may carry per-source `color`)
        for f in scene.fluids:
            _seed_one(f)
    else:
        _seed_one(scene.fluid)
    print(f"           particles: {solver.n_particles}")

    cache_dir = Path(scene.output.cache_dir)
    if not cache_dir.is_absolute() and scene.config_dir is not None:
        cache_dir = scene.config_dir / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = cache_dir / "mesh"; mesh_dir.mkdir(exist_ok=True)
    parts_dir = cache_dir / "particles"
    if scene.output.particles:
        parts_dir.mkdir(exist_ok=True)

    extractor = MeshExtractor(nx, ny, nz, dx) if scene.output.mesh else None
    usd_frames: list = []  # collected only if scene.output.usd

    # W7.x — whitewater
    ww_sys = None
    ww_dir = None
    if scene.output.whitewater:
        from ..sim.whitewater import WhitewaterSystem, WhitewaterConfig
        # Round-23: pre-round-23 we set the legacy `gravity` +
        # `lifetime_sec` fields and they were silently dropped — the
        # actual step() reads only the per-class `gravity_foam/spray
        # /bubble` and `lifetime_foam/spray/bubble` fields. User-set
        # scene.simulation.gravity (low-G scene) or
        # whitewater_lifetime_sec had ZERO effect. Round-23 fans them
        # out: `gravity_spray` = scene gravity (full G), bubble/foam
        # scaled relative to their defaults (-9.81); lifetime_sec
        # applied uniformly to all three classes.
        g_scene = float(scene.simulation.gravity)
        # Round-26 self-fix: pre-round-26 the guard was `if g_scene !=
        # 0.0 else 1.0` — meant to dodge a non-existent div-by-zero,
        # but it inverted user intent: setting gravity=0 (zero-G scene)
        # snapped ratio back to 1.0 and per-class gravities to their
        # full defaults. Now ratio is just `g_scene / -9.81`; zero
        # gravity correctly zeros all three per-class gravities.
        ratio = g_scene / -9.81
        life = float(scene.output.whitewater_lifetime_sec)
        ww_cfg = WhitewaterConfig(
            speed_threshold=scene.output.whitewater_speed_threshold,
            emit_per_frame_max=scene.output.whitewater_emit_per_frame_max,
            total_cap=scene.output.whitewater_total_cap,
            lifetime_sec=life,
            lifetime_foam=life,
            lifetime_spray=life,
            lifetime_bubble=life,
            gravity=g_scene,
            gravity_spray=-9.81 * ratio,   # full gravity (down)
            gravity_foam=-0.5 * ratio,     # near-neutral
            gravity_bubble=+3.0 * ratio,   # buoyant (sign preserved)
        )
        ww_sys = WhitewaterSystem(ww_cfg)
        ww_dir = cache_dir / "whitewater"; ww_dir.mkdir(exist_ok=True)

    steps_per_frame = max(1, int(round(1.0 / (scene.simulation.fps * scene.simulation.dt))))
    print(f"           {steps_per_frame} solver steps per frame")

    t_sim_total = 0.0; t_mesh_total = 0.0
    rng_reseed = np.random.default_rng(123)
    reseed_cfg = None
    if scene.simulation.reseed:
        from ..sim.reseed import reseed_particles, ReseedConfig
        reseed_cfg = ReseedConfig(
            min_per_cell=scene.simulation.reseed_min_per_cell,
            max_per_cell=scene.simulation.reseed_max_per_cell,
            every_n_frames=scene.simulation.reseed_every_n_frames,
        )

    start_frame = int(getattr(args, "start_frame", 0))
    checkpoint_every = int(getattr(args, "checkpoint_every", 0))
    # Round-32: FLIP divergence trap (mirror of MPM round-20).
    # FlipSolver3D had NO NaN guard pre-round-32 — divergent pressure
    # oscillation produced NaN particle velocities, propagated, and
    # cache.json reported full frame_count with garbage PLYs.
    from ..solvers.solver3d import FlipDivergenceError
    flip_truncated_at: int | None = None
    for frame in range(start_frame, scene.simulation.frames):
        # v0.5 per-frame hook: anim obstacles + inflow/outflow
        solver.prepare_frame(frame, 1.0 / scene.simulation.fps)
        # S2.11/S2.11.GPU reseeding — pick path by particle count
        if reseed_cfg is not None and (frame % reseed_cfg.every_n_frames == 0) and solver.n_particles > 0:
            from ..sim.reseed import (
                reseed_particles, reseed_particles_gpu, RESEED_GPU_THRESHOLD)
            import warp as wp
            if solver.n_particles >= RESEED_GPU_THRESHOLD:
                (new_pos_wp, new_vel_wp, new_col_wp, new_temp_wp,
                 n_emit, n_cull) = reseed_particles_gpu(
                    solver.pos, solver.vel, solver.marker, solver.dx, reseed_cfg,
                    rng_reseed,
                    attr_color_wp=solver.attr_color,
                    attr_temperature_wp=solver.attr_temperature,
                    device=solver.device)
                if n_emit > 0 or n_cull > 0:
                    solver.pos = new_pos_wp
                    solver.vel = new_vel_wp
                    if new_col_wp is not None:
                        solver.attr_color = new_col_wp
                    if new_temp_wp is not None:
                        solver.attr_temperature = new_temp_wp
                    solver.affine_C = None
                    solver.n_particles = int(new_pos_wp.shape[0])
            else:
                cur_pos = solver.pos.numpy()
                cur_vel = solver.vel.numpy()
                # Refresh marker host snapshot — the per-step P2G marker is only
                # live on GPU, so without this the reseed sees only walls/obstacles.
                current_marker = solver.marker.numpy()
                # Round-36: pass attrs through CPU reseed so they stay
                # row-aligned with pos/vel (round-35 reviewer caught
                # exact §9.6 mirror-drift — GPU path got attr handling
                # in round-34, CPU sibling forgotten). Variant return:
                # 6-tuple when any attr is non-None, else 4-tuple.
                cur_col = (solver.attr_color.numpy()
                           if solver.attr_color is not None else None)
                cur_temp = (solver.attr_temperature.numpy()
                            if solver.attr_temperature is not None else None)
                has_attrs = cur_col is not None or cur_temp is not None
                out = reseed_particles(
                    cur_pos, cur_vel, current_marker, solver.dx, reseed_cfg,
                    rng_reseed,
                    attr_color=cur_col, attr_temperature=cur_temp)
                if has_attrs:
                    new_pos, new_vel, n_emit, n_cull, new_col, new_temp = out
                else:
                    new_pos, new_vel, n_emit, n_cull = out
                    new_col, new_temp = None, None
                if n_emit > 0 or n_cull > 0:
                    solver.pos = wp.array(new_pos, dtype=wp.vec3, device=solver.device)
                    solver.vel = wp.array(new_vel, dtype=wp.vec3, device=solver.device)
                    if new_col is not None:
                        solver.attr_color = wp.array(
                            new_col, dtype=wp.vec3, device=solver.device)
                    if new_temp is not None:
                        solver.attr_temperature = wp.array(
                            new_temp, dtype=float, device=solver.device)
                    solver.affine_C = None
                    solver.n_particles = len(new_pos)
        ts = time.time()
        # σ>0 forces step_cfl path: explicit CSF needs capillary-wave clamping
        # (S2.14.5). Otherwise the user's `cfl` flag decides.
        use_cfl = scene.simulation.cfl or (scene.simulation.surface_tension > 0.0)
        if use_cfl:
            frame_dt = scene.simulation.dt * steps_per_frame
            solver.step_cfl(frame_dt,
                            pressure_iters=scene.simulation.pressure_iters,
                            cfl=scene.simulation.cfl_factor,
                            max_substeps=scene.simulation.cfl_max_substeps,
                            pressure_solver=scene.simulation.pressure_solver)
        else:
            for _ in range(steps_per_frame):
                solver.step(scene.simulation.dt,
                            pressure_iters=scene.simulation.pressure_iters,
                            pressure_solver=scene.simulation.pressure_solver)
        t_sim_total += time.time() - ts
        # Round-32: NaN check once per frame. step+step_cfl don't raise
        # on divergence — we sample positions and break out so the
        # cache.json can be written with truncated_at_frame instead of
        # the misleading full frame_count.
        if solver.has_diverged():
            flip_truncated_at = frame
            print(
                f"[flip] solver diverged at frame {frame} "
                f"(last dumped: {frame}/{scene.simulation.frames}). "
                f"Recovery: lower dt, raise pressure_iters, or shrink "
                f"resolution.", file=sys.stderr)
            break

        if extractor is not None:
            tm = time.time()
            verts, faces = extractor.extract(
                solver.pos,
                iso_level=scene.output.iso_level,
                smooth_passes=scene.output.smooth_passes,
                mesh_smooth_passes=scene.output.mesh_smooth_passes,
                mesh_smooth_method=scene.output.mesh_smooth_method,
                wall_margin_cells=scene.output.wall_margin_cells,
                decimate_ratio=scene.output.decimate_ratio,
                mesh_method=scene.output.mesh_method,
                cubic_radius_cells=scene.output.cubic_radius_cells,
                sdf_particle_radius_cells=scene.output.sdf_particle_radius_cells,
                sdf_search_radius_cells=scene.output.sdf_search_radius_cells,
            )
            t_mesh_total += time.time() - tm
            # B2 — per-vertex colour for FLIP (mirrors B18.4/B18.5 on MPM).
            # When the solver carries per-particle colour, build vertex colours
            # via KNN blend against the current particle cloud; optionally remix
            # in pigment space (Mixbox) for green-not-grey contact zones.
            vc_u8 = None
            if (verts is not None and verts.shape[0] > 0
                    and solver.attr_color is not None
                    and solver.n_particles > 0):
                search_r_world = float(scene.output.sdf_search_radius_cells) * float(solver.dx)
                cols_np = solver.attr_color.numpy().astype(np.float32)
                if cols_np.shape[0] == int(solver.pos.shape[0]):
                    vc_u8 = extractor.compute_vertex_colors(
                        verts, solver.pos, solver.attr_color, search_r_world,
                    )
                    if str(scene.output.color_mix_mode) == "mixbox":
                        from ..meshing.mixbox import remix_vertices_mixbox
                        pos_np = solver.pos.numpy().astype(np.float32)
                        vc_u8 = remix_vertices_mixbox(
                            verts, pos_np, cols_np, search_r_world,
                        )
            if verts is not None:
                write_ply(mesh_dir / f"frame_{frame:04d}.ply", verts, faces,
                          vertex_colors=vc_u8)
                if scene.output.usd:
                    usd_frames.append((frame, verts.copy(), faces.copy()))
            else:
                # write an empty placeholder so frame indices line up
                write_ply(mesh_dir / f"frame_{frame:04d}.ply",
                          np.zeros((0, 3), dtype=np.float32),
                          np.zeros((0, 3), dtype=np.int32))
                if scene.output.usd:
                    usd_frames.append((frame,
                                       np.zeros((0, 3), dtype=np.float32),
                                       np.zeros((0, 3), dtype=np.int32)))

        if scene.output.particles:
            pos, _ = solver.get_particles()
            write_particles_npy(parts_dir / f"frame_{frame:04d}.npy", pos)
            # S2.15: dump per-particle color sidecar when colored sources are in play
            if solver.attr_color is not None:
                colors_dir = parts_dir.parent / "colors"
                colors_dir.mkdir(parents=True, exist_ok=True)
                np.save(colors_dir / f"frame_{frame:04d}.npy",
                        solver.attr_color.numpy().astype(np.float32))
            # B11.3 / S2.18: per-particle temperature sidecar (lava demo etc.)
            if solver.attr_temperature is not None:
                temps_dir = parts_dir.parent / "temperatures"
                temps_dir.mkdir(parents=True, exist_ok=True)
                np.save(temps_dir / f"frame_{frame:04d}.npy",
                        solver.attr_temperature.numpy().astype(np.float32))

        if ww_sys is not None:
            fpos, fvel = solver.get_particles()
            # W7.4 — share the M5 density grid so whitewater can classify
            # spray/foam/bubble at emit time and pop bubbles at the surface.
            ww_density = extractor.dens.numpy() if extractor is not None else None
            # B3.3 — optionally compute the W7.7 trapped-air potential and
            # pass it as the emit weight, so genuine turbulence pockets emit
            # whitewater rather than every fast-moving bulk particle.
            ww_potential = None
            if scene.output.whitewater_use_potential and fpos.shape[0] > 0:
                from ..sim.whitewater_potentials import trapped_air_potential
                r = scene.output.whitewater_potential_radius
                if r <= 0.0:
                    r = 2.5 * float(solver.dx)
                alpha = float(scene.output.whitewater_trapped_air_weight)
                beta = float(scene.output.whitewater_wave_crest_weight)
                ww_potential = alpha * trapped_air_potential(
                    fpos, fvel, radius=r,
                    v_max=scene.output.whitewater_potential_v_max,
                )
                # B3.2 — wave-crest blend. Reuses solver grid resolution so
                # the indicator scatter matches the simulation's spatial
                # scale; gives surface curvature on top of velocity-divergence.
                if beta > 0.0:
                    from ..sim.whitewater_potentials import wave_crest_potential
                    grid_res = max(int(max(solver.nx, solver.ny, solver.nz)), 16)
                    iwc = wave_crest_potential(
                        fpos, grid_res=grid_res, dx=float(solver.dx),
                        n_blur=int(scene.simulation.csf_smoothing_passes or 2),
                    )
                    ww_potential = ww_potential + beta * iwc
            ww_sys.emit_from_fluid(fpos, fvel, density=ww_density, dx=solver.dx,
                                   potential=ww_potential)
            ww_sys.step(1.0 / scene.simulation.fps,
                        dom=tuple(scene.domain_size),
                        density=ww_density, dx=solver.dx)
            write_particles_npy(ww_dir / f"frame_{frame:04d}.npy", ww_sys.pos)
            # W7.6 — dump kind sidecar so the renderer can colour-code classes.
            if hasattr(ww_sys, "kind") and ww_sys.kind.size > 0:
                kinds_dir = ww_dir.parent / "whitewater_kinds"
                kinds_dir.mkdir(parents=True, exist_ok=True)
                np.save(kinds_dir / f"frame_{frame:04d}.npy",
                        ww_sys.kind.astype(np.int32))

        if frame % 10 == 0 or frame == scene.simulation.frames - 1:
            mesh_info = f" V={len(verts) if extractor and verts is not None else 0}"
            print(f"  frame {frame:04d}/{scene.simulation.frames}{mesh_info}")

        if checkpoint_every > 0 and (frame + 1) % checkpoint_every == 0:
            ckpt = cache_dir / f"checkpoint_frame_{frame:04d}.npz"
            solver.save_checkpoint(ckpt)
            print(f"           [ckpt] wrote {ckpt.name}")

    if scene.output.usd and usd_frames:
        usd_path = cache_dir / "cache.usdc"
        write_usd_mesh_sequence(usd_path, usd_frames, fps=scene.simulation.fps)
        print(f"           wrote USD: {usd_path}")

    # Round-32: if FLIP diverged, frame_count reports ACTUAL written
    # frames (not the cfg target), and we inject truncation_reason
    # via the manifest. The dataclass-typed manifest filters unknown
    # keys (round-31 reviewer #3 schema-drift finding), so we have to
    # write through raw JSON for the truncation marker. Done below.
    flip_frame_count = (
        flip_truncated_at + 1 if flip_truncated_at is not None
        else scene.simulation.frames)
    manifest = CacheManifest(
        fps=scene.simulation.fps,
        frame_count=flip_frame_count,
        mesh_pattern="mesh/frame_{:04d}.ply",
        particles_pattern="particles/frame_{:04d}.npy" if scene.output.particles else None,
        domain_size=list(scene.domain_size),
        resolution=list(scene.domain.resolution),
        dx=scene.dx,
        notes=f"sim={t_sim_total:.1f}s mesh={t_mesh_total:.1f}s",
        solver="flip",
        truncated_at_frame=(int(flip_truncated_at)
                            if flip_truncated_at is not None else None),
        truncation_reason=("flip_divergence"
                           if flip_truncated_at is not None else None),
    )
    manifest_path = write_cache_manifest(cache_dir, manifest)
    print(f"[gpufluid] done. cache: {cache_dir}  manifest: {manifest_path.name}")
    print(f"           sim total: {t_sim_total:.2f}s  mesh total: {t_mesh_total:.2f}s")
    # B12 — per-section profiler report (only when --timings was passed)
    timings = solver._prof.summarize()
    if timings:
        ms_total = sum(timings.values())
        print("[gpufluid] per-section timings (--timings):")
        for name, ms in sorted(timings.items(), key=lambda kv: -kv[1]):
            pct = (100.0 * ms / ms_total) if ms_total > 0 else 0.0
            calls = solver._prof.call_counts.get(name, 0)
            print(f"             {name:<18s} {ms:>9.1f} ms  ({pct:5.1f}%)  x{calls}")
        # Persist alongside the manifest so downstream tooling can ingest.
        import json
        timings_path = cache_dir / "timings.json"
        timings_path.write_text(json.dumps(
            {"sections_ms": timings,
             "call_counts": solver._prof.call_counts,
             "sim_total_s": t_sim_total,
             "mesh_total_s": t_mesh_total},
            indent=2, sort_keys=True))
        print(f"             dumped to {timings_path}")
    if solver._enable_cuda_graphs:
        hits = solver._cuda_graph_hits; misses = solver._cuda_graph_misses
        total = hits + misses
        if total > 0:
            pct = 100.0 * hits / total
            print(f"[gpufluid] cuda-graph: {hits}/{total} steps replayed "
                  f"({pct:.0f}% hit rate, {misses} captures)")
        else:
            print("[gpufluid] cuda-graph: not eligible for any step "
                  "(PCG and pressure_block_sparse=true still fall through)")
    return 0 if flip_truncated_at is None else 2


# ---------------------------------------------------------------------------
# A8.12 — render
# ---------------------------------------------------------------------------

# [BLK A8.12]
@block("A8.12",
       "Headless render CLI command — spawns Blender with the addon render bridge")
def cmd_render(args: argparse.Namespace) -> int:
    """Spawn a headless Blender process to render a baked cache.

    The actual Eevee work happens inside Blender's Python via the
    addon's ``render_bridge`` module (A8.9-A8.11). This command just
    constructs the subprocess command line.
    """
    import subprocess, json as _json
    cache_dir = Path(args.cache).resolve()
    scene_toml = Path(args.scene).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = cache_dir / "cache.json"
    if not cache_json.exists():
        raise BlockError("A8.12", f"cache.json not found in {cache_dir}")
    blender = args.blender or "blender"
    # Locate the bundled render driver script
    driver = Path(__file__).resolve().parents[3] / "addon" / "gpufluid_blender" / "_headless_render.py"
    if not driver.exists():
        raise BlockError("A8.12", f"render driver missing: {driver}")
    payload = _json.dumps({
        "cache":  str(cache_dir),
        "scene":  str(scene_toml),
        "out":    str(out_dir),
        "label":  args.label,
        "color":  args.color,
        "samples": args.samples,
        "fps":    args.fps,
        "frames": args.frames,
    })
    cmd = [blender, "--background", "--python", str(driver), "--", payload]
    print(f"[render] launching: {' '.join(cmd[:3])} ... ({len(payload)} bytes config)")
    r = subprocess.run(cmd, check=False)
    return r.returncode


# ---------------------------------------------------------------------------
# C7.3 — bench
# ---------------------------------------------------------------------------

# [BLK C7.3]
@block("C7.3", "bench command: solver throughput")
def cmd_bench(args: argparse.Namespace) -> int:
    cfgs = [(48, 8, 50, 100), (64, 8, 50, 100), (96, 4, 40, 50)]
    print("[gpufluid] solver throughput (3D, pure GPU)")
    print(f"  {'resolution':>10} {'particles':>10} {'iters':>5} {'steps/s':>10} {'time/step':>10}")
    for n, ppc, iters, n_steps in cfgs:
        s = FlipSolver3D(nx=n, ny=n, nz=n, dx=1.0 / n)
        s.seed_box(lo=(0.05, 0.05, 0.05), hi=(0.40, 0.70, 0.40), ppc=ppc)
        # Round-50: pre-round-50 the warmup loop AND the timed loop ran
        # without `wp.synchronize()` — `FlipSolver3D.step()` enqueues
        # Warp kernels asynchronously and time.time() bracketed only
        # the Python-side dispatch latency. Reported steps/s was
        # inflated 5-50× (often by orders of magnitude); anyone
        # comparing perf branches with `gpufluid bench` steered on
        # noise. Fix: sync before start_time AND after the loop so
        # the bracket covers actual GPU work.
        for _ in range(3):
            s.step(0.005, pressure_iters=iters)
        wp.synchronize()       # warmup queue drained
        t = time.time()
        for _ in range(n_steps):
            s.step(0.005, pressure_iters=iters)
        wp.synchronize()       # timed queue drained before clock-stop
        dt = time.time() - t
        print(f"  {n:>4}^3      {s.n_particles:>10} {iters:>5} {n_steps/dt:>10.1f} {1000*dt/n_steps:>9.2f}ms")
    return 0


# ---------------------------------------------------------------------------
# C7.4 — info
# ---------------------------------------------------------------------------

# [BLK C7.4]
@block("C7.4", "info command: version, devices, registry counts")
def cmd_info(args: argparse.Namespace) -> int:
    print(f"gpufluid version: {__version__}")
    wp.init()
    print("Warp devices:")
    for d in wp.get_devices():
        print(f"  - {d}")
    # Force-load modules that register blocks but aren't imported by top-level
    # gpufluid (W7 whitewater, S2.11 reseed, D4.3.GPU.BVH mesh_sdf_gpu)
    import gpufluid.sim.whitewater   # noqa
    import gpufluid.sim.whitewater_potentials  # noqa — registers W7.7
    import gpufluid.sim.reseed       # noqa
    import gpufluid.domain.mesh_sdf_gpu  # noqa
    reg = get_registry()
    total = sum(len(v) for v in reg.values())
    print(f"Registered blocks: {len(reg)} unique IDs, {total} callables")
    for layer in ["G1", "S2", "F3", "D4", "M5", "I6", "C7", "A8", "W7"]:
        items = by_layer(layer)
        if items:
            print(f"  {layer}: {len(items):>2}  " + ", ".join(b.block_id for b in items))
    return 0


def cmd_blocks(args: argparse.Namespace) -> int:
    """(DESIGN.md §3.2) Registry contract enforcement entry point."""
    from ..blocks.check import cli_main
    sub_argv: list[str] = []
    if args.check:
        sub_argv.append("--check")
    elif args.regen_index:
        sub_argv.append("--regen-index")
    elif args.list:
        sub_argv.append("--list")
        if args.layer:
            sub_argv += ["--layer", args.layer]
    return cli_main(sub_argv)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser("gpufluid", description="GPU FLIP fluid simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("simulate", help="run a TOML scene config")
    p_sim.add_argument("config", help="path to scene.toml")
    p_sim.add_argument("--resume", default=None, help="checkpoint .npz to resume from")
    p_sim.add_argument("--start-frame", type=int, default=0, help="frame to start bake from (used with --resume)")
    p_sim.add_argument("--checkpoint-every", type=int, default=0, help="write checkpoint every N frames (0 = off)")
    p_sim.add_argument("--timings", action="store_true",
                       help="(B12) wrap each major solver phase in wp.ScopedTimer, "
                            "print per-section totals at the end + write timings.json next to cache.json")
    p_sim.add_argument("--enable-cuda-graphs", action="store_true",
                       help="(B5) capture step() into a CUDA graph and replay it "
                            "across substeps with the same topology. ~2x speedup "
                            "at 64^3 for jacobi/gsrb + no CSF + no block-sparse. "
                            "Other configs fall through to direct execution.")
    p_sim.add_argument("--enable-sub-dense", action="store_true",
                       help="(B7-alt) deferred-dense field storage: shrink "
                            "u/v/w/p/p_tmp/div + scratch to the active 8^3-tile "
                            "bbox + sub_dilation. 256^3/10%%-fill dam-break sees "
                            "~6x memory drop. Rebuild every --sub-rebuild-every "
                            "frames or when fluid encroaches on the edge.")
    p_sim.add_argument("--sub-rebuild-every", type=int, default=8,
                       help="(B7-alt) frames between forced sub-dense rebuilds")
    p_sim.add_argument("--sub-dilation", type=int, default=4,
                       help="(B7-alt) cells of safety margin around the active "
                            "fluid bbox; raise if particles drift past edge "
                            "between rebuilds (B7-alt.4 prints a warning)")
    p_sim.set_defaults(func=cmd_simulate)

    p_render = sub.add_parser("render",
        help="Headless Eevee render of a baked cache (A8.12)")
    p_render.add_argument("cache", help="path to cache_dir (must contain cache.json + mesh/)")
    p_render.add_argument("scene", help="path to scene.toml (read for cube obstacle geometry)")
    p_render.add_argument("--out", required=True, help="output directory for PNG frames")
    p_render.add_argument("--label", default="gpufluid",
        help="overlay label drawn into each frame")
    p_render.add_argument("--color", nargs=3, type=float, default=[0.20, 0.50, 0.70],
        metavar=("R", "G", "B"), help="fluid surface RGB in [0,1]")
    p_render.add_argument("--samples", type=int, default=16,
        help="Eevee TAA samples per frame (A8.9 preset)")
    p_render.add_argument("--fps", type=int, default=60)
    p_render.add_argument("--frames", type=int, default=0,
        help="number of frames to render; 0 = read from cache.json")
    p_render.add_argument("--blender", default=None,
        help="path to Blender executable; falls back to 'blender' on $PATH")
    p_render.set_defaults(func=cmd_render)

    p_bench = sub.add_parser("bench", help="solver throughput benchmark")
    p_bench.set_defaults(func=cmd_bench)

    p_info = sub.add_parser("info", help="device + registry info")
    p_info.set_defaults(func=cmd_info)

    p_blocks = sub.add_parser(
        "blocks",
        help="(DESIGN.md §3.2) verify or regenerate the block registry index")
    p_blocks_g = p_blocks.add_mutually_exclusive_group(required=True)
    p_blocks_g.add_argument("--check", action="store_true",
                            help="run all 6 registry contract checks; exit 1 on any error")
    p_blocks_g.add_argument("--regen-index", action="store_true",
                            help="rewrite docs/BLOCKS.md from the live registry")
    p_blocks_g.add_argument("--list", action="store_true",
                            help="pretty-print the registry")
    p_blocks.add_argument("--layer", default=None,
                          help="filter --list by layer prefix (e.g. S2)")
    p_blocks.set_defaults(func=cmd_blocks)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except BlockError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
