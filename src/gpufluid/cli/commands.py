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
from ..domain.sdf import (
    sdf_sphere, sdf_box, sdf_cylinder_y, sdf_plane, sdf_union,
)
from ..domain.regions import InflowBox, OutflowBox
from ..domain.animation import LinearMotion, KeyframeMotion
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
def cmd_simulate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    scene = load_scene(config_path)
    nx, ny, nz = scene.domain.resolution
    dx = scene.dx

    print(f"[gpufluid] simulate: {config_path}")
    print(f"           resolution {nx}x{ny}x{nz}   dx={dx:.5f}   domain={scene.domain_size}")
    print(f"           frames={scene.simulation.frames}  fps={scene.simulation.fps}  dt={scene.simulation.dt}")

    solver = FlipSolver3D(nx=nx, ny=ny, nz=nz, dx=dx,
                          gravity=scene.simulation.gravity,
                          flip_blend=scene.simulation.flip_blend,
                          rho=scene.simulation.rho,
                          viscosity=scene.simulation.viscosity,
                          viscosity_iters=scene.simulation.viscosity_iters,
                          surface_tension=scene.simulation.surface_tension,
                          csf_smoothing_passes=scene.simulation.csf_smoothing_passes,
                          transfer_mode=scene.simulation.transfer_mode)
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
        solver.add_inflow(InflowBox(
            lo=inf.lo, hi=inf.hi, velocity=inf.velocity,
            rate_per_sec=inf.rate_per_sec,
            frame_start=inf.frame_start, frame_end=inf.frame_end,
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
            solver.seed_box(f.lo, f.hi, ppc=f.ppc, color=f.color)
        elif isinstance(f, FluidMeshCfg):
            mp = f.path
            if scene.config_dir is not None and not Path(mp).is_absolute():
                mp = str(scene.config_dir / mp)
            solver.seed_mesh(mp, ppc=f.ppc, scale=f.scale,
                             translate=f.translate, rotate_deg=f.rotate_deg,
                             color=f.color)

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
        ww_cfg = WhitewaterConfig(
            speed_threshold=scene.output.whitewater_speed_threshold,
            emit_per_frame_max=scene.output.whitewater_emit_per_frame_max,
            total_cap=scene.output.whitewater_total_cap,
            lifetime_sec=scene.output.whitewater_lifetime_sec,
            gravity=scene.simulation.gravity,
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
    for frame in range(start_frame, scene.simulation.frames):
        # v0.5 per-frame hook: anim obstacles + inflow/outflow
        solver.prepare_frame(frame, 1.0 / scene.simulation.fps)
        # S2.11/S2.11.GPU reseeding — pick path by particle count
        if reseed_cfg is not None and (frame % reseed_cfg.every_n_frames == 0) and solver.n_particles > 0:
            from ..sim.reseed import (
                reseed_particles, reseed_particles_gpu, RESEED_GPU_THRESHOLD)
            import warp as wp
            if solver.n_particles >= RESEED_GPU_THRESHOLD:
                new_pos_wp, new_vel_wp, new_col_wp, n_emit, n_cull = reseed_particles_gpu(
                    solver.pos, solver.vel, solver.marker, solver.dx, reseed_cfg,
                    rng_reseed,
                    attr_color_wp=solver.attr_color,
                    device=solver.device)
                if n_emit > 0 or n_cull > 0:
                    solver.pos = new_pos_wp
                    solver.vel = new_vel_wp
                    if new_col_wp is not None:
                        solver.attr_color = new_col_wp
                    solver.affine_C = None
                    solver.n_particles = int(new_pos_wp.shape[0])
            else:
                cur_pos = solver.pos.numpy()
                cur_vel = solver.vel.numpy()
                # Refresh marker host snapshot — the per-step P2G marker is only
                # live on GPU, so without this the reseed sees only walls/obstacles.
                current_marker = solver.marker.numpy()
                new_pos, new_vel, n_emit, n_cull = reseed_particles(
                    cur_pos, cur_vel, current_marker, solver.dx, reseed_cfg, rng_reseed)
                if n_emit > 0 or n_cull > 0:
                    solver.pos = wp.array(new_pos, dtype=wp.vec3, device=solver.device)
                    solver.vel = wp.array(new_vel, dtype=wp.vec3, device=solver.device)
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
            )
            t_mesh_total += time.time() - tm
            if verts is not None:
                write_ply(mesh_dir / f"frame_{frame:04d}.ply", verts, faces)
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

        if ww_sys is not None:
            fpos, fvel = solver.get_particles()
            # W7.4 — share the M5 density grid so whitewater can classify
            # spray/foam/bubble at emit time and pop bubbles at the surface.
            ww_density = extractor.dens.numpy() if extractor is not None else None
            ww_sys.emit_from_fluid(fpos, fvel, density=ww_density, dx=solver.dx)
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

    manifest = CacheManifest(
        fps=scene.simulation.fps,
        frame_count=scene.simulation.frames,
        mesh_pattern="mesh/frame_{:04d}.ply",
        particles_pattern="particles/frame_{:04d}.npy" if scene.output.particles else None,
        domain_size=list(scene.domain_size),
        resolution=list(scene.domain.resolution),
        dx=scene.dx,
        notes=f"sim={t_sim_total:.1f}s mesh={t_mesh_total:.1f}s",
    )
    manifest_path = write_cache_manifest(cache_dir, manifest)
    print(f"[gpufluid] done. cache: {cache_dir}  manifest: {manifest_path.name}")
    print(f"           sim total: {t_sim_total:.2f}s  mesh total: {t_mesh_total:.2f}s")
    return 0


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
        for _ in range(3):
            s.step(0.005, pressure_iters=iters)
        t = time.time()
        for _ in range(n_steps):
            s.step(0.005, pressure_iters=iters)
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
    p_sim.set_defaults(func=cmd_simulate)

    p_bench = sub.add_parser("bench", help="solver throughput benchmark")
    p_bench.set_defaults(func=cmd_bench)

    p_info = sub.add_parser("info", help="device + registry info")
    p_info.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except BlockError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
