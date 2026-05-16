"""[BLK A8.5] Bake operator — modal subprocess invocation of `gpufluid simulate`.

Pipeline:
    1. Scan the scene for the Domain object, all Fluid and Obstacle objects.
    2. Compute world-space domain box anchored at the Domain object's bounding box.
       All coordinates are translated so the simulator works in [0, dom_size].
    3. Build a scene dict (config_builder.build_toml) and write `scene.toml`
       into the cache directory.
    4. Spawn the user-configured Python interpreter with `-m gpufluid.cli simulate`.
    5. Modal timer reads stdout, parses "frame N/M" lines, updates progress.
    6. On exit success, auto-attach the cache to the target object (A8.6).
"""
from __future__ import annotations
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Optional

import bpy
import mathutils

from ..config_builder import build_toml
from ..preferences import get_prefs


_FRAME_RE = re.compile(r"frame\s+(\d+)/(\d+)")


def _bbox_world(obj) -> tuple:
    """Return (lo, hi) world-space AABB of an object.

    For mesh-like objects uses Blender's ``bound_box``. For an Empty the
    `bound_box` is degenerate (all zeros), so we derive the AABB from
    ``empty_display_size`` (a half-extent) and the object's transform.
    """
    if obj.type == "EMPTY":
        # Empty visualises as a unit shape of half-extent empty_display_size.
        d = obj.empty_display_size
        local = [mathutils.Vector(c) for c in (
            (-d, -d, -d), ( d, -d, -d), ( d,  d, -d), (-d,  d, -d),
            (-d, -d,  d), ( d, -d,  d), ( d,  d,  d), (-d,  d,  d),
        )]
        pts = [obj.matrix_world @ v for v in local]
    else:
        pts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def collect_scene(context, domain_obj):
    """Build the bpy-free scene dict used by config_builder.

    Translates everything so the Domain bbox starts at origin (sim space).
    """
    scene = context.scene
    dom_lo_w, dom_hi_w = _bbox_world(domain_obj)
    origin = mathutils.Vector(dom_lo_w)
    dom_size = mathutils.Vector((dom_hi_w[0] - dom_lo_w[0],
                                  dom_hi_w[1] - dom_lo_w[1],
                                  dom_hi_w[2] - dom_lo_w[2]))
    # uniform dx from longest axis & resolution; non-cubic domains get equal dx in all axes
    dprops = domain_obj.gpufluid_domain
    nmax = max(dom_size.x, dom_size.y, dom_size.z)
    dx = float(nmax) / float(dprops.resolution)
    res = (max(8, int(round(dom_size.x / dx))),
           max(8, int(round(dom_size.y / dx))),
           max(8, int(round(dom_size.z / dx))))

    def to_sim(v):
        return (float(v[0] - origin.x), float(v[1] - origin.y), float(v[2] - origin.z))

    # fluid sources — emit one entry per source so each can carry its own
    # ppc + color. Pre-B1.3 these were unioned into one bbox.
    fluid_objs = [o for o in scene.objects if o.gpufluid_fluid.is_fluid and o is not domain_obj]
    if not fluid_objs:
        raise RuntimeError("no fluid source objects in scene (mark one with 'Mark Fluid')")
    fluid_sources = []
    eps = 1.5 * dx
    for fobj in fluid_objs:
        flow, fhiw = _bbox_world(fobj)
        flo_sim = list(to_sim(flow))
        fhi_sim = list(to_sim(fhiw))
        flo_sim = [max(eps, v) for v in flo_sim]
        fhi_sim = [min(dom_size[i] - eps, fhi_sim[i]) for i in range(3)]
        fprops = fobj.gpufluid_fluid
        entry = {"kind": "box", "lo": tuple(flo_sim), "hi": tuple(fhi_sim),
                 "ppc": int(fprops.ppc)}
        if getattr(fprops, "use_color", False):
            entry["color"] = tuple(float(c) for c in fprops.color)
        fluid_sources.append(entry)

    # obstacles
    obstacles = []
    for o in scene.objects:
        if not o.gpufluid_obstacle.is_obstacle or o is domain_obj:
            continue
        oprops = o.gpufluid_obstacle
        olow, ohiw = _bbox_world(o)
        centre_w = ((olow[0] + ohiw[0]) * 0.5,
                    (olow[1] + ohiw[1]) * 0.5,
                    (olow[2] + ohiw[2]) * 0.5)
        if oprops.obstacle_type == "BBOX":
            obstacles.append({
                "type": "box",
                "center": to_sim(centre_w),
                "half_size": ((ohiw[0] - olow[0]) * 0.5,
                              (ohiw[1] - olow[1]) * 0.5,
                              (ohiw[2] - olow[2]) * 0.5),
            })
        elif oprops.obstacle_type == "PLANE":
            # object's local Z axis (world-space) = plane normal
            mw = o.matrix_world.to_3x3()
            normal = mw @ mathutils.Vector((0, 0, 1))
            obstacles.append({
                "type": "plane",
                "point": to_sim(centre_w),
                "normal": (normal.x, normal.y, normal.z),
                "bbox_lo": to_sim(olow),
                "bbox_hi": to_sim(ohiw),
            })
        elif oprops.obstacle_type == "SPHERE":
            r = max((ohiw[i] - olow[i]) * 0.5 for i in range(3))
            obstacles.append({"type": "sphere", "center": to_sim(centre_w), "radius": r})
        elif oprops.obstacle_type == "CYLINDER_Y":
            r = max((ohiw[0] - olow[0]) * 0.5, (ohiw[2] - olow[2]) * 0.5)
            hh = (ohiw[1] - olow[1]) * 0.5
            obstacles.append({"type": "cylinder_y", "center": to_sim(centre_w),
                              "radius": r, "half_height": hh})
        elif oprops.obstacle_type == "MESH":
            # export object's mesh as OBJ next to scene.toml
            cache_dir = bpy.path.abspath(dprops.cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            mesh_path = os.path.join(cache_dir, f"obstacle_{o.name}.obj")
            _export_obj(o, mesh_path)
            obstacles.append({
                "type": "mesh", "path": mesh_path,
                "scale": 1.0, "translate": (-origin.x, -origin.y, -origin.z),
            })
        # v0.5 — animated obstacle motion
        if oprops.motion_type == "LINEAR":
            obstacles[-1]["motion"] = {
                "kind": "linear",
                "velocity": tuple(oprops.motion_velocity),
            }

    # v0.5 — inflow / outflow regions (region taken from object bbox)
    inflows = []
    outflows = []
    for o in scene.objects:
        if o is domain_obj: continue
        if o.gpufluid_inflow.is_inflow:
            ilow, ihiw = _bbox_world(o)
            inflows.append({
                "lo": to_sim(ilow), "hi": to_sim(ihiw),
                "velocity": tuple(o.gpufluid_inflow.velocity),
                "rate_per_sec": float(o.gpufluid_inflow.rate_per_sec),
                "frame_start": int(o.gpufluid_inflow.frame_start),
                "frame_end": int(o.gpufluid_inflow.frame_end),
            })
        if o.gpufluid_outflow.is_outflow:
            olow, ohiw = _bbox_world(o)
            outflows.append({
                "lo": to_sim(olow), "hi": to_sim(ohiw),
                "frame_start": int(o.gpufluid_outflow.frame_start),
                "frame_end": int(o.gpufluid_outflow.frame_end),
            })

    stg = getattr(dprops, "surface_tension_group", None)
    surface_tension = float(stg.surface_tension) if stg is not None else 0.0
    csf_passes = int(stg.csf_smoothing_passes) if stg is not None else 2

    return {
        "domain": {"resolution": res, "dx": dx, "origin": list(origin)},
        "fluids": fluid_sources,
        "obstacles": obstacles,
        "inflows": inflows,
        "outflows": outflows,
        "simulation": {
            "dt": dprops.dt, "frames": dprops.frames, "fps": dprops.fps,
            "pressure_iters": dprops.pressure_iters,
            "pressure_solver": dprops.pressure_solver,
            "cfl": dprops.use_cfl or (surface_tension > 0.0),
            "cfl_factor": dprops.cfl_factor,
            "cfl_max_substeps": dprops.cfl_max_substeps,
            "flip_blend": dprops.flip_blend, "gravity": dprops.gravity,
            "surface_tension": surface_tension,
            "csf_smoothing_passes": csf_passes,
            "reseed": dprops.reseed,
            "reseed_every_n_frames": dprops.reseed_every_n_frames,
            "reseed_min_per_cell": dprops.reseed_min_per_cell,
            "reseed_max_per_cell": dprops.reseed_max_per_cell,
        },
        "output": _output_dict(dprops),
    }, dom_size, origin


def _output_dict(dprops):
    out = {
        "cache_dir": bpy.path.abspath(dprops.cache_dir),
        "iso_level": dprops.iso_level, "smooth_passes": dprops.smooth_passes,
        "mesh_smooth_passes": dprops.mesh_smooth_passes,
        "mesh_smooth_method": dprops.mesh_smooth_method,
        "decimate_ratio": dprops.decimate_ratio,
        "wall_margin_cells": dprops.wall_margin_cells,
        "usd": dprops.write_usd,
        "particles": dprops.write_particles, "preview": False,
    }
    ww = getattr(dprops, "whitewater_group", None)
    if ww is not None and ww.enable:
        out["whitewater"] = True
        out["whitewater_speed_threshold"] = float(ww.speed_threshold)
        out["whitewater_lifetime_sec"] = float(ww.lifetime_sec)
        out["whitewater_emit_per_frame_max"] = int(ww.emit_per_frame_max)
        out["whitewater_total_cap"] = int(ww.total_cap)
    return out


def _export_obj(obj, path):
    """Quick OBJ export of a single object's evaluated mesh."""
    dg = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(dg)
    mesh = eval_obj.to_mesh()
    try:
        with open(path, "w") as f:
            f.write(f"# gpufluid obstacle export: {obj.name}\n")
            mw = obj.matrix_world
            for v in mesh.vertices:
                wv = mw @ v.co
                f.write(f"v {wv.x} {wv.y} {wv.z}\n")
            for p in mesh.polygons:
                idx = " ".join(str(i + 1) for i in p.vertices)
                f.write(f"f {idx}\n")
    finally:
        eval_obj.to_mesh_clear()


# [BLK A8.5]
class GPUFLUID_OT_bake(bpy.types.Operator):
    bl_idname = "gpufluid.bake"
    bl_label = "Bake gpufluid Simulation"
    bl_description = "Spawns the gpufluid CLI subprocess and bakes the cache"
    bl_options = {"REGISTER"}

    _timer = None
    _proc: Optional[subprocess.Popen] = None
    _current_frame = 0
    _total_frames = 0
    _domain_obj_name = ""

    @classmethod
    def poll(cls, context):
        return any(o.gpufluid_domain.is_domain for o in context.scene.objects)

    def execute(self, context):
        prefs = get_prefs(context)
        interp = bpy.path.abspath(prefs.interpreter_path).strip()
        if not interp or not Path(interp).exists():
            self.report({"ERROR"}, "Set a valid Python interpreter in Addon Preferences (must have gpufluid installed).")
            return {"CANCELLED"}

        domain = next((o for o in context.scene.objects if o.gpufluid_domain.is_domain), None)
        if domain is None:
            self.report({"ERROR"}, "No Domain object in scene.")
            return {"CANCELLED"}
        self._domain_obj_name = domain.name

        try:
            scene_dict, dom_size, origin = collect_scene(context, domain)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        toml_str = build_toml(scene_dict)
        cache_dir = Path(scene_dict["output"]["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        toml_path = cache_dir / "scene.toml"
        toml_path.write_text(toml_str, encoding="utf-8")

        self._total_frames = scene_dict["simulation"]["frames"]
        self._current_frame = 0

        self._proc = subprocess.Popen(
            [interp, "-m", "gpufluid.cli", "simulate", str(toml_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, encoding="utf-8", errors="replace",
        )
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(self)
        domain["gpufluid_origin"] = list(origin)
        domain["gpufluid_dom_size"] = list(dom_size)
        domain["gpufluid_cache_dir"] = str(cache_dir)
        context.workspace.status_text_set("gpufluid baking…")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._proc is None:
            return self._finish(context, ok=False)
        rc = self._proc.poll()
        # drain a few lines
        if self._proc.stdout is not None:
            for _ in range(20):
                line = self._proc.stdout.readline()
                if not line:
                    break
                m = _FRAME_RE.search(line)
                if m:
                    self._current_frame = int(m.group(1))
                    self._total_frames = int(m.group(2))
                print("[gpufluid]", line.rstrip())
                context.workspace.status_text_set(
                    f"gpufluid: frame {self._current_frame}/{self._total_frames}")
        if rc is None:
            return {"PASS_THROUGH"}
        return self._finish(context, ok=(rc == 0))

    def _finish(self, context, ok):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        if not ok:
            self.report({"ERROR"}, "gpufluid bake failed — see system console")
            return {"CANCELLED"}
        # auto-attach cache to target object
        domain = bpy.data.objects.get(self._domain_obj_name)
        if domain is not None:
            target = domain.gpufluid_domain.target_object
            cache_dir = domain.get("gpufluid_cache_dir", "")
            origin = domain.get("gpufluid_origin", [0, 0, 0])
            if cache_dir:
                try:
                    bpy.ops.gpufluid.attach_cache(
                        cache_dir=str(cache_dir),
                        target_name=target.name if target else "",
                        origin_x=float(origin[0]),
                        origin_y=float(origin[1]),
                        origin_z=float(origin[2]),
                    )
                except Exception as e:
                    self.report({"WARNING"}, f"bake ok, but auto-attach failed: {e}")
        self.report({"INFO"}, "gpufluid bake complete")
        return {"FINISHED"}
