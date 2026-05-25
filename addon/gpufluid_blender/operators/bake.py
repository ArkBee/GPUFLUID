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
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional

import bpy
import mathutils

from .. import logger
from ..config_builder import build_toml
from ..preferences import get_prefs
from ._animation import _bbox_world, _bbox_world_at_frame, _is_animated
from ._collect import _export_obj, _output_dict
from .helpers import subprocess_drain


_FRAME_RE = re.compile(r"frame\s+(\d+)/(\d+)")


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
    # Resolution along each axis (proportional to world extent).
    nmax_w = max(dom_size.x, dom_size.y, dom_size.z)
    res = (max(8, int(round(dom_size.x / nmax_w * dprops.resolution))),
           max(8, int(round(dom_size.y / nmax_w * dprops.resolution))),
           max(8, int(round(dom_size.z / nmax_w * dprops.resolution))))
    # Geometry is normalised to [0,1]³ (see to_sim() below), so dx must be in
    # the same unit-cube basis. FLIP uses dx for kernel sizes; MPM ignores it.
    dx = 1.0 / float(max(res))

    # The MPM solver runs in a fixed [0,1]³ unit cube (see MpmDomainWalls in
    # src/gpufluid/sim/mpm/solver.py — hardcoded lo=0.05, hi=0.95). The mesher
    # likewise outputs vertices in normalized [0,1]³. So all positions the
    # solver receives must be normalized: world coord → translate by domain's
    # world-origin → divide by domain size. When the addon attaches the cache
    # it scales the cache object by `dom_size` and translates it by `origin`,
    # so the inverse transform makes mesh world coords match emitter world
    # coords exactly. Domains scaled to ≠ 1×1×1 used to silently desync —
    # everything was in metres for the solver, in [0,1] for the mesh.
    inv_size = (1.0 / float(dom_size.x), 1.0 / float(dom_size.y), 1.0 / float(dom_size.z))

    def to_sim(v):
        return (float((v[0] - origin.x) * inv_size[0]),
                float((v[1] - origin.y) * inv_size[1]),
                float((v[2] - origin.z) * inv_size[2]))

    # fluid sources — emit one entry per source so each can carry its own
    # ppc + color. Pre-B1.3 these were unioned into one bbox.
    fluid_objs = [o for o in scene.objects if o.gpufluid_fluid.is_fluid and o is not domain_obj]
    inflow_objs = [o for o in scene.objects if o.gpufluid_inflow.is_inflow and o is not domain_obj]
    if not fluid_objs and not inflow_objs:
        raise RuntimeError(
            "no fluid source in scene — mark a mesh as Fluid (initial volume) "
            "or as Inflow (continuous emitter, MPM-only)"
        )
    fluid_sources = []
    warnings: list[str] = []
    # eps margin is in normalised units now (positions are too). Use a
    # 1.5-cell margin so the seeded box sits one and a half cells inside
    # the [0,1]³ unit cube the solver enforces.
    eps_norm = 1.5 / float(dprops.resolution)

    def _out_of_domain(name, lo, hi, kind):
        """Append a warning if a source's AABB lies (even partly) outside
        the unit cube the solver simulates in. Without this, an emitter
        above the domain ceiling silently produces fluid that materialises
        at the wall after the solver clamps it."""
        for axis, axis_name in enumerate("xyz"):
            if hi[axis] < 0.0 or lo[axis] > 1.0:
                warnings.append(
                    f"{kind} '{name}' is fully outside the Domain on {axis_name} "
                    f"(normalised range [{lo[axis]:.2f},{hi[axis]:.2f}] vs [0,1]); "
                    "particles will be wall-clamped to the nearest face. "
                    "Move it inside the Domain bounds.")
                return
            if lo[axis] < eps_norm or hi[axis] > 1.0 - eps_norm:
                warnings.append(
                    f"{kind} '{name}' touches/clips the Domain wall on "
                    f"{axis_name} (normalised [{lo[axis]:.2f},{hi[axis]:.2f}]). "
                    "Solver will wall-clamp those particles — move the source "
                    "deeper inside to avoid visual artefacts.")
                return

    for fobj in fluid_objs:
        flow, fhiw = _bbox_world(fobj)
        flo_sim_raw = list(to_sim(flow))
        fhi_sim_raw = list(to_sim(fhiw))
        _out_of_domain(fobj.name, flo_sim_raw, fhi_sim_raw, "Fluid source")
        flo_sim = [max(eps_norm, v) for v in flo_sim_raw]
        fhi_sim = [min(1.0 - eps_norm, fhi_sim_raw[i]) for i in range(3)]
        fprops = fobj.gpufluid_fluid
        entry = {"kind": "box", "lo": tuple(flo_sim), "hi": tuple(fhi_sim),
                 "ppc": int(fprops.ppc)}
        if getattr(fprops, "use_color", False):
            entry["color"] = tuple(float(c) for c in fprops.color)
        if getattr(fprops, "use_temperature", False):
            entry["temperature"] = float(fprops.temperature)
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
        # World-space half-extents → normalised (divide by dom_size per axis).
        hx_w = (ohiw[0] - olow[0]) * 0.5
        hy_w = (ohiw[1] - olow[1]) * 0.5
        hz_w = (ohiw[2] - olow[2]) * 0.5
        hx = hx_w * inv_size[0]
        hy = hy_w * inv_size[1]
        hz = hz_w * inv_size[2]
        # Uniform-radius shapes (sphere/cylinder) need a single scalar — use
        # the dominant axis after normalisation so the obstacle stays inside
        # the unit cube even when the domain is anisotropic.
        avg_inv = (inv_size[0] + inv_size[1] + inv_size[2]) / 3.0
        if oprops.obstacle_type == "BBOX":
            obstacles.append({
                "type": "box",
                "center": to_sim(centre_w),
                "half_size": (hx, hy, hz),
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
            r = max(hx_w, hy_w, hz_w) * avg_inv
            obstacles.append({"type": "sphere", "center": to_sim(centre_w), "radius": r})
        elif oprops.obstacle_type == "CYLINDER_Y":
            r = max(hx_w, hz_w) * avg_inv
            obstacles.append({"type": "cylinder_y", "center": to_sim(centre_w),
                              "radius": r, "half_height": hy})
        elif oprops.obstacle_type == "MESH":
            # export object's mesh as OBJ next to scene.toml
            cache_dir = bpy.path.abspath(dprops.cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            mesh_path = os.path.join(cache_dir, f"obstacle_{o.name}.obj")
            _export_obj(o, mesh_path)
            # OBJ verts are in world metres — scale to the unit cube so the
            # solver's [0,1]³ collision sampling lands on the mesh.
            obstacles.append({
                "type": "mesh", "path": mesh_path,
                "scale": avg_inv,
                "translate": (-origin.x * avg_inv, -origin.y * avg_inv,
                              -origin.z * avg_inv),
            })
        # v0.5 — animated obstacle motion — velocity in m/s gets normalised
        # the same way positions are.
        if oprops.motion_type == "LINEAR":
            v = oprops.motion_velocity
            obstacles[-1]["motion"] = {
                "kind": "linear",
                "velocity": (float(v[0]) * inv_size[0],
                             float(v[1]) * inv_size[1],
                             float(v[2]) * inv_size[2]),
            }

    # v0.5 — inflow / outflow regions (region taken from object bbox).
    # S2.17.7.MOVING: if the inflow object is animated (has an action or any
    # F-curves), sample its world AABB at every integer frame in
    # [frame_start, frame_end] and emit as `keyframes` so the MPM solver can
    # spawn each particle at the source's position at *its* spawn time.
    # Animation/bbox helpers live in operators/_animation.py (extracted in
    # Phase 4 to keep this file under the 500-line cap).
    inflows = []
    outflows = []

    for o in scene.objects:
        if o is domain_obj: continue
        if o.gpufluid_inflow.is_inflow:
            fs = int(o.gpufluid_inflow.frame_start)
            fe = int(o.gpufluid_inflow.frame_end)
            # Use fcurve-evaluated AABB at fs (no depsgraph rebuild).
            ilow, ihiw = _bbox_world_at_frame(o, fs)
            lo_sim = to_sim(ilow); hi_sim = to_sim(ihiw)
            _out_of_domain(o.name, list(lo_sim), list(hi_sim), "Inflow")
            v = o.gpufluid_inflow.velocity
            entry = {
                "lo": lo_sim, "hi": hi_sim,
                "velocity": (float(v[0]) * inv_size[0],
                             float(v[1]) * inv_size[1],
                             float(v[2]) * inv_size[2]),
                "rate_per_sec": float(o.gpufluid_inflow.rate_per_sec),
                "frame_start": fs, "frame_end": fe,
            }
            # B18 — per-inflow attrs (only emit keys when toggles are on, so
            # the TOML stays minimal and the zero-overhead path in MpmSolver
            # engages for scenes that don't use colour/temperature).
            iprops = o.gpufluid_inflow
            if getattr(iprops, "use_color", False):
                entry["color"] = tuple(float(c) for c in iprops.color)
            if getattr(iprops, "use_temperature", False):
                entry["temperature"] = float(iprops.temperature)
            if _is_animated(o, context) and fe > fs:
                kfs = []
                for f in range(fs, fe + 1):
                    lf, hf = _bbox_world_at_frame(o, f)
                    ls = to_sim(lf); hs = to_sim(hf)
                    kfs.append([f, ls[0], ls[1], ls[2], hs[0], hs[1], hs[2]])
                entry["keyframes"] = kfs
            inflows.append(entry)
        if o.gpufluid_outflow.is_outflow:
            ilow_, ihiw_ = _bbox_world(o)
            outflows.append({
                "lo": to_sim(ilow_), "hi": to_sim(ihiw_),
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
            "solver": dprops.solver,  # B17.12 dispatch: "flip" | "mpm"
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
            # MPM-specific params (config_builder emits only when solver=mpm)
            "mpm_bulk_modulus": dprops.mpm_bulk_modulus,
            "mpm_rpic_damping": dprops.mpm_rpic_damping,
            "mpm_grid_v_damping": dprops.mpm_grid_v_damping,
            "mpm_cube_friction": dprops.mpm_cube_friction,
            "mpm_v_terminal": dprops.mpm_v_terminal,
            "mpm_vz_max_splash": dprops.mpm_vz_max_splash,
            # m/s in UI → unit/s in solver basis. Matches the inflow-velocity
            # convention (bake.py inflow path, ~line 321) so the column pours
            # at the same speed regardless of domain world size.
            "mpm_initial_velocity": float(dprops.mpm_initial_velocity) * inv_size[1],
        },
        "output": _output_dict(dprops),
        # Phase 1 escape-hatch: raw TOML string deep-merged in config_builder.
        "toml_overrides": str(getattr(dprops, "toml_overrides", "") or ""),
    }, dom_size, origin, warnings


# [BLK A8.5]
class GPUFLUID_OT_bake(bpy.types.Operator):
    bl_idname = "gpufluid.bake"
    bl_label = "Bake gpufluid Simulation"
    bl_description = "Spawns the gpufluid CLI subprocess and bakes the cache"
    bl_options = {"REGISTER"}

    _timer = None
    _proc: Optional[subprocess.Popen] = None
    _stdout_q: Optional[queue.Queue] = None
    _stdout_thread: Optional[threading.Thread] = None
    _current_frame = 0
    _total_frames = 0
    _domain_obj_name = ""
    # Class-level reentrance guard. Set True by execute() when modal arms,
    # cleared by _abort()/_finish(). Prevents a second click on Bake while
    # the first subprocess is still running — without this, two CLI
    # processes write to the same cache_dir, racing each other and
    # corrupting the output (live-found 2026-05-25 during round-3 res
    # change test). One bake at a time; second click reports WARNING.
    _is_running: bool = False

    @classmethod
    def poll(cls, context):
        return any(o.gpufluid_domain.is_domain for o in context.scene.objects)

    def execute(self, context):
        if GPUFLUID_OT_bake._is_running:
            self.report({"WARNING"},
                        "A gpufluid bake is already running — wait for it "
                        "to finish or press Esc to cancel before starting "
                        "a new one.")
            return {"CANCELLED"}
        prefs = get_prefs(context)
        interp = bpy.path.abspath(prefs.interpreter_path).strip()
        if not interp or not Path(interp).exists():
            self.report({"ERROR"}, "Set a valid Python interpreter in Addon Preferences (must have gpufluid installed).")
            return {"CANCELLED"}

        # Pick the domain: prefer the SELECTED domain if it's one (standard
        # Blender pattern); otherwise fall back to the first domain in scene.
        # Reports clearly when multiple domains exist so user knows which
        # one is being baked.
        all_domains = [o for o in context.scene.objects if o.gpufluid_domain.is_domain]
        if not all_domains:
            self.report({"ERROR"}, "No Domain object in scene. "
                        "Click `Add gpufluid Domain` in the sidebar first.")
            return {"CANCELLED"}
        active = context.active_object
        if active is not None and active.gpufluid_domain.is_domain:
            domain = active
        else:
            domain = all_domains[0]
        if len(all_domains) > 1:
            other = [o.name for o in all_domains if o.name != domain.name]
            self.report({"INFO"},
                f"Baking domain '{domain.name}'. Other domains in scene "
                f"({', '.join(other)}) are ignored — select one to switch.")
        self._domain_obj_name = domain.name

        # Reject obviously-broken scenes early so the CLI doesn't waste
        # time launching, init'ing, then crashing silently. Live-found
        # 2026-05-25 (round-4 test #24): inflow with frame_start > frame_end
        # produced valid TOML, CLI accepted it, then bake quit with 0
        # mesh frames written and no error report.
        for o in context.scene.objects:
            if o.gpufluid_inflow.is_inflow:
                fs = int(o.gpufluid_inflow.frame_start)
                fe = int(o.gpufluid_inflow.frame_end)
                if fs > fe:
                    self.report({"ERROR"},
                        f"Inflow '{o.name}' has frame_start ({fs}) > "
                        f"frame_end ({fe}) — emission window is empty. "
                        f"Fix the inflow's frame range and re-bake.")
                    return {"CANCELLED"}
            if o.gpufluid_outflow.is_outflow:
                fs = int(o.gpufluid_outflow.frame_start)
                fe = int(o.gpufluid_outflow.frame_end)
                if fs > fe:
                    self.report({"ERROR"},
                        f"Outflow '{o.name}' has frame_start ({fs}) > "
                        f"frame_end ({fe}). Fix the range and re-bake.")
                    return {"CANCELLED"}
        if int(domain.gpufluid_domain.frames) <= 0:
            self.report({"ERROR"},
                f"Domain frames is {domain.gpufluid_domain.frames} — "
                f"set it to a positive integer.")
            return {"CANCELLED"}

        # Auto-fill empty cache_dir so the user gets a working bake on first
        # click even if they marked an existing Empty as Domain manually
        # (instead of using `Add Domain` which sets cache_dir for them).
        if not domain.gpufluid_domain.cache_dir.strip():
            import tempfile
            blend_path = bpy.data.filepath
            if blend_path:
                fallback = bpy.path.abspath(f"//cache/{domain.name}")
            else:
                fallback = os.path.join(tempfile.gettempdir(),
                                        f"gpufluid_cache_{domain.name}")
            domain.gpufluid_domain.cache_dir = fallback
            self.report({"INFO"},
                f"Cache Directory was empty — auto-filled to {fallback}")

        try:
            scene_dict, dom_size, origin, warnings = collect_scene(context, domain)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        # Surface every wall-clip / out-of-domain warning before the heavy
        # bake kicks off. The user can cancel and reposition without paying
        # the 30+ second simulation cost on a misconfigured scene.
        for w in warnings:
            self.report({"WARNING"}, w)

        try:
            toml_str = build_toml(scene_dict)
        except ValueError as e:
            self.report({"ERROR"}, f"TOML overrides invalid: {e}")
            return {"CANCELLED"}
        cache_dir = Path(scene_dict["output"]["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        toml_path = cache_dir / "scene.toml"
        toml_path.write_text(toml_str, encoding="utf-8")

        self._total_frames = scene_dict["simulation"]["frames"]
        self._current_frame = 0

        # CRITICAL: Detach any MeshSequenceCache modifiers BEFORE the CLI
        # subprocess runs. The CLI writes <cache_dir>/cache.abc as part of
        # the bake, but Windows won't let it overwrite a file Blender has
        # mmap'd via MSC — the new .abc write fails silently, the user
        # sees the OLD simulation on next play. We strip MSCs + drop
        # orphan cache_files, then re-attach in _finish() after CLI exits.
        for obj in context.scene.objects:
            for m in list(obj.modifiers):
                if m.type == "MESH_SEQUENCE_CACHE":
                    m.cache_file = None
                    obj.modifiers.remove(m)
        # Direct data-API drop (no dependency on Outliner context — the
        # original bpy.ops.outliner.orphans_purge failed poll outside an
        # Outliner area and the try/except masked it, leaving .abc mmap'd).
        for cf in list(bpy.data.cache_files):
            if cf.users == 0:
                try:
                    bpy.data.cache_files.remove(cf)
                except Exception as e:
                    logger.warning(
                        "addon.bake.cache_files_remove_failed",
                        extra={"name": cf.name, "err": str(e)})

        # Strip stale per-frame artefacts so a re-bake with a shorter range
        # (or a different solver) doesn't leave previous-bake .ply files in
        # mesh/, particles_raw/, colors/. Without this the preloader would
        # pick up the leftovers and the viewport would show a frame from
        # the prior bake at frames beyond the new frame_count.
        # Live-found 2026-05-25 during round-3 testing.
        import shutil as _shutil
        out_root = Path(str(scene_dict["output"]["cache_dir"]))
        for sub in ("mesh", "particles_raw", "colors"):
            d = out_root / sub
            if d.is_dir():
                try:
                    _shutil.rmtree(d)
                except Exception as e:
                    logger.warning(
                        "addon.bake.stale_subdir_clean_failed",
                        extra={"dir": str(d), "err": str(e)})

        self._proc = subprocess.Popen(
            [interp, "-m", "gpufluid.cli", "simulate", str(toml_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, encoding="utf-8", errors="replace",
        )
        # Drain stdout in a background thread → queue, so modal() never
        # blocks on readline() (which freezes Blender UI when the subprocess
        # spends time silently, e.g. inside a nested Blender headless run
        # for Alembic conversion). Helper extracted to operators/helpers.py
        # so OT_render shares the exact same drain semantics (Phase 4).
        self._stdout_q = queue.Queue()
        self._stdout_thread = threading.Thread(
            target=subprocess_drain, args=(self._proc, self._stdout_q),
            daemon=True)
        self._stdout_thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(self)
        domain["gpufluid_origin"] = list(origin)
        domain["gpufluid_dom_size"] = list(dom_size)
        domain["gpufluid_cache_dir"] = str(cache_dir)
        context.workspace.status_text_set("gpufluid baking…")
        GPUFLUID_OT_bake._is_running = True
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        # ESC: user-requested abort. Without this branch Blender's modal loop
        # never delivers cancel to us — only TIMER events get acted on — so
        # the gpufluid.cli subprocess kept running through Esc, right-click,
        # and even Blender window close (orphan process). Live-found
        # 2026-05-25 during round-3 cancellation test.
        if event.type == "ESC":
            return self._abort(context, reason="user pressed Esc")
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._proc is None:
            return self._finish(context, ok=False)
        rc = self._proc.poll()
        # Drain queue (non-blocking — thread already collected lines)
        if self._stdout_q is not None:
            for _ in range(50):
                try:
                    line = self._stdout_q.get_nowait()
                except queue.Empty:
                    break
                if line is None:   # sentinel: stdout EOF
                    break
                m = _FRAME_RE.search(line)
                if m:
                    self._current_frame = int(m.group(1))
                    self._total_frames = int(m.group(2))
                logger.info("bake: %s", line.rstrip())
                context.workspace.status_text_set(
                    f"gpufluid: frame {self._current_frame}/{self._total_frames}")
        if rc is None:
            return {"PASS_THROUGH"}
        return self._finish(context, ok=(rc == 0))

    def _abort(self, context, reason: str):
        """Terminate the bake subprocess + restore UI. Idempotent — safe
        to call from modal(ESC) and from Blender's cancel() callback."""
        GPUFLUID_OT_bake._is_running = False
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            except Exception as e:
                logger.warning("addon.bake.terminate_failed",
                               extra={"err": str(e)})
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        self.report({"WARNING"}, f"gpufluid bake aborted ({reason})")
        return {"CANCELLED"}

    def cancel(self, context):
        # Blender calls this on right-click / window close / Esc when no
        # custom modal handler caught the event. We route both paths
        # through the same _abort to guarantee subprocess teardown.
        self._abort(context, reason="cancelled by Blender")

    def _finish(self, context, ok):
        GPUFLUID_OT_bake._is_running = False
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
                dom_size = list(domain.get("gpufluid_dom_size", [1.0, 1.0, 1.0]))
                try:
                    bpy.ops.gpufluid.attach_cache(
                        cache_dir=str(cache_dir),
                        target_name=target.name if target else "",
                        origin=tuple(float(c) for c in origin),
                        dom_size=tuple(float(c) for c in dom_size),
                    )
                except Exception as e:
                    self.report({"WARNING"}, f"bake ok, but auto-attach failed: {e}")
                # Auto-attach whitewater cache when the bake produced one.
                ww_dir = os.path.join(str(cache_dir), "whitewater")
                if os.path.isdir(ww_dir):
                    try:
                        bpy.ops.gpufluid.attach_ww_cache(
                            cache_dir=str(cache_dir),
                            target_name="",   # always make a new mesh
                            origin_x=float(origin[0]),
                            origin_y=float(origin[1]),
                            origin_z=float(origin[2]),
                        )
                    except Exception as e:
                        self.report({"WARNING"}, f"bake ok, but ww auto-attach failed: {e}")
        self.report({"INFO"}, "gpufluid bake complete")
        return {"FINISHED"}
