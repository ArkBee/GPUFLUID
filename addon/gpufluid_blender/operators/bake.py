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
from ..domain_transform import DomainTransform
from ..preferences import get_prefs
from ..scene_validator import out_of_domain_warning
from ._animation import _bbox_world, _bbox_world_at_frame, _is_animated
from ._collect import _export_obj, _output_dict
from .helpers import subprocess_drain
from ._runner import ModalSubprocessRunner


_FRAME_RE = re.compile(r"frame\s+(\d+)/(\d+)")


def collect_scene(context, domain_obj):
    """Build the bpy-free scene dict used by config_builder.

    Translates everything so the Domain bbox starts at origin (sim space).
    """
    scene = context.scene
    dom_lo_w, dom_hi_w = _bbox_world(domain_obj)
    dprops = domain_obj.gpufluid_domain

    # Round-17: domain math extracted into a pure dataclass — unit-
    # testable, single source of truth for the world↔unit-cube mapping.
    # See addon/gpufluid_blender/domain_transform.py.
    transform = DomainTransform.from_world_aabb(
        tuple(dom_lo_w), tuple(dom_hi_w), int(dprops.resolution))
    origin = mathutils.Vector(transform.origin)
    dom_size = mathutils.Vector(transform.dom_size)
    res = transform.resolution
    dx = transform.dx
    inv_size = transform.inv_size
    to_sim = transform.to_sim

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
    # Round-17: eps_norm now lives on DomainTransform; pure validation
    # extracted to scene_validator.out_of_domain_warning.
    eps_norm = transform.eps_norm

    def _out_of_domain(name, lo, hi, kind):
        w = out_of_domain_warning(name, lo, hi, kind, eps_norm)
        if w:
            warnings.append(w)

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
        # the same way positions are. Round-17: via DomainTransform.
        if oprops.motion_type == "LINEAR":
            obstacles[-1]["motion"] = {
                "kind": "linear",
                "velocity": transform.normalize_velocity(oprops.motion_velocity),
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
            entry = {
                "lo": lo_sim, "hi": hi_sim,
                "velocity": transform.normalize_velocity(o.gpufluid_inflow.velocity),
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

    # Synchronous mode (round-6): skip the modal/timer dance and block
    # in Popen.wait() until the CLI finishes. Required for scripted /
    # CI / MCP-driven workflows where the Blender event loop never
    # ticks between successive bpy.ops calls — without this, callers
    # had to sleep+poll the filesystem and still races on cache.json
    # writeback timing. UI button defaults to sync=False (preserves
    # the cancellable, status-bar-progress modal UX).
    sync: bpy.props.BoolProperty(
        name="Wait for completion",
        description="Block until the CLI subprocess finishes (no modal). "
                    "Useful for batch scripts. UI defaults to False.",
        default=False,
        options={'SKIP_SAVE', 'HIDDEN'},
    )
    sync_timeout_sec: bpy.props.IntProperty(
        name="Sync timeout (seconds)",
        description="Round-8 hardening: hard cap on sync-mode Popen.wait(). "
                    "Without this, a hung CLI freezes Blender forever "
                    "(no ESC, no modal). 0 = no limit. Default 600s = 10min "
                    "(generous for 256³ bakes; tune down for CI).",
        default=600, min=0, soft_max=7200,
        options={'SKIP_SAVE', 'HIDDEN'},
    )

    # Round-14: subprocess lifecycle (Popen/Queue/Thread/timer/cancel)
    # extracted into ModalSubprocessRunner. Operator owns scene-collection
    # + UI; runner owns process plumbing. `self._runner` is lazy-inited
    # in execute() (Blender creates a fresh op instance per click).
    _runner: Optional[ModalSubprocessRunner] = None
    _domain_obj_name = ""
    _cache_dir_str = ""    # for sync post-bake auto-attach
    _origin_tuple = (0.0, 0.0, 0.0)
    _dom_size_tuple = (1.0, 1.0, 1.0)

    # Class-level reentrance guard. Set True by execute() before subprocess
    # spawn, cleared by ModalSubprocessRunner abort/finish paths. Prevents
    # a second click on Bake while the first subprocess is still running.
    # Live-found round-3 (res-change test → 2 CLIs racing same cache_dir).
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
                    bpy.data.cache_files.remove(cf, do_unlink=True)
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

        # Set reentrance flag BEFORE spawn (round-6 race fix). Sync mode
        # also wants the guard so two parallel scripted bakes don't both
        # Popen into the same cache_dir.
        GPUFLUID_OT_bake._is_running = True
        # Round-19: bake-trace via cache_binding (single source of key strings).
        from .. import cache_binding as _cb
        _cb.set_bake_trace(domain, str(cache_dir),
                            tuple(origin), tuple(dom_size))

        # Remember for auto-attach (sync path uses inline; modal calls
        # _finish_auto_attach via tick_modal → _finish_complete below).
        self._cache_dir_str = str(cache_dir)
        self._origin_tuple = tuple(float(c) for c in origin)
        self._dom_size_tuple = tuple(float(c) for c in dom_size)
        self._toml_str_snapshot = toml_str   # for sync truncation sanity

        argv = [interp, "-m", "gpufluid.cli", "simulate", str(toml_path)]
        self._runner = ModalSubprocessRunner("bake")

        # ─── SYNC PATH ────────────────────────────────────────────────
        if self.sync:
            res = self._runner.start_sync(
                self, argv, int(self.sync_timeout_sec),
                log_prefix="bake", logger=logger,
                on_complete=self._sync_post_bake_then_attach,
                friendly_error_for_rc=self._friendly_error_for_rc,
            )
            return res

        # ─── MODAL PATH ───────────────────────────────────────────────
        return self._runner.start_modal(
            self, context, argv, status_msg="gpufluid baking…")

    def modal(self, context, event):
        # Delegate to runner. tick_modal handles ESC + TIMER + drain +
        # progress regex. Returns None to PASS_THROUGH or a final set.
        result = self._runner.tick_modal(
            self, context, event,
            frame_regex=_FRAME_RE,
            on_progress=self._update_status,
            logger=logger, log_prefix="bake",
            friendly_error_for_rc=self._friendly_error_for_rc,
        )
        if result is None:
            return {"PASS_THROUGH"}
        if "FINISHED" in result:
            # Round-28: re-arm the class-level reentrance mutex around
            # the auto-attach call. tick_modal._finish() cleared
            # `_is_running=False` before returning FINISHED, opening a
            # single-tick race window where the user clicking Bake
            # again here would spawn a fresh subprocess while
            # `_auto_attach_post_bake` is mid-mutation of `_PRELOAD`
            # (live-found by round-27 reviewer). Hold the mutex for
            # the attach call too; clear after.
            cls = self.__class__
            try:
                cls._is_running = True
                self._auto_attach_post_bake()
                self.report({"INFO"}, "gpufluid bake complete")
            finally:
                cls._is_running = False
        return result

    def cancel(self, context):
        # Blender callback (right-click / window-close). Route through
        # runner.abort to guarantee subprocess teardown.
        if self._runner is not None:
            self._runner.cancel(self, context)

    # ── Operator-specific callbacks for the runner ──────────────────

    def _update_status(self, current: int, total: int) -> None:
        """Called by runner.tick_modal on each parsed `frame N/M` line."""
        bpy.context.workspace.status_text_set(
            f"gpufluid: frame {current}/{total}")

    def _friendly_error_for_rc(self, rc: int):
        """Round-20: translate CLI rc into actionable error message.
        Reads ``<cache_dir>/cache.json`` for the rc=2 (MPM divergence)
        case — surfaces the truncated frame + recovery hint. Returns
        ``None`` when no specific guidance applies (runner falls back
        to generic 'rc=N' wording)."""
        if rc != 2:
            return None
        try:
            import json as _json
            manifest = _json.loads(
                (Path(self._cache_dir_str) / "cache.json").read_text())
            if manifest.get("truncation_reason") != "mpm_divergence":
                return None
            frame = manifest.get("truncated_at_frame", "?")
            total = manifest.get("frame_count", "?")
            return (
                f"gpufluid bake: MPM solver diverged at frame {frame}; "
                f"{total} valid frames written to {self._cache_dir_str}. "
                f"Recovery: lower Bulk Modulus (default 1500), lower dt, "
                f"or shrink resolution, then re-bake. Use 'Attach Cache' "
                f"to load the partial result.")
        except Exception:
            return None

    def _sync_post_bake_then_attach(self) -> None:
        """Sync on_complete: run truncation sanity, then auto-attach.
        Errors here downgrade to WARNING (don't fail the whole bake)."""
        self._sync_truncation_sanity()
        self._auto_attach_post_bake()

    def _sync_truncation_sanity(self) -> None:
        """Round-10 stress-test finding: CLI can exit 0 with fewer
        frames than requested. Compare mesh count vs MERGED [simulation]
        frames (round-11 reviewer fix — overrides path).

        Round-12 reviewer #6: tomllib only, no tomli fallback (Blender
        4.5+ ships Python 3.11+ which guarantees tomllib).
        """
        try:
            mesh_dir = Path(self._cache_dir_str) / "mesh"
            actual = (len(list(mesh_dir.glob("frame_*.ply")))
                      if mesh_dir.is_dir() else 0)
            import tomllib
            emitted = tomllib.loads(self._toml_str_snapshot)
            expected = int(emitted.get("simulation", {}).get("frames", 0))
            if expected > 0 and 0 < actual < expected:
                self.report({"WARNING"},
                    f"gpufluid bake produced {actual}/{expected} frames "
                    f"— CLI exited cleanly but truncated (likely solver "
                    f"OOM or early-stop). Cache attached as-is.")
        except Exception:
            pass

    def _auto_attach_post_bake(self) -> None:
        """Shared auto-attach for sync + modal success paths.
        Mirrors what _finish used to do inline."""
        domain = bpy.data.objects.get(self._domain_obj_name)
        if domain is None:
            return
        target = domain.gpufluid_domain.target_object
        from .. import cache_binding as _cb
        trace = _cb.get_bake_trace(domain)
        if trace is None:
            return
        cache_dir = trace["cache_dir"]
        origin = list(trace["origin"])
        dom_size = list(trace["dom_size"])
        try:
            bpy.ops.gpufluid.attach_cache(
                cache_dir=str(cache_dir),
                target_name=target.name if target else "",
                origin=tuple(float(c) for c in origin),
                dom_size=tuple(float(c) for c in dom_size),
            )
        except Exception as e:
            self.report({"WARNING"}, f"bake ok, but auto-attach failed: {e}")
        ww_dir = os.path.join(str(cache_dir), "whitewater")
        if os.path.isdir(ww_dir):
            try:
                bpy.ops.gpufluid.attach_ww_cache(
                    cache_dir=str(cache_dir),
                    target_name="",
                    origin_x=float(origin[0]),
                    origin_y=float(origin[1]),
                    origin_z=float(origin[2]),
                )
            except Exception as e:
                self.report({"WARNING"},
                            f"bake ok, but ww auto-attach failed: {e}")
