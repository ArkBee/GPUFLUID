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
from ..preferences import get_prefs, _detect_interpreter, _context_roots
from ..scene_validator import out_of_domain_warning
from ._animation import _bbox_world, _bbox_world_at_frame, _is_animated
from ._collect import _export_obj, _output_dict, _safe_obj_name
from .helpers import subprocess_drain
from ._runner import ModalSubprocessRunner

try:
    from .._blocks import block
except ImportError:
    # dodge: tests load this file under a stub package with no _blocks
    # (spec_from_file_location harnesses) — registration is irrelevant
    # there; the real registration happens on package import. See _blocks.py.
    def block(_bid, _desc=""):
        def _w(fn):
            return fn
        return _w


# Round-62: match BOTH the FLIP per-frame line ("  frame 0005/40") and
# the MPM per-frame line ("  [MPM] frame 5/40"), tolerant of spaces around
# the slash and case. The legacy "step N/M" form is also accepted so any
# solver/path that still prints steps advances the bar instead of freezing
# it (the "no bake indicator" complaint was MPM printing "step", which the
# old `frame\s+...` regex never matched).
_FRAME_RE = re.compile(r"(?:frame|step)\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


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

    # (FU-030: the S2.17.10 `avg_inv` that lived here is gone — fluid
    # sources now use per-axis inv_size; obstacles compute their own
    # avg_inv inside the obstacle loop where uniform shapes still need it.)
    # Round-53: hoist cache_dir once — the obstacle MESH branch and the
    # round-53 BBOX→MESH auto-promote need it before any obstacle is
    # written. Idempotent makedirs.
    # audit-20260610: hoisted further, ABOVE the fluid-sources loop — the
    # S2.17.10 fluid MESH branch exports its .obj into cache_dir too, and
    # referencing it before the (previous) obstacles-section assignment
    # raised UnboundLocalError on every source_type=MESH bake (and on
    # legacy fill_mesh=True .blends via the BBOX→MESH promotion).
    cache_dir = bpy.path.abspath(dprops.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    for fobj in fluid_objs:
        flow, fhiw = _bbox_world(fobj)
        flo_sim_raw = list(to_sim(flow))
        fhi_sim_raw = list(to_sim(fhiw))
        _out_of_domain(fobj.name, flo_sim_raw, fhi_sim_raw, "Fluid source")
        flo_sim = [max(eps_norm, v) for v in flo_sim_raw]
        fhi_sim = [min(1.0 - eps_norm, fhi_sim_raw[i]) for i in range(3)]
        fprops = fobj.gpufluid_fluid
        # S2.17.10: source shape. "Mark as Source" works on any object; emit
        # the shape so the bake fills it (not just a cube). Legacy fill_mesh
        # bool still promotes BBOX -> MESH for back-compat.
        stype = getattr(fprops, "source_type", "BBOX")
        if stype == "BBOX" and getattr(fprops, "fill_mesh", False):
            stype = "MESH"
        if stype == "SPHERE":
            centre_w = ((flow[0] + fhiw[0]) * 0.5, (flow[1] + fhiw[1]) * 0.5,
                        (flow[2] + fhiw[2]) * 0.5)
            # FU-030: per-axis radii (world half-extent per axis × that
            # axis's inv_size) — exact on non-cubic domains. The previous
            # scalar `radius = max(half_extents) * avg_inv` only matched the
            # visible sphere on a ~cubic domain (audit-20260610 added a
            # warning for that; the warning is now obsolete and removed —
            # the CLI schema carries `radii = [rx, ry, rz]`, scalar `radius`
            # stays as back-compat for hand-written TOMLs only).
            entry = {"kind": "sphere", "center": tuple(to_sim(centre_w)),
                     "radii": tuple(
                         float((fhiw[i] - flow[i]) * 0.5 * inv_size[i])
                         for i in range(3)),
                     "ppc": int(fprops.ppc)}
        elif stype == "MESH":
            mesh_path = os.path.join(
                cache_dir, f"fluid_{_safe_obj_name(fobj.name)}.obj")
            _export_obj(fobj, mesh_path)
            # FU-030: per-axis scale/translate. The world→sim map is
            # sim[i] = (v[i] - origin[i]) * inv_size[i]
            #        = v[i]*inv_size[i] + (-origin[i]*inv_size[i]),
            # so scale = inv_size (vec3) and translate = -origin*inv_size
            # PER AXIS — exact on non-cubic domains. The previous uniform
            # avg_inv squashed/offset the mesh on the short axis
            # (audit-20260610 warned about it; warning now obsolete and
            # removed — schema accepts scale as scalar OR [sx, sy, sz]).
            entry = {"kind": "mesh", "path": mesh_path,
                     "scale": tuple(float(s) for s in inv_size),
                     "translate": (-origin.x * inv_size[0],
                                   -origin.y * inv_size[1],
                                   -origin.z * inv_size[2]),
                     "ppc": int(fprops.ppc)}
        else:  # BBOX
            entry = {"kind": "box", "lo": tuple(flo_sim), "hi": tuple(fhi_sim),
                     "ppc": int(fprops.ppc)}
        if getattr(fprops, "use_color", False):
            entry["color"] = tuple(float(c) for c in fprops.color)
        if getattr(fprops, "use_temperature", False):
            entry["temperature"] = float(fprops.temperature)
        fluid_sources.append(entry)

    # obstacles
    obstacles = []
    # (cache_dir is hoisted above the fluid-sources loop — audit-20260610;
    #  the round-53 hoist used to live here.)
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
        # §9.13 (live-MCP 2026-06-11): the OOTB flow marks Blender's default
        # 2 m Cube as obstacle around a domain it completely engulfs — the
        # solver then has ~no free volume, fluid freezes into nonsense
        # plates, and every operator still reported plain FINISHED. Warn
        # when this obstacle's sim-space bbox covers ≥90% of the domain
        # volume (intersection with the [0,1]³ unit cube, so an obstacle
        # bigger than the domain still reads as 100%, not >100%).
        _olo_s = to_sim(olow)
        _ohi_s = to_sim(ohiw)
        _covered = 1.0
        for _i in range(3):
            _covered *= max(0.0, min(_ohi_s[_i], 1.0) - max(_olo_s[_i], 0.0))
        if _covered >= 0.9:
            warnings.append(
                f"obstacle '{o.name}' covers ~the whole domain "
                f"({_covered * 100.0:.0f}% of its volume) — fluid has "
                f"nowhere to flow; did you mean a floor/container? Shrink "
                f"the obstacle or enlarge the Domain.")
        # Uniform-radius shapes (sphere/cylinder) need a single scalar — use
        # the dominant axis after normalisation so the obstacle stays inside
        # the unit cube even when the domain is anisotropic.
        avg_inv = (inv_size[0] + inv_size[1] + inv_size[2]) / 3.0
        if oprops.obstacle_type == "BBOX":
            # Round-52/53/54: detect "this BBOX won't fit the geometry"
            # and auto-promote to MESH. Two trigger paths:
            #   (a) `rotation_euler != 0` (round-53): visible tilt
            #       not yet baked into mesh data.
            #   (b) round-54 — Apply Rotation (Ctrl+A → R) zeroes
            #       rotation_euler but BAKES the rotation into mesh
            #       vertex data. round-53 detection misses this case
            #       and falls back to AABB-of-rotated-mesh (bigger fat
            #       box again). Now we ALSO check if the mesh's loop
            #       triangles all have axis-aligned face normals — if
            #       any face normal isn't ±X/±Y/±Z within 5°, the mesh
            #       isn't an axis-aligned cube and BBOX type will
            #       butcher it. Promote.
            rx, ry, rz = (float(c) for c in o.rotation_euler)
            rotation_nonzero = abs(rx) + abs(ry) + abs(rz) > 1e-4

            non_axis_aligned = False
            if o.type == "MESH" and not rotation_nonzero:
                try:
                    me = o.data
                    me.calc_loop_triangles()
                    for tri in me.loop_triangles:
                        n = tri.normal
                        # axis-aligned if dominant axis component is
                        # essentially ±1 (other two near zero). cos(5°)
                        # ≈ 0.996. Use 0.99 for a safe margin.
                        if max(abs(n.x), abs(n.y), abs(n.z)) < 0.99:
                            non_axis_aligned = True
                            break
                except Exception:
                    pass

            # Round-57: MPM now has a native OBB collider — for
            # rotated cubes we emit `type=box` + `rotation` matrix and
            # the MPM kernel handles tilt correctly. The pre-round-57
            # auto-promote-to-MESH path stays for FLIP (which uses the
            # SDF mesh-marker D4.3.GPU.BVH path) AND for non-cube tilted
            # meshes on MPM (where bbox + rotation isn't a faithful
            # representation of the geometry).
            is_mpm = (dprops.solver == "mpm")
            # MPM-OBB is faithful only when the source object is a
            # roughly cube-shaped mesh — for arbitrary tilted geometry
            # (a bunny, a torus) we still fall back to FLIP's mesh path.
            # Heuristic: trust BBOX when the user explicitly chose
            # obstacle_type=BBOX (this branch); they're telling us
            # "treat me as a box". So MPM always gets the OBB box here.
            if is_mpm and o.type == "MESH":
                # Build OBB: decompose matrix_world to separate the
                # rotation (orthonormal R) from scale; half_size is the
                # OBJECT-LOCAL bbox half-extent times effective scale.
                # This is what makes the rotated cube the SAME SIZE as
                # the visible mesh (not the world-AABB-of-rotated which
                # is bigger and was the round-52 silent-fail symptom).
                _, rot_q, scl = o.matrix_world.decompose()
                R3 = rot_q.to_matrix()
                # local bbox in object-local coords (already includes
                # mesh-data scale — that's what's drawn on screen).
                local_corners = [mathutils.Vector(c) for c in o.bound_box]
                lo_l = mathutils.Vector((
                    min(v.x for v in local_corners),
                    min(v.y for v in local_corners),
                    min(v.z for v in local_corners),
                ))
                hi_l = mathutils.Vector((
                    max(v.x for v in local_corners),
                    max(v.y for v in local_corners),
                    max(v.z for v in local_corners),
                ))
                centre_l = (lo_l + hi_l) * 0.5
                half_l = (hi_l - lo_l) * 0.5
                # Effective world-frame half-extent (post object scale).
                hx_obb = abs(half_l.x * scl.x)
                hy_obb = abs(half_l.y * scl.y)
                hz_obb = abs(half_l.z * scl.z)
                # World-space centre = matrix_world @ local_centre.
                centre_obb_world = o.matrix_world @ centre_l
                # Convert to sim space.
                cw_sim = to_sim((centre_obb_world.x,
                                 centre_obb_world.y,
                                 centre_obb_world.z))
                # Half-extents into sim space. FU-019 (2026-06-01 deep-dive):
                #  - Axis-aligned box (identity rotation): use PER-AXIS
                #    inv_size — exact, and identical to the hx/hy/hz already
                #    computed above. Pre-FU-019 this used the averaged inv_avg
                #    even for unrotated boxes, which mis-sized the collider on
                #    a non-cubic domain (e.g. a wide tank). The common BBOX
                #    case is axis-aligned, so this is the one that mattered.
                #  - Rotated box: per-axis scaling would SHEAR the box (the
                #    rotation matrix R is orthonormal only under a UNIFORM
                #    scale). Keep the averaged inv_avg so the collider stays a
                #    proper oriented box; the non-cubic+rotated case is
                #    genuinely approximate and is warned below.
                # On a cubic domain inv_avg == inv_size[i], so both branches
                # agree — no behaviour change for the common cubic scene.
                rot_is_identity = not (rotation_nonzero or non_axis_aligned)
                if rot_is_identity:
                    hx_sim, hy_sim, hz_sim = hx, hy, hz
                else:
                    inv_avg = (inv_size[0] + inv_size[1] + inv_size[2]) / 3.0
                    hx_sim = hx_obb * inv_avg
                    hy_sim = hy_obb * inv_avg
                    hz_sim = hz_obb * inv_avg
                if (rotation_nonzero or non_axis_aligned) and (
                    max(inv_size) / min(inv_size) > 1.05
                ):
                    warnings.append(
                        f"obstacle '{o.name}' is rotated and the "
                        f"domain is non-cubic — OBB collider assumes "
                        f"isotropic domain, the tilt may render at "
                        f"the wrong angle. Workaround: make domain "
                        f"cubic, or rebuild obstacle as a stack of "
                        f"axis-aligned primitives.")
                # Apply-Rotation case: mesh data was rotated, then
                # Ctrl+A → R zeroed rotation_euler. We cannot recover
                # the original orientation — the mesh bound_box is now
                # an axis-aligned bbox of the tilted geometry (bigger
                # than the visible shape). Warn the user that for OBB
                # to work they need to NOT apply rotation.
                if non_axis_aligned and not rotation_nonzero:
                    warnings.append(
                        f"obstacle '{o.name}' has Apply'd rotation — "
                        f"the original orientation is lost in mesh "
                        f"data, OBB collider falls back to AABB of "
                        f"tilted vertices (bigger than the visible "
                        f"shape). To get faithful tilt: undo the "
                        f"Apply (Object → Apply, with rotation NOT "
                        f"checked) and rotate via the R hotkey only.")
                # R as nested rows (matrix_world @ local axes).
                rot_rows = tuple(
                    (R3[i][0], R3[i][1], R3[i][2]) for i in range(3)
                )
                obstacles.append({
                    "type": "box",
                    "center": cw_sim,
                    "half_size": (hx_sim, hy_sim, hz_sim),
                    "rotation": rot_rows,
                })
            elif (rotation_nonzero or non_axis_aligned) and o.type == "MESH":
                if rotation_nonzero:
                    deg = (f"({rx*57.2958:+.1f}°, {ry*57.2958:+.1f}°, "
                           f"{rz*57.2958:+.1f}°)")
                    why = f"rotation {deg}"
                else:
                    why = ("mesh has non-axis-aligned faces — "
                           "rotation was Apply'd into mesh data")
                warnings.append(
                    f"obstacle '{o.name}' BBOX type + {why} → "
                    f"auto-promoted to MESH so the solver sees the "
                    f"real shape (was: axis-aligned bbox bigger than "
                    f"the mesh + orientation lost). For axis-aligned "
                    f"primitives use BBOX directly; for anything "
                    f"tilted/non-cube, MESH is the right type.")
                # Inline the MESH branch instead of jumping with goto-
                # like control flow.
                mesh_path = os.path.join(
                    cache_dir,
                    f"obstacle_{_safe_obj_name(o.name)}.obj")
                _export_obj(o, mesh_path)
                # audit-2026-06-15r14 (§9.6 mirror of the explicit MESH branch's
                # FU-019 warning below): this auto-promoted mesh obstacle uses the
                # SAME uniform scalar scale=avg_inv, so it has the SAME non-cubic
                # squash defect — warn here too, else the mis-scaling is loud via
                # the explicit-MESH path but silent via auto-promote (the r4 #2
                # patch mirrored scale/translate but forgot this diagnostic).
                if max(inv_size) / min(inv_size) > 1.05:
                    warnings.append(
                        f"obstacle '{o.name}' auto-promoted to MESH on a "
                        f"non-cubic domain — the mesh is scaled uniformly "
                        f"(single-scalar limitation), so the collider is "
                        f"squashed on the short domain axis and won't match the "
                        f"visible mesh. Make the domain cubic for an exact fit "
                        f"(per-axis mesh scaling is a known follow-up).")
                obstacles.append({
                    "type": "mesh",
                    # round-55: key name must match the original MESH
                    # branch + what config_builder reads (KeyError otherwise)
                    "path": mesh_path,
                    # audit-2026-06-14r4 #2 (§9.6 mirror of the explicit MESH
                    # branch below): the OBJ verts are world metres, so the
                    # collider MUST carry the world→[0,1]³ map. Without scale/
                    # translate, config_builder defaults to scale=1/translate=0
                    # and the mesh lands at raw metre coords — entirely outside
                    # the [0,1]³ domain — so the auto-promoted obstacle silently
                    # vanished from the sim. Keep identical to lines 445-447.
                    "scale": avg_inv,
                    "translate": (-origin.x * avg_inv, -origin.y * avg_inv,
                                  -origin.z * avg_inv),
                    "center": to_sim(centre_w),
                    "half_size": (hx, hy, hz),  # bbox-equiv for legacy
                })
            else:
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
            # Round-61: a single sim-space radius from the AVERAGE inverse
            # domain size only matches the visible sphere on a ~cubic
            # domain; on a non-cubic domain the world shape is an ellipsoid
            # the collider won't track. Mirrors the OBB anisotropy warning.
            if max(inv_size) / min(inv_size) > 1.05:
                warnings.append(
                    f"obstacle '{o.name}' is a SPHERE on a non-cubic domain "
                    f"— its radius is averaged across axes, so the collider "
                    f"won't match the visible sphere. Use a MESH obstacle, "
                    f"or make the domain cubic, for an exact fit.")
        elif oprops.obstacle_type == "CYLINDER_Y":
            r = max(hx_w, hz_w) * avg_inv
            obstacles.append({"type": "cylinder_y", "center": to_sim(centre_w),
                              "radius": r, "half_height": hy})
            # Round-61: same anisotropy caveat as SPHERE (radius averaged
            # across the X/Z axes).
            if max(inv_size) / min(inv_size) > 1.05:
                warnings.append(
                    f"obstacle '{o.name}' is a CYLINDER on a non-cubic "
                    f"domain — its radius is averaged across X/Z, so the "
                    f"collider won't match the visible cylinder. Use a MESH "
                    f"obstacle, or make the domain cubic, for an exact fit.")
        elif oprops.obstacle_type == "MESH":
            # export object's mesh as OBJ next to scene.toml.
            # cache_dir hoisted in round-53 — no need to re-resolve.
            # Round-38: sanitise the object name for filesystem use.
            # Blender allows `/ \ : * ? < > | "` in obj.name; Windows
            # `open()` raises OSError on any of them.
            mesh_path = os.path.join(
                cache_dir, f"obstacle_{_safe_obj_name(o.name)}.obj")
            _export_obj(o, mesh_path)
            # OBJ verts are in world metres — scale to the unit cube so the
            # solver's [0,1]³ collision sampling lands on the mesh.
            # FU-019 (2026-06-01 deep-dive): `scale` is a single scalar in the
            # ObstacleMeshCfg schema, so this can only apply a UNIFORM scale.
            # On a non-cubic domain the correct map is per-axis
            # (vert*inv_size - origin*inv_size); a uniform avg_inv squashes the
            # mesh wrong on the short axis. MESH is advertised as the "exact
            # fit" workaround for SPHERE/CYL, so silently mis-scaling it breaks
            # the one escape hatch we recommend. Proper per-axis mesh scaling
            # needs a schema + solver change (FU-019 follow-up); until then,
            # warn so the "exact fit" claim isn't a silent lie.
            if max(inv_size) / min(inv_size) > 1.05:
                warnings.append(
                    f"obstacle '{o.name}' is a MESH on a non-cubic domain — "
                    f"the mesh is scaled uniformly (single-scalar limitation), "
                    f"so the collider is squashed on the short domain axis and "
                    f"won't match the visible mesh. Make the domain cubic for "
                    f"an exact fit (per-axis mesh scaling is a known follow-up).")
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
                # Round-38: clamp the keyframe window to what the bake
                # will actually advance through. Pre-round-38 we
                # iterated `range(fs, fe+1)` blindly — at the default
                # `frame_end=10000` (properties.py) any inflow whose
                # PARENT chain has an fcurve (camera ancestor, common
                # constraint helper) triggered 10001 calls to
                # `_matrix_world_at_frame` (each walks parent chain ×
                # fcurves × evaluate) → 10–60s "collect" hang + ~MB
                # scene.toml bloat. Bake only ever advances through
                # `dprops.frames` frames; the rest are dead-letter
                # rows the solver never reaches.
                fe_eff = min(fe, fs + int(dprops.frames))
                # Pre-public-test audit (2026-06-01): when frame_end (default
                # 10000) overshoots the bake range, the keyframe list below is
                # clipped to fe_eff but entry["frame_end"] still carried the
                # original fe. The solver then emitted past the last keyframe,
                # freezing the emitter at its final position — looked like a
                # solver bug. Warn, and write the clamped end so emission stops
                # exactly when the animated keyframes run out.
                if fe_eff < fe:
                    warnings.append(
                        f"inflow '{o.name}' frame_end ({fe}) is beyond the "
                        f"bake range — emission is clamped to frame {fe_eff} "
                        f"({int(dprops.frames)} baked frames from start {fs}). "
                        f"Lower frame_end, or raise Domain Frames, to emit "
                        f"for longer.")
                entry["frame_end"] = fe_eff
                kfs = []
                for f in range(fs, fe_eff + 1):
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
        # FU-016: forward the real world extent (metres) so the CLI scales
        # gravity/velocity by metres, not the resolution aspect-ratio. dom_size
        # is DomainTransform.dom_size (world hi - lo).
        "domain": {"resolution": res, "dx": dx, "origin": list(origin),
                   "size_world": [float(dom_size[0]), float(dom_size[1]),
                                  float(dom_size[2])]},
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
            # Round-30: pre-round-30 this pre-scaled by `inv_size[1]` (Y)
            # — DOUBLE wrong: (1) the CLI then rescales by `inv_dom_z`
            # (Z) at commands.py:238, so the resulting initial_velocity_z
            # was multiplied by `inv_size[1] * inv_size[2]` — off by
            # 4× on a 2:1 Y:Z aspect domain; (2) wrong axis entirely
            # (Y not Z). Other MPM velocity knobs (mpm_v_terminal,
            # mpm_vz_max_splash) ship UNSCALED from here and ride the
            # CLI's single Z-rescaling — consistent. The pre-round-30
            # comment claimed "matches inflow-velocity convention" but
            # inflow uses per-axis `normalize_velocity`, not a single-
            # axis pre-multiply. Now: ship UNSCALED, let the CLI do
            # the single correct Z scaling once.
            "mpm_initial_velocity": float(dprops.mpm_initial_velocity),
        },
        "output": _output_dict(dprops),
        # Phase 1 escape-hatch: raw TOML string deep-merged in config_builder.
        "toml_overrides": str(getattr(dprops, "toml_overrides", "") or ""),
    }, dom_size, origin, warnings


# [BLK A8.5]
@block("A8.5", "Bake operator (sync + modal subprocess, ESC abort, "
               "reentrance guard, watchdog)")
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
            # Last-chance auto-detect using project-adjacent roots (the .blend
            # dir + this Domain's cache_dir) — Blender's cwd is the Start-menu
            # launch dir so register-time detection often misses the project
            # .venv. If we find one, persist it so the button "just works".
            guess = _detect_interpreter(_context_roots())
            if guess:
                prefs.interpreter_path = guess
                interp = bpy.path.abspath(guess).strip()
                self.report({"INFO"}, f"auto-detected Python interpreter: {guess}")
                try:
                    bpy.ops.wm.save_userpref()
                except Exception:
                    # dodge: headless/CI Blender or a locked/read-only
                    # userpref file makes save_userpref raise — persisting
                    # the detected path is best-effort and must not abort an
                    # otherwise-ready bake (the in-session prefs value is
                    # already set above). Note the side effect when it DOES
                    # succeed: save_userpref writes ALL pending preference
                    # changes, not just interpreter_path (audit-20260610).
                    pass
        if not interp or not Path(interp).exists():
            self.report({"ERROR"},
                        "No Python with gpufluid found. Set the interpreter in "
                        "Addon Preferences (or click Detect), or set "
                        "$GPUFLUID_PYTHON. It must be a venv where "
                        "`pip install -e .` was run on the gpufluid repo.")
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
        except (RuntimeError, ValueError) as e:
            # ValueError: DomainTransform.from_world_aabb rejects a degenerate
            # world AABB (Domain Empty scaled to 0 on an axis, or mirrored so
            # world_lo > world_hi). Pre-audit only RuntimeError was caught, so
            # the ValueError escaped collect_scene as a raw traceback popup
            # instead of the actionable message it already carries.
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
        # audit-2026-06-15r7 #2: use the round-60 _robust_rmtree (retry + chmod
        # + backoff for the Windows WinError 5/32/145 async-delete / AV-handle
        # race), NOT a bare shutil.rmtree. OT_clear_cache was hardened against
        # exactly this race; the bake's own stale-strip was the un-hardened twin
        # (§9.6 mirror drift). A swallowed WinError 145 here leaves the prior
        # run's tail frames on disk, which then inflate the glob-derived
        # frame_count and defeat the preloader's anti-stale cap — the precise
        # stale-frame bug this strip exists to prevent.
        # Lazy import: standalone test loads stub `helpers` with only
        # subprocess_drain, so keep _robust_rmtree off the module-load path.
        from .helpers import _robust_rmtree
        out_root = Path(str(scene_dict["output"]["cache_dir"]))
        # whitewater + whitewater_kinds added per pre-public-test audit
        # (2026-06-01): they were omitted, so a shorter re-bake left the
        # prior run's tail whitewater frames on disk and the loader served
        # them silently (the surface dirs were cleaned, whitewater was not).
        for sub in ("mesh", "particles_raw", "colors",
                    "whitewater", "whitewater_kinds"):
            d = out_root / sub
            if d.is_dir():
                try:
                    _robust_rmtree(str(d))
                except OSError as e:
                    # Surfaced loudly (not a bare warning): proceeding past a
                    # failed strip risks a mixed-length cache whose tail frames
                    # leak into the new bake.
                    logger.warning(
                        "addon.bake.stale_subdir_clean_failed",
                        extra={"dir": str(d), "err": str(e),
                               "hint": "re-bake may show stale tail frames; "
                                       "close any attached cache / file lock"})

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
                # Round-61: gate auto-attach + the 'complete' message on
                # the bake having actually produced frames. A 0-frame
                # bake now reports ERROR (via _post_bake_sanity) instead
                # of a misleading green 'complete' over an empty viewport.
                actual = self._post_bake_sanity()
                if actual > 0:
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
        """Called by runner.tick_modal on each parsed `frame N/M` line.
        Round-62: show a percentage + 'Esc to cancel' so the status bar
        reads as a live indicator, not a frozen label."""
        pct = int(100 * current / total) if total else 0
        bpy.context.workspace.status_text_set(
            f"gpufluid: baking frame {current}/{total} ({pct}%) — Esc to cancel")

    def _friendly_error_for_rc(self, rc: int):
        """Round-20: translate CLI rc into actionable error message.
        Reads ``<cache_dir>/cache.json`` for the rc=2 divergence case —
        surfaces the truncated frame + a solver-specific recovery hint.
        Returns ``None`` when no specific guidance applies (runner falls
        back to generic 'rc=N' wording).

        audit-2026-06-15r7 #1: handle BOTH truncation reasons. Round-32
        added the FLIP divergence rc=2 path to the CLI + manifest schema +
        cache_loader, but this twin still recognised only 'mpm_divergence'
        (§9.6 mirror drift), so a diverged FLIP bake lost its recovery
        message — the user was never told N valid frames were written and
        recoverable via Attach Cache, and the MPM-only 'lower Bulk Modulus'
        text doesn't even apply to FLIP."""
        if rc != 2:
            return None
        try:
            import json as _json
            manifest = _json.loads(
                (Path(self._cache_dir_str) / "cache.json").read_text())
            reason = manifest.get("truncation_reason")
            frame = manifest.get("truncated_at_frame", "?")
            total = manifest.get("frame_count", "?")
            if reason == "mpm_divergence":
                recovery = ("lower Bulk Modulus (default 1500), lower dt, "
                            "or shrink resolution")
                solver = "MPM solver"
            elif reason == "flip_divergence":
                recovery = ("lower dt / CFL factor, soften inflow velocities, "
                            "or raise pressure iterations")
                solver = "FLIP solver"
            else:
                return None
            return (
                f"gpufluid bake: {solver} diverged at frame {frame}; "
                f"{total} valid frames written to {self._cache_dir_str}. "
                f"Recovery: {recovery}, then re-bake. Use 'Attach Cache' "
                f"to load the partial result.")
        except Exception:
            return None

    def _sync_post_bake_then_attach(self) -> bool:
        """Sync on_complete: run frame-count sanity, then auto-attach —
        but only when the bake actually produced frames. Errors here
        downgrade so they don't crash the operator.

        audit-20260610: returns False on a 0-frame bake so the runner
        suppresses its unconditional "(sync) finished" INFO — that line
        used to print AFTER the 0-frame ERROR, leaving the Info log
        ending on a success message over an empty result."""
        actual = self._post_bake_sanity()
        if actual > 0:
            self._auto_attach_post_bake()
            return True
        return False

    def _post_bake_sanity(self) -> int:
        """Round-61: count baked mesh frames vs requested and report any
        shortfall. SHARED by the sync + modal paths so the two can't
        drift (§9.6).

        History: round-10 added a truncation check to the SYNC path
        only; the modal (UI-default) path never got one. Worse, the sync
        check bounded the shortfall below by a ``0 < actual`` clause,
        which EXCLUDED the actual==0 case, so a bake that wrote ZERO was
        reported as a clean 'complete' and the user saw an empty viewport
        with a green success message (live-found 2026-05-28). The
        decision now lives in the bpy-free ``cache_sanity`` module
        (unit-tested) and escalates the zero case to ERROR.

        Returns the actual mesh-frame count so the caller can skip the
        auto-attach + 'complete' message when nothing was produced.

        Round-12 reviewer #6: tomllib only, no tomli fallback (Blender
        4.5+ ships Python 3.11+ which guarantees tomllib).
        """
        from ..cache_sanity import bake_frame_sanity, count_mesh_frames
        actual = count_mesh_frames(self._cache_dir_str)
        expected = 0
        try:
            import tomllib
            emitted = tomllib.loads(self._toml_str_snapshot)
            expected = int(emitted.get("simulation", {}).get("frames", 0))
        except Exception:
            # TOML unparseable → expected stays 0, so only the 0-frame
            # disaster is flagged (never a false "truncated" verdict).
            expected = 0
        level, msg = bake_frame_sanity(actual, expected)
        if level is not None:
            self.report({level}, msg)
        return actual

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
                    dom_size=tuple(float(c) for c in dom_size),  # audit #3
                )
            except Exception as e:
                self.report({"WARNING"},
                            f"bake ok, but ww auto-attach failed: {e}")
