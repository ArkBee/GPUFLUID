"""[BLK A8.6] Mesh-cache import via per-frame PLY swap handler.

Frame-by-frame strategy (matches Stop-Motion-OBJ / FLIP Fluids approach):

    On every frame change, look up the current frame's PLY file for any
    object that has a `gpufluid_cache_dir` custom property, parse it, and
    rebuild the object's mesh in place (verts + faces). Object data is
    overwritten — no link-bumping.

Why not MeshSequenceCache modifier?
    MSC supports Alembic and USD, not raw PLY. Future work (I6.4 / I6.5)
    will add Alembic/USD export which can use the native modifier.
"""
import os
import struct
from collections import OrderedDict
import bpy
import numpy as np

from . import logger as _addon_logger


# ---------------------------------------------------------------------------
# Minimal PLY reader (matches gpufluid.io.ply binary format).
# ---------------------------------------------------------------------------

def _read_ply_minimal(path):
    """Vectorised binary-PLY reader matching gpufluid.io.ply format.

    Returns ``(verts, faces, colors_or_None)``. ``colors`` is a ``(N, 3)``
    uint8 array when the PLY has ``red/green/blue uchar`` properties
    (B18.4) — otherwise ``None``. Pre-B18 cache files (xyz-only verts)
    keep working unchanged.
    """
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), None)
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            if not line:
                return empty
            header += line
            if line.strip() == b"end_header":
                break
        text = header.decode("ascii", errors="replace")
        n_v = 0; n_f = 0
        has_color = False
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("element vertex"):
                n_v = int(s.split()[2])
            elif s.startswith("element face"):
                n_f = int(s.split()[2])
            elif s == "property uchar red":
                has_color = True
        if has_color:
            vbuf = f.read(n_v * 15)
            arr = np.frombuffer(vbuf, dtype=np.uint8).reshape(n_v, 15)
            verts = (np.ascontiguousarray(arr[:, :12])
                       .view(np.float32).reshape(n_v, 3).copy())
            colors = arr[:, 12:].copy()
        else:
            verts = np.frombuffer(f.read(n_v * 12), dtype=np.float32).reshape(n_v, 3).copy()
            colors = None
        if n_f == 0:
            return verts, np.zeros((0, 3), np.int32), colors
        face_bytes = f.read(n_f * 13)
        rows = np.frombuffer(face_bytes, dtype=np.uint8).reshape(n_f, 13)
        faces = np.ascontiguousarray(rows[:, 1:]).view(np.int32).reshape(n_f, 3).copy()
        return verts, faces, colors


def _rebuild_mesh(obj, verts, faces, origin):
    """Replace obj.data via Blender's battle-tested `from_pydata` API.

    Two previous attempts crashed:
      v1: clear_geometry() + vertices.add() + foreach_set(...) — viewport's
          parallel `normals_calc_faces` raced with mid-modify state and
          dereferenced freed polygon-array pointers.
      v2: build new datablock + obj.data swap + bpy.data.meshes.remove(old)
          inside frame_change_pre — depsgraph crashed on COW copy of the
          new mesh's AttributeStorage (yanking bpy.data entries inside a
          handler is unsupported).

    v3 (this): plain `mesh.from_pydata()` — the official supported path
    Blender ships for runtime mesh creation. ~100-300ms per 30k-vert frame
    (the .tolist() conversion), but rock-solid: this is what Stop Motion
    OBJ, FLIP Fluids, and every other cache addon uses.
    """
    n_v = len(verts)
    n_f = len(faces)
    me = obj.data
    me.clear_geometry()
    if n_v == 0 or n_f == 0:
        return
    # Strip out-of-range face indices (MC degenerate triangles)
    if faces.size:
        mask = ((faces[:, 0] < n_v) & (faces[:, 1] < n_v) & (faces[:, 2] < n_v)
                & (faces[:, 0] >= 0) & (faces[:, 1] >= 0) & (faces[:, 2] >= 0))
        if not mask.all():
            faces = np.ascontiguousarray(faces[mask])
            n_f = faces.shape[0]
            if n_f == 0:
                return
    v_world = (verts + np.asarray(origin, dtype=np.float32)).astype(np.float32)
    # from_pydata expects list-of-tuples; .tolist() is the standard idiom.
    me.from_pydata(v_world.tolist(), [], faces.astype(np.int32).tolist())
    me.update()
    # Flag every polygon for smooth shading so the marching-cubes facets
    # disappear under interpolated normals. foreach_set on a uint8 array
    # of 1s sets `use_smooth` for all polys in one C call.
    if len(me.polygons):
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), dtype=np.uint8))
        me.update()


# ---------------------------------------------------------------------------
# Whitewater point-cloud loader
# ---------------------------------------------------------------------------

def _rebuild_ww_points_inplace(me, positions, kinds, origin, visible_kinds):
    """Identical to old in-place rebuild, but split out so the double-buffer
    swap path can reuse it on a fresh datablock."""
    me.clear_geometry()
    if positions is None or positions.shape[0] == 0:
        return
    if kinds is not None:
        keep = np.ones(positions.shape[0], dtype=bool)
        if not visible_kinds[0]:
            keep &= kinds != 0
        if not visible_kinds[1]:
            keep &= kinds != 1
        if not visible_kinds[2]:
            keep &= kinds != 2
        positions = positions[keep]
        kinds = kinds[keep]
    if positions.shape[0] == 0:
        return
    pts = (positions + np.asarray(origin, dtype=np.float32)).astype(np.float32)
    me.vertices.add(pts.shape[0])
    me.vertices.foreach_set("co", pts.ravel())
    if kinds is not None and kinds.size > 0:
        attr = me.attributes.get("gpufluid_kind")
        if attr is None or attr.domain != "POINT" or attr.data_type != "INT":
            if attr is not None:
                me.attributes.remove(attr)
            attr = me.attributes.new(name="gpufluid_kind", type="INT", domain="POINT")
        attr.data.foreach_set("value", kinds.astype(np.int32))
    me.update()


def _rebuild_ww_points(obj, positions, kinds, origin, visible_kinds):
    """Replace `obj.data` with a vertex-only mesh of whitewater positions.

    `kinds` is an int32 array (foam=0, spray=1, bubble=2) of length N, or None.
    `visible_kinds` is a tuple of three booleans (show_foam, show_spray,
    show_bubble) consumed at filter time. Vertices for hidden classes are
    dropped completely so downstream Geometry Nodes / particle instancing
    doesn't see them.

    A per-vertex integer attribute ``gpufluid_kind`` is written when
    `kinds` is provided — Geometry Nodes / shaders can branch on it to
    instance foam vs spray vs bubble assets.
    """
    # WW path uses in-place helper directly — point-only data, no faces,
    # avoids the depsgraph race seen in the surface mesh path.
    _rebuild_ww_points_inplace(obj.data, positions, kinds, origin, visible_kinds)


# ---------------------------------------------------------------------------
# Per-frame handler
# ---------------------------------------------------------------------------

def _domain_whitewater_visibility(scene):
    """Return (show_foam, show_spray, show_bubble) from the Domain object,
    defaulting to all-visible if there is no Domain in the scene."""
    for o in scene.objects:
        try:
            dom = o.gpufluid_domain
        except AttributeError:
            continue
        if dom.is_domain:
            try:
                ww = dom.whitewater_group
                return (bool(ww.show_foam), bool(ww.show_spray), bool(ww.show_bubble))
            except AttributeError:
                break
    return (True, True, True)


# Per-object pre-loaded mesh datablocks: {obj_name: {frame_idx: mesh_datablock}}.
# Populated by `_preload_sequence`. Frame_change_handler then does a pointer-
# swap `obj.data = _PRELOAD[obj.name][idx]` (~microseconds vs ~5s rebuild).
#
# Phase 2: OrderedDict with LRU eviction. Each entry can hold thousands of
# mesh datablocks; an unbounded dict on a long Blender session leaks RAM
# until the process is killed. Cap is read from preferences.preload_cap;
# `_lru_touch` and `_lru_evict_one` move/drop entries.
_PRELOAD: "OrderedDict[str, dict[int, bpy.types.Mesh]]" = OrderedDict()


def _preload_cap() -> int:
    """Read the LRU cap from addon preferences. Falls back to 8 when prefs
    can't be reached (e.g. during pytest or first-tick before register)."""
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        return int(getattr(prefs, "preload_cap", 8))
    except (AttributeError, KeyError, RuntimeError):
        return 8


def _preload_max_frames() -> int:
    """Read the per-call max_frames bound from prefs. Falls back to 10000."""
    try:
        prefs = bpy.context.preferences.addons[__package__].preferences
        return int(getattr(prefs, "preload_max_frames", 10000))
    except (AttributeError, KeyError, RuntimeError):
        return 10000


def _free_table(table: dict) -> None:
    """Release every mesh datablock in a preload table, but only when no
    live object still uses it (mesh.users == 0). Datablocks the user has
    pinned to a fluid object stay; we just drop our handle."""
    for mesh in list(table.values()):
        if mesh is None:
            continue
        try:
            if getattr(mesh, "users", 0) == 0:
                bpy.data.meshes.remove(mesh, do_unlink=True)
        except (ReferenceError, RuntimeError) as exc:
            _addon_logger.warning(
                "cache_loader: could not free preload mesh '%s' (%s)",
                getattr(mesh, "name", "?"), exc,
            )


def _lru_touch(obj_name: str) -> None:
    """Mark this entry as most-recently-used (move to end)."""
    if obj_name in _PRELOAD:
        _PRELOAD.move_to_end(obj_name)


def _lru_install(obj_name: str, table: dict) -> None:
    """Insert (or replace) a preload table for ``obj_name`` and evict the
    oldest entries until len(_PRELOAD) <= preload_cap."""
    if obj_name in _PRELOAD:
        _free_table(_PRELOAD[obj_name])
        del _PRELOAD[obj_name]
    _PRELOAD[obj_name] = table
    cap = _preload_cap()
    while len(_PRELOAD) > cap:
        old_name, old_table = _PRELOAD.popitem(last=False)
        _free_table(old_table)
        _addon_logger.info(
            "cache_loader: LRU evicted preload entry '%s' (cap=%d)",
            old_name, cap,
        )


def _preload_sequence(obj, cache_dir, pattern, origin, max_frames=None,
                       dom_size=(1.0, 1.0, 1.0)):
    """Build all mesh datablocks for the sequence up-front.

    Why preload: turns 5-7s per render frame (Python from_pydata + subsurf
    re-eval) into a sub-millisecond pointer swap. Plus we copy materials
    onto each preloaded mesh datablock so the render keeps its shader
    across the swap, and we also disable the SubSurf modifier — re-evaluating
    it on a fresh mesh every frame was the actual bottleneck (Blender
    invalidates the modifier cache when obj.data changes). Smooth shading
    alone is enough at 96³ MPM resolution.
    """
    import bpy as _bpy
    table = {}
    name_base = obj.name + "_seq_"
    # Snapshot current materials so every preloaded mesh has the lava shader.
    src_materials = [m for m in obj.data.materials]
    n_loaded = 0
    if max_frames is None:
        max_frames = _preload_max_frames()
    for idx in range(max_frames):
        path = os.path.join(cache_dir, pattern.format(idx))
        if not os.path.exists(path):
            if idx == 0:
                continue
            break
        try:
            verts, faces, colors = _read_ply_minimal(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[gpufluid cache] preload error at frame {idx}: {exc}")
            continue
        me = _bpy.data.meshes.new(name_base + f"{idx:04d}")
        # Carry the current material slot onto the new datablock.
        for m in src_materials:
            me.materials.append(m)
        n_v = len(verts); n_f = len(faces)
        if n_v == 0 or n_f == 0:
            table[idx] = me
            continue
        if faces.size:
            mask = ((faces[:, 0] < n_v) & (faces[:, 1] < n_v) & (faces[:, 2] < n_v)
                    & (faces[:, 0] >= 0) & (faces[:, 1] >= 0) & (faces[:, 2] >= 0))
            if not mask.all():
                faces = np.ascontiguousarray(faces[mask])
        # PLY verts are normalised [0,1]³; map to world by scaling with the
        # domain extent and translating by the domain origin. Identical
        # registration to the Alembic path (obj.scale=dom_size, obj.loc=origin)
        # but baked into the mesh data so the obj stays at the world origin.
        ds = np.asarray(dom_size, dtype=np.float32)
        v_world = (verts * ds + np.asarray(origin, dtype=np.float32)).astype(np.float32)
        me.from_pydata(v_world.tolist(), [], faces.astype(np.int32).tolist())
        me.update()
        if len(me.polygons):
            me.polygons.foreach_set("use_smooth",
                                     np.ones(len(me.polygons), dtype=np.uint8))
            me.update()
        # B18.6 — attach per-vertex RGB as a POINT-domain colour attribute
        # named "Col" (Blender's default vertex-colour layer name). The
        # bake's PLY has uint8 [0,255]; Blender's color_attributes expect
        # float [0,1] RGBA, so we convert + extend with alpha=1.
        if colors is not None and colors.shape[0] == n_v:
            try:
                ca = me.color_attributes.new(
                    name="Col", type="FLOAT_COLOR", domain="POINT",
                )
                rgba = np.empty(n_v * 4, dtype=np.float32)
                cf = colors.astype(np.float32) / 255.0
                rgba[0::4] = cf[:, 0]
                rgba[1::4] = cf[:, 1]
                rgba[2::4] = cf[:, 2]
                rgba[3::4] = 1.0
                ca.data.foreach_set("color", rgba)
            except Exception as exc:  # noqa: BLE001
                # Don't kill the bake-attach if a single frame's colour
                # attr fails — log once and move on.
                if idx == 0:
                    print(f"[gpufluid cache] vertex-colour attach failed: {exc}")
        table[idx] = me
        n_loaded += 1
    _lru_install(obj.name, table)

    # Disable SubSurf modifier(s): they re-evaluate on every obj.data swap
    # and were the dominant render cost. Smooth shading + 96³ MPM mesh is
    # plenty without subdivision.
    for mod in list(obj.modifiers):
        if mod.type == "SUBSURF":
            mod.show_render = False
            mod.show_viewport = False

    print(f"[gpufluid cache] preloaded {n_loaded} frames for '{obj.name}'")
    return n_loaded


def _frame_change_handler(scene, depsgraph=None):
    f = scene.frame_current
    visible_kinds = _domain_whitewater_visibility(scene)
    for obj in scene.objects:
        # Cache loading only applies to mesh objects. The Domain Empty also
        # carries a `gpufluid_cache_dir` custom prop (for the bake operator's
        # own bookkeeping) but is not a render target.
        if obj.type != "MESH":
            continue
        # Surface mesh path
        cache_dir = obj.get("gpufluid_cache_dir")
        if cache_dir:
            pattern = obj.get("gpufluid_cache_pattern", "mesh/frame_{:04d}.ply")
            offset = int(obj.get("gpufluid_cache_frame_offset", 0))
            origin = list(obj.get("gpufluid_cache_origin", [0.0, 0.0, 0.0]))
            idx = f - offset
            if idx >= 0:
                # FAST PATH — pre-loaded sequence: pointer swap only.
                # Mesh datablocks in _PRELOAD can be freed externally (e.g.
                # by orphans_purge after Alembic switchover) — guard the
                # swap with try/except and drop the stale table on first
                # ReferenceError so the handler doesn't spam its full log.
                table = _PRELOAD.get(obj.name)
                if table is not None and idx in table:
                    try:
                        obj.data = table[idx]
                        _lru_touch(obj.name)
                        continue
                    except (ReferenceError, RuntimeError):
                        _PRELOAD.pop(obj.name, None)
                        # fall through to slow path / no-op
                # SLOW PATH — legacy per-frame rebuild (preload not active yet)
                path = os.path.join(cache_dir, pattern.format(idx))
                if os.path.exists(path):
                    try:
                        verts, faces, _colors = _read_ply_minimal(path)
                        # Slow path doesn't reattach colour attribute (the
                        # preload path is the production code path for
                        # coloured caches); colours arrive on the next
                        # full attach_cache call. Keeps the handler-time
                        # cost down for the legacy fallback.
                        _rebuild_mesh(obj, verts, faces, origin)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[gpufluid cache] error at frame {f} for '{obj.name}': {exc}")
        # Whitewater point-cloud path
        ww_dir = obj.get("gpufluid_ww_cache_dir")
        if ww_dir:
            offset = int(obj.get("gpufluid_ww_cache_frame_offset", 0))
            origin = list(obj.get("gpufluid_ww_cache_origin", [0.0, 0.0, 0.0]))
            idx = f - offset
            if idx >= 0:
                pos_path = os.path.join(ww_dir, "whitewater", f"frame_{idx:04d}.npy")
                kind_path = os.path.join(ww_dir, "whitewater_kinds", f"frame_{idx:04d}.npy")
                if os.path.exists(pos_path):
                    try:
                        pos = np.load(pos_path).astype(np.float32)
                        kinds = (np.load(kind_path).astype(np.int32)
                                 if os.path.exists(kind_path) else None)
                        _rebuild_ww_points(obj, pos, kinds, origin, visible_kinds)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[gpufluid ww] error at frame {f} for '{obj.name}': {exc}")


def register_handler():
    """Register frame_change handler and strip any orphans from past reloads.

    When the addon is hot-reloaded (`del sys.modules[*]` then re-enable),
    the OLD module's `_frame_change_handler` function object stays in
    `bpy.app.handlers.frame_change_pre` referencing a dead module. Firing
    it later either no-ops (handler reads None _PRELOAD) or in worst-case
    triggers a hard crash inside Blender's depsgraph when accessing freed
    Python state. Strip every handler that *looks like ours* by name, then
    install the fresh one.
    """
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre
        if getattr(h, "__name__", "") != "_frame_change_handler"
    ]
    bpy.app.handlers.frame_change_pre.append(_frame_change_handler)


def unregister_handler():
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre
        if getattr(h, "__name__", "") != "_frame_change_handler"
    ]


# ---------------------------------------------------------------------------
# Attach / detach operators
# ---------------------------------------------------------------------------

class GPUFLUID_OT_attach_cache(bpy.types.Operator):
    bl_idname = "gpufluid.attach_cache"
    bl_label = "Attach gpufluid Cache"
    bl_description = "Bind a PLY mesh cache directory to an object (or create a new one)"
    bl_options = {"REGISTER", "UNDO"}

    cache_dir: bpy.props.StringProperty(name="Cache Dir", subtype="DIR_PATH")
    target_name: bpy.props.StringProperty(name="Target object")
    # World-space lower corner of the domain bbox. PLY meshes live in
    # normalised [0,1]³; this `origin` plus `dom_size` is how we map them
    # back to the user's world (verts*dom_size + origin baked into the
    # mesh data — see _preload_sequence).
    origin: bpy.props.FloatVectorProperty(
        name="Domain origin (world)", size=3, default=(0.0, 0.0, 0.0),
        subtype="TRANSLATION",
    )
    dom_size: bpy.props.FloatVectorProperty(
        name="Domain size (world)", size=3, default=(1.0, 1.0, 1.0),
        subtype="XYZ", min=1e-6,
    )
    frame_offset: bpy.props.IntProperty(name="Cache starts at scene frame", default=1)
    # The Alembic fast path is currently disabled (see commands.py); keep
    # this flag so external scripts can still force the PLY path if needed.
    force_ply: bpy.props.BoolProperty(name="Force PLY preload", default=True)

    def execute(self, context):
        cache_dir = bpy.path.abspath(self.cache_dir)
        if not os.path.isdir(cache_dir):
            self.report({"ERROR"}, f"cache dir not found: {cache_dir}")
            return {"CANCELLED"}

        target = context.scene.objects.get(self.target_name) if self.target_name else None
        if target is None:
            # No explicit target — reuse an existing gpufluid_cache in the
            # scene (so re-bakes / re-attaches don't spawn .001/.002/...).
            target = context.scene.objects.get("gpufluid_cache")
        if target is None:
            # First-time attach: create a new mesh object.
            mesh = bpy.data.meshes.new("gpufluid_cache_mesh")
            target = bpy.data.objects.new("gpufluid_cache", mesh)
            context.scene.collection.objects.link(target)
        target["gpufluid_cache_dir"] = cache_dir
        target["gpufluid_cache_pattern"] = "mesh/frame_{:04d}.ply"
        target["gpufluid_cache_frame_offset"] = self.frame_offset
        target["gpufluid_cache_origin"] = list(self.origin)
        target["gpufluid_cache_dom_size"] = list(self.dom_size)

        # FAST PATH (I6.2.ABC) — Alembic + MeshSequenceCache. CURRENTLY
        # DISABLED because Blender 5.1's MSC modifier on a multi-sub-object
        # .abc applies a phantom origin offset we have not yet reverse-
        # engineered (see chat log 2026-05-20 + commands.py for the trace).
        # The legacy PLY-preload path below produces identical visual output
        # at ~150 MB peak RAM and ~14 s one-shot attach cost, which is fine.
        # Re-enable once we have a single-object animated-vertex Alembic
        # writer (likely via pyalembic directly, not wm.alembic_export).
        abc_path = os.path.join(cache_dir, "cache.abc")
        if os.path.isfile(abc_path) and not self.force_ply:
            _PRELOAD.pop(target.name, None)
            for m in list(target.modifiers):
                if m.type in ("MESH_SEQUENCE_CACHE", "SUBSURF"):
                    target.modifiers.remove(m)
            target.location = tuple(self.origin)
            target.rotation_euler = (0.0, 0.0, 0.0)
            target.scale = tuple(self.dom_size)
            before = {cf.name for cf in bpy.data.cache_files}
            bpy.ops.cachefile.open(filepath=abc_path)
            cf = next((bpy.data.cache_files[n] for n in bpy.data.cache_files.keys()
                       if n not in before), None)
            if cf is None and len(bpy.data.cache_files):
                cf = bpy.data.cache_files[-1]
            mod = target.modifiers.new(name="MeshSequenceCache",
                                        type="MESH_SEQUENCE_CACHE")
            mod.cache_file = cf
            paths = [p.path for p in cf.object_paths] if cf else []
            if paths:
                mod.object_path = paths[0]
            lava = bpy.data.materials.get("LavaProd") or bpy.data.materials.get("LavaMat")
            if lava is not None:
                if not target.data.materials:
                    target.data.materials.append(lava)
                if target.material_slots:
                    target.material_slots[0].link = "OBJECT"
                    target.material_slots[0].material = lava
            self.report({"INFO"},
                        f"cache attached via Alembic ({len(paths)} objects)")
            return {"FINISHED"}

        # LEGACY PATH — preload PLY sequence frame-by-frame, baking the
        # domain transform (origin + dom_size) directly into the vertices so
        # the object can stay at world origin with identity scale. This
        # mirrors what the Alembic path achieves via obj.location / obj.scale.
        target.location = (0.0, 0.0, 0.0)
        target.rotation_euler = (0.0, 0.0, 0.0)
        target.scale = (1.0, 1.0, 1.0)
        import time as _time
        t0 = _time.time()
        n = _preload_sequence(
            target, cache_dir, "mesh/frame_{:04d}.ply",
            list(self.origin),
            dom_size=tuple(self.dom_size),
        )
        _frame_change_handler(context.scene)
        self.report({"INFO"},
                    f"cache attached (PLY preload) — {n} frames in {_time.time()-t0:.1f}s")
        return {"FINISHED"}


class GPUFLUID_OT_attach_ww_cache(bpy.types.Operator):
    bl_idname = "gpufluid.attach_ww_cache"
    bl_label = "Attach gpufluid Whitewater Cache"
    bl_description = ("Bind a whitewater point cache (cache_dir/whitewater/*.npy + "
                      "whitewater_kinds/*.npy) to an object. Verts get a per-vertex "
                      "INT attribute 'gpufluid_kind' (0=foam, 1=spray, 2=bubble) for "
                      "Geometry Nodes branching")
    bl_options = {"REGISTER", "UNDO"}

    cache_dir: bpy.props.StringProperty(name="Cache Dir", subtype="DIR_PATH")
    target_name: bpy.props.StringProperty(name="Target object")
    origin_x: bpy.props.FloatProperty(name="Origin X", default=0.0)
    origin_y: bpy.props.FloatProperty(name="Origin Y", default=0.0)
    origin_z: bpy.props.FloatProperty(name="Origin Z", default=0.0)
    frame_offset: bpy.props.IntProperty(name="Cache starts at scene frame", default=1)

    def execute(self, context):
        cache_dir = bpy.path.abspath(self.cache_dir)
        if not os.path.isdir(os.path.join(cache_dir, "whitewater")):
            self.report({"ERROR"},
                        f"no 'whitewater' subdir in {cache_dir} — bake whitewater first")
            return {"CANCELLED"}
        target = context.scene.objects.get(self.target_name) if self.target_name else None
        if target is None:
            mesh = bpy.data.meshes.new("gpufluid_ww_mesh")
            target = bpy.data.objects.new("gpufluid_whitewater", mesh)
            context.scene.collection.objects.link(target)
        target["gpufluid_ww_cache_dir"] = cache_dir
        target["gpufluid_ww_cache_frame_offset"] = self.frame_offset
        target["gpufluid_ww_cache_origin"] = [self.origin_x, self.origin_y, self.origin_z]
        _frame_change_handler(context.scene)
        self.report({"INFO"}, f"whitewater cache attached to '{target.name}'")
        return {"FINISHED"}


class GPUFLUID_OT_detach_cache(bpy.types.Operator):
    bl_idname = "gpufluid.detach_cache"
    bl_label = "Detach gpufluid Cache"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and (
            o.get("gpufluid_cache_dir") is not None
            or o.get("gpufluid_ww_cache_dir") is not None
        )

    def execute(self, context):
        obj = context.active_object
        for k in ("gpufluid_cache_dir", "gpufluid_cache_pattern",
                  "gpufluid_cache_frame_offset", "gpufluid_cache_origin",
                  "gpufluid_ww_cache_dir", "gpufluid_ww_cache_frame_offset",
                  "gpufluid_ww_cache_origin"):
            if k in obj.keys():
                del obj[k]
        return {"FINISHED"}
