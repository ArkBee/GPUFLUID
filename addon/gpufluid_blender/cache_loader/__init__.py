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
from collections import OrderedDict
import bpy
import numpy as np

from .. import logger as _addon_logger
from ._ply import _read_ply_minimal  # noqa: F401 — re-exported for callers


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


# Blender's addon prefs are keyed by the addon-root package, which is the
# parent of this module — NOT __package__ verbatim and NOT the top-level
# segment. Examples seen live:
#   * Legacy install: __package__ == "gpufluid_blender.cache_loader"
#     → addon-root = "gpufluid_blender"
#   * 4.2+ extension: __package__ == "bl_ext.user_default.gpufluid_blender.cache_loader"
#     → addon-root = "bl_ext.user_default.gpufluid_blender"
# rsplit one level off — works for both layouts.
_ADDON_PKG = (__package__ or "").rsplit(".", 1)[0]


def _preload_cap() -> int:
    """Read the LRU cap from addon preferences. Falls back to 8 when prefs
    can't be reached (e.g. during pytest or first-tick before register)."""
    try:
        prefs = bpy.context.preferences.addons[_ADDON_PKG].preferences
        return int(getattr(prefs, "preload_cap", 8))
    except (AttributeError, KeyError, RuntimeError):
        return 8


def _preload_max_frames() -> int:
    """Read the per-call max_frames bound from prefs. Falls back to 10000."""
    try:
        prefs = bpy.context.preferences.addons[_ADDON_PKG].preferences
        return int(getattr(prefs, "preload_max_frames", 10000))
    except (AttributeError, KeyError, RuntimeError):
        return 10000


def _free_table(table: dict) -> None:
    """Release every mesh datablock in a preload table, but only when no
    live object still uses it (mesh.users == 0). Datablocks the user has
    pinned to a fluid object stay; we just drop our handle.

    External code (user manual orphan-purge, addon disable/enable cycle,
    file load) can invalidate the Mesh StructRNA out from under us — in
    that case ``getattr(mesh, "users", ...)`` raises ReferenceError, not
    AttributeError, so the default doesn't kick in. Catch defensively
    and resolve the mesh name BEFORE entering the except handler so the
    handler itself can't ReferenceError on ``mesh.name``.
    """
    for mesh in list(table.values()):
        if mesh is None:
            continue
        # Resolve name eagerly; if the struct is already dead the table
        # entry is just a stale pointer we should drop silently.
        try:
            mesh_name = mesh.name
        except ReferenceError:
            continue
        try:
            if getattr(mesh, "users", 0) == 0:
                bpy.data.meshes.remove(mesh, do_unlink=True)
        except (ReferenceError, RuntimeError) as exc:
            _addon_logger.warning(
                "cache_loader: could not free preload mesh '%s' (%s)",
                mesh_name, exc,
            )


def _lru_touch(obj_name: str) -> None:
    """Mark this entry as most-recently-used (move to end)."""
    if obj_name in _PRELOAD:
        _PRELOAD.move_to_end(obj_name)


def _lru_install(obj_name: str, table: dict) -> None:
    """Insert (or replace) a preload table for ``obj_name`` and evict the
    oldest entries until len(_PRELOAD) <= preload_cap.

    Live-found 2026-05-25: when REPLACING an entry, the previously-installed
    table's current-frame mesh has ``users==1`` (the live object holds it),
    so ``_free_table`` skips it and the mesh leaks into ``bpy.data.meshes``
    forever — visible after a few attach cycles as a pile of stale
    ``*_seq_NNNN`` datablocks ``OT_clear_cache`` can no longer reach.
    Redirect the live object onto a mesh from the NEW table first so the
    old current-frame mesh's users drop to 0 and ``_free_table`` can
    actually remove it.
    """
    if obj_name in _PRELOAD:
        if table:
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                try:
                    obj.data = next(iter(table.values()))
                except (ReferenceError, RuntimeError, TypeError) as exc:
                    _addon_logger.warning(
                        "cache_loader: pre-swap obj.data failed for %s (%s)",
                        obj_name, exc)
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
    # Respect cache.json:frame_count — without this, a re-bake that shrank
    # the frame range leaks stale .ply files from the previous bake into
    # the preview (live-found 2026-05-25: 20-frame MPM re-bake over a
    # 30-frame FLIP bake → 29 frames preloaded with mixed solver output).
    cache_json_path = os.path.join(cache_dir, "cache.json")
    if os.path.isfile(cache_json_path):
        try:
            import json
            with open(cache_json_path, "r", encoding="utf-8") as fh:
                fc = int(json.load(fh).get("frame_count", max_frames))
            if fc > 0:
                max_frames = min(max_frames, fc)
        except Exception as exc:  # noqa: BLE001
            _addon_logger.warning(
                "cache: cache.json read failed (%s) — falling back to dir scan",
                exc)
    for idx in range(max_frames):
        path = os.path.join(cache_dir, pattern.format(idx))
        if not os.path.exists(path):
            if idx == 0:
                continue
            break
        try:
            verts, faces, colors = _read_ply_minimal(path)
        except Exception as exc:  # noqa: BLE001
            _addon_logger.warning(
                "cache: preload error at frame %d: %s", idx, exc)
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
                    _addon_logger.warning(
                        "cache: vertex-colour attach failed: %s", exc)
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

    _addon_logger.info(
        "cache: preloaded %d frames for '%s'", n_loaded, obj.name)
    return n_loaded


@bpy.app.handlers.persistent
def _frame_change_handler(scene, depsgraph=None):
    # @persistent so Blender doesn't clear us on file-load. Without it,
    # opening a .blend that has a previously-baked gpufluid_cache obj
    # leaves the mesh frozen at whichever frame the file was saved on
    # — scrubbing the timeline does nothing because no handler is
    # registered to swap obj.data anymore. Live-found 2026-05-25
    # (round-4 test #23 save/load).
    f = scene.frame_current
    visible_kinds = _domain_whitewater_visibility(scene)
    # Round-8 reviewer flag #5: round-7 added stale-key prune here, but
    # it builds `{o.name for o in scene.objects}` on EVERY frame_change.
    # For a 5000-object production scene + 24fps playback that's 120k
    # set-insertions/sec just to defend a corner case. Moved to
    # `_prune_stale()` which is called from OT_detach, OT_clear, and
    # _on_load_post — the only paths that legitimately invalidate
    # _PRELOAD keys behind our back. Hot frame loop now skips it.
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
                        _addon_logger.warning(
                            "cache: error at frame %d for '%s': %s",
                            f, obj.name, exc)
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
                        _addon_logger.warning(
                            "ww: error at frame %d for '%s': %s",
                            f, obj.name, exc)


def _prune_stale(scene=None):
    """Drop _PRELOAD entries whose obj.name no longer exists in any scene.
    Call from detach/clear ops and load_post — paths that legitimately
    invalidate keys. NOT from frame_change (too hot)."""
    if not _PRELOAD:
        return
    live_names = set()
    if scene is not None:
        live_names = {o.name for o in scene.objects}
    else:
        for s in bpy.data.scenes:
            live_names.update(o.name for o in s.objects)
    stale = [k for k in _PRELOAD.keys() if k not in live_names]
    for k in stale:
        _free_table(_PRELOAD[k])
        del _PRELOAD[k]


@bpy.app.handlers.persistent
def _on_load_post(_dummy):
    """File-load recovery: wipe the stale _PRELOAD dict (its mesh pointers
    refer to freed StructRNAs after .blend reload) and re-attach every
    cache-bearing object using its saved custom props. Without this, a
    saved-and-reopened scene shows the cache at exactly one frame
    (the frame the file was saved on) and scrubbing does nothing —
    live-found 2026-05-25 (round-4 test #23).
    """
    _PRELOAD.clear()
    # _prune_stale not needed after clear, but pattern: any path that
    # invalidates obj→table mapping should ensure the dict is sane.
    for obj in bpy.data.objects:
        cache_dir = obj.get("gpufluid_cache_dir")
        if not cache_dir or not os.path.isdir(str(cache_dir)):
            continue
        pattern = obj.get("gpufluid_cache_pattern", "mesh/frame_{:04d}.ply")
        origin_key = "gpufluid_cache_origin"
        dom_size_key = "gpufluid_cache_dom_size"
        # Round-5 reviewer caught: silently defaulting dom_size to
        # (1, 1, 1) when the prop is missing positions every preloaded
        # mesh in the [0,1]³ corner of world — worse UX than the
        # pre-fix behaviour of leaving the mesh frozen at its last
        # baked state. .blend files saved before round-3 (when
        # gpufluid_cache_dom_size became a per-object prop) hit this.
        # Skip reattach with a warning instead of silently miscaling.
        if origin_key not in obj.keys() or dom_size_key not in obj.keys():
            _addon_logger.warning(
                "cache_loader: '%s' has gpufluid_cache_dir but no saved "
                "origin/dom_size — skipping load_post auto-reattach. "
                "Re-bake (or click Attach Surface) to repopulate.",
                obj.name)
            continue
        origin = list(obj[origin_key])
        dom_size = tuple(obj[dom_size_key])
        try:
            _preload_sequence(obj, str(cache_dir), str(pattern),
                              origin, dom_size=dom_size)
        except Exception as exc:  # noqa: BLE001
            _addon_logger.warning(
                "cache_loader: load_post auto-reattach failed for '%s': %s",
                obj.name, exc)


def register_handler():
    """Register frame_change + load_post handlers and strip any orphans
    from past reloads.

    When the addon is hot-reloaded (`del sys.modules[*]` then re-enable),
    the OLD module's handler function objects stay in
    `bpy.app.handlers.*` referencing a dead module. Firing them later
    either no-ops (handler reads None _PRELOAD) or in worst-case
    triggers a hard crash inside Blender's depsgraph when accessing freed
    Python state. Strip every handler that *looks like ours* by name,
    then install the fresh ones.
    """
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre
        if getattr(h, "__name__", "") != "_frame_change_handler"
    ]
    bpy.app.handlers.frame_change_pre.append(_frame_change_handler)
    bpy.app.handlers.load_post[:] = [
        h for h in bpy.app.handlers.load_post
        if getattr(h, "__name__", "") != "_on_load_post"
    ]
    bpy.app.handlers.load_post.append(_on_load_post)


def unregister_handler():
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre
        if getattr(h, "__name__", "") != "_frame_change_handler"
    ]
    bpy.app.handlers.load_post[:] = [
        h for h in bpy.app.handlers.load_post
        if getattr(h, "__name__", "") != "_on_load_post"
    ]


# ---------------------------------------------------------------------------
# Attach / detach operators live in ``cache_loader._ops`` (Phase 4 split).
# Re-export here so external callers keep importing the operator classes
# from the package root, e.g. ``from .cache_loader import GPUFLUID_OT_attach_cache``.
# ---------------------------------------------------------------------------
from ._ops import (  # noqa: E402, F401  (late import: _ops depends on names above)
    GPUFLUID_OT_attach_cache,
    GPUFLUID_OT_attach_ww_cache,
    GPUFLUID_OT_detach_cache,
)
