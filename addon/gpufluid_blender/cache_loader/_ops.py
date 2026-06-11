"""[BLK A8.6] Attach / detach operators for the cache_loader package.

Phase 4 split: lifted out of ``cache_loader/__init__.py`` so the package
stays under the 500-line cap. The operator classes are still re-exported
at the package level (see ``cache_loader/__init__.py``) so
``__init__.py`` of the addon can keep registering them by name.
"""
from __future__ import annotations

import os
import time as _time

import bpy

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

from . import (
    _PRELOAD,
    _free_table,
    _frame_change_handler,
    _preload_sequence,
    _prune_stale,
)


@block("A8.6", "Cache import — attach operator (PLY preload bind)")
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
    frame_offset: bpy.props.IntProperty(
        name="Cache starts at scene frame", default=0,
        description="Scene frame at which cache frame_0000 appears. "
                    "frame_0000 is the empty pre-sim initial state (no "
                    "surface yet), so the default 0 maps scene frame 1 to "
                    "the first SIMULATED frame (frame_0001) — pre-round-61 "
                    "default 1 put the empty frame on scene frame 1 (looked "
                    "like a blank/failed bake) AND dropped the final frame.")
    # The Alembic fast path is currently disabled (see commands.py); keep
    # this flag so external scripts can still force the PLY path if needed.
    force_ply: bpy.props.BoolProperty(name="Force PLY preload", default=True)

    def execute(self, context):
        # Fallback when invoked from the Attach Surface button without
        # the F6 popup args filled (live test 2026-05-25): if cache_dir
        # is empty AND there's an active Domain in the scene, derive
        # cache_dir + origin + dom_size from it. The post-bake auto-attach
        # in operators/bake.py always passes explicit args, so this only
        # kicks in for the standalone button flow.
        cache_dir_raw = self.cache_dir
        if not cache_dir_raw:
            active = context.view_layer.objects.active
            # `gpufluid_domain` attr is missing on objects from .blends
            # that were saved before the addon registered its property
            # groups — getattr-guard so we degrade to the explicit-path
            # case instead of AttributeError'ing.
            active_dom = getattr(active, "gpufluid_domain", None)
            if active is not None and active_dom is not None and active_dom.is_domain:
                cache_dir_raw = active.gpufluid_domain.cache_dir
                # Recompute world AABB from the live Domain so origin and
                # dom_size match what Bake would have written. Duplicated
                # rather than imported from operators._animation to keep
                # cache_loader package free of cross-package coupling.
                import mathutils
                if active.type == "EMPTY":
                    d = active.empty_display_size
                    local = [mathutils.Vector(c) for c in (
                        (-d,-d,-d),(d,-d,-d),(d,d,-d),(-d,d,-d),
                        (-d,-d,d),(d,-d,d),(d,d,d),(-d,d,d))]
                    pts = [active.matrix_world @ v for v in local]
                else:
                    pts = [active.matrix_world @ mathutils.Vector(c)
                           for c in active.bound_box]
                xs=[p.x for p in pts]; ys=[p.y for p in pts]; zs=[p.z for p in pts]
                lo = (min(xs), min(ys), min(zs))
                hi = (max(xs), max(ys), max(zs))
                self.origin = lo
                self.dom_size = (hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2])
        cache_dir = bpy.path.abspath(cache_dir_raw)
        if not os.path.isdir(cache_dir):
            self.report({"ERROR"}, f"cache dir not found: {cache_dir or '(empty — no active Domain)'}")
            return {"CANCELLED"}

        target = (context.scene.objects.get(self.target_name)
                  if self.target_name else None)
        if target is None:
            # No explicit target — reuse an existing gpufluid_cache in the
            # scene (so re-bakes / re-attaches don't spawn .001/.002/...).
            target = context.scene.objects.get("gpufluid_cache")
        if target is None:
            # First-time attach: create a new mesh object.
            mesh = bpy.data.meshes.new("gpufluid_cache_mesh")
            target = bpy.data.objects.new("gpufluid_cache", mesh)
            context.scene.collection.objects.link(target)
        # Round-19: centralised via cache_binding.set_cache_binding —
        # no more magic strings sprayed across 3 files.
        from .. import cache_binding as _cb
        _cb.set_cache_binding(
            target, cache_dir, list(self.origin), list(self.dom_size),
            frame_offset=self.frame_offset)

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
            cf = next((bpy.data.cache_files[n]
                       for n in bpy.data.cache_files.keys()
                       if n not in before), None)
            if cf is None and len(bpy.data.cache_files):
                cf = bpy.data.cache_files[-1]
            mod = target.modifiers.new(
                name="MeshSequenceCache", type="MESH_SEQUENCE_CACHE")
            mod.cache_file = cf
            paths = [p.path for p in cf.object_paths] if cf else []
            if paths:
                mod.object_path = paths[0]
            lava = (bpy.data.materials.get("LavaProd")
                    or bpy.data.materials.get("LavaMat"))
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
        t0 = _time.time()
        n = _preload_sequence(
            target, cache_dir, "mesh/frame_{:04d}.ply",
            list(self.origin),
            dom_size=tuple(self.dom_size),
        )
        _frame_change_handler(context.scene)
        # Round-61: 0 frames loaded means the bind points at an empty /
        # cleared / torn cache — the user would see nothing on screen.
        # Report WARNING (not the old unconditional INFO) so the empty
        # result is visible instead of masquerading as success.
        if n == 0:
            self.report({"WARNING"},
                        f"cache attached but 0 renderable frames found in "
                        f"{os.path.join(cache_dir, 'mesh')} — the bake "
                        f"produced no mesh output, or the cache was "
                        f"cleared. Re-bake, or check the system console "
                        f"for the CLI error.")
        else:
            self.report({"INFO"},
                        f"cache attached (PLY preload) — {n} frames in "
                        f"{_time.time() - t0:.1f}s")
        # Round-62: force a 3D-viewport redraw. A MODAL bake's auto-attach
        # swaps obj.data inside the frame_change handler, but the viewport
        # kept showing the pre-bake state until the user moved the mouse /
        # scrubbed — read as "I pressed Bake and nothing happened". Tag all
        # VIEW_3D areas so the result appears the instant the bake finishes.
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
        return {"FINISHED"}


@block("A8.6.1", "Whitewater point-cloud import")
class GPUFLUID_OT_attach_ww_cache(bpy.types.Operator):
    bl_idname = "gpufluid.attach_ww_cache"
    bl_label = "Attach gpufluid Whitewater Cache"
    bl_description = (
        "Bind a whitewater point cache (cache_dir/whitewater/*.npy + "
        "whitewater_kinds/*.npy) to an object. Verts get a per-vertex "
        "INT attribute 'gpufluid_kind' (0=foam, 1=spray, 2=bubble) for "
        "Geometry Nodes branching")
    bl_options = {"REGISTER", "UNDO"}

    cache_dir: bpy.props.StringProperty(name="Cache Dir", subtype="DIR_PATH")
    target_name: bpy.props.StringProperty(name="Target object")
    origin_x: bpy.props.FloatProperty(name="Origin X", default=0.0)
    origin_y: bpy.props.FloatProperty(name="Origin Y", default=0.0)
    origin_z: bpy.props.FloatProperty(name="Origin Z", default=0.0)
    frame_offset: bpy.props.IntProperty(
        name="Cache starts at scene frame", default=0,
        description="Scene frame at which cache frame_0000 appears. "
                    "frame_0000 is the empty pre-sim initial state (no "
                    "surface yet), so the default 0 maps scene frame 1 to "
                    "the first SIMULATED frame (frame_0001) — pre-round-61 "
                    "default 1 put the empty frame on scene frame 1 (looked "
                    "like a blank/failed bake) AND dropped the final frame.")

    def execute(self, context):
        cache_dir = bpy.path.abspath(self.cache_dir)
        if not os.path.isdir(os.path.join(cache_dir, "whitewater")):
            self.report(
                {"ERROR"},
                f"no 'whitewater' subdir in {cache_dir} — bake whitewater first")
            return {"CANCELLED"}
        target = (context.scene.objects.get(self.target_name)
                  if self.target_name else None)
        if target is None:
            mesh = bpy.data.meshes.new("gpufluid_ww_mesh")
            target = bpy.data.objects.new("gpufluid_whitewater", mesh)
            context.scene.collection.objects.link(target)
        from .. import cache_binding as _cb
        _cb.set_ww_binding(
            target, cache_dir,
            (self.origin_x, self.origin_y, self.origin_z),
            frame_offset=self.frame_offset)
        _frame_change_handler(context.scene)
        # Pre-public-test audit (2026-06-01, §9.6 mirror-drift): the surface
        # attach WARNINGs on a 0-frame bind, but this path reported plain INFO
        # success even when whitewater/ was empty — "attached" then nothing on
        # playback. Count via the shared cache_sanity helper so both honesty
        # paths can't drift.
        from ..cache_sanity import attach_ww_sanity, count_ww_frames
        n = count_ww_frames(cache_dir)
        level, msg = attach_ww_sanity(n)
        if level is not None:
            self.report({level}, msg)
        else:
            self.report({"INFO"},
                        f"whitewater cache attached to '{target.name}' "
                        f"({n} frames)")
        return {"FINISHED"}


@block("A8.6", "Cache import — detach operator")
class GPUFLUID_OT_detach_cache(bpy.types.Operator):
    bl_idname = "gpufluid.detach_cache"
    bl_label = "Detach gpufluid Cache"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        # Pass if EITHER the active object has cache props OR a
        # gpufluid_cache / gpufluid_whitewater object exists in scene.
        # Round-19: use cache_binding helpers instead of raw key lookup.
        if context.scene is None:
            return False
        from .. import cache_binding as _cb
        for o in context.scene.objects:
            if _cb.has_cache_binding(o) or _cb.has_ww_binding(o):
                return True
        for n in ("gpufluid_cache", "gpufluid_whitewater"):
            if n in context.scene.objects:
                return True
        return False

    def execute(self, context):
        # Live-found 2026-05-25 (round-5): original execute operated on
        # context.active_object, which is typically Domain after a bake.
        # Domain carries bake-trace props with deliberately DIFFERENT
        # names from the cache-obj per-object props — the for-loop
        # silently no-op'd on the real cache object. Round-19 centralised
        # both name groups in `cache_binding.clear_all_bindings` so this
        # operator can't drift again.
        from .. import cache_binding as _cb
        n_cleaned = 0
        for o in list(context.scene.objects):
            if _cb.clear_all_bindings(o):
                n_cleaned += 1
            # Drop preload table — match cache_loader behaviour
            if o.name in _PRELOAD:
                _free_table(_PRELOAD[o.name])
                del _PRELOAD[o.name]
        # Round-8: also sweep stale _PRELOAD entries (objects that may
        # have been removed from the scene since last attach).
        _prune_stale(context.scene)
        self.report({"INFO"}, f"detached cache from {n_cleaned} object(s)")
        return {"FINISHED"}
