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
import bpy
import numpy as np


# ---------------------------------------------------------------------------
# Minimal PLY reader (matches gpufluid.io.ply binary format).
# ---------------------------------------------------------------------------

def _read_ply_minimal(path):
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            if not line:
                return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
            header += line
            if line.strip() == b"end_header":
                break
        text = header.decode("ascii", errors="replace")
        n_v = 0; n_f = 0
        for ln in text.splitlines():
            if ln.startswith("element vertex"):
                n_v = int(ln.split()[2])
            elif ln.startswith("element face"):
                n_f = int(ln.split()[2])
        verts = np.frombuffer(f.read(n_v * 12), dtype=np.float32).reshape(n_v, 3)
        verts = verts.copy()
        face_bytes = f.read(n_f * 13)
        faces = np.empty((n_f, 3), dtype=np.int32)
        for i in range(n_f):
            base = i * 13
            faces[i] = np.frombuffer(face_bytes[base + 1: base + 13], dtype=np.int32)
        return verts, faces


def _rebuild_mesh(obj, verts, faces, origin):
    """Replace obj.data with the given mesh, translated by `origin`."""
    me = obj.data
    me.clear_geometry()
    n_v = len(verts)
    n_f = len(faces)
    if n_v == 0 or n_f == 0:
        return
    # apply origin offset so sim-space coordinates land at the right place in the world
    v_world = (verts + np.asarray(origin, dtype=np.float32))
    me.vertices.add(n_v)
    me.vertices.foreach_set("co", v_world.ravel())
    loop_total = n_f * 3
    me.loops.add(loop_total)
    me.polygons.add(n_f)
    loop_starts = np.arange(0, loop_total, 3, dtype=np.int32)
    loop_totals = np.full(n_f, 3, dtype=np.int32)
    me.polygons.foreach_set("loop_start", loop_starts)
    me.polygons.foreach_set("loop_total", loop_totals)
    me.polygons.foreach_set("vertices", faces.ravel())
    me.update(calc_edges=True)


# ---------------------------------------------------------------------------
# Whitewater point-cloud loader
# ---------------------------------------------------------------------------

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
    me = obj.data
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
                path = os.path.join(cache_dir, pattern.format(idx))
                if os.path.exists(path):
                    try:
                        verts, faces = _read_ply_minimal(path)
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
    if _frame_change_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_frame_change_handler)


def unregister_handler():
    while _frame_change_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_frame_change_handler)


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
    origin_x: bpy.props.FloatProperty(name="Origin X", default=0.0)
    origin_y: bpy.props.FloatProperty(name="Origin Y", default=0.0)
    origin_z: bpy.props.FloatProperty(name="Origin Z", default=0.0)
    frame_offset: bpy.props.IntProperty(name="Cache starts at scene frame", default=1)

    def execute(self, context):
        cache_dir = bpy.path.abspath(self.cache_dir)
        if not os.path.isdir(cache_dir):
            self.report({"ERROR"}, f"cache dir not found: {cache_dir}")
            return {"CANCELLED"}

        target = context.scene.objects.get(self.target_name) if self.target_name else None
        if target is None:
            # create a new mesh object
            mesh = bpy.data.meshes.new("gpufluid_cache_mesh")
            target = bpy.data.objects.new("gpufluid_cache", mesh)
            context.scene.collection.objects.link(target)
        target["gpufluid_cache_dir"] = cache_dir
        target["gpufluid_cache_pattern"] = "mesh/frame_{:04d}.ply"
        target["gpufluid_cache_frame_offset"] = self.frame_offset
        target["gpufluid_cache_origin"] = [self.origin_x, self.origin_y, self.origin_z]
        # trigger an immediate refresh
        _frame_change_handler(context.scene)
        self.report({"INFO"}, f"cache attached to '{target.name}'")
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
