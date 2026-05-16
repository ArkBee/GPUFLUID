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
# Per-frame handler
# ---------------------------------------------------------------------------

def _frame_change_handler(scene, depsgraph=None):
    f = scene.frame_current
    for obj in scene.objects:
        # Cache loading only applies to mesh objects. The Domain Empty also
        # carries a `gpufluid_cache_dir` custom prop (for the bake operator's
        # own bookkeeping) but is not a render target.
        if obj.type != "MESH":
            continue
        cache_dir = obj.get("gpufluid_cache_dir")
        if not cache_dir:
            continue
        pattern = obj.get("gpufluid_cache_pattern", "mesh/frame_{:04d}.ply")
        offset = int(obj.get("gpufluid_cache_frame_offset", 0))
        origin = list(obj.get("gpufluid_cache_origin", [0.0, 0.0, 0.0]))
        idx = f - offset
        if idx < 0:
            continue
        path = os.path.join(cache_dir, pattern.format(idx))
        if not os.path.exists(path):
            continue
        try:
            verts, faces = _read_ply_minimal(path)
            _rebuild_mesh(obj, verts, faces, origin)
        except Exception as exc:  # noqa: BLE001
            print(f"[gpufluid cache] error at frame {f} for '{obj.name}': {exc}")


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


class GPUFLUID_OT_detach_cache(bpy.types.Operator):
    bl_idname = "gpufluid.detach_cache"
    bl_label = "Detach gpufluid Cache"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and \
               context.active_object.get("gpufluid_cache_dir") is not None

    def execute(self, context):
        obj = context.active_object
        for k in ("gpufluid_cache_dir", "gpufluid_cache_pattern",
                  "gpufluid_cache_frame_offset", "gpufluid_cache_origin"):
            if k in obj.keys():
                del obj[k]
        return {"FINISHED"}
