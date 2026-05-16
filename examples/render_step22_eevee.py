"""step22 — whitewater splash (W7.4/W7.5 + W7.7 potential selector) Eevee renderer.

Renders the `whitewater_splash` bake (`examples/scenes/whitewater_splash.toml`,
v0.8 with `whitewater_use_potential = true`) by:

  * Drawing the fluid surface mesh from `<cache>/mesh/frame_NNNN.ply`
    as a translucent water material.
  * Reading per-frame whitewater positions from
    `<cache>/whitewater/frame_NNNN.npy` AND per-particle kind labels
    from `<cache>/whitewater_kinds/frame_NNNN.npy`, splitting them into
    THREE separate vertex-instance buckets — one per class — so each
    class can carry its own emissive material:
      - kind 0 = foam   → near-white
      - kind 1 = spray  → cyan
      - kind 2 = bubble → saturated blue

Run from inside Blender:

    blender --background --python examples/render_step22_eevee.py -- \\
            --cache out/whitewater_splash \\
            --out   out/step22_eevee_frames

Stitch to mp4 (same pattern as step24/step25):

    .venv/Scripts/python -c "import imageio.v2 as io, glob; \\
        w = io.get_writer('out/videos/step22.mp4', fps=24, codec='libx264', \\
                          quality=8, macro_block_size=1); \\
        [w.append_data(io.imread(p)) for p in sorted(glob.glob('out/step22_eevee_frames/f_*.png'))]; \\
        w.close()"
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path

import bpy
import mathutils
import numpy as np


def _argv_after_doubledash():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]


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
        n_v = n_f = 0
        for ln in text.splitlines():
            if ln.startswith("element vertex"):
                n_v = int(ln.split()[2])
            elif ln.startswith("element face"):
                n_f = int(ln.split()[2])
        verts = np.frombuffer(f.read(n_v * 12), dtype=np.float32).reshape(n_v, 3).copy()
        faces = np.empty((n_f, 3), dtype=np.int32)
        fb = f.read(n_f * 13)
        for i in range(n_f):
            base = i * 13
            faces[i] = np.frombuffer(fb[base + 1: base + 13], dtype=np.int32)
        return verts, faces


def _rebuild_surface_mesh(obj, verts, faces):
    me = obj.data
    me.clear_geometry()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return
    me.vertices.add(verts.shape[0])
    me.vertices.foreach_set("co", verts.ravel())
    loop_total = faces.shape[0] * 3
    me.loops.add(loop_total)
    me.polygons.add(faces.shape[0])
    me.polygons.foreach_set("loop_start", np.arange(0, loop_total, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(faces.shape[0], 3, dtype=np.int32))
    me.polygons.foreach_set("vertices", faces.ravel())
    me.update(calc_edges=True)


def _rebuild_points(obj, positions: np.ndarray):
    me = obj.data
    me.clear_geometry()
    if positions.shape[0] == 0:
        return
    me.vertices.add(positions.shape[0])
    me.vertices.foreach_set("co", positions.ravel())
    me.update()


def _make_class_material(name, base_rgb, emission_strength, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*base_rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        # Emission Color (4.x) or Emission (legacy)
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*base_rgb, 1.0)
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = (*base_rgb, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def build_scene(cache: Path):
    sc = bpy.data.scenes.get("step22_render") or bpy.data.scenes.new("step22_render")
    bpy.context.window.scene = sc
    for o in list(sc.objects):
        sc.collection.objects.unlink(o)
    for nm in [n for n in bpy.data.materials.keys() if n.startswith("step22_")]:
        bpy.data.materials.remove(bpy.data.materials[nm])
    for nm in [n for n in bpy.data.meshes.keys() if n.startswith("step22_")]:
        bpy.data.meshes.remove(bpy.data.meshes[nm])

    sc.frame_start = 1
    sc.frame_end = 90
    sc.render.fps = 24
    sc.render.resolution_x = 1600
    sc.render.resolution_y = 900
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except (TypeError, ValueError):
        sc.render.engine = "BLENDER_EEVEE"
    sc.render.image_settings.file_format = "PNG"

    cam_data = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_data)
    sc.collection.objects.link(cam)
    cam.location = (2.0, -2.0, 1.5)
    target = mathutils.Vector((0.0, 0.25, 0.3))
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 35
    sc.camera = cam

    key = bpy.data.lights.new("Key", "SUN"); key.energy = 3.5
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(-25))
    sc.collection.objects.link(key_obj)
    fill_d = bpy.data.lights.new("Fill", "AREA"); fill_d.energy = 60.0
    fill_d.size = 2.0
    fill = bpy.data.objects.new("Fill", fill_d)
    fill.location = (-2.5, -2.0, 2.0)
    fill.rotation_euler = (math.radians(60), 0, math.radians(45))
    sc.collection.objects.link(fill)

    world = bpy.data.worlds.get("step22_world") or bpy.data.worlds.new("step22_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.03, 0.06, 0.10, 1.0)
        bg.inputs["Strength"].default_value = 0.6
    sc.world = world

    pivot = bpy.data.objects.new("step22_pivot", None)
    pivot.rotation_euler = (math.radians(90), 0, 0)
    pivot.location = (-0.5, 0.0, 0.0)
    sc.collection.objects.link(pivot)

    # Surface mesh — translucent blueish water
    surf_me = bpy.data.meshes.new("step22_surface")
    surf = bpy.data.objects.new("step22_surface", surf_me)
    sc.collection.objects.link(surf)
    surf.parent = pivot
    water = bpy.data.materials.new("step22_water")
    water.use_nodes = True
    wbsdf = water.node_tree.nodes.get("Principled BSDF")
    if wbsdf:
        wbsdf.inputs["Base Color"].default_value = (0.18, 0.42, 0.55, 1.0)
        wbsdf.inputs["Roughness"].default_value = 0.08
        if "Transmission Weight" in wbsdf.inputs:
            wbsdf.inputs["Transmission Weight"].default_value = 0.6
        elif "Transmission" in wbsdf.inputs:
            wbsdf.inputs["Transmission"].default_value = 0.6
        if "IOR" in wbsdf.inputs:
            wbsdf.inputs["IOR"].default_value = 1.33
    surf.data.materials.append(water)

    # Three whitewater buckets — foam / spray / bubble — each with its
    # own vertex-instanced sphere glyph + material.
    class_specs = [
        ("foam",   (0.95, 0.97, 1.00), 2.5, 0.014),
        ("spray",  (0.40, 0.95, 1.00), 4.0, 0.011),
        ("bubble", (0.10, 0.30, 0.95), 2.0, 0.013),
    ]
    bucket_objs = {}
    for kind_idx, (name, rgb, em_strength, radius) in enumerate(class_specs):
        pts_me = bpy.data.meshes.new(f"step22_ww_{name}")
        pts_obj = bpy.data.objects.new(f"step22_ww_{name}", pts_me)
        sc.collection.objects.link(pts_obj)
        pts_obj.parent = pivot
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=radius,
                                              location=(0, 0, -10))
        glyph = bpy.context.active_object
        glyph.name = f"step22_ww_{name}_glyph"
        glyph.parent = pts_obj
        pts_obj.instance_type = "VERTS"
        mat = _make_class_material(f"step22_ww_{name}_mat", rgb, em_strength, 0.4)
        glyph.data.materials.append(mat)
        bucket_objs[kind_idx] = pts_obj

    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "step22_ground"
    gmat = bpy.data.materials.new("step22_ground_mat")
    gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    if gb:
        gb.inputs["Base Color"].default_value = (0.06, 0.06, 0.08, 1.0)
        gb.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(gmat)

    return sc, surf, bucket_objs


class FrameLoader:
    def __init__(self, cache: Path, surf, buckets):
        self.cache = cache
        self.surf = surf
        self.buckets = buckets  # dict {kind_idx: object}

    def __call__(self, scene, depsgraph=None):
        f = scene.frame_current
        idx = f - 1
        ply_path = self.cache / "mesh" / f"frame_{idx:04d}.ply"
        if ply_path.exists():
            verts, faces = _read_ply_minimal(str(ply_path))
            _rebuild_surface_mesh(self.surf, verts, faces)
        ww_pos = self.cache / "whitewater" / f"frame_{idx:04d}.npy"
        ww_kind = self.cache / "whitewater_kinds" / f"frame_{idx:04d}.npy"
        if ww_pos.exists() and ww_kind.exists():
            pos = np.load(ww_pos).astype(np.float32)
            kind = np.load(ww_kind).astype(np.int32)
            for kidx, obj in self.buckets.items():
                mask = (kind == kidx)
                _rebuild_points(obj, pos[mask] if mask.any() else np.zeros((0, 3), np.float32))
        else:
            for obj in self.buckets.values():
                _rebuild_points(obj, np.zeros((0, 3), np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(_argv_after_doubledash())
    cache = Path(args.cache).resolve()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    sc, surf, buckets = build_scene(cache)
    loader = FrameLoader(cache, surf, buckets)
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(loader)

    sc.render.filepath = str(out_dir / "f_")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
