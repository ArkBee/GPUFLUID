"""step26 — CUDA-graphs feature showcase (Options A + B) Eevee renderer.

Bakes the `big_pcg.toml` scene (96^3 waterfall with PCG pressure) and
renders it in Eevee Next as a single-pane animation with a corner text
overlay that calls out the perf win:

    PCG dense + CUDA graphs (Option A)
    sim 4.29s -> 1.05s  (4.09x, 88% hit rate)
    eligibility matrix: 9/9 (Options A + B)

The visual content (water hitting the cylinder obstacle) is identical
to a non-graph bake — the *feature* is the timing on the overlay. The
overlay is two text objects parented to the camera so they stick to
the same screen position every frame.

Run from inside Blender:

    blender --background --python examples/render_step26_eevee.py -- \\
            --cache out/big_pcg_graph_on \\
            --out   out/step26_eevee_frames

Stitch:

    .venv/Scripts/python -c "import imageio.v2 as io, glob; \\
        w = io.get_writer('out/videos/step26.mp4', fps=24, codec='libx264', \\
                          quality=8, macro_block_size=1); \\
        [w.append_data(io.imread(p)) for p in sorted(glob.glob('out/step26_eevee_frames/f_*.png'))]; \\
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


def _add_text(name, body, location, scale=0.10, color=(1.0, 1.0, 1.0, 1.0),
              emission=4.0, parent=None):
    """Create a 3D Text object and attach an emissive material so it
    stays bright regardless of scene lighting."""
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.body = body
    obj.data.size = scale
    obj.data.align_x = "LEFT"
    obj.data.align_y = "TOP"
    if parent is not None:
        obj.parent = parent
    mat = bpy.data.materials.new(f"{name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = color
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = color
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission
    obj.data.materials.append(mat)
    return obj


def build_scene(cache: Path):
    sc = bpy.data.scenes.get("step26_render") or bpy.data.scenes.new("step26_render")
    bpy.context.window.scene = sc
    for o in list(sc.objects):
        sc.collection.objects.unlink(o)
    for nm in [n for n in bpy.data.materials.keys() if n.startswith("step26_")]:
        bpy.data.materials.remove(bpy.data.materials[nm])
    for nm in [n for n in bpy.data.meshes.keys() if n.startswith("step26_")]:
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
    cam.location = (2.2, -2.2, 1.6)
    target = mathutils.Vector((0.0, 0.20, 0.3))
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 35
    sc.camera = cam

    key = bpy.data.lights.new("Key", "SUN"); key.energy = 4.0
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(-25))
    sc.collection.objects.link(key_obj)
    fill_d = bpy.data.lights.new("Fill", "AREA"); fill_d.energy = 70.0
    fill_d.size = 2.0
    fill = bpy.data.objects.new("Fill", fill_d)
    fill.location = (-2.5, -2.0, 2.0)
    fill.rotation_euler = (math.radians(60), 0, math.radians(45))
    sc.collection.objects.link(fill)

    world = bpy.data.worlds.get("step26_world") or bpy.data.worlds.new("step26_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.03, 0.06, 0.10, 1.0)
        bg.inputs["Strength"].default_value = 0.6
    sc.world = world

    pivot = bpy.data.objects.new("step26_pivot", None)
    pivot.rotation_euler = (math.radians(90), 0, 0)
    pivot.location = (-0.5, 0.0, 0.0)
    sc.collection.objects.link(pivot)

    # Surface mesh — clear water.
    surf_me = bpy.data.meshes.new("step26_surface")
    surf = bpy.data.objects.new("step26_surface", surf_me)
    sc.collection.objects.link(surf)
    surf.parent = pivot
    water = bpy.data.materials.new("step26_water")
    water.use_nodes = True
    wbsdf = water.node_tree.nodes.get("Principled BSDF")
    if wbsdf:
        wbsdf.inputs["Base Color"].default_value = (0.20, 0.50, 0.70, 1.0)
        wbsdf.inputs["Roughness"].default_value = 0.08
        if "Transmission Weight" in wbsdf.inputs:
            wbsdf.inputs["Transmission Weight"].default_value = 0.5
        elif "Transmission" in wbsdf.inputs:
            wbsdf.inputs["Transmission"].default_value = 0.5
        if "IOR" in wbsdf.inputs:
            wbsdf.inputs["IOR"].default_value = 1.33
    surf.data.materials.append(water)

    # Obstacle: cylinder_y from big_pcg.toml — center=(0.55, 0.30, 0.50),
    # radius=0.08, half_height=0.30.
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.08, depth=0.60, vertices=32,
        location=(0.55, 0.30, 0.50),
    )
    obs = bpy.context.active_object
    obs.name = "step26_obstacle"
    obs.parent = pivot
    bpy.ops.object.shade_smooth()
    omat = bpy.data.materials.new("step26_obstacle_mat")
    omat.use_nodes = True
    ob = omat.node_tree.nodes.get("Principled BSDF")
    if ob:
        ob.inputs["Base Color"].default_value = (0.65, 0.55, 0.40, 1.0)
        ob.inputs["Roughness"].default_value = 0.6
    obs.data.materials.append(omat)

    # Ground plane.
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "step26_ground"
    gmat = bpy.data.materials.new("step26_ground_mat")
    gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    if gb:
        gb.inputs["Base Color"].default_value = (0.07, 0.07, 0.09, 1.0)
        gb.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(gmat)

    # ---------- Title overlay parented to the camera so it sticks in 2D.
    # The text-on-cam trick: parent text to the camera at a small Z offset
    # in the camera's local frame. Anchors at top-left of the viewport.
    overlay = bpy.data.objects.new("step26_overlay", None)
    overlay.parent = cam
    overlay.location = (-0.42, 0.22, -1.0)  # camera-space: left, up, in front
    sc.collection.objects.link(overlay)

    _add_text("step26_t1", "PCG dense + CUDA graphs (Option A)",
              (0.0, 0.00, 0.0), scale=0.05, parent=overlay,
              color=(0.95, 0.97, 1.00, 1.0))
    _add_text("step26_t2", "sim 4.29s  ->  1.05s   (4.09x, 88% hit rate)",
              (0.0, -0.08, 0.0), scale=0.045, parent=overlay,
              color=(1.00, 0.85, 0.30, 1.0))
    _add_text("step26_t3", "+ sparse pressure now graph-eligible (Option B)",
              (0.0, -0.15, 0.0), scale=0.040, parent=overlay,
              color=(0.85, 0.85, 0.95, 1.0))
    _add_text("step26_t4", "eligibility matrix: 9 / 9  (v0.9 polish closed)",
              (0.0, -0.21, 0.0), scale=0.040, parent=overlay,
              color=(0.40, 0.95, 1.00, 1.0))

    return sc, surf


class FrameLoader:
    def __init__(self, cache: Path, surf):
        self.cache = cache
        self.surf = surf

    def __call__(self, scene, depsgraph=None):
        f = scene.frame_current
        idx = f - 1
        ply_path = self.cache / "mesh" / f"frame_{idx:04d}.ply"
        if ply_path.exists():
            verts, faces = _read_ply_minimal(str(ply_path))
            _rebuild_surface_mesh(self.surf, verts, faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(_argv_after_doubledash())
    cache = Path(args.cache).resolve()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    sc, surf = build_scene(cache)
    loader = FrameLoader(cache, surf)
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(loader)

    sc.render.filepath = str(out_dir / "f_")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
