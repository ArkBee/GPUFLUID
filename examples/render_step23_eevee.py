"""step23 alternate renderer — Blender Eevee headless, ~9x faster than
the matplotlib equivalent (`render_step23.py`).

Run inside Blender (not via the project venv):

    blender --background --python examples/render_step23_eevee.py -- \
            --left  out/step23_legacy  \
            --right out/step23_potential \
            --out   out/step23_eevee_frames

The script then writes one PNG per frame to ``--out``. Stitch to mp4 with::

    .venv/Scripts/python -c "import imageio.v2 as io, glob; \
        w = io.get_writer('out/videos/step23_eevee.mp4', fps=24, \
                          codec='libx264', quality=8, macro_block_size=1); \
        [w.append_data(io.imread(p)) for p in sorted(glob.glob('out/step23_eevee_frames/f_*.png'))]; \
        w.close()"

Known polish gap vs the matplotlib version: this Eevee setup renders all
whitewater as a single white emissive sphere. The matplotlib version
colour-codes foam (white) / spray (cyan) / bubble (blue) and overlays
per-class counts. A follow-up micro can drive material colour from the
per-vertex `gpufluid_kind` INT attribute via a Geometry Nodes "Attribute
to Color" graph; not done here to keep the speed-win patch small.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import mathutils


def _argv_after_doubledash():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]


def build_scene(left_cache: Path, right_cache: Path) -> bpy.types.Scene:
    sc = bpy.data.scenes.get("step23_render") or bpy.data.scenes.new("step23_render")
    bpy.context.window.scene = sc

    # Wipe step23-prefixed objects/materials/meshes from any prior run
    for o in list(sc.objects):
        if o.name.startswith("step23_") or o.name in ("Cam", "Sun"):
            sc.collection.objects.unlink(o)
    for nm in [n for n in bpy.data.materials.keys() if n.startswith("step23_")]:
        bpy.data.materials.remove(bpy.data.materials[nm])
    for nm in [n for n in bpy.data.meshes.keys() if n.startswith("step23_")]:
        bpy.data.meshes.remove(bpy.data.meshes[nm])

    sc.frame_start = 1
    sc.frame_end = 60
    sc.render.fps = 24
    sc.render.resolution_x = 1600
    sc.render.resolution_y = 700
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.image_settings.file_format = "PNG"

    # Camera
    cam_data = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_data)
    sc.collection.objects.link(cam)
    cam.location = (0.0, -3.2, 1.6)
    target = mathutils.Vector((0.0, 0.3, 0.4))
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 30
    sc.camera = cam

    # Sun
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 5.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(45), math.radians(20), math.radians(-30))
    sc.collection.objects.link(sun)

    # World
    world = bpy.data.worlds.get("step23_world") or bpy.data.worlds.new("step23_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.03, 0.04, 0.06, 1.0)
        bg.inputs["Strength"].default_value = 0.4
    sc.world = world

    for side, cache_dir, x_offset in [("legacy", left_cache, -1.15),
                                       ("pot", right_cache, 0.15)]:
        # Surface mesh
        m = bpy.data.meshes.new(f"step23_{side}_mesh")
        o = bpy.data.objects.new(f"step23_{side}_mesh", m)
        sc.collection.objects.link(o)
        o.location = (x_offset, 0, 0)
        o.rotation_euler = (math.radians(90), 0, 0)  # sim-Y → Blender-Z
        o["gpufluid_cache_dir"] = str(cache_dir)
        o["gpufluid_cache_pattern"] = "mesh/frame_{:04d}.ply"
        o["gpufluid_cache_frame_offset"] = 1
        o["gpufluid_cache_origin"] = [0.0, 0.0, 0.0]
        mat = bpy.data.materials.new(f"step23_water_{side}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.10, 0.40, 0.65, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.08
        o.data.materials.append(mat)

        # Whitewater point-cloud + vertex-instance sphere
        wm = bpy.data.meshes.new(f"step23_{side}_ww")
        wo = bpy.data.objects.new(f"step23_{side}_ww", wm)
        sc.collection.objects.link(wo)
        wo.location = (x_offset, 0, 0)
        wo.rotation_euler = (math.radians(90), 0, 0)
        wo["gpufluid_ww_cache_dir"] = str(cache_dir)
        wo["gpufluid_ww_cache_frame_offset"] = 1
        wo["gpufluid_ww_cache_origin"] = [0.0, 0.0, 0.0]

        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.02,
                                              location=(0, 0, -10))
        sphere = bpy.context.active_object
        sphere.name = f"step23_{side}_ww_glyph"
        sphere.parent = wo
        wo.instance_type = "VERTS"
        wmat = bpy.data.materials.new(f"step23_ww_{side}")
        wmat.use_nodes = True
        wb = wmat.node_tree.nodes.get("Principled BSDF")
        if wb:
            wb.inputs["Base Color"].default_value = (0.95, 0.97, 1.0, 1.0)
            wb.inputs["Roughness"].default_value = 0.6
            if "Emission Strength" in wb.inputs:
                wb.inputs["Emission Strength"].default_value = 1.2
        sphere.data.materials.append(wmat)

    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left",  required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out",   required=True)
    args = ap.parse_args(_argv_after_doubledash())
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    sc = build_scene(Path(args.left).resolve(), Path(args.right).resolve())
    sc.render.filepath = str(out_dir / "f_")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
