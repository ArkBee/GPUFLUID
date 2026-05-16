"""step24 kitchen-sink v0.8 demo — single-pane Eevee renderer.

Showcases the v0.8 feature stack in one image:
  * S2.15 multi-source colour — per-vertex colour on the mesh, computed
    each frame from the nearest fluid particle's RGB (so the mixing
    zone shows up purple where red meets blue).
  * S2.14 surface tension — visible rounding on the falling drops.
  * S2.12 APIC transfer mode (B6: known safe with σ at this scale).
  * D4 sphere obstacle — re-instanced in Blender from the same coords
    used in the TOML.
  * W7.7 trapped-air potential whitewater — small emissive spheres
    instanced on a vertex-only mesh that updates each frame.

Run from inside Blender (not from the project venv):

    blender --background --python examples/render_step24_eevee.py -- \
            --cache out/step24_kitchen_sink \
            --out   out/step24_eevee_frames

Then stitch to mp4:

    .venv/Scripts/python -c "import imageio.v2 as io, glob; \
        w = io.get_writer('out/videos/step24.mp4', fps=24, codec='libx264', \
                          quality=8, macro_block_size=1); \
        [w.append_data(io.imread(p)) for p in sorted(glob.glob('out/step24_eevee_frames/f_*.png'))]; \
        w.close()"
"""
from __future__ import annotations
import argparse
import math
import struct
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np


# ---------------------------------------------------------------- argv helpers
def _argv_after_doubledash():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]


# ---------------------------------------------------------------- PLY reader
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


# ---------------------------------------------------------------- nearest-particle vertex colour
def _vert_colours_from_particles(verts: np.ndarray, particles: np.ndarray,
                                  colours: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Assign each mesh vertex the colour of its nearest fluid particle.

    Memory-bounded: processes `batch` verts at a time so the V×P distance
    matrix never blows past ~200 MB.
    """
    V = verts.shape[0]
    if V == 0 or particles.shape[0] == 0:
        return np.ones((V, 3), dtype=np.float32)
    out = np.zeros((V, 3), dtype=np.float32)
    p2 = (particles ** 2).sum(axis=1)
    for i in range(0, V, batch):
        v = verts[i:i + batch]
        v2 = (v ** 2).sum(axis=1)
        sq = v2[:, None] - 2.0 * (v @ particles.T) + p2[None, :]
        out[i:i + batch] = colours[np.argmin(sq, axis=1)]
    return out


# ---------------------------------------------------------------- mesh rebuild
def _rebuild_surface_mesh(obj, verts: np.ndarray, faces: np.ndarray,
                           vert_rgb: np.ndarray):
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
    # Per-vertex Float Color attribute named "fluidcol". Material reads it
    # via an Attribute node → BSDF Base Color.
    attr = me.attributes.get("fluidcol")
    if attr is None or attr.domain != "POINT" or attr.data_type != "FLOAT_COLOR":
        if attr is not None:
            me.attributes.remove(attr)
        attr = me.attributes.new(name="fluidcol", type="FLOAT_COLOR", domain="POINT")
    rgba = np.concatenate([vert_rgb, np.ones((vert_rgb.shape[0], 1), dtype=np.float32)], axis=1)
    attr.data.foreach_set("color", rgba.ravel())


def _rebuild_ww_points(obj, positions: np.ndarray):
    me = obj.data
    me.clear_geometry()
    if positions.shape[0] == 0:
        return
    me.vertices.add(positions.shape[0])
    me.vertices.foreach_set("co", positions.ravel())
    me.update()


# ---------------------------------------------------------------- scene build
def build_scene(cache: Path) -> bpy.types.Scene:
    sc = bpy.data.scenes.get("step24_render") or bpy.data.scenes.new("step24_render")
    bpy.context.window.scene = sc
    for o in list(sc.objects):
        sc.collection.objects.unlink(o)
    for nm in [n for n in bpy.data.materials.keys() if n.startswith("step24_")]:
        bpy.data.materials.remove(bpy.data.materials[nm])
    for nm in [n for n in bpy.data.meshes.keys() if n.startswith("step24_")]:
        bpy.data.meshes.remove(bpy.data.meshes[nm])

    sc.frame_start = 1
    sc.frame_end = 90
    sc.render.fps = 24
    sc.render.resolution_x = 1600
    sc.render.resolution_y = 900
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.image_settings.file_format = "PNG"

    # Camera looking at the centre of the unit domain. Sim-Y is up, so
    # objects get rotated +90° around X at the empty-parent level.
    cam_data = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_data)
    sc.collection.objects.link(cam)
    cam.location = (1.8, -1.8, 1.6)
    target = mathutils.Vector((0.0, 0.3, 0.4))
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 35
    sc.camera = cam

    # Lights — key sun + fill area
    key = bpy.data.lights.new("Key", "SUN"); key.energy = 4.5
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.rotation_euler = (math.radians(50), math.radians(15), math.radians(-30))
    sc.collection.objects.link(key_obj)

    fill_d = bpy.data.lights.new("Fill", "AREA"); fill_d.energy = 80.0
    fill_d.size = 2.0
    fill = bpy.data.objects.new("Fill", fill_d)
    fill.location = (-2.5, -2.0, 2.0)
    fill.rotation_euler = (math.radians(60), 0, math.radians(45))
    sc.collection.objects.link(fill)

    # World
    world = bpy.data.worlds.get("step24_world") or bpy.data.worlds.new("step24_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.04, 0.05, 0.07, 1.0)
        bg.inputs["Strength"].default_value = 0.5
    sc.world = world

    # World-X rotation empty: everything below it is in sim coords (Y=up).
    pivot = bpy.data.objects.new("step24_pivot", None)
    pivot.rotation_euler = (math.radians(90), 0, 0)
    sc.collection.objects.link(pivot)
    # Translate so the unit-cube domain centre lands at world origin
    pivot.location = (-0.5, 0.0, 0.0)

    # Surface mesh
    m = bpy.data.meshes.new("step24_surface")
    surf = bpy.data.objects.new("step24_surface", m)
    sc.collection.objects.link(surf)
    surf.parent = pivot
    mat = bpy.data.materials.new("step24_water")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 0.08
        attr = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "fluidcol"
        attr.location = (-300, 100)
        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    surf.data.materials.append(mat)

    # Whitewater points + vertex-instance sphere
    wm = bpy.data.meshes.new("step24_ww")
    ww = bpy.data.objects.new("step24_ww", wm)
    sc.collection.objects.link(ww)
    ww.parent = pivot
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.012,
                                          location=(0, 0, -10))
    ww_glyph = bpy.context.active_object
    ww_glyph.name = "step24_ww_glyph"
    ww_glyph.parent = ww
    ww.instance_type = "VERTS"
    wmat = bpy.data.materials.new("step24_ww_mat")
    wmat.use_nodes = True
    wb = wmat.node_tree.nodes.get("Principled BSDF")
    if wb:
        wb.inputs["Base Color"].default_value = (0.95, 0.97, 1.0, 1.0)
        wb.inputs["Roughness"].default_value = 0.6
        if "Emission Strength" in wb.inputs:
            wb.inputs["Emission Strength"].default_value = 1.5
    ww_glyph.data.materials.append(wmat)

    # Obstacle sphere — hard-coded from the scene TOML.
    # Re-using `primitive_uv_sphere_add` keeps the script self-contained
    # rather than parsing the TOML for one number.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, segments=32, ring_count=16,
                                         location=(0.50, 0.25, 0.50))
    obs = bpy.context.active_object
    obs.name = "step24_obstacle"
    obs.parent = pivot
    bpy.ops.object.shade_smooth()
    omat = bpy.data.materials.new("step24_obstacle_mat")
    omat.use_nodes = True
    ob = omat.node_tree.nodes.get("Principled BSDF")
    if ob:
        ob.inputs["Base Color"].default_value = (0.85, 0.65, 0.30, 1.0)
        ob.inputs["Roughness"].default_value = 0.35
    obs.data.materials.append(omat)

    # Ground plane (slight reference)
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "step24_ground"
    gmat = bpy.data.materials.new("step24_ground_mat")
    gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    if gb:
        gb.inputs["Base Color"].default_value = (0.10, 0.10, 0.12, 1.0)
        gb.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(gmat)

    return sc, surf, ww


# ---------------------------------------------------------------- per-frame hook
class FrameLoader:
    def __init__(self, cache: Path, surf_obj, ww_obj):
        self.cache = cache
        self.surf = surf_obj
        self.ww = ww_obj

    def __call__(self, scene, depsgraph=None):
        f = scene.frame_current
        idx = f - 1  # frames are 0-indexed in the cache
        ply_path = self.cache / "mesh" / f"frame_{idx:04d}.ply"
        if not ply_path.exists():
            return
        verts, faces = _read_ply_minimal(str(ply_path))
        # Per-vertex colour via nearest particle
        part_path = self.cache / "particles" / f"frame_{idx:04d}.npy"
        col_path = self.cache / "colors" / f"frame_{idx:04d}.npy"
        if part_path.exists() and col_path.exists():
            part = np.load(part_path).astype(np.float32)
            col = np.load(col_path).astype(np.float32)
            rgb = _vert_colours_from_particles(verts, part, col)
        else:
            rgb = np.tile([0.18, 0.45, 0.65], (verts.shape[0], 1)).astype(np.float32)
        _rebuild_surface_mesh(self.surf, verts, faces, rgb)
        # Whitewater
        ww_path = self.cache / "whitewater" / f"frame_{idx:04d}.npy"
        if ww_path.exists():
            pos = np.load(ww_path).astype(np.float32)
            _rebuild_ww_points(self.ww, pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(_argv_after_doubledash())
    cache = Path(args.cache).resolve()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    sc, surf, ww = build_scene(cache)
    loader = FrameLoader(cache, surf, ww)
    # Reset any pre-existing handlers from prior runs.
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(loader)

    sc.render.filepath = str(out_dir / "f_")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
