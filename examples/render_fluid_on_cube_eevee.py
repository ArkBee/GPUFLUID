"""step30/31/32 — three-fluid demo (water / oil / honey on cube), Eevee renderer.

A single render template parameterised by `--label` and `--color-*` so the
same code drives all three fluids; the per-scene parameters live in
examples/scenes/step30/31/32_*.toml. The video gets two overlays:

  * TOP    "sim time" — running counter, frame_idx / scene.fps in seconds.
            Ticks visibly through the 10-second clip.
  * BOTTOM "wall-clock" — bake numbers parsed from cache.json's `notes`
            field (e.g. "sim=7.4s mesh=6.5s"). Fixed per video.

Run from inside Blender:

    blender --background --python examples/render_fluid_on_cube_eevee.py -- \\
            --cache out/step30_water_on_cube \\
            --out   out/step30_eevee_frames \\
            --label "Water" \\
            --color 0.20 0.50 0.70

Each fluid pass uses a different label/colour but otherwise identical
camera, lighting, ground, cube obstacle so the three videos compare
directly.
"""
from __future__ import annotations
import argparse, json, math, re, sys
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


def _parse_wallclock_from_cache_json(cache: Path) -> tuple[float, float]:
    """Pull (sim_s, mesh_s) from cache.json's notes field.

    Example notes: "sim=7.4s mesh=6.5s". If parsing fails returns (0, 0).
    """
    try:
        data = json.loads((cache / "cache.json").read_text())
        notes = data.get("notes", "")
        m_sim = re.search(r"sim=([\d.]+)s", notes)
        m_mesh = re.search(r"mesh=([\d.]+)s", notes)
        sim_s = float(m_sim.group(1)) if m_sim else 0.0
        mesh_s = float(m_mesh.group(1)) if m_mesh else 0.0
        return sim_s, mesh_s
    except Exception as e:
        print(f"[warn] could not parse cache.json notes: {e}")
        return 0.0, 0.0


def build_scene(cache: Path, label: str, fluid_color: tuple,
                fps: int, n_frames: int,
                sim_s: float, mesh_s: float):
    sc = bpy.data.scenes.get("fluidcube_render") or bpy.data.scenes.new("fluidcube_render")
    bpy.context.window.scene = sc
    for o in list(sc.objects):
        sc.collection.objects.unlink(o)
    for nm in [n for n in bpy.data.materials.keys() if n.startswith("fluidcube_")]:
        bpy.data.materials.remove(bpy.data.materials[nm])
    for nm in [n for n in bpy.data.meshes.keys() if n.startswith("fluidcube_")]:
        bpy.data.meshes.remove(bpy.data.meshes[nm])

    sc.frame_start = 1
    sc.frame_end = n_frames
    sc.render.fps = fps
    sc.render.resolution_x = 1600
    sc.render.resolution_y = 900
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except (TypeError, ValueError):
        sc.render.engine = "BLENDER_EEVEE"
    sc.render.image_settings.file_format = "PNG"

    cam_data = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_data)
    sc.collection.objects.link(cam)
    # Pull back so the full vertical trajectory is in frame: inflow zone
    # at the top, cube in the middle, floor at the bottom. 35mm wide enough
    # to keep the overlay text visible at the edges.
    cam.location = (2.5, -2.5, 1.5)
    target = mathutils.Vector((0.0, 0.0, 0.25))
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

    world = bpy.data.worlds.get("fluidcube_world") or bpy.data.worlds.new("fluidcube_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.03, 0.06, 0.10, 1.0)
        bg.inputs["Strength"].default_value = 0.6
    sc.world = world

    # Pivot rotates the mesh so the sim's Y-up matches Blender's Z-up.
    pivot = bpy.data.objects.new("fluidcube_pivot", None)
    pivot.rotation_euler = (math.radians(90), 0, 0)
    pivot.location = (-0.5, 0.0, 0.0)
    sc.collection.objects.link(pivot)

    # Surface mesh — coloured per-fluid.
    surf_me = bpy.data.meshes.new("fluidcube_surface")
    surf = bpy.data.objects.new("fluidcube_surface", surf_me)
    sc.collection.objects.link(surf)
    surf.parent = pivot
    fluid_mat = bpy.data.materials.new("fluidcube_fluid")
    fluid_mat.use_nodes = True
    fbsdf = fluid_mat.node_tree.nodes.get("Principled BSDF")
    if fbsdf:
        fbsdf.inputs["Base Color"].default_value = (*fluid_color, 1.0)
        # Per-fluid look: honey is glossy + transmissive, oil is half-glossy,
        # water is highly transmissive.
        if label.lower() == "water":
            fbsdf.inputs["Roughness"].default_value = 0.05
            trans = 0.7
        elif label.lower() == "oil":
            fbsdf.inputs["Roughness"].default_value = 0.15
            trans = 0.4
        else:  # honey
            fbsdf.inputs["Roughness"].default_value = 0.20
            trans = 0.6
        if "Transmission Weight" in fbsdf.inputs:
            fbsdf.inputs["Transmission Weight"].default_value = trans
        elif "Transmission" in fbsdf.inputs:
            fbsdf.inputs["Transmission"].default_value = trans
        if "IOR" in fbsdf.inputs:
            fbsdf.inputs["IOR"].default_value = 1.45
    surf.data.materials.append(fluid_mat)

    # Cube obstacle — TOML places it at center=(0.5, 0.4, 0.5) with half_size=0.15.
    # In sim-space the domain is (0,0,0)..(1,1,1) and the pivot rotates Y->Z,
    # then translates -0.5 on X. So the cube ends up at world ~(0.0, 0.0, 0.4).
    bpy.ops.mesh.primitive_cube_add(size=0.30,
                                    location=(0.0, 0.0, 0.4))
    cube = bpy.context.active_object
    cube.name = "fluidcube_obstacle"
    cube.parent = pivot
    # Cancel the pivot's Y-up rotation on the cube so it stays axis-aligned
    # in world frame regardless of the pivot transform.
    cube.rotation_euler = (math.radians(-90), 0, 0)
    cube.location = (0.5, 0.5, 0.4)  # in pivot-local sim coords
    cmat = bpy.data.materials.new("fluidcube_cube_mat")
    cmat.use_nodes = True
    cbsdf = cmat.node_tree.nodes.get("Principled BSDF")
    if cbsdf:
        cbsdf.inputs["Base Color"].default_value = (0.55, 0.45, 0.35, 1.0)
        cbsdf.inputs["Roughness"].default_value = 0.7
    cube.data.materials.append(cmat)

    # Ground plane.
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "fluidcube_ground"
    gmat = bpy.data.materials.new("fluidcube_ground_mat")
    gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    if gb:
        gb.inputs["Base Color"].default_value = (0.07, 0.07, 0.09, 1.0)
        gb.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(gmat)

    # ---------- Title + sim-time overlay (top of frame).
    overlay_top = bpy.data.objects.new("fluidcube_overlay_top", None)
    overlay_top.parent = cam
    overlay_top.location = (-0.42, 0.22, -1.0)
    sc.collection.objects.link(overlay_top)

    # Label uses fluid colour for visual continuity.
    _add_text("fluidcube_t_label", label,
              (0.0, 0.00, 0.0), scale=0.06, parent=overlay_top,
              color=(*fluid_color, 1.0))
    # Sim time is updated per-frame by the FrameLoader callback.
    sim_time_text = _add_text("fluidcube_t_simtime", "sim time:  0.000 s",
                              (0.0, -0.09, 0.0), scale=0.045, parent=overlay_top,
                              color=(0.95, 0.97, 1.00, 1.0))

    # ---------- Wall-clock overlay (bottom of frame).
    overlay_bot = bpy.data.objects.new("fluidcube_overlay_bot", None)
    overlay_bot.parent = cam
    overlay_bot.location = (-0.42, -0.25, -1.0)
    sc.collection.objects.link(overlay_bot)

    total_s = sim_s + mesh_s
    _add_text("fluidcube_t_wall1",
              f"wall-clock  sim: {sim_s:.1f}s   mesh: {mesh_s:.1f}s",
              (0.0, 0.00, 0.0), scale=0.035, parent=overlay_bot,
              color=(0.85, 0.85, 0.95, 1.0))
    _add_text("fluidcube_t_wall2",
              f"total bake: {total_s:.1f}s  for {n_frames} frames @ {fps}fps",
              (0.0, -0.06, 0.0), scale=0.035, parent=overlay_bot,
              color=(0.70, 0.85, 0.70, 1.0))

    return sc, surf, sim_time_text


class FrameLoader:
    def __init__(self, cache: Path, surf, sim_time_text, fps: int):
        self.cache = cache
        self.surf = surf
        self.sim_time_text = sim_time_text
        self.fps = fps

    def __call__(self, scene, depsgraph=None):
        f = scene.frame_current
        idx = f - 1
        # Update sim-time overlay: this frame represents (idx / fps) seconds
        # of simulated time. Counter ticks visibly through the clip.
        t = idx / float(self.fps)
        self.sim_time_text.data.body = f"sim time: {t:6.3f} s"
        # Swap the surface mesh for the baked frame's PLY.
        ply_path = self.cache / "mesh" / f"frame_{idx:04d}.ply"
        if ply_path.exists():
            verts, faces = _read_ply_minimal(str(ply_path))
            _rebuild_surface_mesh(self.surf, verts, faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True,
                    help="Water / Oil / Honey — drives fluid material + overlay text")
    ap.add_argument("--color", nargs=3, type=float, required=True,
                    metavar=("R", "G", "B"),
                    help="Fluid colour (RGB in [0,1])")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args(_argv_after_doubledash())
    cache = Path(args.cache).resolve()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    sim_s, mesh_s = _parse_wallclock_from_cache_json(cache)
    print(f"[render] {args.label}: cache={cache}  "
          f"sim={sim_s:.1f}s mesh={mesh_s:.1f}s")

    sc, surf, sim_time_text = build_scene(
        cache, args.label, tuple(args.color),
        args.fps, args.frames, sim_s, mesh_s,
    )
    loader = FrameLoader(cache, surf, sim_time_text, args.fps)
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(loader)

    sc.render.filepath = str(out_dir / "f_")
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
