"""step25 — lava demo (B11.3) single-pane Eevee renderer.

Renders the lava bake (`examples/scenes/lava_drop.toml`) by driving the
per-vertex surface colour AND emission strength from the per-particle
temperature scalar (S2.18). Each surface vertex picks up the temperature
of its nearest fluid particle (cdist nearest-neighbour, same pattern as
step24's nearest-particle colour transfer), then a piecewise-linear
"blackbody-ish" colormap maps T → RGB:

    T = 1500 K (hot lava)   →  near-white, strong emission
    T = 1000 K              →  bright orange
    T =  600 K              →  deep red
    T =  300 K (crust)      →  near-black, low emission

So when the hot drop hits the cool basin you see the contact zone cool
visibly as the P2G→G2P pass mixes the two reservoirs — the visual is
driven entirely by the new per-particle scalar pipeline.

Run from inside Blender (not the project venv):

    blender --background --python examples/render_step25_eevee.py -- \\
            --cache out/step25_lava \\
            --out   out/step25_eevee_frames

Then stitch to mp4:

    .venv/Scripts/python -c "import imageio.v2 as io, glob; \\
        w = io.get_writer('out/videos/step25.mp4', fps=24, codec='libx264', \\
                          quality=8, macro_block_size=1); \\
        [w.append_data(io.imread(p)) for p in sorted(glob.glob('out/step25_eevee_frames/f_*.png'))]; \\
        w.close()"
"""
from __future__ import annotations
import argparse
import math
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


# ---------------------------------------------------------------- lava colormap
# Piecewise-linear "blackbody-ish" ramp. Anchor points in (T_K, R, G, B).
# Linear interpolation between anchors; clamps at the ends.
_LAVA_ANCHORS = np.array([
    [ 300.0, 0.04, 0.02, 0.02],    # crust
    [ 600.0, 0.55, 0.05, 0.03],    # deep red
    [ 900.0, 0.95, 0.25, 0.05],    # red-orange
    [1200.0, 1.00, 0.65, 0.10],    # bright orange
    [1500.0, 1.00, 0.95, 0.65],    # near-white hot
], dtype=np.float32)


def _temperature_to_rgb(temps: np.ndarray) -> np.ndarray:
    """Vectorised piecewise-linear lookup on `_LAVA_ANCHORS`. Returns
    (N, 3) float32 RGB in [0, 1]. Clamped to the anchor endpoints."""
    t = temps.astype(np.float32)
    out = np.empty((t.shape[0], 3), dtype=np.float32)
    ts = _LAVA_ANCHORS[:, 0]
    cs = _LAVA_ANCHORS[:, 1:]
    # Below first anchor / above last anchor: clamp.
    out[:] = cs[0]
    out[t >= ts[-1]] = cs[-1]
    # Interior segments.
    for i in range(len(ts) - 1):
        lo, hi = ts[i], ts[i + 1]
        mask = (t >= lo) & (t < hi)
        if not mask.any():
            continue
        u = ((t[mask] - lo) / (hi - lo))[:, None]
        out[mask] = (1.0 - u) * cs[i] + u * cs[i + 1]
    return out


# ---------------------------------------------------------------- nearest-particle temperature
def _vert_temps_from_particles(verts: np.ndarray, particles: np.ndarray,
                                temps: np.ndarray, batch: int = 1024) -> np.ndarray:
    """For each mesh vertex, return the temperature of its nearest fluid
    particle. Memory-bounded by the V×P chunk size.

    Falls back to a uniform "warm crust" value (600 K) when either side
    is empty so the material still has something to read."""
    V = verts.shape[0]
    if V == 0:
        return np.zeros(0, dtype=np.float32)
    if particles.shape[0] == 0:
        return np.full(V, 600.0, dtype=np.float32)
    out = np.zeros(V, dtype=np.float32)
    p2 = (particles ** 2).sum(axis=1)
    for i in range(0, V, batch):
        v = verts[i:i + batch]
        v2 = (v ** 2).sum(axis=1)
        sq = v2[:, None] - 2.0 * (v @ particles.T) + p2[None, :]
        out[i:i + batch] = temps[np.argmin(sq, axis=1)]
    return out


# ---------------------------------------------------------------- mesh rebuild
def _rebuild_surface_mesh(obj, verts: np.ndarray, faces: np.ndarray,
                           vert_rgb: np.ndarray, vert_emit: np.ndarray):
    """Stamp positions + topology, plus TWO POINT attributes:
      - `lavacol`      FLOAT_COLOR — hot→cold ramp from temperature.
      - `lavaemit`     FLOAT       — emission strength in [0, 1]
                                      (drives the BSDF Emission Strength via
                                       a multiplier in the material).
    """
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

    # FLOAT_COLOR — base colour ramp.
    cattr = me.attributes.get("lavacol")
    if cattr is None or cattr.domain != "POINT" or cattr.data_type != "FLOAT_COLOR":
        if cattr is not None:
            me.attributes.remove(cattr)
        cattr = me.attributes.new(name="lavacol", type="FLOAT_COLOR", domain="POINT")
    rgba = np.concatenate([vert_rgb, np.ones((vert_rgb.shape[0], 1), dtype=np.float32)], axis=1)
    cattr.data.foreach_set("color", rgba.ravel())

    # FLOAT — emission strength (hotter = brighter glow).
    eattr = me.attributes.get("lavaemit")
    if eattr is None or eattr.domain != "POINT" or eattr.data_type != "FLOAT":
        if eattr is not None:
            me.attributes.remove(eattr)
        eattr = me.attributes.new(name="lavaemit", type="FLOAT", domain="POINT")
    eattr.data.foreach_set("value", vert_emit.ravel())


# ---------------------------------------------------------------- scene build
def build_scene(cache: Path):
    sc = bpy.data.scenes.get("step25_render") or bpy.data.scenes.new("step25_render")
    bpy.context.window.scene = sc
    for o in list(sc.objects):
        sc.collection.objects.unlink(o)
    for nm in [n for n in bpy.data.materials.keys() if n.startswith("step25_")]:
        bpy.data.materials.remove(bpy.data.materials[nm])
    for nm in [n for n in bpy.data.meshes.keys() if n.startswith("step25_")]:
        bpy.data.meshes.remove(bpy.data.meshes[nm])

    sc.frame_start = 1
    sc.frame_end = 90
    sc.render.fps = 24
    sc.render.resolution_x = 1600
    sc.render.resolution_y = 900
    # Prefer Eevee Next (Blender 4.2+ / 5.x) — falls back to legacy Eevee
    # so the script still works on 4.1 boxes. The engine ID changed; the
    # render-settings API for the bits we touch did not.
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except (TypeError, ValueError):
        sc.render.engine = "BLENDER_EEVEE"
    sc.render.image_settings.file_format = "PNG"

    # Camera, slightly lower so the splash sits in frame.
    cam_data = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_data)
    sc.collection.objects.link(cam)
    cam.location = (1.8, -1.8, 1.4)
    target = mathutils.Vector((0.0, 0.2, 0.3))
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 35
    sc.camera = cam

    # Lighting — keep it MUTED so the lava's own emission dominates.
    key = bpy.data.lights.new("Key", "SUN"); key.energy = 1.2
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.rotation_euler = (math.radians(60), math.radians(15), math.radians(-30))
    sc.collection.objects.link(key_obj)
    fill_d = bpy.data.lights.new("Fill", "AREA"); fill_d.energy = 20.0
    fill_d.size = 2.0
    fill = bpy.data.objects.new("Fill", fill_d)
    fill.location = (-2.5, -2.0, 2.0)
    fill.rotation_euler = (math.radians(60), 0, math.radians(45))
    sc.collection.objects.link(fill)

    # World — near-black so the lava emission reads.
    world = bpy.data.worlds.get("step25_world") or bpy.data.worlds.new("step25_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.01, 0.01, 0.02, 1.0)
        bg.inputs["Strength"].default_value = 0.3
    sc.world = world

    # Sim-Y is up — rotate +90° around X at the pivot, like step24.
    pivot = bpy.data.objects.new("step25_pivot", None)
    pivot.rotation_euler = (math.radians(90), 0, 0)
    sc.collection.objects.link(pivot)
    pivot.location = (-0.5, 0.0, 0.0)

    # Surface mesh — drives BOTH Base Color and Emission via vertex attrs.
    m = bpy.data.meshes.new("step25_surface")
    surf = bpy.data.objects.new("step25_surface", m)
    sc.collection.objects.link(surf)
    surf.parent = pivot
    mat = bpy.data.materials.new("step25_lava")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 0.45
        # Base Color ← lavacol attribute
        col_attr = nodes.new("ShaderNodeAttribute")
        col_attr.attribute_name = "lavacol"
        col_attr.location = (-500, 200)
        links.new(col_attr.outputs["Color"], bsdf.inputs["Base Color"])
        # Emission Color ← same ramp (so glow matches surface tone)
        if "Emission Color" in bsdf.inputs:
            links.new(col_attr.outputs["Color"], bsdf.inputs["Emission Color"])
        elif "Emission" in bsdf.inputs:
            links.new(col_attr.outputs["Color"], bsdf.inputs["Emission"])
        # Emission Strength ← lavaemit * 6 (so peak-hot vertices glow strongly)
        emit_attr = nodes.new("ShaderNodeAttribute")
        emit_attr.attribute_name = "lavaemit"
        emit_attr.location = (-500, -50)
        mul = nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 6.0
        mul.location = (-250, -50)
        links.new(emit_attr.outputs["Fac"], mul.inputs[0])
        if "Emission Strength" in bsdf.inputs:
            links.new(mul.outputs["Value"], bsdf.inputs["Emission Strength"])
    surf.data.materials.append(mat)

    # Obstacle sphere — hard-coded from lava_drop.toml.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, segments=32, ring_count=16,
                                         location=(0.50, 0.30, 0.50))
    obs = bpy.context.active_object
    obs.name = "step25_obstacle"
    obs.parent = pivot
    bpy.ops.object.shade_smooth()
    omat = bpy.data.materials.new("step25_obstacle_mat")
    omat.use_nodes = True
    ob = omat.node_tree.nodes.get("Principled BSDF")
    if ob:
        # Cool grey rock — lit only by lava glow + dim key.
        ob.inputs["Base Color"].default_value = (0.18, 0.18, 0.22, 1.0)
        ob.inputs["Roughness"].default_value = 0.7
    obs.data.materials.append(omat)

    # Ground plane (slight reference)
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "step25_ground"
    gmat = bpy.data.materials.new("step25_ground_mat")
    gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    if gb:
        gb.inputs["Base Color"].default_value = (0.04, 0.03, 0.03, 1.0)
        gb.inputs["Roughness"].default_value = 0.95
    ground.data.materials.append(gmat)

    # Eevee bloom / glare — make the hot peaks bloom over the dark
    # background. Legacy Eevee exposed `use_bloom`; Eevee Next removed it
    # in favour of the Compositor "Glare" node OR per-shader bloom under
    # `use_raytracing` + glow strength. Set what we can; missing knobs are
    # OK — Eevee Next still renders the emissive surface fine without it.
    eevee = getattr(sc, "eevee", None)
    if eevee is not None:
        if hasattr(eevee, "use_bloom"):
            eevee.use_bloom = True
            if hasattr(eevee, "bloom_intensity"):
                eevee.bloom_intensity = 0.15
        # Eevee Next prefers ray-tracing on for nicer reflections off the
        # obstacle. Cheap at 1600x900 / 90 frames.
        if hasattr(eevee, "use_raytracing"):
            eevee.use_raytracing = True

    return sc, surf


# ---------------------------------------------------------------- per-frame hook
class FrameLoader:
    def __init__(self, cache: Path, surf_obj):
        self.cache = cache
        self.surf = surf_obj
        # Anchors used both for colormap interp and emission normalisation.
        self.t_min = float(_LAVA_ANCHORS[0, 0])
        self.t_max = float(_LAVA_ANCHORS[-1, 0])

    def __call__(self, scene, depsgraph=None):
        f = scene.frame_current
        idx = f - 1
        ply_path = self.cache / "mesh" / f"frame_{idx:04d}.ply"
        if not ply_path.exists():
            return
        verts, faces = _read_ply_minimal(str(ply_path))
        part_path = self.cache / "particles" / f"frame_{idx:04d}.npy"
        temp_path = self.cache / "temperatures" / f"frame_{idx:04d}.npy"
        if part_path.exists() and temp_path.exists() and verts.shape[0] > 0:
            part = np.load(part_path).astype(np.float32)
            temp = np.load(temp_path).astype(np.float32)
            # Robustness: if the dump has a different particle count than
            # the position file (shouldn't, but be defensive), fall back to
            # the warm-crust value so we don't crash the render.
            if temp.shape[0] != part.shape[0]:
                temp = np.full(part.shape[0], 600.0, dtype=np.float32)
            vt = _vert_temps_from_particles(verts, part, temp)
        else:
            vt = np.full(verts.shape[0], 600.0, dtype=np.float32)
        rgb = _temperature_to_rgb(vt)
        # Emission scales as ((T - T_min) / (T_max - T_min))^1.5 so cool
        # crust stays nearly dark while hot core glows strongly.
        norm = np.clip((vt - self.t_min) / (self.t_max - self.t_min), 0.0, 1.0)
        emit = (norm ** 1.5).astype(np.float32)
        _rebuild_surface_mesh(self.surf, verts, faces, rgb, emit)


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
