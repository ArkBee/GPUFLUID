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
            --scene examples/scenes/step30_water_on_cube.toml \\
            --out   out/step30_eevee_frames \\
            --label "Water" \\
            --color 0.20 0.50 0.70

Each fluid pass uses a different label/colour but otherwise identical
camera, lighting, ground, cube obstacle so the three videos compare
directly.

Iteration mode (for tuning passes — render 24 frames spread across the
bake instead of all 600, ~25x faster):

    blender --background --python examples/render_fluid_on_cube_eevee.py -- \\
            --cache out/step30_water_on_cube \\
            --scene examples/scenes/step30_water_on_cube.toml \\
            --out   C:/out/step30_test \\
            --label "Water" --color 0.20 0.50 0.70 \\
            --test-frames 24

Output files are named test_fXXXX.png where XXXX is the sim-frame index.
"""
from __future__ import annotations
import argparse, json, math, re, sys, tomllib
from pathlib import Path

import bpy
import mathutils
import numpy as np

# Make gpufluid + the addon importable when running inside Blender. The
# example lives next to addon/ and src/, so walk up to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
# Include examples/ (this script's own dir) so sibling `_render_camera`
# resolves whether run via the headless driver OR `blender --python` directly.
for p in (_REPO_ROOT / "src", _REPO_ROOT / "addon", _REPO_ROOT / "examples"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Library helpers — promoted from inline copies (B17.8 / A8.9 / A8.10 / A8.11)
from gpufluid.io.ply import read_ply  # noqa: E402
from gpufluid_blender.render_bridge import (  # noqa: E402
    apply_eevee_preset,
    FrameMeshLoader,
    rebuild_surface_mesh,
)
from _render_camera import frame_pose_for_box  # noqa: E402  pure FU-035 maths


def _load_obstacles_from_scene(scene_toml: Path) -> list[dict]:
    """Read EVERY [[obstacle]] from a scene TOML and return the raw dicts
    (sim-space geometry, untouched). Renderer authoritative source: TOML,
    never hardcoded — otherwise bake collides with invisible obstacles at
    sim coords while the renderer draws something elsewhere, producing a
    positional desync that looks like "chaotic" fluid to the viewer.

    FU-033: pre-fix this read only ``obs_list[0]`` and RAISED
    ``"only box obstacles are renderable here"`` on anything non-box — so
    after FU-019/FU-030 a sphere/cylinder/mesh obstacle baked correctly
    but CRASHED the render, and multi-obstacle scenes only ever drew
    obstacle #0. Now: return ALL entries of ANY type, no raise.
    ``build_scene`` spawns the matching primitive for each. An
    obstacle-free scene legitimately returns ``[]``.
    """
    data = tomllib.loads(scene_toml.read_text())
    return list(data.get("obstacle", []))


def _obb_rows_to_quat(rows):
    """FU-033: convert an OBB rotation matrix (3 row-vectors, sim space,
    orthonormal — round-57 schema) to a mathutils.Quaternion.

    `bake.py` emits ``rotation`` as the rows of the world→local rotation R
    such that a sim-axis-aligned half-box rotated by R is the visible box.
    mathutils.Matrix() takes ROW vectors, matching that layout, so we
    build it directly and read its quaternion. Returns identity for a None
    / malformed value (defensive — a box with no rotation is a plain AABB).

    Pure (no bpy beyond mathutils, which is CPU-importable) so the
    conversion is unit-testable without Blender.
    """
    if not rows:
        return mathutils.Quaternion()
    try:
        m = mathutils.Matrix((tuple(rows[0]), tuple(rows[1]), tuple(rows[2])))
        return m.to_quaternion()
    except (TypeError, ValueError, IndexError):
        return mathutils.Quaternion()


def _import_obstacle_obj(path: Path, idx: int):
    """FU-033: import a mesh-obstacle OBJ, tolerant of the operator rename.

    Blender 4.0+ ships ``bpy.ops.wm.obj_import``; older builds only have
    ``bpy.ops.import_scene.obj``. Returns the imported object (the single
    new selected object), or None on any failure (caller skips + warns).
    """
    before = set(bpy.data.objects)
    try:
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:  # pragma: no cover — legacy Blender
            bpy.ops.import_scene.obj(filepath=str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] FU-033: obj_import raised for obstacle {idx}: {exc}")
        return None
    new = [o for o in bpy.data.objects if o not in before]
    if not new:
        return None
    # The OBJ may import as multiple objects; if so, the first mesh is the
    # body. Keep it simple — return the first new object.
    return new[0]


def _argv_after_doubledash():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]


def _read_ply_minimal(path):
    # Thin wrapper around library reader (I6.1 + I6.1.MESH vectorized face parse).
    return read_ply(path)


def _rebuild_surface_mesh(obj, verts, faces):
    # Thin wrapper around library helper (A8.11).
    rebuild_surface_mesh(obj, verts, faces)


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


def _frame_camera_to_domain(cam, pivot, sc, margin: float = 1.18):
    """FU-035: aim + pull the camera back so the whole sim [0,1]^3 domain
    (mapped to world by ``pivot``) fits the frame, from the canonical high
    3/4 view.

    The target/distance used to be hardcoded for a typical centred scene, so
    a tall or off-centre body (e.g. a waterfall column reaching sim y~0.95)
    clipped the top edge — every non-tuned scene framed wrong (demo30 hero v1
    spilled into the corner). Deriving the frame from the domain box fixes
    any scene with no per-scene tuning. Thin bpy wrapper over
    :func:`_render_camera.frame_pose_for_box`.
    """
    import itertools
    # An unparented empty's world matrix == its basis (translate @ rotate);
    # compute it explicitly so we don't depend on a depsgraph update yet.
    pivot_mw = (mathutils.Matrix.Translation(pivot.location)
                @ pivot.rotation_euler.to_matrix().to_4x4())
    corners = [tuple(pivot_mw @ mathutils.Vector(c))
               for c in itertools.product((0.0, 1.0), repeat=3)]
    pose = frame_pose_for_box(
        corners, (2.5, -2.5, 1.25), cam.data.sensor_width, cam.data.lens,
        max(1, sc.render.resolution_x), max(1, sc.render.resolution_y), margin)
    cam.location = mathutils.Vector(pose["location"])
    cam.rotation_euler = (
        mathutils.Vector(pose["centroid"]) - cam.location
    ).to_track_quat('-Z', 'Y').to_euler()


def build_scene(cache: Path, label: str, fluid_color: tuple,
                fps: int, n_frames: int,
                sim_s: float, mesh_s: float,
                obstacles: list,
                samples: int = 16, material: str = "water"):
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

    # A8.9 Eevee perf preset (single line — library handles the version drift)
    apply_eevee_preset(sc, samples=samples)

    # Pivot maps the sim's Y-up [0,1]^3 to Blender's Z-up world. Created here
    # (before the camera) so the camera can be framed to the domain box.
    pivot = bpy.data.objects.new("fluidcube_pivot", None)
    pivot.rotation_euler = (math.radians(90), 0, 0)
    pivot.location = (-0.5, 0.0, 0.0)
    sc.collection.objects.link(pivot)

    cam_data = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_data)
    sc.collection.objects.link(cam)
    cam.data.lens = 35
    # FU-035: frame the whole domain from the canonical high 3/4 angle instead
    # of a hardcoded target/distance (which clipped tall/off-centre scenes).
    _frame_camera_to_domain(cam, pivot, sc)
    sc.camera = cam

    key = bpy.data.lights.new("Key", "SUN"); key.energy = 4.0
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(-25))
    sc.collection.objects.link(key_obj)
    fill_d = bpy.data.lights.new("Fill", "AREA"); fill_d.energy = 110.0
    fill_d.size = 2.0
    fill = bpy.data.objects.new("Fill", fill_d)
    fill.location = (-2.5, -2.0, 2.0)
    fill.rotation_euler = (math.radians(60), 0, math.radians(45))
    sc.collection.objects.link(fill)
    # demo30 (2026-06-13): a TALL water body (e.g. the waterfall curtain) casts
    # itself into shadow and the camera sees the unlit back face — with the old
    # single key + weak ambient it rendered NEAR-BLACK (caught reading the v3
    # frames). A camera-side fill lights the face the camera actually sees.
    fill2_d = bpy.data.lights.new("Fill2", "AREA"); fill2_d.energy = 90.0
    fill2_d.size = 3.0
    fill2 = bpy.data.objects.new("Fill2", fill2_d)
    fill2.location = (2.5, -2.5, 2.2)
    fill2.rotation_euler = (math.radians(55), 0, math.radians(-30))
    sc.collection.objects.link(fill2)

    world = bpy.data.worlds.get("fluidcube_world") or bpy.data.worlds.new("fluidcube_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        # demo30: ambient was (0.03,0.06,0.10)*0.6 — almost no sky fill, so any
        # self-shadowed surface crushed to black. Lift it (brighter + bluer)
        # so shadow sides read as dim water, not holes. Kept low enough not to
        # wash out the lit bench scenes.
        bg.inputs["Color"].default_value = (0.06, 0.11, 0.18, 1.0)
        bg.inputs["Strength"].default_value = 1.6
    sc.world = world

    # Surface mesh — coloured per-fluid.
    surf_me = bpy.data.meshes.new("fluidcube_surface")
    surf = bpy.data.objects.new("fluidcube_surface", surf_me)
    sc.collection.objects.link(surf)
    surf.parent = pivot
    fluid_mat = bpy.data.materials.new("fluidcube_fluid")
    fluid_mat.use_nodes = True
    nt = fluid_mat.node_tree
    fbsdf = nt.nodes.get("Principled BSDF")
    if fbsdf:
        fbsdf.inputs["Base Color"].default_value = (*fluid_color, 1.0)
        # Wire mesh.color_attributes["Col"] into Base Color when the
        # surface has it, so per-vertex colours from inflow (mixbox etc.)
        # actually reach the rendered PNG. Without this the renderer
        # ignores per-vertex data and everything looks --color uniform.
        # Live-found 2026-05-25 (round-3 test #17: multi-source mixbox
        # showed clear red↔blue blending at the mesh level but rendered
        # as flat grey because no Attribute node was wired). The MixRGB
        # node lets us fall back to --color when the mesh has no Col
        # layer (e.g. obstacle-only or temperature-colormap previews
        # that use a different attribute name).
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = "GEOMETRY"
        attr.attribute_name = "Col"
        attr.location = (fbsdf.location.x - 380, fbsdf.location.y)
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        mix.location = (fbsdf.location.x - 180, fbsdf.location.y)
        # factor = Attribute.Alpha — Attribute node sets Alpha=1.0 when
        # the named layer exists, 0.0 when it doesn't. So mix picks
        # per-vertex Col where present, --color elsewhere, no extra logic.
        # Address Mix node sockets by name. With data_type='RGBA' the
        # node exposes Factor + A (color) + B (color) + Result (color).
        # Indexing by int is unstable across Blender versions because
        # multiple typed Factor/A/B sockets coexist hidden by data_type.
        def _named_input(node, name):
            # On 4.x+ duplicate socket names exist (one per data_type);
            # pick the first ENABLED one matching the name.
            for s in node.inputs:
                if s.name == name and s.enabled:
                    return s
            for s in node.inputs:
                if s.name == name:
                    return s
            return None
        def _named_output(node, name):
            for s in node.outputs:
                if s.name == name and s.enabled:
                    return s
            for s in node.outputs:
                if s.name == name:
                    return s
            return None
        sock_factor = _named_input(mix, "Factor")
        sock_a = _named_input(mix, "A")
        sock_b = _named_input(mix, "B")
        sock_result = _named_output(mix, "Result")
        if all([sock_factor, sock_a, sock_b, sock_result]):
            nt.links.new(attr.outputs["Alpha"], sock_factor)
            sock_a.default_value = (*fluid_color, 1.0)
            nt.links.new(attr.outputs["Color"], sock_b)
            nt.links.new(sock_result, fbsdf.inputs["Base Color"])
        else:
            # If the Mix node layout changed, fall back to the original
            # uniform colour — better than a crashing renderer.
            fbsdf.inputs["Base Color"].default_value = (*fluid_color, 1.0)
            nt.nodes.remove(mix)
            nt.nodes.remove(attr)
        # Per-fluid look — FU-032: keyed on the EXPLICIT `material` arg,
        # never on the overlay label. Pre-FU-032 this if/else keyed on
        # `label`, with a honey catch-all `else` — any custom label (the
        # CLI default is "gpufluid"!) silently rendered honey
        # (Transmission 0.6), which is near-BLACK under the headless
        # Eevee preset (SSR/refraction off). main() resolves the
        # precedence: --material wins; else legacy label inference; else
        # water.
        if material == "oil":
            fbsdf.inputs["Roughness"].default_value = 0.15
            trans = 0.4
        elif material == "honey":
            fbsdf.inputs["Roughness"].default_value = 0.20
            trans = 0.6
        else:  # water (the honest default — bright diffuse, trans 0)
            fbsdf.inputs["Roughness"].default_value = 0.05
            trans = 0.0  # TEMP for B16.2 visibility eval
        if "Transmission Weight" in fbsdf.inputs:
            fbsdf.inputs["Transmission Weight"].default_value = trans
        elif "Transmission" in fbsdf.inputs:
            fbsdf.inputs["Transmission"].default_value = trans
        if "IOR" in fbsdf.inputs:
            fbsdf.inputs["IOR"].default_value = 1.45
    surf.data.materials.append(fluid_mat)

    # Obstacles — read from the scene TOML so the rendered geometry tracks
    # whatever the bake actually collided with. (Past bug: hardcoded
    # location had Y/Z swapped, so the visible cube was 10cm off from where
    # the solver placed the invisible obstacle — fluid behaviour looked
    # chaotic.) FU-033: render EVERY obstacle of ANY type, not just box #0.
    #
    # Pivot/rotation contract (round-57/59 §9.6): every primitive is
    # parented to `pivot` (which maps sim Y-up → Blender Z-up) and gets the
    # counter-rotation rot_euler=(-90°,0,0) so it stays axis-aligned in the
    # WORLD frame regardless of the pivot. Any OBB rotation composes ON TOP
    # of that counter-rotation.
    #
    # One shared tan material for all obstacle primitives (minimal).
    cmat = bpy.data.materials.new("fluidcube_cube_mat")
    cmat.use_nodes = True
    cbsdf = cmat.node_tree.nodes.get("Principled BSDF")
    if cbsdf:
        cbsdf.inputs["Base Color"].default_value = (0.55, 0.45, 0.35, 1.0)
        cbsdf.inputs["Roughness"].default_value = 0.7

    _counter = mathutils.Euler((math.radians(-90), 0, 0)).to_quaternion()

    def _finish_obstacle(ob, idx, extra_rot=None):
        """Parent + counter-rotate (+ optional extra rotation) + material."""
        ob.name = f"fluidcube_obstacle_{idx}"
        ob.parent = pivot
        ob.rotation_mode = "QUATERNION"
        # extra_rot is applied in the obstacle's pre-counter sim frame, so
        # compose counter * extra (counter on the left = outermost).
        ob.rotation_quaternion = (
            _counter @ extra_rot if extra_rot is not None else _counter)
        ob.data.materials.append(cmat)

    for idx, obs in enumerate(obstacles):
        otype = obs.get("type", "box")
        if otype == "box":
            bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
            ob = bpy.context.active_object
            # unit cube has half-extent 1.0 (size=2) → scale to half_size.
            hs = obs.get("half_size", (0.05, 0.05, 0.05))
            ob.scale = (float(hs[0]), float(hs[1]), float(hs[2]))
            ob.location = tuple(obs.get("center", (0, 0, 0)))  # pivot-local sim
            # round-57 OBB: rotate the box if a rotation matrix is present.
            extra = _obb_rows_to_quat(obs.get("rotation"))
            _finish_obstacle(ob, idx, extra_rot=extra)
        elif otype == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=float(obs.get("radius", 0.05)), location=(0, 0, 0))
            ob = bpy.context.active_object
            ob.location = tuple(obs.get("center", (0, 0, 0)))
            _finish_obstacle(ob, idx)
        elif otype == "cylinder_y":
            # FU-033 cylinder-axis resolution: primitive_cylinder_add builds
            # the cylinder along its LOCAL Z. The solver's cylinder_y axis is
            # sim-Y. The counter-rotation (-90° about X) ALONE already maps
            # local-Z → sim-Y, which the pivot (+90° about X) then maps to
            # world-Z (vertical) — exactly what cylinder_y wants. So NO
            # extra rotation is needed: the cylinder follows the same
            # parent+counter pattern as the box/sphere. (Verified live: an
            # added Z→Y rotation laid the cylinder on its side; plain
            # counter-rotation stands it upright. Computed: counter@local-Z
            # = sim-Y, pivot@sim-Y = world-Z.)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=float(obs.get("radius", 0.05)),
                depth=2.0 * float(obs.get("half_height", 0.1)),
                location=(0, 0, 0))
            ob = bpy.context.active_object
            ob.location = tuple(obs.get("center", (0, 0, 0)))
            _finish_obstacle(ob, idx)
        elif otype == "mesh":
            path = obs.get("path", "")
            if not path or not Path(path).exists():
                # dodge: the bake writes the obstacle OBJ next to scene.toml;
                # a cache copied/moved WITHOUT its sidecar .obj (or an old
                # cache) leaves a dangling path. Skip with a warning rather
                # than crash the whole render — the other obstacles + fluid
                # still draw. (FU-033: pre-fix any non-box raised here.)
                print(f"[warn] FU-033: obstacle {idx} mesh path missing, "
                      f"skipping: {path!r}")
                continue
            ob = _import_obstacle_obj(Path(path), idx)
            if ob is None:
                print(f"[warn] FU-033: obstacle {idx} OBJ import failed, "
                      f"skipping: {path!r}")
                continue
            # OBJ verts are in WORLD METRES (see addon _export_obj); the bake
            # maps them to sim space via scale (scalar OR per-axis vec3) +
            # translate. Apply the SAME map here so the drawn mesh lands
            # where the solver sampled it.
            scale = obs.get("scale", 1.0)
            if isinstance(scale, (list, tuple)):
                ob.scale = (float(scale[0]), float(scale[1]), float(scale[2]))
            else:
                ob.scale = (float(scale),) * 3
            ob.location = tuple(obs.get("translate", (0, 0, 0)))
            _finish_obstacle(ob, idx)
        else:
            print(f"[warn] FU-033: unknown obstacle type {otype!r} "
                  f"(obstacle {idx}) — skipping")
            continue

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


# FrameLoader was promoted to library — see gpufluid_blender.render_bridge.FrameMeshLoader (A8.10).
FrameLoader = FrameMeshLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--scene", required=True,
                    help="Path to the scene TOML — read for obstacle geometry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True,
                    help="Overlay text drawn into each frame. Legacy: a "
                         "label of water/oil/honey ALSO infers --material "
                         "when that flag is absent (step30-32 demo flows); "
                         "any other label no longer touches the shader "
                         "(FU-032 — it used to fall through to honey).")
    ap.add_argument("--material", choices=["water", "oil", "honey"],
                    default=None,
                    help="Fluid material preset. Precedence (FU-032): this "
                         "flag wins; else inferred from --label when the "
                         "label IS one of water/oil/honey; else water.")
    ap.add_argument("--color", nargs=3, type=float, required=True,
                    metavar=("R", "G", "B"),
                    help="Fluid colour (RGB in [0,1])")
    ap.add_argument("--fps", type=int, default=60)
    # `None` (not 600) so we can detect "user didn't ask for a count" and fall
    # back to cache.json's frame_count. The headless wrapper (A8.12) only
    # forwards --frames when it's a positive int, so this default kicks in for
    # both the CLI `gpufluid render` path and direct invocations.
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--samples", type=int, default=16,
                    help="Eevee TAA samples per frame (A8.9 preset). 16 is "
                         "the speed-tuned default; bump to 64 for hero shots.")
    ap.add_argument("--test-frames", type=int, default=0,
                    help="Iteration mode: render N frames spaced evenly across "
                         "the full bake (e.g. 24 instead of 600). Output names "
                         "are test_fXXXX.png so the sim-frame index stays "
                         "visible. ~25x faster than a full render for picking "
                         "between tuning iterations.")
    args = ap.parse_args(_argv_after_doubledash())
    cache = Path(args.cache).resolve()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    obstacles = _load_obstacles_from_scene(Path(args.scene).resolve())
    sim_s, mesh_s = _parse_wallclock_from_cache_json(cache)

    # Frame count fallback: if --frames was omitted (or 0) read cache.json's
    # frame_count. This is what the headless wrapper relies on — see Bug #2
    # in _headless_render.py.
    n_frames = int(args.frames) if (args.frames and args.frames > 0) else 0
    if n_frames <= 0:
        try:
            cache_meta = json.loads((cache / "cache.json").read_text())
            n_frames = int(cache_meta.get("frame_count", 0))
        except Exception as e:
            print(f"[warn] could not read frame_count from cache.json: {e}")
        if n_frames <= 0:
            n_frames = 600  # last-ditch fallback matches the old hardcoded default
    # FU-033: summarise count + types instead of a single cube's geometry.
    from collections import Counter as _Counter
    _types = _Counter(o.get("type", "box") for o in obstacles)
    _obs_summary = (", ".join(f"{n}×{t}" for t, n in sorted(_types.items()))
                    if obstacles else "none")
    print(f"[render] {args.label}: cache={cache}  "
          f"sim={sim_s:.1f}s mesh={mesh_s:.1f}s  "
          f"obstacles=[{_obs_summary}]  frames={n_frames} samples={args.samples}")

    # FU-032 precedence: explicit --material > legacy label inference
    # (back-compat for the step30-32 demo flows where the label WAS the
    # material) > water. The old behaviour fell through to HONEY for any
    # custom label — near-black with the headless Eevee preset.
    material = args.material or (
        args.label.lower()
        if args.label.lower() in ("water", "oil", "honey") else "water")
    sc, surf, sim_time_text = build_scene(
        cache, args.label, tuple(args.color),
        args.fps, n_frames, sim_s, mesh_s,
        obstacles,
        samples=args.samples, material=material,
    )
    loader = FrameLoader(cache, surf, sim_time_text, args.fps)
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(loader)

    if args.test_frames > 0:
        n = args.test_frames
        last = max(n_frames - 1, 0)
        indices = [round(i * last / max(n - 1, 1)) for i in range(n)]
        print(f"[render] test mode: {n} frames @ indices {indices}")
        for sim_idx in indices:
            sc.frame_current = sim_idx + 1
            loader(sc)
            sc.render.filepath = str(out_dir / f"test_f{sim_idx:04d}")
            bpy.ops.render.render(write_still=True)
    else:
        sc.render.filepath = str(out_dir / "f_")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
