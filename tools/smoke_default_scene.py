"""Headless default-scene smoke — permanent cert step (§9.13).

Run:  blender -b --factory-startup -P tools/smoke_default_scene.py
Exit: 0 = all scenarios baked >0 frames; 1 = any scenario failed.

Promoted from the 2026-06-10 audit, where two runs caught two production
bugs all three code reviewers had missed (sphere-source TOML emit
KeyError 'lo'; get_prefs crash under manual register()). Run before
every merge touching addon defaults, operators, or the bake pipeline.
Cache output goes to tmp/smoke_* (gitignored).

Builds the scene a first-time user would: Add Domain, mark a sphere as
Fluid, mark a cube as Obstacle — defaults untouched except frames/res
(kept tiny for speed) and cache_dir (tmp). Then bakes sync and asserts
frames actually exist on disk. Scenario B switches the source to MESH —
locks the audit-20260610 cache_dir UnboundLocalError fix live.
"""
import os
import shutil
import sys
import traceback

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "addon"))

import gpufluid_blender  # noqa: E402

FAILURES = []


def _fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _build_default_scene(cache_dir, source_type):
    # what a user does: Add Domain button
    bpy.ops.gpufluid.add_domain()
    domain = next(o for o in bpy.context.scene.objects
                  if getattr(o, "gpufluid_domain", None)
                  and o.gpufluid_domain.is_domain)
    d = domain.gpufluid_domain
    d.cache_dir = cache_dir
    d.frames = 8          # tiny for smoke speed; everything else OOTB
    d.resolution = 48

    # fluid source: a sphere, like the docs suggest
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.6, location=(0.0, 0.0, 1.2))
    src = bpy.context.active_object
    bpy.ops.gpufluid.mark_fluid()
    if source_type != "BBOX":
        src.gpufluid_fluid.source_type = source_type

    # obstacle: a cube under the stream
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.8))
    bpy.ops.gpufluid.mark_obstacle()


def _count_frames(cache_dir):
    mesh_dir = os.path.join(cache_dir, "mesh")
    if not os.path.isdir(mesh_dir):
        return 0
    return len([f for f in os.listdir(mesh_dir) if f.endswith(".ply")])


def run_scenario(name, source_type):
    cache_dir = os.path.join(REPO, "tmp", f"smoke_{name}")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    _fresh_scene()
    _build_default_scene(cache_dir, source_type)
    try:
        # OOTB: no interpreter_path set anywhere -> the c5834f1 auto-detect
        # must find the repo-adjacent .venv on its own.
        rv = bpy.ops.gpufluid.bake(sync=True, sync_timeout_sec=600)
    except Exception:
        FAILURES.append(f"{name}: bake raised:\n{traceback.format_exc()}")
        return
    n = _count_frames(cache_dir)
    print(f"[smoke] {name}: bake -> {rv}, frames on disk: {n}")
    if "FINISHED" not in rv or n == 0:
        FAILURES.append(f"{name}: rv={rv}, frames={n} (expected FINISHED + >0)")


def main():
    gpufluid_blender.register()
    run_scenario("sphere_default", "SPHERE")
    run_scenario("mesh_source", "MESH")
    if FAILURES:
        print("[smoke] FAILURES:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("[smoke] ALL GREEN")
    sys.exit(0)


main()
