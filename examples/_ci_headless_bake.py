"""Headless CI smoke test for the gpufluid_blender addon.

Run from any shell:
    blender -b -P examples/_ci_headless_bake.py -- <cache_dir>

What it does (in order, fails-fast on first error with rc=1):

  1. Enable the addon (bl_ext.user_default.gpufluid_blender).
  2. Build a minimal scene: Empty Domain at origin, sphere Inflow,
     box Obstacle.
  3. Sync-bake (`bpy.ops.gpufluid.bake('EXEC_DEFAULT', sync=True)`)
     and verify cache.json + non-zero mesh frames appear.
  4. Sync-render (`bpy.ops.gpufluid.render(...)`) and verify N PNGs.
  5. Exit 0 on full success, 1 on any step failure.

This is the proof that the addon is usable from render-farm /
CI workflows where the Blender event loop never ticks between
operator calls.

Not a pytest — pytest can't drive `bpy` outside Blender. Invoke
from a wrapper test if needed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Skip Blender's own argv parsing (everything before --)
_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
CACHE_DIR = Path(_args[0]) if _args else Path(tempfile.gettempdir()) / "gpufluid_ci_test"

# Round-24: WIPE the cache dir before bake. Pre-round-24 a stale dir
# from a previous successful run could satisfy the `n_mesh >= 6` /
# `n_png >= 6` count checks even if the current bake/render produced
# 0 new files. CI-green-for-wrong-reason (lesson 9.2). The
# `FINISHED` op-result guards most cases, but a Blender that aborts
# mid-bake without raising would still slip through.
import shutil
if CACHE_DIR.exists():
    try:
        shutil.rmtree(CACHE_DIR)
    except OSError as _e:
        print(f"[CI-WARN] could not wipe {CACHE_DIR}: {_e}", file=sys.stderr)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fail(step: str, msg: str) -> None:
    print(f"[CI-FAIL] {step}: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(step: str, msg: str = "") -> None:
    print(f"[CI-OK]   {step}{(' — ' + msg) if msg else ''}", file=sys.stderr)


def main() -> None:
    import bpy
    import addon_utils

    addon_name = "bl_ext.user_default.gpufluid_blender"

    # ── Step 1: enable addon ────────────────────────────────────────────
    try:
        addon_utils.enable(addon_name, default_set=True)
        loaded_default, loaded_state = addon_utils.check(addon_name)
        if not loaded_state:
            _fail("enable_addon", f"check() says not loaded: {loaded_state}")
    except Exception as exc:
        _fail("enable_addon", f"exception: {exc}\n{traceback.format_exc()}")
    _ok("enable_addon", addon_name)

    # ── Step 2: build minimal scene ──────────────────────────────────────
    # Strip default scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scn = bpy.context.scene
    scn.frame_start = 1
    scn.frame_end = 6
    scn.frame_set(1)

    # Domain via the addon's add_domain op
    bpy.ops.gpufluid.add_domain()
    dom = next((o for o in scn.objects if o.gpufluid_domain.is_domain), None)
    if dom is None:
        _fail("scene_build", "add_domain did not create a domain")
    dom.name = "CIDomain"
    dom.location = (0, 0, 0)
    dom.empty_display_size = 1.0
    dp = dom.gpufluid_domain
    dp.resolution = 48
    dp.frames = 6
    dp.fps = 24
    dp.solver = "mpm"
    dp.cache_dir = str(CACHE_DIR)
    dp.mpm_initial_velocity = -1.0

    # Inflow sphere at top
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(0, 0, 0.7))
    inflow = bpy.context.active_object
    inflow.name = "CIInflow"
    bpy.ops.gpufluid.mark_inflow()
    inflow.gpufluid_inflow.rate_per_sec = 25000
    inflow.gpufluid_inflow.frame_start = 1
    inflow.gpufluid_inflow.frame_end = 6
    inflow.gpufluid_inflow.use_color = True
    inflow.gpufluid_inflow.color = (0.3, 0.7, 0.95)

    # Plate obstacle at bottom
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.6))
    plate = bpy.context.active_object
    plate.name = "CIPlate"
    plate.scale = (0.6, 0.6, 0.08)
    bpy.ops.gpufluid.mark_obstacle()
    _ok("scene_build",
        f"Domain={dom.name} Inflow={inflow.name} Obstacle={plate.name}")

    # ── Step 3: sync bake ────────────────────────────────────────────────
    bpy.ops.object.select_all(action="DESELECT")
    dom.select_set(True)
    bpy.context.view_layer.objects.active = dom

    bake_res = bpy.ops.gpufluid.bake("EXEC_DEFAULT", sync=True)
    if "FINISHED" not in bake_res:
        _fail("sync_bake", f"unexpected result: {bake_res}")

    cache_json = CACHE_DIR / "cache.json"
    if not cache_json.exists():
        _fail("sync_bake", f"cache.json missing in {CACHE_DIR}")
    mesh_dir = CACHE_DIR / "mesh"
    n_mesh = len(list(mesh_dir.glob("frame_*.ply"))) if mesh_dir.is_dir() else 0
    if n_mesh < 6:
        _fail("sync_bake",
              f"expected ≥6 mesh frames, got {n_mesh} in {mesh_dir}")
    _ok("sync_bake", f"{n_mesh} mesh frames + cache.json")

    # ── Step 4: sync render ──────────────────────────────────────────────
    render_out = CACHE_DIR / "ci_render"
    render_res = bpy.ops.gpufluid.render(
        "EXEC_DEFAULT",
        sync=True,
        samples=4,                         # min quality, max speed
        label="CI",
        color=(0.3, 0.7, 0.95),
        blender_path=bpy.app.binary_path,  # use this very Blender
        out_subdir=str(render_out),
    )
    if "FINISHED" not in render_res:
        _fail("sync_render", f"unexpected result: {render_res}")
    n_png = len(list(render_out.glob("*.png")))
    if n_png < 6:
        _fail("sync_render",
              f"expected ≥6 PNG frames, got {n_png} in {render_out}")
    _ok("sync_render", f"{n_png} PNG frames")

    print(f"[CI-PASS] all steps green — addon usable headless. cache={CACHE_DIR}",
          file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
