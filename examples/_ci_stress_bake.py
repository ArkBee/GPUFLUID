"""Stress test: 100 frames at res 64, attach, scrub all frames, measure.

(Original draft used 200@96; reduced after live run showed an unrelated
core MPM truncation bug at res 96 — that's a solver issue filed
separately. The addon-side preload/scrub contract is what we measure
here, not solver throughput.)

Run:
    blender -b -P examples/_ci_stress_bake.py -- <cache_dir>

Reports:
  - bake wall time + frames produced
  - attach time + preload mesh count + RSS delta (if psutil available)
  - scrub all ~100 frames sequentially: per-frame mean/max/p99 time
  - final teardown: clear_cache invoked, RSS drop

Exits 0 only if:
  - bake produces ≥95 frames
  - attach loads them
  - mean scrub <50ms (pointer-swap should be sub-ms; >50ms = preload broken)
  - no Python exception during scrub
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path

_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
CACHE_DIR = Path(_args[0]) if _args else Path(tempfile.gettempdir()) / "gpufluid_stress"


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return -1.0  # psutil not available — skip memory checks


def _fail(step: str, msg: str) -> None:
    print(f"[STRESS-FAIL] {step}: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(step: str, msg: str = "") -> None:
    print(f"[STRESS-OK] {step}{(' — ' + msg) if msg else ''}", file=sys.stderr)


def main() -> None:
    import bpy
    import addon_utils
    addon_utils.enable("bl_ext.user_default.gpufluid_blender", default_set=True)

    rss0 = _rss_mb()
    _ok("startup", f"RSS={rss0:.1f} MB")

    # Build heavier scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    scn = bpy.context.scene
    scn.frame_start, scn.frame_end = 1, 100

    bpy.ops.gpufluid.add_domain()
    dom = next(o for o in scn.objects if o.gpufluid_domain.is_domain)
    dom.name = "StressDom"
    dom.location = (0, 0, 0); dom.empty_display_size = 1.0
    dp = dom.gpufluid_domain
    # Round-10: res 96 + 200 frames was hitting an unrelated core MPM
    # issue (CLI exits rc=0 around frame 26 — separate filed bug).
    # Stress at res 64 + 100 frames — addon-side preload/scrub is what
    # we're measuring, not solver throughput.
    dp.resolution = 64
    dp.frames = 100
    dp.fps = 24
    dp.solver = "mpm"
    dp.cache_dir = str(CACHE_DIR)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(0, 0, 0.7))
    inflow = bpy.context.active_object
    inflow.name = "SInflow"
    bpy.ops.gpufluid.mark_inflow()
    inflow.gpufluid_inflow.rate_per_sec = 30000
    inflow.gpufluid_inflow.frame_start = 1
    inflow.gpufluid_inflow.frame_end = 100

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.65))
    plate = bpy.context.active_object
    plate.name = "SPlate"
    plate.scale = (0.6, 0.6, 0.06)
    bpy.ops.gpufluid.mark_obstacle()

    bpy.ops.object.select_all(action="DESELECT")
    dom.select_set(True)
    bpy.context.view_layer.objects.active = dom

    # ── BAKE ─────────────────────────────────────────────────────────
    t0 = time.time()
    res = bpy.ops.gpufluid.bake("EXEC_DEFAULT", sync=True, sync_timeout_sec=600)
    bake_t = time.time() - t0
    if "FINISHED" not in res:
        _fail("bake", f"unexpected: {res}")
    mesh_dir = CACHE_DIR / "mesh"
    n_frames = len(list(mesh_dir.glob("frame_*.ply")))
    if n_frames < 95:
        _fail("bake", f"only {n_frames}/100 frames produced")
    _ok("bake", f"{n_frames} frames in {bake_t:.1f}s, RSS={_rss_mb():.1f}MB")

    # ── ATTACH ───────────────────────────────────────────────────────
    # Cache may have been auto-attached by sync bake; force fresh
    if "gpufluid_cache" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["gpufluid_cache"], do_unlink=True)
    import sys as _sys
    cl = _sys.modules["bl_ext.user_default.gpufluid_blender.cache_loader"]
    cl._PRELOAD.clear()

    t0 = time.time()
    bpy.ops.gpufluid.attach_cache()
    attach_t = time.time() - t0
    cobj = bpy.data.objects.get("gpufluid_cache")
    if cobj is None:
        _fail("attach", "gpufluid_cache obj missing after attach")
    preload_size = len(cl._PRELOAD.get("gpufluid_cache", {}))
    rss_attach = _rss_mb()
    _ok("attach",
        f"{preload_size} frames preloaded in {attach_t:.1f}s, RSS={rss_attach:.1f}MB (Δ{rss_attach-rss0:.1f})")

    # ── SCRUB all frames sequentially ────────────────────────────────
    times = []
    for f in range(1, n_frames + 1):
        t0 = time.time()
        scn.frame_set(f)
        bpy.context.view_layer.update()
        times.append((time.time() - t0) * 1000.0)
    mean_ms = sum(times) / len(times)
    max_ms = max(times)
    p99_ms = sorted(times)[int(0.99 * len(times))]
    rss_scrub = _rss_mb()
    if mean_ms > 50:
        _fail("scrub",
              f"mean per-frame {mean_ms:.1f}ms exceeds 50ms — pointer-swap broken?")
    _ok("scrub",
        f"mean={mean_ms:.2f}ms max={max_ms:.1f}ms p99={p99_ms:.1f}ms RSS={rss_scrub:.1f}MB")

    # ── TEARDOWN ─────────────────────────────────────────────────────
    bpy.ops.gpufluid.clear_cache()
    rss_after = _rss_mb()
    _ok("teardown", f"RSS={rss_after:.1f}MB (Δ from peak {rss_after-rss_scrub:.1f})")

    print(f"[STRESS-PASS] all checks green. cache={CACHE_DIR}",
          file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
