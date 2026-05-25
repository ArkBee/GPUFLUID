"""[Layer A8.9-A8.12] Headless Blender Eevee render bridge.

Three reusable helpers used both by the addon Bake operator (A8.5) and
by the standalone CLI command ``gpufluid render`` (A8.12):

  A8.9  — :func:`apply_eevee_preset` — explicit perf tuning for headless
          renders. Without it Blender uses photo-quality defaults (~3×
          slower).
  A8.10 — :class:`FrameMeshLoader` — frame-change handler that reads
          ``cache/mesh/frame_NNNN.ply`` (via I6.1) and rebuilds the
          surface object's vertex/face buffers in-place. Vectorised
          face parse (I6.1.MESH) makes the read <2ms per 10k-face mesh.
  A8.11 — :func:`rebuild_surface_mesh` — utility to swap vertex/face
          buffers on an existing Blender mesh object via the fastest
          available API (``foreach_set``).

These helpers must be importable both inside a Blender Python process
(when called from operators) AND inside a regular Python process
(for unit tests via ``bpy`` if present, or via ``unittest.mock`` if
not). Imports of ``bpy`` are deferred to function bodies so the module
loads cleanly outside Blender.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ─── A8.11 mesh rebuild ────────────────────────────────────────────────

def rebuild_surface_mesh(obj: Any, verts: np.ndarray, faces: np.ndarray) -> None:
    """Replace ``obj.data`` geometry with the given vert/face arrays.

    Uses :py:func:`bpy.types.Mesh.foreach_set` for the fastest possible
    bulk write — ~10× faster than ``from_pydata`` for >1000-vert meshes.

    Parameters
    ----------
    obj : ``bpy.types.Object``
    verts : (N, 3) float32-castable
    faces : (M, 3) int32-castable
    """
    me = obj.data
    me.clear_geometry()
    n_v = len(verts)
    n_f = len(faces)
    if n_v == 0 or n_f == 0:
        return
    me.vertices.add(n_v)
    me.vertices.foreach_set("co", np.asarray(verts, dtype=np.float32).ravel())
    loop_total = n_f * 3
    me.loops.add(loop_total)
    me.polygons.add(n_f)
    me.polygons.foreach_set(
        "loop_start", np.arange(0, loop_total, 3, dtype=np.int32))
    me.polygons.foreach_set(
        "loop_total", np.full(n_f, 3, dtype=np.int32))
    me.polygons.foreach_set(
        "vertices", np.asarray(faces, dtype=np.int32).ravel())
    me.update(calc_edges=True)


# ─── A8.10 frame loader ────────────────────────────────────────────────

class FrameMeshLoader:
    """Frame-change handler: load PLY for current frame into a surface mesh.

    Install via::

        loader = FrameMeshLoader(cache_dir, surf_obj)
        bpy.app.handlers.frame_change_pre.clear()
        bpy.app.handlers.frame_change_pre.append(loader)

    Reads ``cache_dir/mesh/frame_NNNN.ply`` for the current scene frame.
    Uses :func:`gpufluid.io.ply.read_ply` (which includes the I6.1.MESH
    vectorised face parse — ~50× faster than naïve PLY parse).

    Optionally updates a text object's ``data.body`` with the sim time
    derived from ``frame / fps``.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        surf_obj: Any,
        sim_time_text: Any | None = None,
        fps: int = 60,
    ):
        self.cache_dir = Path(cache_dir)
        self.surf = surf_obj
        self.sim_time_text = sim_time_text
        self.fps = fps

    def __call__(self, scene: Any, depsgraph: Any | None = None) -> None:
        # Defer the heavy import so this module loads outside Blender.
        from gpufluid.io.ply import read_ply
        f = scene.frame_current
        idx = f - 1
        if self.sim_time_text is not None:
            t = idx / float(self.fps)
            self.sim_time_text.data.body = f"sim time: {t:6.3f} s"
        ply_path = self.cache_dir / "mesh" / f"frame_{idx:04d}.ply"
        if not ply_path.exists():
            return
        verts, faces = read_ply(str(ply_path))
        rebuild_surface_mesh(self.surf, verts, faces)


# ─── A8.9 Eevee perf preset ────────────────────────────────────────────

def apply_eevee_preset(scene: Any, samples: int = 16) -> dict:
    """Configure the Blender scene's Eevee renderer for headless speed.

    Default settings in Blender Eevee are tuned for photo-quality. For
    a simple fluid+cube scene the photo-quality defaults are ~3× slower
    than necessary. This helper:

      * Sets TAA samples to ``samples`` (default 16; valid range 4-64).
      * Disables bloom, screen-space reflections, GTAO, volumetric
        lights — none of these are useful for an opaque fluid mesh.
      * Lowers shadow ray/step counts if those attributes exist
        (Blender Eevee Next).

    Idempotent. Returns a dict logging which attrs were actually set
    so tests can verify across Blender version differences.
    """
    log: dict[str, Any] = {}
    ee = getattr(scene, "eevee", None)
    if ee is None:
        return log
    for attr in ("taa_render_samples", "taa_samples"):
        if hasattr(ee, attr):
            setattr(ee, attr, samples)
            log[attr] = samples
    for attr in ("use_bloom", "use_ssr", "use_ssr_refraction",
                 "use_gtao", "use_volumetric_lights"):
        if hasattr(ee, attr):
            setattr(ee, attr, False)
            log[attr] = False
    if hasattr(ee, "shadow_ray_count"):
        ee.shadow_ray_count = 1
        log["shadow_ray_count"] = 1
    if hasattr(ee, "shadow_step_count"):
        ee.shadow_step_count = 1
        log["shadow_step_count"] = 1
    return log
