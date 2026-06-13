"""Pure camera-framing geometry for the headless renderer (no bpy).

Split out of ``render_fluid_on_cube_eevee.py`` so the FU-035 auto-framing
maths is unit-testable outside Blender. The renderer keeps the thin bpy
wrapper (builds the world-space domain corners via the pivot matrix, then
applies the pose returned here to the camera object).
"""
from __future__ import annotations
import math


def frame_pose_for_box(corners, view_dir, sensor_w, lens, rx, ry,
                       margin=1.18):
    """Given 8 world-space box ``corners`` as (x,y,z) tuples and a
    (need-not-be-normalised) ``view_dir``, return the camera pose that frames
    every corner from that direction.

    Returns a dict with ``location`` + ``centroid`` (look-at) plus the
    ``radius``/``distance``/``half_fov`` used. The box's bounding sphere is
    bounded inside the limiting (smaller of horizontal/vertical) half-FOV with
    ``margin`` head-room, so no corner can clip regardless of where the content
    sits in the domain (FU-035: the old hardcoded target/distance clipped tall
    or off-centre scenes).
    """
    if not corners:
        raise ValueError("frame_pose_for_box: need at least one corner")
    n = len(corners)
    cen = (sum(c[0] for c in corners) / n,
           sum(c[1] for c in corners) / n,
           sum(c[2] for c in corners) / n)
    radius = max(((c[0] - cen[0]) ** 2 + (c[1] - cen[1]) ** 2
                  + (c[2] - cen[2]) ** 2) ** 0.5 for c in corners)
    half_h = math.atan((sensor_w * 0.5) / lens)
    half_v = math.atan((sensor_w * 0.5 * (ry / rx)) / lens)
    half_fov = min(half_h, half_v)
    # radius==0 (degenerate single-point box) would divide fine but give 0
    # distance; clamp so the camera never lands on the subject.
    dist = max(radius, 1e-6) / math.sin(half_fov) * margin
    vlen = (view_dir[0] ** 2 + view_dir[1] ** 2 + view_dir[2] ** 2) ** 0.5
    u = (view_dir[0] / vlen, view_dir[1] / vlen, view_dir[2] / vlen)
    loc = (cen[0] + u[0] * dist, cen[1] + u[1] * dist, cen[2] + u[2] * dist)
    return {"location": loc, "centroid": cen, "radius": radius,
            "distance": dist, "half_fov": half_fov}
