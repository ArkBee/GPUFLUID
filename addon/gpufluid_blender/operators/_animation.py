"""[BLK A8.5] Animation / bbox helpers extracted from operators/bake.py.

Split out in Phase 4 to keep bake.py under the 500-line cap. Two clusters
live here:

* ``_bbox_world`` and ``_bbox_world_at_frame`` — world-space AABB of any
  object, optionally evaluated at a specific frame via F-curve sampling.
* ``_is_animated``, ``_iter_fcurves``, ``_matrix_world_at_frame`` — the
  fcurve-only animation evaluator that powers MPM inflow keyframe
  emission without paying ``scene.frame_set``'s depsgraph-rebuild cost
  per frame.

All functions are deliberately bpy-leaf so they can be unit-tested with
a stubbed ``mathutils`` if/when needed; for now they only run inside a
live Blender session called by ``operators.bake.collect_scene``.
"""
from __future__ import annotations

import bpy
import mathutils


def _bbox_world(obj) -> tuple:
    """Return (lo, hi) world-space AABB of an object.

    For mesh-like objects uses Blender's ``bound_box``. For an Empty the
    `bound_box` is degenerate (all zeros), so we derive the AABB from
    ``empty_display_size`` (a half-extent) and the object's transform.
    """
    if obj.type == "EMPTY":
        # Empty visualises as a unit shape of half-extent empty_display_size.
        d = obj.empty_display_size
        local = [mathutils.Vector(c) for c in (
            (-d, -d, -d), ( d, -d, -d), ( d,  d, -d), (-d,  d, -d),
            (-d, -d,  d), ( d, -d,  d), ( d,  d,  d), (-d,  d,  d),
        )]
        pts = [obj.matrix_world @ v for v in local]
    else:
        pts = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _is_animated(obj, context=None):
    """Animated = has F-curves on self OR any ancestor in the parent chain.

    Earlier Phase 2 replaced the original f-curve check with a depsgraph
    probe (scene.frame_set ×2 + restore). It was correct but cost two
    full depsgraph rebuilds *per inflow per bake* — measurable regression
    on 5+ inflow scenes. We revert to F-curve scanning and additionally
    walk the parent chain, which fixes the original bug (parent-keyframed
    location was being missed). Constraints/drivers without backing
    F-curves remain a known limitation — documented in BACKLOG.md.

    ``context`` accepted for API parity with earlier callers; unused.
    """
    del context  # signature parity; F-curve scan is context-free
    o = obj
    seen = set()
    while o is not None and id(o) not in seen:
        seen.add(id(o))
        ad = getattr(o, "animation_data", None)
        if ad is not None:
            act = getattr(ad, "action", None)
            for _ in _iter_fcurves(act):
                return True
        o = getattr(o, "parent", None)
    return False


def _iter_fcurves(action):
    """Yield F-curves from an action across Blender 4.x and 5.x layouts.

    Blender 5.x slot-based actions expose curves under
    ``action.layers[*].strips[*].channelbags[*].fcurves``; legacy actions
    still have ``action.fcurves``. We try both so the addon works on
    either runtime.
    """
    if action is None:
        return
    if hasattr(action, "fcurves") and len(action.fcurves):
        for fc in action.fcurves:
            yield fc
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for cbag in getattr(strip, "channelbags", []):
                for fc in cbag.fcurves:
                    yield fc


def _matrix_world_at_frame(obj, frame):
    """Compose obj.matrix_world at integer ``frame`` WITHOUT
    scene.frame_set — evaluates the F-curves of obj (and its parent
    chain) directly. Avoids the 50-100ms-per-frame depsgraph rebuild
    that ``scene.frame_set`` triggers.

    Falls back to current ``obj.matrix_world`` if the object has no
    action (constraints aren't covered — those need depsgraph).
    """
    def _local_matrix(o):
        loc = list(o.location)
        rot = list(o.rotation_euler)
        scl = list(o.scale)
        act = o.animation_data.action if o.animation_data else None
        for fc in _iter_fcurves(act):
            dp = fc.data_path
            i = fc.array_index
            v = fc.evaluate(frame)
            if dp == "location" and 0 <= i < 3:
                loc[i] = v
            elif dp == "rotation_euler" and 0 <= i < 3:
                rot[i] = v
            elif dp == "scale" and 0 <= i < 3:
                scl[i] = v
        m_loc = mathutils.Matrix.Translation(loc)
        rot_mode = o.rotation_mode if o.rotation_mode in {
            "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"} else "XYZ"
        m_rot = mathutils.Euler(rot, rot_mode).to_matrix().to_4x4()
        m_scl = mathutils.Matrix.Diagonal((scl[0], scl[1], scl[2], 1.0))
        return m_loc @ m_rot @ m_scl
    # Walk parent chain bottom-up, compose top-down.
    chain = []
    cur = obj
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    m = mathutils.Matrix.Identity(4)
    for o in reversed(chain):
        m = m @ _local_matrix(o)
    return m


def _bbox_world_at_frame(obj, frame):
    """AABB of ``obj`` at integer ``frame`` via fcurve-evaluated matrix."""
    mw = _matrix_world_at_frame(obj, frame)
    if obj.type == "EMPTY":
        d = obj.empty_display_size
        local = [mathutils.Vector(c) for c in (
            (-d, -d, -d), ( d, -d, -d), ( d,  d, -d), (-d,  d, -d),
            (-d, -d,  d), ( d, -d,  d), ( d,  d,  d), (-d,  d,  d),
        )]
        pts = [mw @ v for v in local]
    else:
        pts = [mw @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
