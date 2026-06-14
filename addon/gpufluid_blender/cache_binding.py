"""[Round-19] Cache-binding helpers — central source for inter-op data passing.

Senior architect code-smell #4 (last of the five day-1+code-smell list).
Round 1-18 had ~10 magic-string custom-property keys scattered across
`bake.py`, `render.py`, `cache_loader/_ops.py`, `cache_loader/__init__.py`.
Three subtle name groups co-existed:

  - **cache binding** on the gpufluid_cache mesh object — set by
    OT_attach_cache, read by frame_change_handler:
    `gpufluid_cache_dir`, `gpufluid_cache_pattern`,
    `gpufluid_cache_frame_offset`, `gpufluid_cache_origin`,
    `gpufluid_cache_dom_size`.
  - **whitewater binding** on the gpufluid_whitewater object — set
    by OT_attach_ww_cache: `gpufluid_ww_cache_dir`,
    `gpufluid_ww_cache_frame_offset`, `gpufluid_ww_cache_origin`.
  - **bake-trace** on the Domain — set by OT_bake for sync auto-
    attach + post-bake sanity. Deliberately DIFFERENT name set from
    cache binding (one shared name `gpufluid_cache_dir` + two
    different `gpufluid_origin` / `gpufluid_dom_size` — NOT
    `_cache_origin` / `_cache_dom_size`).

Round-5 reviewer (commit 49dfb0c) found a missing-key bug in
OT_detach_cache via exactly this name-set confusion. Round-19
centralises every key string here + provides typed get/set/clear
helpers so consumers can't accidentally re-introduce the same trap.

**Backward compatibility:** key strings UNCHANGED from rounds 1-18.
.blend files saved before round-19 keep working — this module is a
rename of magic strings to named constants, not a format migration.
"""
from __future__ import annotations

from typing import Optional, Tuple

Vec3 = Tuple[float, float, float]


# ─── Key constants (the ONLY place these strings appear) ───────────────

# Cache binding (gpufluid_cache mesh obj)
KEY_CACHE_DIR           = "gpufluid_cache_dir"
KEY_CACHE_PATTERN       = "gpufluid_cache_pattern"
KEY_CACHE_FRAME_OFFSET  = "gpufluid_cache_frame_offset"
KEY_CACHE_ORIGIN        = "gpufluid_cache_origin"
KEY_CACHE_DOM_SIZE      = "gpufluid_cache_dom_size"

# Whitewater binding (gpufluid_whitewater obj)
KEY_WW_CACHE_DIR        = "gpufluid_ww_cache_dir"
KEY_WW_CACHE_FRAME_OFFSET = "gpufluid_ww_cache_frame_offset"
KEY_WW_CACHE_ORIGIN     = "gpufluid_ww_cache_origin"
# audit-2026-06-14 #3: whitewater positions are in the SAME normalised [0,1]^3
# space as the surface mesh, so they MUST be scaled by dom_size too — this key
# was missing, so foam/spray collapsed into a 1 m cube on any non-unit domain.
KEY_WW_CACHE_DOM_SIZE   = "gpufluid_ww_cache_dom_size"

# Bake-trace binding (on Domain Empty — note: NOT prefixed
# `_cache_origin`/`_cache_dom_size` like the cache-binding set).
# This deliberate asymmetry is why round-5 caught a missing-key bug.
KEY_BAKE_TRACE_CACHE_DIR = "gpufluid_cache_dir"   # shared name, different obj
KEY_BAKE_TRACE_ORIGIN    = "gpufluid_origin"      # NOT _cache_origin
KEY_BAKE_TRACE_DOM_SIZE  = "gpufluid_dom_size"    # NOT _cache_dom_size


# Tuples for bulk operations (detach iterates these).
ALL_CACHE_KEYS = (
    KEY_CACHE_DIR, KEY_CACHE_PATTERN, KEY_CACHE_FRAME_OFFSET,
    KEY_CACHE_ORIGIN, KEY_CACHE_DOM_SIZE,
)
ALL_WW_KEYS = (
    KEY_WW_CACHE_DIR, KEY_WW_CACHE_FRAME_OFFSET, KEY_WW_CACHE_ORIGIN,
    KEY_WW_CACHE_DOM_SIZE,
)
ALL_BAKE_TRACE_KEYS = (
    KEY_BAKE_TRACE_CACHE_DIR, KEY_BAKE_TRACE_ORIGIN, KEY_BAKE_TRACE_DOM_SIZE,
)


# ─── Read helpers (return Optional / typed default) ─────────────────────

def get_cache_dir(obj) -> Optional[str]:
    """Read cache_dir from a gpufluid_cache obj. Returns None if not bound."""
    v = obj.get(KEY_CACHE_DIR)
    return str(v) if v is not None else None


def get_cache_pattern(obj, default: str = "mesh/frame_{:04d}.ply") -> str:
    return str(obj.get(KEY_CACHE_PATTERN, default))


def get_cache_frame_offset(obj, default: int = 0) -> int:
    return int(obj.get(KEY_CACHE_FRAME_OFFSET, default))


def get_cache_origin(obj, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    v = obj.get(KEY_CACHE_ORIGIN, default)
    return (float(v[0]), float(v[1]), float(v[2]))


def get_cache_dom_size(obj, default: Vec3 = (1.0, 1.0, 1.0)) -> Vec3:
    v = obj.get(KEY_CACHE_DOM_SIZE, default)
    return (float(v[0]), float(v[1]), float(v[2]))


def has_cache_binding(obj) -> bool:
    """True iff this object has been .attach()'d (cache_dir prop present)."""
    return obj.get(KEY_CACHE_DIR) is not None


# Whitewater equivalents
def get_ww_cache_dir(obj) -> Optional[str]:
    v = obj.get(KEY_WW_CACHE_DIR)
    return str(v) if v is not None else None


def get_ww_cache_frame_offset(obj, default: int = 0) -> int:
    return int(obj.get(KEY_WW_CACHE_FRAME_OFFSET, default))


def get_ww_cache_origin(obj, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    v = obj.get(KEY_WW_CACHE_ORIGIN, default)
    return (float(v[0]), float(v[1]), float(v[2]))


def get_ww_cache_dom_size(obj, default: Vec3 = (1.0, 1.0, 1.0)) -> Vec3:
    v = obj.get(KEY_WW_CACHE_DOM_SIZE, default)
    return (float(v[0]), float(v[1]), float(v[2]))


def has_ww_binding(obj) -> bool:
    return obj.get(KEY_WW_CACHE_DIR) is not None


# Bake-trace (on Domain) — returned as dict because all three fields
# travel together (caller passes them to attach_cache as one batch).
def get_bake_trace(obj) -> Optional[dict]:
    """Read the Domain's bake-trace props (set by OT_bake._finish).
    Returns dict {cache_dir, origin, dom_size} or None if no bake yet."""
    cd = obj.get(KEY_BAKE_TRACE_CACHE_DIR)
    if cd is None:
        return None
    origin = obj.get(KEY_BAKE_TRACE_ORIGIN, (0.0, 0.0, 0.0))
    dom_size = obj.get(KEY_BAKE_TRACE_DOM_SIZE, (1.0, 1.0, 1.0))
    return {
        "cache_dir": str(cd),
        "origin": (float(origin[0]), float(origin[1]), float(origin[2])),
        "dom_size": (float(dom_size[0]), float(dom_size[1]), float(dom_size[2])),
    }


# ─── Write helpers ────────────────────────────────────────────────────

def set_cache_binding(
    obj,
    cache_dir: str,
    origin: Vec3,
    dom_size: Vec3,
    pattern: str = "mesh/frame_{:04d}.ply",
    frame_offset: int = 0,
) -> None:
    """Write all 5 cache-binding props in one call. Used by OT_attach_cache."""
    obj[KEY_CACHE_DIR] = str(cache_dir)
    obj[KEY_CACHE_PATTERN] = pattern
    obj[KEY_CACHE_FRAME_OFFSET] = int(frame_offset)
    obj[KEY_CACHE_ORIGIN] = list(origin)
    obj[KEY_CACHE_DOM_SIZE] = list(dom_size)


def set_ww_binding(
    obj, cache_dir: str, origin: Vec3, frame_offset: int = 0,
    dom_size: Vec3 = (1.0, 1.0, 1.0),
) -> None:
    obj[KEY_WW_CACHE_DIR] = str(cache_dir)
    obj[KEY_WW_CACHE_FRAME_OFFSET] = int(frame_offset)
    obj[KEY_WW_CACHE_ORIGIN] = list(origin)
    # audit-2026-06-14 #3: persist dom_size so the rebuild scales normalised
    # whitewater positions to world like the surface mesh does.
    obj[KEY_WW_CACHE_DOM_SIZE] = list(dom_size)


def set_bake_trace(obj, cache_dir: str, origin: Vec3, dom_size: Vec3) -> None:
    """Write the 3 bake-trace props on Domain. Used by OT_bake._finish
    to record what the sync auto-attach needs."""
    obj[KEY_BAKE_TRACE_CACHE_DIR] = str(cache_dir)
    obj[KEY_BAKE_TRACE_ORIGIN] = list(origin)
    obj[KEY_BAKE_TRACE_DOM_SIZE] = list(dom_size)


# ─── Clear helpers ───────────────────────────────────────────────────

def clear_all_bindings(obj) -> bool:
    """Drop every cache/ww/bake-trace key from obj. Returns True if
    anything was present. Used by OT_detach_cache to scrub one object."""
    cleared = False
    for k in ALL_CACHE_KEYS + ALL_WW_KEYS + ALL_BAKE_TRACE_KEYS:
        if k in obj.keys():
            del obj[k]
            cleared = True
    return cleared
