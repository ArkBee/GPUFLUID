"""FU-035 — headless renderer camera auto-frames the whole domain.

The renderer's camera target + distance used to be hardcoded for a centred
scene, so tall / off-centre content (a waterfall column at sim y~0.95) clipped
the top edge. ``frame_pose_for_box`` now derives the pose from the domain box;
these tests pin the framing invariant (every corner inside the FOV cone) on
plain geometry — no bpy/Blender needed.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path

import pytest

_CAM = Path(__file__).resolve().parents[1] / "examples" / "_render_camera.py"
_spec = importlib.util.spec_from_file_location("_render_camera_under_test", _CAM)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
frame_pose_for_box = _mod.frame_pose_for_box

# canonical renderer settings
VIEW = (2.5, -2.5, 1.25)
SENSOR, LENS, RX, RY = 36.0, 35.0, 1600, 900


def _unit_cube():
    return [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]


def _corner_angles(pose, corners):
    """Angle (rad) of each corner off the camera->centroid view axis."""
    loc, cen = pose["location"], pose["centroid"]
    axis = tuple(cen[i] - loc[i] for i in range(3))
    alen = math.sqrt(sum(a * a for a in axis))
    out = []
    for c in corners:
        v = tuple(c[i] - loc[i] for i in range(3))
        vlen = math.sqrt(sum(t * t for t in v))
        dot = sum(v[i] * axis[i] for i in range(3))
        out.append(math.acos(max(-1.0, min(1.0, dot / (vlen * alen)))))
    return out


def test_unit_domain_centroid_and_radius():
    pose = frame_pose_for_box(_unit_cube(), VIEW, SENSOR, LENS, RX, RY)
    assert pose["centroid"] == pytest.approx((0.5, 0.5, 0.5))
    assert pose["radius"] == pytest.approx(math.sqrt(0.75))  # half space-diagonal


def test_every_corner_inside_fov_cone():
    corners = _unit_cube()
    pose = frame_pose_for_box(corners, VIEW, SENSOR, LENS, RX, RY)
    half_fov = pose["half_fov"]
    angles = _corner_angles(pose, corners)
    # FU-035 invariant: nothing clips — every corner within the limiting FOV.
    assert max(angles) <= half_fov + 1e-9, (
        f"FU-035: corner at {math.degrees(max(angles)):.2f} deg exceeds "
        f"half-FOV {math.degrees(half_fov):.2f} deg")
    # margin>1 means strict head-room, not a frame-filling tangent.
    assert max(angles) < half_fov, "FU-035: margin should leave head-room"


def test_distance_matches_bounding_sphere_formula():
    pose = frame_pose_for_box(_unit_cube(), VIEW, SENSOR, LENS, RX, RY, margin=1.18)
    expect = pose["radius"] / math.sin(pose["half_fov"]) * 1.18
    assert pose["distance"] == pytest.approx(expect)
    # camera sits along the view direction from the centroid, at that distance.
    cen, loc = pose["centroid"], pose["location"]
    off = math.sqrt(sum((loc[i] - cen[i]) ** 2 for i in range(3)))
    assert off == pytest.approx(pose["distance"])


def test_tall_offcentre_box_still_framed():
    """A box NOT at the domain centre and taller than wide (the exact case the
    hardcoded camera clipped) must still have every corner inside the cone."""
    corners = [(x, y, z) for x in (0.30, 0.55)
               for y in (0.10, 0.95) for z in (0.30, 0.55)]
    pose = frame_pose_for_box(corners, VIEW, SENSOR, LENS, RX, RY)
    angles = _corner_angles(pose, corners)
    assert max(angles) <= pose["half_fov"] + 1e-9
    # centroid actually tracks the content, not a hardcoded point.
    assert pose["centroid"][1] == pytest.approx(0.525)


def test_limiting_fov_is_the_smaller_axis():
    """16:9 render → vertical FOV is the binding constraint; the helper must
    pick the smaller half-FOV or a wide-but-short box would clip top/bottom."""
    pose = frame_pose_for_box(_unit_cube(), VIEW, SENSOR, LENS, RX, RY)
    half_h = math.atan((SENSOR * 0.5) / LENS)
    half_v = math.atan((SENSOR * 0.5 * (RY / RX)) / LENS)
    assert pose["half_fov"] == pytest.approx(min(half_h, half_v))
    assert half_v < half_h  # sanity: landscape => vertical is tighter
