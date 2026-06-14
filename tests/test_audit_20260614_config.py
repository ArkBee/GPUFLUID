"""Audit 2026-06-14 — cli/config.py validation holes (findings #8, #9, #10).

Found by the adversarial bug-hunt workflow; each was confirmed 3/3 skeptics.
All three are "passes silently / crashes opaquely" cases that the named-field
validation (`_tuple`) was supposed to catch but didn't.
"""
from __future__ import annotations

import pytest

from gpufluid.blocks import BlockError
from gpufluid.cli.config import _tuple, load_scene


# ---- #8: a quoted scalar string must NOT pass as an N-vector --------------

def test_tuple_rejects_3char_string():
    # "0.5" is a typing.Sequence of len 3 -> used to become ('0','.','5')
    with pytest.raises(BlockError, match="expected list of 3"):
        _tuple("0.5", 3, "obstacle.center")


def test_tuple_rejects_bytes():
    with pytest.raises(BlockError):
        _tuple(b"abc", 3, "x")


def test_tuple_still_accepts_a_real_list():
    assert _tuple([0.1, 0.2, 0.3], 3, "x") == (0.1, 0.2, 0.3)


# ---- end-to-end via load_scene -------------------------------------------

_BASE = """
[domain]
resolution = [16, 16, 16]
[[fluids]]
type = "box"
lo = [0.1, 0.1, 0.1]
hi = [0.3, 0.3, 0.3]
[simulation]
solver = "mpm"
dt = 0.005
frames = 2
fps = 24
[output]
cache_dir = "out/x"
"""


def _scene(tmp_path, extra=""):
    p = tmp_path / "s.toml"
    p.write_text(_BASE + extra, encoding="utf-8")
    return p


def test_resolution_float_is_coerced_to_int(tmp_path):
    # #9: [16.0, 16.0, 16.0] must land as int grid cell counts
    p = tmp_path / "s.toml"
    p.write_text(_BASE.replace("[16, 16, 16]", "[16.0, 16.0, 16.0]"),
                 encoding="utf-8")
    scene = load_scene(p)
    assert scene.domain.resolution == (16, 16, 16)
    assert all(type(v) is int for v in scene.domain.resolution)


def test_obstacle_center_string_gives_named_error(tmp_path):
    # #8 end-to-end: center = "0.5" -> named BlockError, not garbage tuple
    p = _scene(tmp_path, '\n[[obstacle]]\ntype = "sphere"\n'
                         'center = "0.5"\nradius = 0.1\n')
    with pytest.raises(BlockError, match="obstacle.center"):
        load_scene(p)


def test_obstacle_rotation_scalar_gives_named_error(tmp_path):
    # #10: rotation = 1.0 -> BlockError("must be 3×3"), not bare TypeError
    p = _scene(tmp_path, '\n[[obstacle]]\ntype = "box"\n'
                         'center = [0.5, 0.5, 0.5]\nhalf_size = [0.1, 0.1, 0.1]\n'
                         'rotation = 1.0\n')
    with pytest.raises(BlockError, match="3×3"):
        load_scene(p)
