"""Round-59 — TOML writer must propagate R57's OBB rotation key.

§9.6 mirror-drift bug in production: R57 updated `bake.py` to put
`rotation` into the obstacle dict. `config_builder.py` (the
scene-dict -> scene.toml writer) was NOT updated and silently
stripped the key. Symptom on live MPM bake (2026-05-28): tilted
cube in viewport, fluid treated it as AABB because scene.toml
never carried the rotation.

Contract: the `box` branch in `config_builder.py` must emit
`rotation = [[...], [...], [...]]` when the dict carries it, AND
must not crash when absent (legacy AABB obstacles).
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_CB = (_REPO / "addon" / "gpufluid_blender" / "config_builder.py")


def _load_cb():
    spec = importlib.util.spec_from_file_location("addon_cb", _CB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_box_branch_emits_rotation_when_present():
    cb = _load_cb()
    scene = {
        "domain": {"resolution": [64, 64, 64], "dx": 1.0 / 64},
        "obstacles": [{
            "type": "box",
            "center": [0.5, 0.5, 0.5],
            "half_size": [0.1, 0.1, 0.1],
            "rotation": [
                [0.866025, -0.5, 0.0],
                [0.5, 0.866025, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }],
        # Need at least one source so build_scene_toml doesn't raise.
        "inflows": [{
            "lo": [0.45, 0.45, 0.8], "hi": [0.55, 0.55, 0.9],
            "velocity": [0, 0, -0.5],
            "rate_per_sec": 1000.0,
            "frame_start": 0, "frame_end": 100,
        }],
        "simulation": {"solver": "mpm", "frames": 60, "fps": 24, "dt": 0.005,
                       "gravity": -9.81, "cfl": False,
                       "cfl_factor": 0.5, "cfl_max_substeps": 16},
        "output": {"cache_dir": "/tmp/test", "mesh": True,
                   "iso_level": 0.6, "smooth_passes": 2,
                   "mesh_smooth_passes": 2, "mesh_smooth_method": "taubin",
                   "decimate_ratio": 1.0, "wall_margin_cells": 2,
                   "usd": True, "particles": False, "preview": False},
    }
    toml = cb.build_toml(scene)
    assert "rotation = [" in toml, (
        "round-59 regressed: config_builder.py box branch must emit "
        "rotation key when the dict carries it (R57 OBB path)")
    # _fmt_vec3 uses :g which drops trailing zeros — match substrings
    # that survive that formatting.
    for needle in ("0.866025", "-0.5", "0.5"):
        assert needle in toml, f"rotation element {needle} missing"


def test_box_branch_omits_rotation_when_absent():
    cb = _load_cb()
    scene = {
        "domain": {"resolution": [64, 64, 64], "dx": 1.0 / 64},
        "obstacles": [{
            "type": "box",
            "center": [0.5, 0.5, 0.5],
            "half_size": [0.1, 0.1, 0.1],
        }],
        "inflows": [{
            "lo": [0.45, 0.45, 0.8], "hi": [0.55, 0.55, 0.9],
            "velocity": [0, 0, -0.5],
            "rate_per_sec": 1000.0,
            "frame_start": 0, "frame_end": 100,
        }],
        "simulation": {"solver": "mpm", "frames": 60, "fps": 24, "dt": 0.005,
                       "gravity": -9.81, "cfl": False,
                       "cfl_factor": 0.5, "cfl_max_substeps": 16},
        "output": {"cache_dir": "/tmp/test", "mesh": True,
                   "iso_level": 0.6, "smooth_passes": 2,
                   "mesh_smooth_passes": 2, "mesh_smooth_method": "taubin",
                   "decimate_ratio": 1.0, "wall_margin_cells": 2,
                   "usd": True, "particles": False, "preview": False},
    }
    toml = cb.build_toml(scene)
    # AABB-only obstacle: no rotation key emitted (pre-R57 boxes stay
    # untouched, no spurious identity matrix).
    assert "rotation =" not in toml, (
        "round-59 regressed: rotation key must be OMITTED for boxes "
        "without explicit rotation (don't emit identity-by-default)")
