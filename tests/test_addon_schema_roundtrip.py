"""Phase-1 addon contract test: scene_dict → config_builder → tomllib → load_scene.

The addon's `config_builder.build_toml` is the only thing between Blender
UI state and the gpufluid CLI. This test fixes the contract by round-
tripping representative scene dicts through:

    config_builder.build_toml(scene_dict)        # addon side
    tomllib.loads(...)                            # storage layer
    gpufluid.cli.config.load_scene(tmp.toml)      # core side

If any of these break the chain, the bake silently produces a scene with
wrong solver semantics or missing knobs. Whitelist coverage is implicit:
`load_scene` raises BlockError on schema violations (missing required
keys, bad types), so a green test means every emitted field has a parser.

Test runs entirely without bpy — the addon's config_builder is bpy-free
by design (it accepts plain dicts).
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

# Ensure addon/ is on sys.path — conftest.py already does this, but be
# explicit so this test can also run via `python -m pytest tests/test_addon_schema_roundtrip.py`.
_ADDON_DIR = Path(__file__).resolve().parents[1] / "addon"
if str(_ADDON_DIR) not in sys.path:
    sys.path.insert(0, str(_ADDON_DIR))

from gpufluid_blender.config_builder import build_toml  # noqa: E402
from gpufluid.cli.config import load_scene  # noqa: E402


def _base_domain():
    return {"resolution": (64, 64, 64), "dx": 1.0 / 64, "origin": [0.0, 0.0, 0.0]}


def _base_output(tmp_path: Path):
    return {
        "cache_dir": str(tmp_path / "cache"),
        "iso_level": 0.6,
        "smooth_passes": 2,
        "mesh_smooth_passes": 0,
        "mesh_smooth_method": "taubin",
        "decimate_ratio": 1.0,
        "wall_margin_cells": 0,
        "usd": False,
        "particles": False,
        "preview": False,
        "color_mix_mode": "linear",
    }


def _flip_sim():
    return {
        "solver": "flip",
        "dt": 0.005,
        "frames": 10,
        "fps": 24,
        "pressure_iters": 60,
        "pressure_solver": "pcg",
        "cfl": False,
        "cfl_factor": 0.5,
        "cfl_max_substeps": 16,
        "flip_blend": 0.95,
        "gravity": -9.81,
        "surface_tension": 0.0,
        "csf_smoothing_passes": 2,
        "reseed": False,
        "reseed_every_n_frames": 5,
        "reseed_min_per_cell": 4,
        "reseed_max_per_cell": 16,
    }


def _mpm_sim():
    sim = _flip_sim()
    sim["solver"] = "mpm"
    sim.update({
        "mpm_bulk_modulus": 1500.0,
        "mpm_rpic_damping": 0.15,
        "mpm_grid_v_damping": 0.998,
        "mpm_cube_friction": 0.6,
        "mpm_v_terminal": -1.0,
        "mpm_vz_max_splash": 0.3,
        "mpm_initial_velocity": -0.3 * (1.0 / 1.0),  # already normalised
    })
    return sim


def _roundtrip(scene_dict, tmp_path: Path):
    toml_str = build_toml(scene_dict)
    # Layer 1: must parse as TOML.
    parsed = tomllib.loads(toml_str)
    # Layer 2: must load through the core schema (raises BlockError on
    # missing required keys / bad types).
    p = tmp_path / "scene.toml"
    p.write_text(toml_str, encoding="utf-8")
    cfg = load_scene(p)
    return toml_str, parsed, cfg


# ---------------------------------------------------------------------------

def test_flip_minimal(tmp_path):
    scene = {
        "domain": _base_domain(),
        "fluids": [{"kind": "box", "lo": (0.1, 0.1, 0.1), "hi": (0.4, 0.4, 0.4), "ppc": 8}],
        "obstacles": [],
        "inflows": [],
        "outflows": [],
        "simulation": _flip_sim(),
        "output": _base_output(tmp_path),
    }
    toml_str, parsed, cfg = _roundtrip(scene, tmp_path)
    assert cfg.simulation.solver == "flip"
    # FLIP-only knobs present.
    assert parsed["simulation"]["pressure_solver"] == "pcg"
    assert parsed["simulation"]["flip_blend"] == pytest.approx(0.95)
    # MPM subsection should NOT appear.
    assert "mpm" not in parsed["simulation"]


def test_mpm_minimal(tmp_path):
    scene = {
        "domain": _base_domain(),
        "fluids": [{"kind": "box", "lo": (0.1, 0.1, 0.1), "hi": (0.4, 0.4, 0.4), "ppc": 8}],
        "obstacles": [],
        "inflows": [],
        "outflows": [],
        "simulation": _mpm_sim(),
        "output": _base_output(tmp_path),
    }
    toml_str, parsed, cfg = _roundtrip(scene, tmp_path)
    assert cfg.simulation.solver == "mpm"
    # FLIP-only knobs MUST be gated out when solver=mpm.
    assert "pressure_solver" not in parsed["simulation"]
    assert "flip_blend" not in parsed["simulation"]
    assert "pressure_iters" not in parsed["simulation"]
    assert "surface_tension" not in parsed["simulation"]
    assert "csf_smoothing_passes" not in parsed["simulation"]
    # MPM subsection round-trips.
    assert parsed["simulation"]["mpm"]["bulk_modulus"] == pytest.approx(1500.0)
    assert cfg.simulation.mpm_bulk_modulus == pytest.approx(1500.0)


def test_obstacle_motion_inline_table(tmp_path):
    scene = {
        "domain": _base_domain(),
        "fluids": [{"kind": "box", "lo": (0.1, 0.1, 0.1), "hi": (0.4, 0.4, 0.4), "ppc": 8}],
        "obstacles": [{
            "type": "box",
            "center": (0.5, 0.5, 0.5),
            "half_size": (0.1, 0.1, 0.1),
            "motion": {"kind": "linear", "velocity": (0.1, 0.0, 0.0)},
        }],
        "inflows": [],
        "outflows": [],
        "simulation": _flip_sim(),
        "output": _base_output(tmp_path),
    }
    toml_str, parsed, cfg = _roundtrip(scene, tmp_path)
    assert len(cfg.obstacle) == 1
    assert cfg.obstacle_motion[0] is not None
    assert cfg.obstacle_motion[0].kind == "linear"


def test_inflow_with_color_and_temperature(tmp_path):
    scene = {
        "domain": _base_domain(),
        "fluids": [],
        "obstacles": [],
        "inflows": [{
            "lo": (0.4, 0.8, 0.4),
            "hi": (0.6, 0.95, 0.6),
            "velocity": (0.0, -0.5, 0.0),
            "rate_per_sec": 3000.0,
            "frame_start": 0,
            "frame_end": 120,
            "color": (0.2, 0.4, 0.9),
            "temperature": 290.0,
        }],
        "outflows": [],
        "simulation": _mpm_sim(),
        "output": _base_output(tmp_path),
    }
    toml_str, parsed, cfg = _roundtrip(scene, tmp_path)
    assert len(cfg.inflow) == 1
    assert cfg.inflow[0].color == (0.2, 0.4, 0.9)
    assert cfg.inflow[0].temperature == pytest.approx(290.0)


def test_overrides_deep_merge(tmp_path):
    scene = {
        "domain": _base_domain(),
        "fluids": [{"kind": "box", "lo": (0.1, 0.1, 0.1), "hi": (0.4, 0.4, 0.4), "ppc": 8}],
        "obstacles": [],
        "inflows": [],
        "outflows": [],
        "simulation": _mpm_sim(),
        "output": _base_output(tmp_path),
        # Override: bump bulk_modulus AND add a new mpm field. Deep-merge
        # should preserve the other [simulation.mpm] keys from the base.
        "toml_overrides": (
            "[simulation.mpm]\n"
            "bulk_modulus = 3000.0\n"
        ),
    }
    toml_str, parsed, cfg = _roundtrip(scene, tmp_path)
    assert cfg.simulation.mpm_bulk_modulus == pytest.approx(3000.0)
    # The other knobs we set in _mpm_sim() must survive the merge.
    assert cfg.simulation.mpm_cube_friction == pytest.approx(0.6)


def test_overrides_invalid_toml_raises_valueerror(tmp_path):
    scene = {
        "domain": _base_domain(),
        "fluids": [{"kind": "box", "lo": (0.1, 0.1, 0.1), "hi": (0.4, 0.4, 0.4), "ppc": 8}],
        "obstacles": [],
        "inflows": [],
        "outflows": [],
        "simulation": _flip_sim(),
        "output": _base_output(tmp_path),
        "toml_overrides": "this is = not = valid toml",
    }
    with pytest.raises(ValueError, match="TOML overrides"):
        build_toml(scene)
