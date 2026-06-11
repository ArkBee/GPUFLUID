"""Tests for A8 addon config_builder (pure Python — no bpy needed)."""
import importlib.util
import sys
import tomllib
from pathlib import Path
import pytest

# Load config_builder.py directly, bypassing the addon's __init__.py
# (which imports bpy and would fail outside Blender).
_CB_PATH = Path(__file__).resolve().parents[1] / "addon" / "gpufluid_blender" / "config_builder.py"
_spec = importlib.util.spec_from_file_location("_gpufluid_config_builder", _CB_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_toml = _mod.build_toml


def _minimal_scene():
    return {
        "domain": {"resolution": (32, 32, 32), "dx": 0.03125, "origin": [0, 0, 0]},
        "fluid": {"lo": (0.10, 0.10, 0.10), "hi": (0.40, 0.70, 0.40), "ppc": 8},
        "obstacles": [],
        "simulation": {"dt": 0.005, "frames": 50, "fps": 24,
                       "pressure_iters": 40, "flip_blend": 0.95, "gravity": -9.81},
        "output": {"cache_dir": "/tmp/cache", "iso_level": 0.6,
                   "smooth_passes": 2, "particles": False, "preview": False},
    }


def test_a8_5_config_round_trip_minimal():
    s = _minimal_scene()
    toml_str = build_toml(s)
    parsed = tomllib.loads(toml_str)
    assert parsed["domain"]["resolution"] == [32, 32, 32]
    assert parsed["fluid"]["ppc"] == 8
    assert parsed["simulation"]["frames"] == 50
    assert parsed["output"]["cache_dir"] == "/tmp/cache"
    assert "obstacle" not in parsed  # empty array of tables omitted


def test_a8_5_config_with_obstacles():
    s = _minimal_scene()
    s["obstacles"] = [
        {"type": "sphere", "center": [0.5, 0.3, 0.5], "radius": 0.10},
        {"type": "box", "center": [0.5, 0.5, 0.5], "half_size": [0.2, 0.05, 0.2]},
        {"type": "cylinder_y", "center": [0.7, 0.3, 0.7], "radius": 0.05, "half_height": 0.30},
        {"type": "mesh", "path": r"C:\path with spaces\bunny.obj", "scale": 0.5, "translate": [0.1, 0.0, 0.1]},
    ]
    toml_str = build_toml(s)
    parsed = tomllib.loads(toml_str)
    obs = parsed["obstacle"]
    assert len(obs) == 4
    assert obs[0]["type"] == "sphere" and obs[0]["radius"] == 0.10
    assert obs[1]["type"] == "box" and obs[1]["half_size"] == [0.2, 0.05, 0.2]
    assert obs[2]["type"] == "cylinder_y" and obs[2]["half_height"] == 0.30
    assert obs[3]["type"] == "mesh"
    assert "bunny.obj" in obs[3]["path"]


def test_a8_5_config_loadable_by_cli():
    """Config produced by addon must parse via gpufluid.cli.load_scene."""
    from gpufluid.cli.config import load_scene
    import tempfile, os
    s = _minimal_scene()
    s["obstacles"] = [{"type": "cylinder_y", "center": [0.5, 0.3, 0.5],
                       "radius": 0.1, "half_height": 0.3}]
    toml_str = build_toml(s)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(toml_str)
        path = f.name
    try:
        scene = load_scene(path)
    finally:
        os.unlink(path)
    assert scene.domain.resolution == (32, 32, 32)
    assert len(scene.obstacle) == 1
    assert scene.obstacle[0].type == "cylinder_y"
    assert scene.simulation.frames == 50


# ---- B1.3 / B1.4 — surface tension + per-source colour pass-through --------

def test_b1_4_surface_tension_round_trips_through_toml():
    s = _minimal_scene()
    s["simulation"]["surface_tension"] = 0.5
    s["simulation"]["csf_smoothing_passes"] = 3
    parsed = tomllib.loads(build_toml(s))
    assert parsed["simulation"]["surface_tension"] == pytest.approx(0.5)
    assert parsed["simulation"]["csf_smoothing_passes"] == 3


def test_b1_4_multi_source_fluids_with_color():
    """fluids list → [[fluids]] array, each entry preserves its own color."""
    s = _minimal_scene()
    s.pop("fluid")
    s["fluids"] = [
        {"kind": "box", "lo": (0.10, 0.10, 0.10), "hi": (0.40, 0.40, 0.40),
         "ppc": 8, "color": (1.0, 0.0, 0.0)},
        {"kind": "box", "lo": (0.60, 0.10, 0.60), "hi": (0.90, 0.40, 0.90),
         "ppc": 8, "color": (0.0, 0.0, 1.0)},
    ]
    parsed = tomllib.loads(build_toml(s))
    assert "fluid" not in parsed, "legacy [fluid] should not appear when [[fluids]] is used"
    arr = parsed["fluids"]
    assert len(arr) == 2
    assert arr[0]["color"] == [1.0, 0.0, 0.0]
    assert arr[1]["color"] == [0.0, 0.0, 1.0]
    assert arr[0]["ppc"] == 8


def test_b1_4_fluids_without_color_omits_field():
    s = _minimal_scene()
    s.pop("fluid")
    s["fluids"] = [{"kind": "box", "lo": (0.1, 0.1, 0.1), "hi": (0.4, 0.4, 0.4), "ppc": 4}]
    parsed = tomllib.loads(build_toml(s))
    assert "color" not in parsed["fluids"][0]


def test_addon_temperature_field_round_trips_through_toml():
    """Addon UI per-source temperature ([[fluids]] temperature = X) makes
    it through build_toml verbatim. Mirrors the colour wiring."""
    s = _minimal_scene()
    s.pop("fluid")
    s["fluids"] = [
        {"kind": "box", "lo": (0.10, 0.10, 0.10), "hi": (0.40, 0.40, 0.40),
         "ppc": 4, "temperature": 1500.0},  # hot lava
        {"kind": "box", "lo": (0.60, 0.10, 0.60), "hi": (0.90, 0.40, 0.90),
         "ppc": 4, "temperature": 300.0,
         "color": (0.2, 0.5, 0.9)},          # cool basin, also tinted
    ]
    parsed = tomllib.loads(build_toml(s))
    arr = parsed["fluids"]
    assert len(arr) == 2
    assert arr[0]["temperature"] == pytest.approx(1500.0)
    assert arr[1]["temperature"] == pytest.approx(300.0)
    assert arr[1]["color"] == [0.2, 0.5, 0.9]
    # First source has no colour → field omitted, only temperature present.
    assert "color" not in arr[0]


def test_addon_temperature_omitted_when_not_set():
    s = _minimal_scene()
    s.pop("fluid")
    s["fluids"] = [{"kind": "box", "lo": (0.1, 0.1, 0.1), "hi": (0.4, 0.4, 0.4), "ppc": 4}]
    parsed = tomllib.loads(build_toml(s))
    assert "temperature" not in parsed["fluids"][0]


def test_b1_4_cli_loads_multi_source_with_color():
    """Config produced by addon (with [[fluids]] + color) must parse via load_scene."""
    from gpufluid.cli.config import load_scene
    import tempfile, os
    s = _minimal_scene()
    s.pop("fluid")
    s["fluids"] = [
        {"kind": "box", "lo": (0.1, 0.1, 0.1), "hi": (0.4, 0.4, 0.4),
         "ppc": 4, "color": (0.2, 0.8, 0.3)},
    ]
    s["simulation"]["surface_tension"] = 0.25
    toml_str = build_toml(s)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(toml_str); path = f.name
    try:
        scene = load_scene(path)
    finally:
        os.unlink(path)
    assert scene.simulation.surface_tension == pytest.approx(0.25)


# ---- B1.5 — whitewater output round-trip -----------------------------------

def test_b1_5_whitewater_disabled_omits_block():
    s = _minimal_scene()
    parsed = tomllib.loads(build_toml(s))
    assert "whitewater" not in parsed["output"], \
        "whitewater key should be absent when disabled (preserves CLI default = False)"


def test_b1_5_whitewater_enabled_round_trips_all_knobs():
    s = _minimal_scene()
    s["output"]["whitewater"] = True
    s["output"]["whitewater_speed_threshold"] = 5.5
    s["output"]["whitewater_lifetime_sec"] = 2.25
    s["output"]["whitewater_emit_per_frame_max"] = 1234
    s["output"]["whitewater_total_cap"] = 56789
    parsed = tomllib.loads(build_toml(s))
    o = parsed["output"]
    assert o["whitewater"] is True
    assert o["whitewater_speed_threshold"] == pytest.approx(5.5)
    assert o["whitewater_lifetime_sec"] == pytest.approx(2.25)
    assert o["whitewater_emit_per_frame_max"] == 1234
    assert o["whitewater_total_cap"] == 56789


def test_b1_5_cli_loads_whitewater_config():
    """Whitewater params produced by addon must parse via load_scene."""
    from gpufluid.cli.config import load_scene
    import tempfile, os
    s = _minimal_scene()
    s["output"]["whitewater"] = True
    s["output"]["whitewater_speed_threshold"] = 3.0
    s["output"]["whitewater_total_cap"] = 42000
    toml_str = build_toml(s)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(toml_str); path = f.name
    try:
        scene = load_scene(path)
    finally:
        os.unlink(path)
    assert scene.output.whitewater is True
    assert scene.output.whitewater_speed_threshold == pytest.approx(3.0)
    assert scene.output.whitewater_total_cap == 42000


def test_a8_5_rejects_unknown_obstacle_type():
    s = _minimal_scene()
    s["obstacles"] = [{"type": "trapezohedron"}]
    with pytest.raises(ValueError, match="trapezohedron"):
        build_toml(s)


def test_sphere_fluid_source_emits_sphere_toml():
    """audit-20260610: collect_scene emits kind="sphere" entries (S2.17.10)
    but _emit_fluid_body only knew mesh/box -> addon-path SPHERE bake
    crashed with KeyError 'lo' (found by the headless default-scene smoke,
    round-59 class: TOML writer drops what collect_scene emits)."""
    s = _minimal_scene()
    s.pop("fluid")
    s["fluids"] = [{"kind": "sphere", "center": (0.5, 0.5, 0.7),
                    "radius": 0.12, "ppc": 8}]
    parsed = tomllib.loads(build_toml(s))
    f = parsed["fluids"][0]
    assert f["type"] == "sphere", "audit-20260610: sphere source must survive the TOML writer"
    assert f["center"] == [0.5, 0.5, 0.7]
    assert f["radius"] == 0.12
    assert "lo" not in f
