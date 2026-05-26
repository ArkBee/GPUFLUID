"""Round-15: SceneDict validator contract tests.

Senior architect (round-12) flagged `scene_dict: Dict[str, Any]` as
the smell that let round-9's `_emit_table` data-loss bug ship. Round-15
introduced `addon/gpufluid_blender/scene_dict.py::validate_scene_dict`
called at the top of `build_toml`. These tests enforce the contract:

  - Malformed dicts raise `SceneDictError` (ValueError subclass) with
    a message that names the offending key path.
  - Well-formed dicts pass through unchanged.
  - FLIP-only gating (no `[simulation.mpm]` under solver=flip) is
    enforced.
  - Reversed inflow/outflow frame_start > frame_end is rejected (same
    invariant the bake op enforces — defence in depth at emit-time too).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_ADDON_DIR = Path(__file__).resolve().parent.parent / "addon"


def _load_scene_dict_module():
    """Load scene_dict.py without booting the full addon (no bpy needed)."""
    spec = importlib.util.spec_from_file_location(
        "_test_scene_dict",
        _ADDON_DIR / "gpufluid_blender" / "scene_dict.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sd():
    return _load_scene_dict_module()


def _minimal_valid_flip():
    return {
        "domain": {"resolution": [64, 64, 64], "dx": 1 / 64},
        "simulation": {
            "solver": "flip",
            "dt": 0.005,
            "frames": 60,
            "fps": 24,
            "gravity": -9.81,
        },
        "output": {"cache_dir": "C:/tmp/cache"},
    }


def _minimal_valid_mpm():
    d = _minimal_valid_flip()
    d["simulation"]["solver"] = "mpm"
    d["simulation"]["mpm"] = {
        "bulk_modulus": 1500, "rpic_damping": 0.15,
        "grid_v_damping": 0.998, "cube_friction": 0.6,
        "v_terminal": -1, "vz_max_splash": 0.3,
        "initial_velocity": -0.5,
    }
    return d


# ─── Happy paths ────────────────────────────────────────────────────────

def test_minimal_flip_passes(sd):
    sd.validate_scene_dict(_minimal_valid_flip())   # no raise


def test_minimal_mpm_passes(sd):
    sd.validate_scene_dict(_minimal_valid_mpm())


def test_with_optional_arrays_passes(sd):
    d = _minimal_valid_flip()
    d["obstacle"] = [{"type": "box", "center": [0.5, 0.5, 0.5],
                       "half_size": [0.2, 0.2, 0.05]}]
    d["inflow"] = [{"lo": [0.4, 0.4, 0.8], "hi": [0.6, 0.6, 1.0],
                     "velocity": [0, -1.5, 0], "rate_per_sec": 30000,
                     "frame_start": 1, "frame_end": 60}]
    d["outflow"] = [{"lo": [0, 0, 0], "hi": [1, 1, 0.1],
                      "frame_start": 1, "frame_end": 60}]
    sd.validate_scene_dict(d)


# ─── Type-not-dict ──────────────────────────────────────────────────────

def test_non_dict_root_raises(sd):
    with pytest.raises(sd.SceneDictError, match="scene_dict must be dict"):
        sd.validate_scene_dict("not a dict")


# ─── Partial-dict tolerance (round-15 contract: validate what's there) ─

def test_empty_dict_passes(sd):
    """Validator must not insist on completeness — only that what's
    present has correct shape. Test fixtures pre-round-15 routinely
    use partial dicts to exercise specific code paths."""
    sd.validate_scene_dict({})


def test_missing_simulation_does_not_raise(sd):
    """A dict without simulation should pass — only emit-time decides
    what's required to produce TOML; validator stays out of completeness
    claims (those belong in validate_complete_scene_dict)."""
    d = _minimal_valid_flip(); del d["simulation"]
    sd.validate_scene_dict(d)


# ─── Wrong types in required ────────────────────────────────────────────

def test_wrong_solver_type_raises(sd):
    d = _minimal_valid_flip(); d["simulation"]["solver"] = 42
    with pytest.raises(sd.SceneDictError, match="simulation.solver"):
        sd.validate_scene_dict(d)


def test_unknown_solver_enum_raises(sd):
    d = _minimal_valid_flip(); d["simulation"]["solver"] = "smoke"
    with pytest.raises(sd.SceneDictError, match="simulation.solver"):
        sd.validate_scene_dict(d)


def test_dx_must_be_numeric(sd):
    d = _minimal_valid_flip(); d["domain"]["dx"] = "0.015"
    with pytest.raises(sd.SceneDictError, match="domain.dx"):
        sd.validate_scene_dict(d)


def test_resolution_not_three_raises(sd):
    d = _minimal_valid_flip(); d["domain"]["resolution"] = [64, 64]
    with pytest.raises(sd.SceneDictError, match="domain.resolution"):
        sd.validate_scene_dict(d)


def test_resolution_negative_raises(sd):
    d = _minimal_valid_flip(); d["domain"]["resolution"] = [64, -64, 64]
    with pytest.raises(sd.SceneDictError, match="domain.resolution"):
        sd.validate_scene_dict(d)


def test_frames_zero_raises(sd):
    d = _minimal_valid_flip(); d["simulation"]["frames"] = 0
    with pytest.raises(sd.SceneDictError, match="simulation.frames"):
        sd.validate_scene_dict(d)


# ─── Array-of-tables shape ──────────────────────────────────────────────

def test_obstacle_not_list_raises(sd):
    d = _minimal_valid_flip(); d["obstacle"] = {"type": "box"}
    with pytest.raises(sd.SceneDictError, match="'obstacle' must be a list"):
        sd.validate_scene_dict(d)


def test_inflow_entry_not_dict_raises(sd):
    d = _minimal_valid_flip(); d["inflow"] = ["not_a_dict"]
    with pytest.raises(sd.SceneDictError, match=r"inflow\[0\]"):
        sd.validate_scene_dict(d)


def test_reversed_inflow_range_raises(sd):
    """Same invariant the bake operator enforces (round-5 fix);
    validator gives defence in depth at emit-time."""
    d = _minimal_valid_flip()
    d["inflow"] = [{"frame_start": 30, "frame_end": 10}]
    with pytest.raises(sd.SceneDictError, match=r"inflow\[0\].*frame_start"):
        sd.validate_scene_dict(d)


def test_reversed_outflow_range_raises(sd):
    d = _minimal_valid_flip()
    d["outflow"] = [{"frame_start": 30, "frame_end": 10}]
    with pytest.raises(sd.SceneDictError, match=r"outflow\[0\].*frame_start"):
        sd.validate_scene_dict(d)


# NOTE: an earlier round-15 draft enforced "solver=mpm requires
# [simulation.mpm]" and "solver=flip forbids [simulation.mpm]" at
# validate-time. Both broke real test fixtures because the addon's
# _collect.py only emits the sub-table when MPM-specific knobs differ
# from defaults, and the round-1 overrides escape-hatch can produce
# any cross-solver shape. The actual FLIP-only-vs-MPM gating lives in
# config_builder's emit gating, not in the validator. See
# scene_dict.py:298+ for the explanation.


# ─── build_toml wiring (validator must run before emit) ────────────────

def test_build_toml_invokes_validator():
    """Wiring test: malformed dict passed to build_toml must hit the
    validator and raise BEFORE _emit_toml produces broken output."""
    # Load config_builder via its own loader (bpy-free import works
    # because config_builder.py only imports stdlib + scene_dict).
    sys.modules.pop("gpufluid_blender", None)
    sys.modules.pop("gpufluid_blender.scene_dict", None)
    sys.modules.pop("gpufluid_blender.config_builder", None)

    import types as _types
    pkg = _types.ModuleType("gpufluid_blender")
    pkg.__path__ = [str(_ADDON_DIR / "gpufluid_blender")]
    sys.modules["gpufluid_blender"] = pkg

    spec = importlib.util.spec_from_file_location(
        "gpufluid_blender.scene_dict",
        _ADDON_DIR / "gpufluid_blender" / "scene_dict.py",
    )
    sd_mod = importlib.util.module_from_spec(spec)
    sys.modules["gpufluid_blender.scene_dict"] = sd_mod
    spec.loader.exec_module(sd_mod)

    spec = importlib.util.spec_from_file_location(
        "gpufluid_blender.config_builder",
        _ADDON_DIR / "gpufluid_blender" / "config_builder.py",
    )
    cb_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb_mod)

    # Malformed scene_dict — wrong type for simulation.solver
    bad = {"domain": {"resolution": [64, 64, 64], "dx": 0.015},
           "simulation": {"solver": 42, "frames": 10},
           "output": {"cache_dir": "/tmp/x"}}
    with pytest.raises(sd_mod.SceneDictError, match="simulation.solver"):
        cb_mod.build_toml(bad)
