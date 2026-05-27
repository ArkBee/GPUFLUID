"""Round-23 regression tests for findings from round-23 reviewer:

  - whitewater: gravity applies on Z (not Y) — tests verify per-class
    dynamics diverge on the correct axis.
  - whitewater plumbing: scene.simulation.gravity propagates into
    per-class gravity fields (was: silently dropped via legacy field).
  - config_builder: emits whitewater_trapped_air_weight (was: missing
    from key list → silent default).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


# ─── whitewater Z-up dynamics ───────────────────────────────────────────

def test_whitewater_gravity_acts_on_z_axis():
    """Round-23: per-class step() must move particles on Z, not Y."""
    from gpufluid.sim.whitewater import (
        WhitewaterConfig, WhitewaterSystem,
        KIND_FOAM, KIND_SPRAY, KIND_BUBBLE,
    )
    ws = WhitewaterSystem(WhitewaterConfig(speed_threshold=0.0))
    ws.pos = np.array([[0.5, 0.5, 0.5]] * 3, dtype=np.float32)
    ws.vel = np.zeros((3, 3), dtype=np.float32)
    ws.age = np.zeros(3, dtype=np.float32)
    ws.kind = np.array([KIND_FOAM, KIND_SPRAY, KIND_BUBBLE], dtype=np.int32)
    for _ in range(20):
        ws.step(0.01, dom=(10.0, 10.0, 10.0))
    # All movement must be on Z; X+Y stay at the starting 0.5.
    assert np.allclose(ws.pos[:, 0], 0.5, atol=1e-4), (
        f"X must not move; got {ws.pos[:, 0]}")
    assert np.allclose(ws.pos[:, 1], 0.5, atol=1e-4), (
        f"Y must not move (round-23 fix: gravity is Z, not Y); "
        f"got {ws.pos[:, 1]}")
    # Spray < foam < bubble on Z
    z_foam, z_spray, z_bubble = ws.pos[:, 2]
    assert z_spray < z_foam < z_bubble, (
        f"Z ordering wrong: spray={z_spray:.3f} foam={z_foam:.3f} "
        f"bubble={z_bubble:.3f}")


def test_whitewater_spawn_offset_is_z():
    """spawn_offset on emit moves new particles upward on Z."""
    from gpufluid.sim.whitewater import (
        WhitewaterConfig, WhitewaterSystem,
    )
    cfg = WhitewaterConfig(speed_threshold=0.0, spawn_offset=0.1)
    ws = WhitewaterSystem(cfg)
    pos = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    vel = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    ws.emit_from_fluid(pos, vel)
    assert ws.n == 1
    # Z should be 0.5 + spawn_offset, X/Y untouched.
    assert abs(ws.pos[0, 0] - 0.5) < 1e-4
    assert abs(ws.pos[0, 1] - 0.5) < 1e-4
    assert ws.pos[0, 2] > 0.55, (
        f"spawn_offset must hit Z; got pos={ws.pos[0]}")


# ─── addon config_builder emits trapped_air_weight ──────────────────────

def _load_config_builder():
    sys.modules.pop("gpufluid_blender", None)
    sys.modules.pop("gpufluid_blender.scene_dict", None)
    sys.modules.pop("gpufluid_blender.config_builder", None)
    pkg = types.ModuleType("gpufluid_blender")
    pkg.__path__ = [str(_ADDON_DIR / "gpufluid_blender")]
    sys.modules["gpufluid_blender"] = pkg
    for name in ("scene_dict", "config_builder"):
        spec = importlib.util.spec_from_file_location(
            f"gpufluid_blender.{name}",
            _ADDON_DIR / "gpufluid_blender" / f"{name}.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"gpufluid_blender.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["gpufluid_blender.config_builder"]


def test_config_builder_emits_trapped_air_weight():
    """Round-23: setting whitewater_trapped_air_weight in scene_dict's
    output table must appear in the emitted TOML (pre-round-23 it was
    silently dropped)."""
    cb = _load_config_builder()
    scene_dict = {
        "domain": {"resolution": [64, 64, 64], "dx": 1 / 64},
        "simulation": {"solver": "flip", "dt": 0.005, "frames": 30,
                       "fps": 24, "gravity": -9.81},
        "output": {
            "cache_dir": "C:/tmp/x",
            "iso_level": 0.5,
            "smooth_passes": 2,
            "whitewater": True,
            "whitewater_use_potential": True,
            "whitewater_trapped_air_weight": 2.5,
            "whitewater_wave_crest_weight": 1.7,
        },
        "fluid": {"lo": [0.2, 0.2, 0.2], "hi": [0.4, 0.4, 0.4]},
    }
    toml_str = cb.build_toml(scene_dict)
    assert "whitewater_trapped_air_weight = 2.5" in toml_str, (
        f"trapped-air weight missing from emit:\n{toml_str}")
    # Ensure existing keys still emit (no regression).
    assert "whitewater_wave_crest_weight = 1.7" in toml_str


# ─── runner sync orphan-Popen guard ─────────────────────────────────────

def test_start_sync_kills_orphan_on_thread_failure(monkeypatch):
    """Round-23: if Thread() construction fails after Popen succeeds,
    start_sync must kill the now-orphan subprocess (mirroring the
    round-9 guard the modal path has). Pre-round-23 it leaked."""
    # bpy stub (helpers.py imports it at module top).
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.types = types.SimpleNamespace(
        Operator=type("_Op", (), {}),
        Panel=type("_P", (), {}),
    )
    fake_bpy.props = types.SimpleNamespace(
        StringProperty=lambda **k: None,
        BoolProperty=lambda **k: None,
        IntProperty=lambda **k: None,
        FloatProperty=lambda **k: None,
        EnumProperty=lambda **k: None,
    )
    sys.modules["bpy"] = fake_bpy

    for mod_name in ("gpufluid_blender", "gpufluid_blender.operators",
                     "gpufluid_blender.operators._runner",
                     "gpufluid_blender.operators.helpers"):
        sys.modules.pop(mod_name, None)
    pkg = types.ModuleType("gpufluid_blender")
    pkg.__path__ = [str(_ADDON_DIR / "gpufluid_blender")]
    sys.modules["gpufluid_blender"] = pkg
    ops_pkg = types.ModuleType("gpufluid_blender.operators")
    ops_pkg.__path__ = [str(_ADDON_DIR / "gpufluid_blender" / "operators")]
    sys.modules["gpufluid_blender.operators"] = ops_pkg
    # Stub helpers minimally — runner only needs `subprocess_drain`.
    h_mod = types.ModuleType("gpufluid_blender.operators.helpers")
    h_mod.subprocess_drain = lambda *a, **k: None
    sys.modules["gpufluid_blender.operators.helpers"] = h_mod
    # Now runner.
    spec = importlib.util.spec_from_file_location(
        "gpufluid_blender.operators._runner",
        _ADDON_DIR / "gpufluid_blender" / "operators" / "_runner.py",
    )
    runner_mod = importlib.util.module_from_spec(spec)
    sys.modules["gpufluid_blender.operators._runner"] = runner_mod
    spec.loader.exec_module(runner_mod)

    killed = {"count": 0}

    class _FakeProc:
        def __init__(self):
            self.stdout = iter([])
            self.returncode = None
            self._alive = True
        def poll(self):
            return None if self._alive else 0
        def kill(self):
            killed["count"] += 1
            self._alive = False
        def wait(self, timeout=None):
            self._alive = False
            return 0

    proc = _FakeProc()
    monkeypatch.setattr(runner_mod.subprocess, "Popen",
                        lambda *a, **k: proc)

    def _bad_thread(*a, **k):
        raise RuntimeError("simulated resource exhaustion")
    monkeypatch.setattr(runner_mod.threading, "Thread", _bad_thread)

    reports: list = []
    class _Op:
        class __class__:
            _is_running = False
        def report(self, lvl, msg):
            reports.append((lvl, msg))

    op = _Op()
    op.__class__._is_running = True
    runner = runner_mod.ModalSubprocessRunner("bake")
    result = runner.start_sync(
        op, ["echo"], timeout_sec=0,
        log_prefix="bake", on_complete=None, logger=None,
    )
    assert result == {"CANCELLED"}
    assert killed["count"] == 1, "subprocess must have been killed"
    assert any("drainer" in m or "subprocess" in m for _, m in reports)
