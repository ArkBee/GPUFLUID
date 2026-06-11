"""Round-44 regression tests for round-43 reviewer findings.

  - pushback kernels gate on particle_selection (source-grep).
  - OutflowBox rejects reversed AABB at __post_init__.
  - apply_eevee_preset skips when scene.render.engine != Eevee.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


# ─── 1+2. Pushback kernels gate on selection (source-grep) ─────────────
# [BLK S2.17.3] guard: test_wall_pushback_skips_held_particles asserts on
# the body of k_wall_pushback — the S2.17.3 callable (audit-20260610:
# coverage ref added; the test predates the convention).

def test_cube_pushback_skips_held_particles():
    src = (_REPO / "src" / "gpufluid" / "sim" / "mpm"
           / "pushback.py").read_text()
    fn_pos = src.find("def k_cube_pushback")
    next_def = src.find("\n@", fn_pos)
    body = src[fn_pos:next_def] if next_def > 0 else src[fn_pos:]
    assert "state.particle_selection[p] == 1" in body, (
        "round-44 regressed: k_cube_pushback must skip held inflows")


def test_wall_pushback_skips_held_particles():
    src = (_REPO / "src" / "gpufluid" / "sim" / "mpm"
           / "pushback.py").read_text()
    fn_pos = src.find("def k_wall_pushback")
    next_def = src.find("\n@", fn_pos)
    body = src[fn_pos:next_def] if next_def > 0 else src[fn_pos:]
    assert "state.particle_selection[p] == 1" in body, (
        "round-44 regressed: k_wall_pushback must skip held inflows")


# ─── 3. OutflowBox AABB validation ──────────────────────────────────────

def test_outflow_rejects_reversed_aabb():
    """Mirror of round-33 MPM-inflow `lo > hi` validation."""
    from gpufluid.domain.regions import OutflowBox
    # x axis reversed
    with pytest.raises(ValueError, match=r"hi\[x\]"):
        OutflowBox(lo=(0.5, 0.0, 0.0), hi=(0.2, 1.0, 1.0))
    # y axis reversed
    with pytest.raises(ValueError, match=r"hi\[y\]"):
        OutflowBox(lo=(0.0, 0.5, 0.0), hi=(1.0, 0.2, 1.0))
    # z axis reversed
    with pytest.raises(ValueError, match=r"hi\[z\]"):
        OutflowBox(lo=(0.0, 0.0, 0.5), hi=(1.0, 1.0, 0.2))


def test_outflow_accepts_equal_bounds():
    """Zero-volume box is degenerate but not reversed; not the bug class
    round-44 closes. Must not raise."""
    from gpufluid.domain.regions import OutflowBox
    # lo == hi on one axis is OK (degenerate slab, will simply contain
    # no particles — different semantic from the silent-no-op bug).
    OutflowBox(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 0.0))


def test_outflow_normal_aabb_unchanged():
    from gpufluid.domain.regions import OutflowBox
    o = OutflowBox(lo=(0.1, 0.2, 0.3), hi=(0.4, 0.5, 0.6))
    assert o.lo == (0.1, 0.2, 0.3)
    assert o.hi == (0.4, 0.5, 0.6)


# ─── 4. apply_eevee_preset skips non-Eevee engine ───────────────────────
# [BLK A8.9] guard: the two tests below call apply_eevee_preset (the A8.9
# callable) behaviourally — engine gate + applied-log contract
# (audit-20260610: coverage ref added; the tests predate the convention).

def _load_render_bridge():
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    spec = importlib.util.spec_from_file_location(
        "_test_rb_44",
        _ADDON_DIR / "gpufluid_blender" / "render_bridge.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_apply_eevee_preset_skipped_on_cycles_engine():
    rb = _load_render_bridge()
    scene = types.SimpleNamespace(
        render=types.SimpleNamespace(engine="CYCLES"),
        eevee=types.SimpleNamespace(
            taa_render_samples=64, use_bloom=True, use_ssr=True))
    log = rb.apply_eevee_preset(scene)
    assert log == {"skipped_engine": "CYCLES"}
    # Eevee state must NOT have been mutated.
    assert scene.eevee.taa_render_samples == 64
    assert scene.eevee.use_bloom is True


def test_apply_eevee_preset_applies_on_eevee_engine():
    rb = _load_render_bridge()
    scene = types.SimpleNamespace(
        render=types.SimpleNamespace(engine="BLENDER_EEVEE"),
        eevee=types.SimpleNamespace(
            taa_render_samples=64, use_bloom=True, use_ssr=True))
    log = rb.apply_eevee_preset(scene, samples=16)
    assert log.get("taa_render_samples") == 16
    assert log.get("use_bloom") is False
    assert scene.eevee.use_bloom is False


def test_apply_eevee_preset_no_render_engine_applies_default():
    """Round-44 back-compat: when scene.render is absent (test mocks
    without a render block), behave as before — apply the preset."""
    rb = _load_render_bridge()
    scene = types.SimpleNamespace(
        eevee=types.SimpleNamespace(
            taa_render_samples=64, use_bloom=True))
    log = rb.apply_eevee_preset(scene, samples=16)
    assert log.get("taa_render_samples") == 16
