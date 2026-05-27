"""Round-17: DomainTransform invariant suite.

Senior architect day-1 #4 (code smell #2, post-day-1): extracted the
world↔unit-cube math from `bake.collect_scene` into a pure dataclass
so it can be tested without bpy and reasoned about in isolation.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_ADDON_DIR = Path(__file__).resolve().parent.parent / "addon"


def _load():
    # @dataclass machinery looks up cls.__module__ in sys.modules;
    # register the module BEFORE exec so the dataclass decorator can
    # resolve `cls.__module__` (or it raises AttributeError on a None).
    name = "_test_domain_transform"
    spec = importlib.util.spec_from_file_location(
        name, _ADDON_DIR / "gpufluid_blender" / "domain_transform.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.DomainTransform


# ─── Construction invariants ───────────────────────────────────────────

def test_unit_cube_domain():
    """Domain [(0,0,0)→(1,1,1)] at res 64 → identity transform."""
    DT = _load()
    t = DT.from_world_aabb((0, 0, 0), (1, 1, 1), 64)
    assert t.origin == (0.0, 0.0, 0.0)
    assert t.dom_size == (1.0, 1.0, 1.0)
    assert t.resolution == (64, 64, 64)
    assert t.dx == pytest.approx(1.0 / 64)
    assert t.eps_norm == pytest.approx(1.5 / 64)
    assert t.to_sim((0.5, 0.5, 0.5)) == pytest.approx((0.5, 0.5, 0.5))


def test_offset_domain_translates_to_origin():
    """Domain [(-1,-1,-1)→(1,1,1)] (centered at origin, size 2) →
    world (0,0,0) maps to sim (0.5,0.5,0.5)."""
    DT = _load()
    t = DT.from_world_aabb((-1, -1, -1), (1, 1, 1), 64)
    assert t.dom_size == (2.0, 2.0, 2.0)
    assert t.to_sim((0, 0, 0)) == pytest.approx((0.5, 0.5, 0.5))
    assert t.to_sim((-1, -1, -1)) == pytest.approx((0.0, 0.0, 0.0))
    assert t.to_sim((1, 1, 1)) == pytest.approx((1.0, 1.0, 1.0))


def test_anisotropic_domain_proportional_resolution():
    """A 2×1×1 domain at res=64 → (64, 32, 32) cells (proportional to
    extent) — non-cubic domains MUST get equal dx per axis or the
    solver fires kernels at wrong scales."""
    DT = _load()
    t = DT.from_world_aabb((0, 0, 0), (2, 1, 1), 64)
    assert t.resolution == (64, 32, 32)
    # dx is uniform — 1 / max(res) = 1/64 = same world distance per cell
    assert t.dx == pytest.approx(1.0 / 64)


def test_resolution_minimum_8():
    """Even a very thin axis gets at least 8 cells (max(8, ...))."""
    DT = _load()
    t = DT.from_world_aabb((0, 0, 0), (10, 0.001, 1), 64)
    assert t.resolution[1] == 8


def test_degenerate_aabb_raises():
    DT = _load()
    with pytest.raises(ValueError, match="degenerate"):
        DT.from_world_aabb((0, 0, 0), (0, 0, 0), 64)


def test_negative_resolution_raises():
    DT = _load()
    with pytest.raises(ValueError, match="resolution"):
        DT.from_world_aabb((0, 0, 0), (1, 1, 1), 0)


# ─── to_sim / normalize_velocity round-trip ────────────────────────────

def test_to_sim_round_trip_via_inverse_at_corners():
    DT = _load()
    t = DT.from_world_aabb((-1, -2, -3), (1, 2, 3), 64)
    # Corners of [0,1] should map to world corners
    assert t.to_sim((-1, -2, -3)) == pytest.approx((0, 0, 0))
    assert t.to_sim((1, 2, 3)) == pytest.approx((1, 1, 1))


def test_inv_size_property():
    DT = _load()
    t = DT.from_world_aabb((0, 0, 0), (2, 4, 8), 64)
    inv = t.inv_size
    assert inv == pytest.approx((1 / 2, 1 / 4, 1 / 8))


def test_avg_inv_property():
    DT = _load()
    t = DT.from_world_aabb((0, 0, 0), (2, 4, 8), 64)
    expected = (1 / 2 + 1 / 4 + 1 / 8) / 3
    assert t.avg_inv == pytest.approx(expected)


def test_normalize_velocity_uses_inv_size():
    """Velocity in m/s normalises by per-axis inv_size, NOT a single
    scalar — round-1 fix lived in collect_scene; now lives here so
    callers can't accidentally re-introduce the divide-by-extent bug."""
    DT = _load()
    t = DT.from_world_aabb((0, 0, 0), (2, 4, 8), 64)
    v = (1.0, 1.0, 1.0)
    assert t.normalize_velocity(v) == pytest.approx((0.5, 0.25, 0.125))


def test_normalize_velocity_zero():
    DT = _load()
    t = DT.from_world_aabb((0, 0, 0), (1, 1, 1), 64)
    assert t.normalize_velocity((0, 0, 0)) == (0.0, 0.0, 0.0)


# ─── eps_norm (1.5-cell wall margin) ───────────────────────────────────

def test_eps_norm_is_1_5_cells():
    DT = _load()
    t = DT.from_world_aabb((0, 0, 0), (1, 1, 1), 64)
    # 1.5 cells out of 64 = 0.0234375
    assert t.eps_norm == pytest.approx(1.5 / 64)


def test_eps_norm_independent_of_anisotropy():
    """Same resolution → same eps_norm even if domain is anisotropic.
    The MPM unit cube is [0.05, 0.95] regardless of world geometry."""
    DT = _load()
    t1 = DT.from_world_aabb((0, 0, 0), (1, 1, 1), 96)
    t2 = DT.from_world_aabb((0, 0, 0), (1, 4, 16), 96)
    assert t1.eps_norm == t2.eps_norm
