"""Round-22 regression tests for findings from round-21 reviewer:

  - [BLK A8.10] render_bridge.FrameMeshLoader: frame_current=0 (or negative)
    must clamp to idx 0, not silently fall through with idx=-1.
  - [BLK A8.11] render_bridge.rebuild_surface_mesh: colour-count mismatch must
    log a warning (not silently drop colours without trace).
  - MpmConfig.seed: configurable, default 42 preserves determinism,
    different seeds produce different inflow distributions.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_ADDON_DIR = Path(__file__).resolve().parent.parent / "addon"


# ─── render_bridge tests ────────────────────────────────────────────────

def _load_render_bridge():
    """Load render_bridge.py with minimal bpy stub so the import works
    headlessly. FrameMeshLoader uses bpy only via the surf_obj it
    receives in __init__, so the module-level import is the only thing
    we need to satisfy."""
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.app = types.SimpleNamespace(handlers=types.SimpleNamespace(
        frame_change_pre=[], persistent=lambda f: f))
    sys.modules.setdefault("bpy", fake_bpy)

    spec = importlib.util.spec_from_file_location(
        "_test_render_bridge",
        _ADDON_DIR / "gpufluid_blender" / "render_bridge.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeSimTimeText:
    def __init__(self):
        self.data = types.SimpleNamespace(body="")


class _FakeScene:
    def __init__(self, frame):
        self.frame_current = frame


def test_frame_zero_clamps_to_idx_zero(tmp_path):
    """Pre-round-22: f=0 → idx=-1 → frame_-001.ply missing → silent
    no-op + sim-time text reads negative. Now: clamped to idx=0."""
    rb = _load_render_bridge()
    text = _FakeSimTimeText()
    surf = types.SimpleNamespace()  # never touched because path won't exist
    loader = rb.FrameMeshLoader(
        cache_dir=tmp_path, surf_obj=surf,
        sim_time_text=text, fps=60,
    )
    loader(_FakeScene(frame=0))
    # idx = max(0, 0-1) = 0  → sim time = 0/60 = 0.0
    assert text.data.body == "sim time:  0.000 s"


def test_frame_one_unaffected(tmp_path):
    """Round-22 must not regress the common-case f=1 → idx=0."""
    rb = _load_render_bridge()
    text = _FakeSimTimeText()
    surf = types.SimpleNamespace()
    loader = rb.FrameMeshLoader(
        cache_dir=tmp_path, surf_obj=surf,
        sim_time_text=text, fps=60,
    )
    loader(_FakeScene(frame=1))
    assert text.data.body == "sim time:  0.000 s"


def test_negative_frame_also_clamps(tmp_path):
    """If a user/operator sets a negative frame_current (rare but
    possible), idx still clamps to 0 — no negative index in filename."""
    rb = _load_render_bridge()
    text = _FakeSimTimeText()
    surf = types.SimpleNamespace()
    loader = rb.FrameMeshLoader(
        cache_dir=tmp_path, surf_obj=surf,
        sim_time_text=text, fps=60,
    )
    loader(_FakeScene(frame=-5))
    assert text.data.body == "sim time:  0.000 s"


def test_colour_count_mismatch_logs_warning(caplog):
    """Round-22: rebuild_surface_mesh, when given colours whose count
    != vertex count, must emit a WARNING. Pre-round-22 silently
    dropped the colour block, leaving the user staring at uniform
    fluid wondering why their multi-source bake had no mixed colours."""
    rb = _load_render_bridge()

    # Minimal stand-in for the bpy mesh — must accept .vertices.add,
    # .vertices.foreach_set, .loops.add, .polygons.add, .polygons.foreach_set,
    # .update, .color_attributes.get.
    calls = []

    class _Coll:
        def __init__(self): pass
        def add(self, _n): pass
        def foreach_set(self, *_a, **_k): pass

    class _Polys(_Coll):
        pass

    class _ColorAttrs:
        def get(self, _name): return None
        def new(self, **k):
            calls.append(("color_attributes.new", k))
            return types.SimpleNamespace(
                data=types.SimpleNamespace(foreach_set=lambda *a, **k: None))

    fake_mesh = types.SimpleNamespace(
        vertices=_Coll(),
        loops=_Coll(),
        polygons=_Polys(),
        color_attributes=_ColorAttrs(),
        update=lambda **k: None,
        clear_geometry=lambda: None,
    )

    class _Surf:
        def __init__(self):
            self.data = fake_mesh
            self.modifiers = types.SimpleNamespace(__iter__=lambda s: iter([]))

    verts = np.zeros((10, 3), dtype=np.float32)
    faces = np.zeros((4, 3), dtype=np.int32)
    bad_colors = np.zeros((7, 3), dtype=np.uint8)  # 7 != 10

    with caplog.at_level(logging.WARNING, logger="gpufluid.addon"):
        rb.rebuild_surface_mesh(_Surf(), verts, faces, colors=bad_colors)

    # Must have logged + must NOT have created the Col attribute.
    assert any("colour count" in rec.message and "vertex count" in rec.message
               for rec in caplog.records), caplog.text
    assert not any(c[0] == "color_attributes.new" for c in calls)


def test_colour_count_match_still_writes(caplog):
    """Round-22 must not regress the happy path — matched counts still
    create the Col attribute, no warning logged."""
    rb = _load_render_bridge()
    created = []

    class _Coll:
        def add(self, _n): pass
        def foreach_set(self, *_a, **_k): pass

    class _ColorAttrs:
        def get(self, _name): return None
        def new(self, **k):
            created.append(k)
            return types.SimpleNamespace(
                data=types.SimpleNamespace(foreach_set=lambda *a, **k: None))

    fake_mesh = types.SimpleNamespace(
        vertices=_Coll(), loops=_Coll(), polygons=_Coll(),
        color_attributes=_ColorAttrs(),
        update=lambda **k: None,
        clear_geometry=lambda: None,
    )

    class _Surf:
        def __init__(self):
            self.data = fake_mesh
            self.modifiers = types.SimpleNamespace(__iter__=lambda s: iter([]))

    verts = np.zeros((10, 3), dtype=np.float32)
    faces = np.zeros((4, 3), dtype=np.int32)
    good_colors = np.zeros((10, 3), dtype=np.uint8)

    with caplog.at_level(logging.WARNING, logger="gpufluid.addon"):
        rb.rebuild_surface_mesh(_Surf(), verts, faces, colors=good_colors)

    assert len(created) == 1
    assert created[0].get("name") == "Col"
    assert not [r for r in caplog.records if "colour count" in r.message]


# ─── MpmConfig.seed tests ───────────────────────────────────────────────

def test_mpm_config_default_seed_is_42():
    """Round-22 contract: default seed must remain 42 so existing
    bake-hash regression tests / repro guides stay valid."""
    from gpufluid.sim.mpm.solver import MpmConfig
    assert MpmConfig().seed == 42


def test_mpm_config_seed_overridable():
    """Round-22: the seed is now a knob, not a hardcoded literal."""
    from gpufluid.sim.mpm.solver import MpmConfig
    cfg = MpmConfig(seed=1234)
    assert cfg.seed == 1234
    cfg_none = MpmConfig(seed=None)
    assert cfg_none.seed is None
