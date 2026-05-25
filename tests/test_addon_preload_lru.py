"""Phase 2 — preload LRU eviction.

The addon's `_PRELOAD` was an unbounded module-level dict; in long Blender
sessions (compare-bake-iterate workflows) it accumulated thousands of
mesh datablocks until the process was killed. Phase 2 wraps it in an
OrderedDict with an LRU cap pulled from preferences.

This test runs without bpy by monkey-patching the bpy-touching helpers
in `gpufluid_blender.cache_loader` with plain stubs. We exercise the
public LRU surface (`_lru_install`, `_lru_touch`, `_preload_cap` hook)
and assert:
  * len(_PRELOAD) never exceeds cap
  * least-recently-touched entries are evicted first
  * `_free_table` is called on each evicted entry's meshes (release path)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Stub bpy BEFORE importing cache_loader (it does `import bpy` at module top).
_ADDON_DIR = Path(__file__).resolve().parents[1] / "addon"
if str(_ADDON_DIR) not in sys.path:
    sys.path.insert(0, str(_ADDON_DIR))

if "bpy" not in sys.modules:
    # Minimal bpy stub. cache_loader and (transitively) properties.py need
    # bpy.types.PropertyGroup / bpy.types.Operator / bpy.props.* — these
    # only have to be importable, not behave correctly. We import the
    # gpufluid_blender package (rather than cache_loader directly) so we
    # exercise the same import graph the real addon takes.
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.app = types.SimpleNamespace(
        handlers=types.SimpleNamespace(frame_change_pre=[])
    )

    class _FakeMeshes:
        def remove(self, mesh, do_unlink=True):
            pass

    fake_bpy.data = types.SimpleNamespace(meshes=_FakeMeshes())

    class _PG: pass
    class _Op: pass
    class _AddonPrefs: pass
    fake_bpy.types = types.SimpleNamespace(
        Operator=_Op, Mesh=type("Mesh", (), {}),
        PropertyGroup=_PG, AddonPreferences=_AddonPrefs,
        Panel=type("Panel", (), {}),
    )

    def _prop(*a, **k):
        return None
    fake_bpy.props = types.SimpleNamespace(
        StringProperty=_prop, FloatProperty=_prop,
        FloatVectorProperty=_prop, IntProperty=_prop,
        BoolProperty=_prop, EnumProperty=_prop, PointerProperty=_prop,
    )
    fake_bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    fake_bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(addons={})
    )
    sys.modules["bpy"] = fake_bpy

# Stub the gpufluid_blender package WITHOUT running its __init__
# (which would import properties/panels/operators that need a real bpy).
# Then load cache_loader as `gpufluid_blender.cache_loader` so its
# `from . import logger` resolves to our stub.
import importlib.util  # noqa: E402
import logging  # noqa: E402
_pkg = types.ModuleType("gpufluid_blender")
_pkg.__path__ = [str(_ADDON_DIR / "gpufluid_blender")]
_pkg.logger = logging.getLogger("gpufluid.addon.test")
sys.modules["gpufluid_blender"] = _pkg

_spec = importlib.util.spec_from_file_location(
    "gpufluid_blender.cache_loader",
    _ADDON_DIR / "gpufluid_blender" / "cache_loader.py",
)
cache_loader = importlib.util.module_from_spec(_spec)
sys.modules["gpufluid_blender.cache_loader"] = cache_loader
_spec.loader.exec_module(cache_loader)


class _FakeMesh:
    """Stand-in for bpy.types.Mesh with the attrs cache_loader touches."""
    def __init__(self, name: str, users: int = 0):
        self.name = name
        self.users = users


@pytest.fixture(autouse=True)
def _reset_preload(monkeypatch):
    """Each test starts with an empty _PRELOAD and a captured remove() list."""
    cache_loader._PRELOAD.clear()
    removed = []
    monkeypatch.setattr(
        cache_loader.bpy.data, "meshes",
        types.SimpleNamespace(remove=lambda m, do_unlink=True: removed.append(m)),
    )
    yield removed
    cache_loader._PRELOAD.clear()


def _make_table(prefix: str, n: int = 3) -> dict:
    return {i: _FakeMesh(f"{prefix}_{i:04d}") for i in range(n)}


def test_lru_install_below_cap_keeps_all(_reset_preload, monkeypatch):
    monkeypatch.setattr(cache_loader, "_preload_cap", lambda: 4)
    for i in range(3):
        cache_loader._lru_install(f"obj{i}", _make_table(f"obj{i}"))
    assert list(cache_loader._PRELOAD.keys()) == ["obj0", "obj1", "obj2"]
    assert _reset_preload == []  # nothing freed yet


def test_lru_install_above_cap_evicts_oldest(_reset_preload, monkeypatch):
    monkeypatch.setattr(cache_loader, "_preload_cap", lambda: 3)
    for i in range(5):
        cache_loader._lru_install(f"obj{i}", _make_table(f"obj{i}"))
    # cap=3 → oldest two (obj0, obj1) evicted; obj2,obj3,obj4 remain.
    assert list(cache_loader._PRELOAD.keys()) == ["obj2", "obj3", "obj4"]
    # Each evicted table had 3 meshes, all with users=0 → all freed.
    assert len(_reset_preload) == 2 * 3
    freed_names = {m.name for m in _reset_preload}
    assert freed_names == {"obj0_0000", "obj0_0001", "obj0_0002",
                            "obj1_0000", "obj1_0001", "obj1_0002"}


def test_lru_touch_protects_from_eviction(_reset_preload, monkeypatch):
    monkeypatch.setattr(cache_loader, "_preload_cap", lambda: 3)
    for i in range(3):
        cache_loader._lru_install(f"obj{i}", _make_table(f"obj{i}"))
    # Touch obj0 — now obj1 is oldest.
    cache_loader._lru_touch("obj0")
    cache_loader._lru_install("obj3", _make_table("obj3"))
    assert list(cache_loader._PRELOAD.keys()) == ["obj2", "obj0", "obj3"]
    # obj1 evicted.
    freed_names = {m.name for m in _reset_preload}
    assert freed_names == {"obj1_0000", "obj1_0001", "obj1_0002"}


def test_lru_install_skips_mesh_with_live_users(_reset_preload, monkeypatch):
    """A mesh that's still attached to a live object (users>0) must NOT
    be removed by the evictor — only the dict handle is dropped."""
    monkeypatch.setattr(cache_loader, "_preload_cap", lambda: 1)
    pinned_table = {0: _FakeMesh("pinned_0000", users=2)}
    cache_loader._lru_install("pinned", pinned_table)
    cache_loader._lru_install("fresh", _make_table("fresh"))
    # 'pinned' got evicted, but its mesh had users>0 → no remove call.
    assert "pinned" not in cache_loader._PRELOAD
    freed_names = {m.name for m in _reset_preload}
    assert "pinned_0000" not in freed_names
    # The fresh table's meshes are still resident → also no remove yet.
    assert _reset_preload == []


def test_n_plus_5_attaches_settles_at_cap(_reset_preload, monkeypatch):
    """After N+5 LRU installs with cap=N, the dict holds exactly N entries
    — the 5 oldest are evicted in order."""
    N = 4
    monkeypatch.setattr(cache_loader, "_preload_cap", lambda: N)
    for i in range(N + 5):
        cache_loader._lru_install(f"obj{i}", _make_table(f"obj{i}"))
    assert len(cache_loader._PRELOAD) == N
    expected = [f"obj{i}" for i in range(5, N + 5)]
    assert list(cache_loader._PRELOAD.keys()) == expected


def test_preload_max_frames_default_when_no_prefs():
    """When bpy.context.preferences can't reach our addon (e.g. unit test
    without registration), _preload_max_frames falls back to 10000 — not
    the legacy hard-coded 2000."""
    assert cache_loader._preload_max_frames() == 10000
