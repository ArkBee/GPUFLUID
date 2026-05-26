"""Round-13 invariant suite for the PreloadCache class extraction.

Senior architect (round-12) flagged the module-global `_PRELOAD`
OrderedDict as the #1 source of round-3/5/7/8 regressions: every new
feature carried ~50% chance of re-opening a leak/dead-pointer bug
because the mutation surface was 8 helper functions touching one
shared dict. Round-13 wrapped that into `PreloadCache` with explicit
attach/touch/invalidate/prune/free_all/get_frame API.

This file enforces the INVARIANTS the class promises (per its docstring),
not specific bug regressions. If a future refactor breaks an invariant,
these tests should fail BEFORE any new bug-class re-opens.

Each test uses a fake bpy with a counting `bpy.data.meshes.remove` so
the contract "every mesh freed by the cache had users==0" is verifiable
without a real Blender.
"""
from __future__ import annotations

import sys
import types
import pytest
from collections import OrderedDict
from pathlib import Path

_ADDON_DIR = Path(__file__).resolve().parent.parent / "addon"


# ─── Fake bpy fixture ──────────────────────────────────────────────────

class _FakeMesh:
    """Plain mesh stand-in. `users` is a regular attribute; production
    code calls `getattr(m, "users", 0)` which works on plain objects.
    The dead-struct case is modelled by a separate _DeadMesh class."""
    def __init__(self, name, users=0):
        self.name = name
        self.users = users

    def __repr__(self):
        return f"<FakeMesh {self.name} u={self.users}>"


class _DeadMesh:
    """Models a Mesh whose StructRNA has been freed externally (orphan-
    purge, addon disable/enable, file load). Attribute access on a real
    dead Blender struct raises ReferenceError, not AttributeError —
    this stand-in reproduces that."""
    def __init__(self, name):
        # __init__ allowed to touch `name` because we set it via
        # __dict__ before the descriptor magic kicks in.
        object.__setattr__(self, "_real_name", name)

    @property
    def name(self):
        raise ReferenceError("struct has been removed")

    @property
    def users(self):
        raise ReferenceError("struct has been removed")


class _FakeMeshesCollection:
    def __init__(self):
        self.removed: list[str] = []

    def remove(self, mesh, do_unlink=True):
        # Production calls remove() only after checking users == 0
        # (_free_table_static guard). Assert the invariant.
        try:
            u = getattr(mesh, "users", 0)
        except ReferenceError:
            # Already-dead struct — production shouldn't reach .remove()
            # on this, _free_table_static skips via the eager name probe.
            pytest.fail("remove() called on a dead struct — pre-check missing")
        assert u == 0, (
            f"contract violation: removed mesh '{mesh.name}' with users={u}")
        self.removed.append(mesh.name)
        # Flip to dead state for any subsequent attribute access.
        mesh.__class__ = _DeadMesh
        mesh.__dict__["_real_name"] = mesh.__dict__.get("name", "?")


class _FakeScene:
    def __init__(self, obj_names):
        self.objects = [types.SimpleNamespace(name=n) for n in obj_names]


def _make_fake_bpy(scene_objs: list[str] | None = None,
                   live_objs: list[str] | None = None):
    """Mock just enough bpy for cache_loader to import + run.

    scene_objs: object names in the active scene (controls prune_stale).
    live_objs: object names with a `.data = <mesh>` setter so the
               pre-swap pattern in attach() works.
    """
    scene_objs = scene_objs or []
    live_objs = live_objs or []
    meshes_coll = _FakeMeshesCollection()

    # _FakeObject with a `data` descriptor that bumps mesh.users on
    # assignment — mimics bpy's reference counting. Defined per call
    # so different fake bpy instances don't share state.
    class _FakeObject:
        def __init__(self, n):
            self.name = n
            self.__dict__["_data"] = None
        @property
        def data(self):
            return self.__dict__["_data"]
        @data.setter
        def data(self, value):
            prev = self.__dict__["_data"]
            if prev is not None and hasattr(prev, "users"):
                prev.users = max(0, prev.users - 1)
            self.__dict__["_data"] = value
            if value is not None and hasattr(value, "users"):
                value.users += 1

    objects_map: dict[str, _FakeObject] = {}
    for name in live_objs:
        objects_map[name] = _FakeObject(name)

    class _ObjectsAccessor:
        def get(self, name, default=None):
            return objects_map.get(name, default)
        def __iter__(self): return iter(objects_map.values())
        def __contains__(self, name): return name in objects_map
        def remove(self, obj, do_unlink=True):
            objects_map.pop(obj.name, None)

    fake_bpy = types.ModuleType("bpy")
    fake_bpy.app = types.SimpleNamespace(
        handlers=types.SimpleNamespace(
            frame_change_pre=[], load_post=[],
            persistent=lambda f: f,
        )
    )
    fake_bpy.data = types.SimpleNamespace(
        meshes=meshes_coll,
        objects=_ObjectsAccessor(),
        scenes=[_FakeScene(scene_objs)],
    )
    fake_bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(addons={}),
        scene=fake_bpy.data.scenes[0],
    )
    fake_bpy.types = types.SimpleNamespace(
        Operator=type("_Op", (), {}),
        PropertyGroup=type("_PG", (), {}),
        Mesh=_FakeMesh,
        AddonPreferences=type("_AP", (), {}),
        Panel=type("_Panel", (), {}),
    )
    _prop = lambda **k: None
    fake_bpy.props = types.SimpleNamespace(
        StringProperty=_prop, FloatProperty=_prop,
        FloatVectorProperty=_prop, IntProperty=_prop,
        BoolProperty=_prop, EnumProperty=_prop,
        PointerProperty=_prop,
    )
    fake_bpy.path = types.SimpleNamespace(abspath=lambda p: p)
    sys.modules["bpy"] = fake_bpy
    return fake_bpy, meshes_coll, objects_map


def _load_cache_loader():
    """Re-import cache_loader against the current bpy stub."""
    import logging
    pkg = types.ModuleType("gpufluid_blender")
    pkg.__path__ = [str(_ADDON_DIR / "gpufluid_blender")]
    pkg.logger = logging.getLogger("gpufluid.addon.test")
    sys.modules["gpufluid_blender"] = pkg

    # Clear any prior load so stub fresh-binds.
    sys.modules.pop("gpufluid_blender.cache_loader", None)

    import importlib.util
    cl_dir = _ADDON_DIR / "gpufluid_blender" / "cache_loader"
    spec = importlib.util.spec_from_file_location(
        "gpufluid_blender.cache_loader",
        cl_dir / "__init__.py",
        submodule_search_locations=[str(cl_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpufluid_blender.cache_loader"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Invariant 1: attach-replace never orphans old current-mesh ────────

def test_attach_replace_pre_swap_releases_old_current_mesh():
    """When attach() REPLACES an existing entry, the live object's
    .data must be redirected to a fresh mesh from the new table
    BEFORE the old table is freed — so the old current-frame mesh's
    users drops to 0 and _free_table_static actually removes it.

    Without the pre-swap pattern, every re-attach leaked the old
    current-frame _seq_*** datablock (live-found bug, round-3)."""
    bpy_mod, meshes_coll, objects_map = _make_fake_bpy(live_objs=["cobj"])
    cl = _load_cache_loader()

    # Build old table; mesh[0] becomes pinned via the descriptor setter
    old = {i: _FakeMesh(f"old_{i}", users=0) for i in range(3)}
    objects_map["cobj"].data = old[0]   # setter bumps old[0].users 0→1

    cl._PRELOAD.attach("cobj", old, cap=8)
    assert "cobj" in cl._PRELOAD

    # New table for replacement. Pre-swap should redirect obj.data to
    # new[0], dropping old[0].users to 0 via our setter side-effect.
    new = {i: _FakeMesh(f"new_{i}", users=0) for i in range(3)}

    # _FakeObject.data descriptor (built in _make_fake_bpy) already
    # bumps mesh.users on assignment. Re-attach old[0] via setter
    # so users counter is correct.
    cobj = objects_map["cobj"]
    cobj.data = old[0]   # users 0→1
    assert old[0].users == 1, "pre-swap test fixture broken"

    cl._PRELOAD.attach("cobj", new, cap=8)

    # CONTRACT: post-replace, ALL of `old`'s meshes (incl. old[0] that
    # was previously users=1) removed. Pre-swap pattern dropped old[0].
    assert "cobj" in cl._PRELOAD
    assert cl._PRELOAD["cobj"] is new
    for old_name in ("old_0", "old_1", "old_2"):
        assert old_name in meshes_coll.removed, (
            f"INVARIANT BROKEN: '{old_name}' leaked after attach-replace. "
            f"Removed: {meshes_coll.removed}")


# ─── Invariant 2: prune_stale matches scene.objects ────────────────────

def test_prune_stale_removes_only_keys_not_in_scene():
    bpy_mod, meshes_coll, _ = _make_fake_bpy(
        scene_objs=["alive"])
    cl = _load_cache_loader()

    cl._PRELOAD.attach("alive", {0: _FakeMesh("a0", users=0)}, cap=8)
    cl._PRELOAD.attach("dead1", {0: _FakeMesh("d10", users=0)}, cap=8)
    cl._PRELOAD.attach("dead2", {0: _FakeMesh("d20", users=0)}, cap=8)

    pruned = cl._PRELOAD.prune_stale(bpy_mod.data.scenes[0])

    assert pruned == 2
    assert "alive" in cl._PRELOAD
    assert "dead1" not in cl._PRELOAD
    assert "dead2" not in cl._PRELOAD
    assert sorted(meshes_coll.removed) == ["d10", "d20"]


def test_prune_stale_union_mode_with_no_scene_arg():
    bpy_mod, meshes_coll, _ = _make_fake_bpy()
    bpy_mod.data.scenes = [
        _FakeScene(["alive_a"]),
        _FakeScene(["alive_b"]),
    ]
    cl = _load_cache_loader()

    cl._PRELOAD.attach("alive_a", {0: _FakeMesh("aa0", users=0)}, cap=8)
    cl._PRELOAD.attach("alive_b", {0: _FakeMesh("ab0", users=0)}, cap=8)
    cl._PRELOAD.attach("nowhere", {0: _FakeMesh("nw0", users=0)}, cap=8)

    pruned = cl._PRELOAD.prune_stale(scene=None)
    assert pruned == 1
    assert "nowhere" not in cl._PRELOAD
    assert meshes_coll.removed == ["nw0"]


# ─── Invariant 3: free_all clears completely ───────────────────────────

def test_free_all_empties_cache_and_frees_meshes():
    bpy_mod, meshes_coll, _ = _make_fake_bpy()
    cl = _load_cache_loader()

    cl._PRELOAD.attach("a", {i: _FakeMesh(f"a{i}") for i in range(3)}, cap=8)
    cl._PRELOAD.attach("b", {i: _FakeMesh(f"b{i}") for i in range(2)}, cap=8)

    assert len(cl._PRELOAD) == 2
    cl._PRELOAD.free_all()
    assert len(cl._PRELOAD) == 0
    assert sorted(meshes_coll.removed) == ["a0", "a1", "a2", "b0", "b1"]


# ─── Invariant 4: invalidate is idempotent + frees ─────────────────────

def test_invalidate_existing_frees_and_drops():
    bpy_mod, meshes_coll, _ = _make_fake_bpy()
    cl = _load_cache_loader()

    cl._PRELOAD.attach("x", {0: _FakeMesh("x0"), 1: _FakeMesh("x1")}, cap=8)
    assert cl._PRELOAD.invalidate("x") is True
    assert "x" not in cl._PRELOAD
    assert sorted(meshes_coll.removed) == ["x0", "x1"]


def test_invalidate_missing_returns_false_no_op():
    bpy_mod, meshes_coll, _ = _make_fake_bpy()
    cl = _load_cache_loader()
    assert cl._PRELOAD.invalidate("nothing-here") is False
    assert meshes_coll.removed == []


# ─── Invariant 5: LRU evicts oldest when cap exceeded ──────────────────

def test_attach_evicts_oldest_when_over_cap():
    bpy_mod, meshes_coll, _ = _make_fake_bpy()
    cl = _load_cache_loader()

    for i in range(5):
        cl._PRELOAD.attach(f"obj_{i}",
                            {0: _FakeMesh(f"m_{i}_0")},
                            cap=3)

    # Cap=3 → after 5 attaches, only obj_2/3/4 remain; obj_0/1 evicted.
    assert sorted(cl._PRELOAD.keys()) == ["obj_2", "obj_3", "obj_4"]
    assert "m_0_0" in meshes_coll.removed
    assert "m_1_0" in meshes_coll.removed


# ─── Invariant 6: get_frame returns None for missing, not KeyError ─────

def test_get_frame_missing_key_returns_none():
    bpy_mod, _, _ = _make_fake_bpy()
    cl = _load_cache_loader()
    assert cl._PRELOAD.get_frame("nonexistent", 0) is None
    cl._PRELOAD.attach("x", {0: _FakeMesh("x0")}, cap=8)
    assert cl._PRELOAD.get_frame("x", 999) is None
    assert cl._PRELOAD.get_frame("x", 0) is not None


# ─── Invariant 7: _free_table_static defensive against ReferenceError ──

def test_free_table_static_survives_dead_struct():
    """When external code (orphans_purge, file load) frees the Mesh
    StructRNA out from under us, attribute access raises
    ReferenceError. _free_table_static must NOT propagate that;
    it must drop the stale pointer silently."""
    bpy_mod, _, _ = _make_fake_bpy()
    cl = _load_cache_loader()

    # Use _DeadMesh class directly (not a mutation of _FakeMesh —
    # earlier attempt did `type(dead).name = property(...)` which
    # leaked the descriptor onto _FakeMesh itself and broke every
    # subsequent test's `_FakeMesh(name, ...)` constructor).
    dead = _DeadMesh("dead")
    table = {0: dead, 1: _FakeMesh("alive", users=0)}
    # Should not raise.
    cl.PreloadCache._free_table_static(table)


# ─── Back-compat: old module-level shims still work ───────────────────

def test_module_shims_delegate_to_singleton():
    bpy_mod, meshes_coll, _ = _make_fake_bpy(live_objs=["cobj"])
    cl = _load_cache_loader()

    # Old call site pattern from _ops.py / helpers.py
    cl._PRELOAD["x"] = {0: _FakeMesh("x0")}
    assert "x" in cl._PRELOAD
    cl._free_table(cl._PRELOAD["x"])
    del cl._PRELOAD["x"]
    assert "x" not in cl._PRELOAD
    assert "x0" in meshes_coll.removed

    cl._PRELOAD["y"] = {0: _FakeMesh("y0")}
    cl._lru_touch("y")
    assert list(cl._PRELOAD.keys())[-1] == "y"
