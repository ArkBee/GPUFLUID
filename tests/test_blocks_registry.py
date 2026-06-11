"""Registry contract enforcement — see docs/DESIGN.md §3.2.

This file is the executable form of the §3.2 spec. Each `test_check_N_*`
maps 1:1 onto a row in the §3.2.1 table. Checks 1-5 are hard; check 6 is
xfail-strict-false (warnings, not failures).

The tests load EVERY production module under src/gpufluid/ and the addon
(behind a bpy stub) so the registry is fully populated before assertion.
Without this priming step the registry is whatever happened to be imported
by earlier tests, which makes failures order-dependent and flaky.
"""
from __future__ import annotations
import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "gpufluid"
ADDON_ROOT = REPO_ROOT / "addon" / "gpufluid_blender"
DESIGN_MD = REPO_ROOT / "docs" / "DESIGN.md"
BLOCKS_MD = REPO_ROOT / "docs" / "BLOCKS.md"


# ---------------------------------------------------------- bpy stub helper

def _install_bpy_stub() -> None:
    """Install/augment a bpy stub so addon modules import without Blender.

    Earlier addon tests may have installed thinner stubs; we top up any
    missing attributes so the full addon tree imports. Idempotent.
    """
    bpy = sys.modules.get("bpy") or types.ModuleType("bpy")
    bpy.types = getattr(bpy, "types", None) or types.ModuleType("bpy.types")
    bpy.props = getattr(bpy, "props", None) or types.ModuleType("bpy.props")
    bpy.utils = getattr(bpy, "utils", None) or types.ModuleType("bpy.utils")

    def _empty(name):
        return type(name, (), {})

    for cls_name in ("PropertyGroup", "Operator", "Panel", "AddonPreferences",
                     "Scene", "Object"):
        if not hasattr(bpy.types, cls_name):
            setattr(bpy.types, cls_name, _empty(cls_name))

    def _prop_factory(*a, **kw):
        return None
    for name in ("FloatProperty", "IntProperty", "BoolProperty",
                 "StringProperty", "EnumProperty", "FloatVectorProperty",
                 "PointerProperty", "CollectionProperty"):
        if not hasattr(bpy.props, name):
            setattr(bpy.props, name, _prop_factory)
    if not hasattr(bpy.utils, "register_class"):
        bpy.utils.register_class = lambda cls: None
    if not hasattr(bpy.utils, "unregister_class"):
        bpy.utils.unregister_class = lambda cls: None

    # audit-20260610: bpy.app.handlers.persistent is used as a DECORATOR at
    # module-import time (cache_loader/__init__.py) — without it the whole
    # cache_loader package fails to import and its A8.6 blocks never
    # register. Top up rather than replace (earlier tests may have left a
    # thinner namespace-style stub).
    app = getattr(bpy, "app", None) or types.SimpleNamespace()
    handlers = getattr(app, "handlers", None) or types.SimpleNamespace()
    if not hasattr(handlers, "persistent"):
        handlers.persistent = lambda f: f
    for hook in ("frame_change_pre", "load_post"):
        if not hasattr(handlers, hook):
            setattr(handlers, hook, [])
    app.handlers = handlers
    bpy.app = app
    for attr, default in (("path", types.SimpleNamespace(abspath=lambda p: p)),
                          ("data", types.SimpleNamespace(filepath="",
                                                         objects={})),
                          ("context", types.SimpleNamespace()),
                          ("ops", types.SimpleNamespace())):
        if not hasattr(bpy, attr):
            setattr(bpy, attr, default)

    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy.types
    sys.modules["bpy.props"] = bpy.props
    sys.modules["bpy.utils"] = bpy.utils

    # audit-20260610 (root cause of the latent registry failure): the stub
    # never provided `mathutils`, so bake.py / _animation.py / the package
    # __init__ chain ALL failed import with ModuleNotFoundError — silently,
    # because load_all_modules swallowed ImportError (§9.10). Minimal
    # mathutils so the addon tree imports fully; top up an existing thinner
    # stub (e.g. test_audit_20260610_addon leaves one with only Vector).
    mathutils = sys.modules.get("mathutils") or types.ModuleType("mathutils")

    class _Stub:
        def __init__(self, *a, **k):
            pass

    for cls_name in ("Vector", "Matrix", "Euler", "Quaternion"):
        if not hasattr(mathutils, cls_name):
            setattr(mathutils, cls_name, type(cls_name, (_Stub,), {}))
    sys.modules["mathutils"] = mathutils


@pytest.fixture(scope="module")
def populated_registry():
    """Import every production module so the registry is fully populated.

    Yields the CheckReport from a fresh `run_checks()` call so individual
    test functions can inspect specific check results without re-running
    the whole sweep.
    """
    _install_bpy_stub()
    # audit-20260610: earlier tests (e.g. test_addon_preload_lru) leave a
    # STUB `gpufluid_blender` package in sys.modules — a bare ModuleType
    # with __path__ but without the real __init__ body (no ADDON_PKG, no
    # submodule imports). The sweep would then resolve submodule parents
    # against that stub and die with a confusing "cannot import name
    # 'ADDON_PKG' from 'gpufluid_blender' (unknown location)" — exactly
    # the order-dependent flakiness this fixture's docstring promises to
    # avoid. Evict all cached addon entries so the sweep imports the real
    # package fresh.
    for name in [n for n in sys.modules
                 if n == "gpufluid_blender"
                 or n.startswith("gpufluid_blender.")]:
        del sys.modules[name]
    from gpufluid.blocks.check import load_all_modules, run_checks
    loaded, failed = load_all_modules()
    # audit-20260610 (§9.10): a module skipped on ImportError is a module
    # whose @block decorators never fired. Pre-audit this was SILENT: the
    # stub lacked `mathutils`, all 16 addon modules were skipped, zero A8
    # blocks registered, and check 3 emitted 17 misleading
    # "impl-row-without-registry" errors with no hint of the real cause.
    # Addon failures are now a hard fixture failure (the stub above is
    # supposed to make the WHOLE addon importable); core failures are
    # printed loudly but tolerated (optional heavy deps may be absent in
    # minimal environments).
    addon_failed = [f for f in failed if f[0].startswith("gpufluid_blender")]
    if addon_failed:
        pytest.fail(
            "addon modules failed to import under the bpy stub — their A8 "
            "blocks did NOT register (extend _install_bpy_stub for whatever "
            "is missing):\n  "
            + "\n  ".join(f"{name}: {err}" for name, err in addon_failed))
    if failed:
        print(f"\n[populated_registry] {len(failed)} core module(s) skipped "
              f"on ImportError (their blocks did not register):")
        for name, err in failed:
            print(f"  {name}: {err}")
    return run_checks()


# ---------------------------------------------------------- per-check tests

def test_check_1_id_format(populated_registry):
    """Every registered ID matches the spec regex."""
    errs = [e for e in populated_registry.errors if e.check == 1]
    assert not errs, "Bad-format block IDs:\n  " + "\n  ".join(str(e) for e in errs)


def test_check_2_layer_declared(populated_registry):
    """Every layer prefix in the registry has a DESIGN.md §§4-11 heading."""
    errs = [e for e in populated_registry.errors if e.check == 2]
    assert not errs, "Layers used by code but not declared in DESIGN.md:\n  " + \
        "\n  ".join(str(e) for e in errs)


def test_check_3_blocks_md_in_sync(populated_registry):
    """BLOCKS.md rows ↔ registry one-to-one.

    A row marked `plan` must have zero registry entries.
    A row marked `impl`/`impl,test` must have ≥1 registry entry.
    A registered ID must have exactly one BLOCKS.md row.
    """
    errs = [e for e in populated_registry.errors if e.check == 3]
    assert not errs, "BLOCKS.md ↔ registry drift:\n  " + \
        "\n  ".join(str(e) for e in errs)


def test_check_4_no_cross_layer_import_violations(populated_registry):
    """Layer N may only import from layers 1..N-1."""
    errs = [e for e in populated_registry.errors if e.check == 4]
    assert not errs, "Cross-layer import violations:\n  " + \
        "\n  ".join(str(e) for e in errs)


def test_check_5_no_duplicate_id_qualname(populated_registry):
    """Same callable registered twice under the same ID is a bug."""
    errs = [e for e in populated_registry.errors if e.check == 5]
    assert not errs, "Duplicate (id, qualname) registrations:\n  " + \
        "\n  ".join(str(e) for e in errs)


def test_check_6_every_block_has_a_test(populated_registry):
    """Every registered block is mentioned in at least one test (by
    slug in a name OR by literal `[BLK X.Y.Z]` reference in a test
    file's contents).

    Started as warning-level (xfail) per DESIGN.md §3.2.1; flipped to
    hard once the index reached zero warnings, so new blocks must
    carry a test reference from day one.
    """
    warns = [w for w in populated_registry.warnings if w.check == 6]
    assert not warns, "Blocks missing a guard test:\n  " + \
        "\n  ".join(str(w) for w in warns)


# ---------------------------------------------------- meta-checks on tooling

def test_design_md_folder_map_parses():
    """The DESIGN.md §2.1 folder→layer map is parseable.

    This is the SoT for check 4. A malformed §2.1 must fail loudly here,
    not silently weaken the import check.
    """
    from gpufluid.blocks.check import parse_folder_map
    fmap = parse_folder_map(DESIGN_MD)
    assert "primitives" in fmap and fmap["primitives"] == "G1"
    assert "solvers" in fmap and fmap["solvers"] == "F3"
    assert "cli" in fmap and fmap["cli"] == "C7"


def test_design_md_declared_layers_parses():
    """§§4-11 layer headings parse into the expected layer set."""
    from gpufluid.blocks.check import parse_declared_layers
    layers = parse_declared_layers(DESIGN_MD)
    # Eight canonical layers must all be present.
    for l in ("G1", "S2", "F3", "D4", "M5", "I6", "C7", "A8"):
        assert l in layers, f"Layer {l} missing from DESIGN.md §§4-11"


def test_blocks_md_has_auto_gen_header():
    """BLOCKS.md carries the `<!-- generated by ... -->` sentinel."""
    text = BLOCKS_MD.read_text(encoding="utf-8")
    assert "generated by gpufluid blocks --regen-index" in text, \
        "BLOCKS.md missing the auto-gen header. Run `gpufluid blocks --regen-index`."
