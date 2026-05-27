"""Round-18 contract test: ADDON_PKG is single source of truth.

Senior code-smell #5 (lesson 9.10): `__package__.rsplit('.', 1)[0]` in
nested submodules worked by coincidence — one level deep only. A
future nested-deeper module (e.g. `cache_loader/_sub/file.py`) would
silently mis-key the prefs lookup. Round-18 centralised the addon-root
package name on `addon/gpufluid_blender/__init__.py::ADDON_PKG`.

These tests enforce that every prefs lookup goes through ADDON_PKG,
not raw `__package__` or rsplit. Pure source-grep — no bpy needed.
"""
from __future__ import annotations

from pathlib import Path

_ADDON_DIR = Path(__file__).resolve().parent.parent / "addon" / "gpufluid_blender"


def test_addon_pkg_constant_exists_at_root():
    """The constant must live at the addon root (`__init__.py`), not
    in a submodule — otherwise nested modules can't import it via
    `from . import ADDON_PKG` consistently."""
    src = (_ADDON_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "ADDON_PKG: str = __package__" in src, (
        "ADDON_PKG must be declared at the addon root via __package__ "
        "(round-18 contract — see CLAUDE.md lesson 9.10)")


def test_no_rsplit_pattern_in_addon():
    """The anti-pattern `__package__.rsplit('.', 1)[0]` must NOT appear
    anywhere in the addon source. Comments documenting WHY we don't use
    it are allowed (commented mention is fine)."""
    for py in _ADDON_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue   # comment-only mention is OK
            assert "__package__.rsplit" not in line, (
                f"{py.relative_to(_ADDON_DIR.parent.parent)}:{n}: "
                f"rsplit pattern banned — use `from .. import ADDON_PKG`")


def test_prefs_lookup_uses_addon_pkg_not_raw_package():
    """Every `bpy.context.preferences.addons[X].preferences` call must
    use ADDON_PKG (or _ADDON_PKG alias), not `__package__` directly.
    Direct __package__ would mis-key from a submodule.

    Exception: `bl_idname = __package__` on AddonPreferences class IS
    correct — that's Blender's registration key and the class lives
    at addon root by definition.
    """
    forbidden_pattern = "addons[__package__]"
    for py in _ADDON_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            assert forbidden_pattern not in line, (
                f"{py.relative_to(_ADDON_DIR.parent.parent)}:{n}: "
                f"`{forbidden_pattern}` — use ADDON_PKG instead "
                f"(round-18 contract)")


def test_cache_loader_imports_addon_pkg():
    """The cache_loader package's _ADDON_PKG alias must import from
    the addon root as the PRIMARY source. A test-stub fallback
    (rsplit) is allowed only when wrapped in `try/except ImportError`
    — that path fires only under unit-test stub loaders, not real
    Blender."""
    src = (_ADDON_DIR / "cache_loader" / "__init__.py").read_text(encoding="utf-8")
    assert "from .. import ADDON_PKG" in src, (
        "cache_loader must primary-import ADDON_PKG from addon root")
    # Verify the rsplit-fallback is INSIDE a try/except (not at module
    # top level as the primary path).
    rsplit_idx = src.find("_ADDON_PKG = (__package__")
    if rsplit_idx > -1:
        # Check the preceding ~5 lines contain `except ImportError`.
        preceding = src[max(0, rsplit_idx - 300):rsplit_idx]
        assert "except ImportError" in preceding, (
            "rsplit fallback for _ADDON_PKG must be inside "
            "`try/except ImportError`, not at module top level")
