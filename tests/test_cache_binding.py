"""Round-19: cache_binding contract tests.

Senior code-smell #4: ~10 magic-string keys (`obj["gpufluid_cache_dir"]`,
etc) consumed across 5 files with subtly different name sets between
cache binding and bake-trace. Round-19 centralised everything here.

These tests prove:
  1. Round-trip: set_X then get_X returns the same data.
  2. clear_all_bindings really clears every key set we manage.
  3. Source-grep gate: magic strings appear ONLY in cache_binding.py
     (and tests) — any other source file using the raw key strings
     fails this gate (round-5 bug class prevention).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_ADDON_DIR = Path(__file__).resolve().parent.parent / "addon"


def _load_cb():
    name = "_test_cache_binding"
    spec = importlib.util.spec_from_file_location(
        name, _ADDON_DIR / "gpufluid_blender" / "cache_binding.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeObj:
    """Stand-in for bpy.types.Object — supports `obj[key] = v`,
    `obj.get(key)`, `obj.keys()`, `del obj[key]`. That's the full
    bpy custom-prop surface cache_binding touches."""
    def __init__(self):
        self._d: dict = {}
    def __setitem__(self, k, v): self._d[k] = v
    def __getitem__(self, k): return self._d[k]
    def __delitem__(self, k): del self._d[k]
    def get(self, k, default=None): return self._d.get(k, default)
    def keys(self): return self._d.keys()


# ─── Round-trip ─────────────────────────────────────────────────────────

def test_cache_binding_round_trip():
    cb = _load_cb()
    obj = _FakeObj()
    cb.set_cache_binding(obj, cache_dir="/tmp/x",
                         origin=(1.0, 2.0, 3.0),
                         dom_size=(4.0, 5.0, 6.0),
                         frame_offset=7)
    assert cb.has_cache_binding(obj)
    assert cb.get_cache_dir(obj) == "/tmp/x"
    assert cb.get_cache_origin(obj) == (1.0, 2.0, 3.0)
    assert cb.get_cache_dom_size(obj) == (4.0, 5.0, 6.0)
    assert cb.get_cache_frame_offset(obj) == 7
    assert cb.get_cache_pattern(obj) == "mesh/frame_{:04d}.ply"


def test_ww_binding_round_trip():
    cb = _load_cb()
    obj = _FakeObj()
    cb.set_ww_binding(obj, cache_dir="/tmp/ww",
                       origin=(1.5, 2.5, 3.5), frame_offset=2)
    assert cb.has_ww_binding(obj)
    assert cb.get_ww_cache_dir(obj) == "/tmp/ww"
    assert cb.get_ww_cache_origin(obj) == (1.5, 2.5, 3.5)
    assert cb.get_ww_cache_frame_offset(obj) == 2


def test_bake_trace_round_trip():
    cb = _load_cb()
    obj = _FakeObj()
    cb.set_bake_trace(obj, cache_dir="/tmp/bake",
                       origin=(0, 0, 0), dom_size=(2, 2, 2))
    trace = cb.get_bake_trace(obj)
    assert trace == {
        "cache_dir": "/tmp/bake",
        "origin": (0.0, 0.0, 0.0),
        "dom_size": (2.0, 2.0, 2.0),
    }


def test_no_binding_returns_none():
    cb = _load_cb()
    obj = _FakeObj()
    assert cb.has_cache_binding(obj) is False
    assert cb.get_cache_dir(obj) is None
    assert cb.has_ww_binding(obj) is False
    assert cb.get_ww_cache_dir(obj) is None
    assert cb.get_bake_trace(obj) is None


def test_clear_all_removes_every_key_and_returns_true():
    cb = _load_cb()
    obj = _FakeObj()
    cb.set_cache_binding(obj, "/tmp/a", (1, 1, 1), (2, 2, 2))
    cb.set_ww_binding(obj, "/tmp/b", (3, 3, 3))
    cb.set_bake_trace(obj, "/tmp/c", (4, 4, 4), (5, 5, 5))
    assert cb.clear_all_bindings(obj) is True
    # After clear: zero keys left
    assert list(obj.keys()) == []


def test_clear_all_on_empty_returns_false():
    cb = _load_cb()
    obj = _FakeObj()
    assert cb.clear_all_bindings(obj) is False


# ─── Source-grep gate: magic strings ONLY in cache_binding.py ──────────

# These are the literal strings cache_binding.py owns. Any production
# source file outside cache_binding.py mentioning them in a non-comment
# context is a regression (round-19 contract).
_OWNED_KEYS = (
    "gpufluid_cache_dir", "gpufluid_cache_pattern",
    "gpufluid_cache_frame_offset", "gpufluid_cache_origin",
    "gpufluid_cache_dom_size",
    "gpufluid_ww_cache_dir", "gpufluid_ww_cache_frame_offset",
    "gpufluid_ww_cache_origin",
    "gpufluid_origin", "gpufluid_dom_size",
)


def _scan_for_keys(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_no, key, line_text) for non-comment +
    non-docstring hits. Tracks triple-quote state so module/class/
    function docstrings (which legitimately mention key names as
    English explanation) don't trip the gate."""
    hits = []
    in_triple = False
    triple_char = None
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.lstrip()
        # Triple-quote state machine — count both `"""` and `'''` and
        # toggle on each occurrence. Multi-line docstrings span lines
        # we want to skip.
        for q in ('"""', "'''"):
            count = line.count(q)
            if count == 0:
                continue
            if in_triple and triple_char == q:
                if count >= 1:
                    in_triple = False
                    triple_char = None
            elif not in_triple:
                in_triple = True
                triple_char = q
                if count >= 2:   # opens and closes on same line
                    in_triple = False
                    triple_char = None
        if in_triple or stripped.startswith("#"):
            continue
        for key in _OWNED_KEYS:
            if key in line:
                hits.append((n, key, line.strip()))
                break
    return hits


def test_magic_keys_only_in_cache_binding_and_logger_msgs():
    """No source file outside `cache_binding.py` should mention any of
    the cache/ww/bake-trace key strings in non-comment lines. Logger
    warning messages that quote the key name for the user are the
    only exception — those are in addon-side log strings, not lookups."""
    addon_root = _ADDON_DIR / "gpufluid_blender"
    bad: list[str] = []
    for py in addon_root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(addon_root.parent.parent)
        if py.name == "cache_binding.py":
            continue   # the one file that's allowed to own these strings
        for n, key, text in _scan_for_keys(py):
            # Allow user-facing log messages that mention the key name
            # in plain English (round-19 reviewer note: hard exception).
            if any(marker in text for marker in (
                    "logger.warning", "logger.info", "_addon_logger",
                    'f"cache_loader:', 'f"cache:', '"addon.bake',
                    '"cache:', '"ww:', '"bake')):
                continue
            bad.append(f"  {rel}:{n}: '{key}' in `{text[:80]}`")
    assert not bad, (
        "Magic cache-binding keys leaked outside cache_binding.py "
        "(round-19 contract violation):\n" + "\n".join(bad))
