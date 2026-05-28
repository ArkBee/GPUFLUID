"""Round-60 — OT_clear_cache must survive WinError 145 (dir-not-empty).

Live crash (2026-05-28): user clicked Clear Cache, `shutil.rmtree` hit
`OSError [WinError 145] "Папка не пуста: round57_cache\\mesh"` and the
operator crashed. Root cause: the handler caught only `PermissionError`
(WinError 5/32). WinError 145 maps to plain `OSError` (errno ENOTEMPTY)
from the Windows async-delete race (children unlinked but the directory
listing not yet flushed when `os.rmdir` fires), so it escaped uncaught.

Fix: `_robust_rmtree` retries the whole rmtree with backoff + read-only
bit clearing, and the execute() block catches `OSError` broadly with a
clear user-facing message.

Source-grep contract (lesson §9.12) — helpers.py imports bpy at module
top so we can't import the function under pytest; assert on source text.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_HELPERS = _REPO / "addon" / "gpufluid_blender" / "operators" / "helpers.py"


def _code() -> str:
    src = _HELPERS.read_text(encoding="utf-8")
    return "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )


def test_robust_rmtree_helper_exists():
    code = _code()
    assert "def _robust_rmtree(" in code, (
        "round-60 regressed: _robust_rmtree helper missing")
    # Must retry (loop) and catch OSError broadly, not just PermissionError.
    assert "except OSError" in code, (
        "round-60: _robust_rmtree must catch OSError (WinError 145 is "
        "OSError, not PermissionError)")
    # Must clear read-only bits so the retry can unlink (WinError 5).
    assert "stat.S_IWRITE" in code, (
        "round-60: _robust_rmtree must chmod read-only files before retry")


def test_clear_cache_uses_robust_rmtree_and_catches_oserror():
    code = _code()
    # The execute() body must call the robust helper, not bare rmtree.
    assert "_robust_rmtree(cache)" in code, (
        "round-60 regressed: OT_clear_cache.execute must call "
        "_robust_rmtree instead of bare shutil.rmtree")
    # The old PermissionError-only handler must be gone from execute.
    assert "except PermissionError" not in code, (
        "round-60 regressed: clear_cache still has the narrow "
        "PermissionError handler that let WinError 145 escape")
