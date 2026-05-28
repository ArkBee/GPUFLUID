"""Round-62 — the bake must show live progress (and the result must
appear without manual scrubbing).

Live-found 2026-05-28: pressing Bake on an MPM scene looked like "nothing
happens". Two root causes:
  1. The MPM solver printed progress as "[MPM] step k/n" every 200 raw
     steps; the Blender addon's modal parser only matched "frame N/M", so
     the status bar / progress bar never advanced for MPM bakes (and the
     200-step cadence never even fired on short bakes).
  2. A modal bake's auto-attach swapped obj.data but never forced a
     viewport redraw, so the fluid only appeared after the user scrubbed.

Fix: MPM emits "[MPM] frame N/M" on the dump cadence (per output frame);
the addon regex matches frame|step case-insensitively; a wm.progress bar
opens/closes around the modal run; attach tags VIEW_3D for redraw.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_ADDON = _REPO / "addon" / "gpufluid_blender"

# Mirror of the addon's _FRAME_RE — kept in sync by test_bake_regex_broadened.
_PAT = re.compile(r"(?:frame|step)\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def _code(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


def test_regex_matches_mpm_and_flip_progress():
    cases = [
        ("  frame 0005/40", (5, 40)),                       # FLIP
        ("  [MPM] frame 5/40  z_min=0.10 z_max=0.90", (5, 40)),  # round-62 MPM
        ("  [MPM] step 200/1600", (200, 1600)),             # legacy fallback
        ("frame 12 / 60", (12, 60)),                        # spaces around /
    ]
    for line, expect in cases:
        m = _PAT.search(line)
        assert m, f"progress line not matched: {line!r}"
        assert (int(m.group(1)), int(m.group(2))) == expect


def test_bake_regex_broadened():
    code = _code(_ADDON / "operators" / "bake.py")
    assert "(?:frame|step)" in code and "re.IGNORECASE" in code, (
        "round-62 regressed: _FRAME_RE must match frame|step, "
        "case-insensitive, so MPM progress advances the bar")


def test_mpm_solver_prints_frame_on_dump_cadence():
    code = _code(_REPO / "src" / "gpufluid" / "sim" / "mpm" / "solver.py")
    assert "k % cfg.dump_every == 0" in code, (
        "round-62: MPM progress must print on the dump cadence (per output "
        "frame), not every 200 raw steps (never fired on short bakes)")
    assert "[MPM] frame" in code, (
        "round-62: MPM progress must say 'frame N/M' (was 'step', which the "
        "addon parser never matched)")
    assert "step {k}/" not in code, (
        "round-62 regressed: legacy 'step {k}/n' progress line is back")


def test_runner_opens_and_closes_progress_bar():
    code = _code(_ADDON / "operators" / "_runner.py")
    assert "progress_begin" in code and "progress_end" in code, (
        "round-62: modal runner must open + close a wm progress bar")
    # every begin paired with an end via the shared helper on both
    # teardown paths (abort + _finish).
    assert code.count("self._end_progress(context)") >= 2, (
        "round-62: _end_progress must be called from both abort and _finish")


def test_attach_forces_view3d_redraw():
    code = _code(_ADDON / "cache_loader" / "_ops.py")
    assert "tag_redraw()" in code and 'area.type == "VIEW_3D"' in code, (
        "round-62: attach must force a VIEW_3D redraw so the result appears "
        "the instant the bake finishes (no manual scrub)")
