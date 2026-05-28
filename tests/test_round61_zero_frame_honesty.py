"""Round-61 — a bake/render/attach that produces 0 usable frames must
say so, not report a green 'complete'.

Live-found 2026-05-28: a bake finished, auto-attached, and reported
"gpufluid bake complete" while `mesh/` was empty — the user saw a blank
viewport with a success message. Root cause was three converging silent
spots:

  1. The sync truncation check used ``0 < actual < expected`` — the
     ``0 < actual`` clause excluded the actual==0 case (the worst one).
  2. The modal (UI-default) bake path had no frame-count check at all
     (§9.6 mirror-drift: sync got the check in round-10, modal never).
  3. attach_cache reported ``n`` frames as plain INFO even when n==0.
  4. (sibling) render's modal path reported "render complete" on rc=0
     with zero PNGs written.

This suite has two layers:
  * UNIT — the bpy-free ``cache_sanity`` classifier (§9.2: every new
    branch gets a real test).
  * CONTRACT — source-grep guards (§9.12) so the silent forms can't
    creep back into the four operator wiring sites.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_ADDON = _REPO / "addon" / "gpufluid_blender"


def _load_cache_sanity():
    """Load the bpy-free module directly (the package __init__ imports
    bpy, which isn't available under pytest)."""
    name = "_test_cache_sanity"
    spec = importlib.util.spec_from_file_location(
        name, _ADDON / "cache_sanity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _code(path: Path) -> str:
    """Source with comment lines stripped (§9.12) so the fix history in
    comments can't satisfy or trip a grep assertion."""
    src = path.read_text(encoding="utf-8")
    return "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


# ── UNIT: the classifier (§9.2) ──────────────────────────────────────

def test_bake_zero_frames_is_error():
    cs = _load_cache_sanity()
    level, msg = cs.bake_frame_sanity(0, 600)
    assert level == "ERROR", "0 frames must escalate to ERROR, not pass"
    assert "0 mesh frames" in msg and "600" in msg


def test_bake_zero_frames_error_even_when_expected_unknown():
    cs = _load_cache_sanity()
    # toml unparseable → expected=0; the 0-frame disaster must still fire.
    level, _ = cs.bake_frame_sanity(0, 0)
    assert level == "ERROR"


def test_bake_truncated_is_warning():
    cs = _load_cache_sanity()
    level, msg = cs.bake_frame_sanity(120, 600)
    assert level == "WARNING"
    assert "120/600" in msg


def test_bake_complete_is_silent():
    cs = _load_cache_sanity()
    assert cs.bake_frame_sanity(600, 600) == (None, "")
    # actual > expected (the empty frame_0000 pushes file count to
    # frames+1) must NOT be a false truncation.
    assert cs.bake_frame_sanity(601, 600) == (None, "")


def test_render_zero_pngs_is_error():
    cs = _load_cache_sanity()
    level, msg = cs.render_output_sanity(0, 0)
    assert level == "ERROR" and "0 PNG frames" in msg


def test_render_complete_is_silent():
    cs = _load_cache_sanity()
    assert cs.render_output_sanity(48, 0)[0] is None


def test_count_mesh_frames_missing_dir(tmp_path):
    cs = _load_cache_sanity()
    assert cs.count_mesh_frames(str(tmp_path)) == 0  # no mesh/ subdir


def test_count_mesh_frames_counts_ply(tmp_path):
    cs = _load_cache_sanity()
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    for i in range(3):
        (mesh / f"frame_{i:04d}.ply").write_text("ply\n")
    (mesh / "not_a_frame.txt").write_text("x")
    assert cs.count_mesh_frames(str(tmp_path)) == 3


# ── CONTRACT: source-grep guards (§9.12) ─────────────────────────────

def test_bake_no_longer_excludes_zero_frame_case():
    code = _code(_ADDON / "operators" / "bake.py")
    assert "0 < actual < expected" not in code, (
        "round-61 regressed: the sync truncation check again excludes "
        "the actual==0 case")
    assert "_post_bake_sanity" in code, (
        "round-61: shared sanity helper missing from bake.py")
    # modal path must gate auto-attach + 'complete' on actual > 0.
    assert "if actual > 0:" in code, (
        "round-61: modal FINISHED must gate auto-attach on a non-zero "
        "frame count")


def test_attach_warns_on_zero_frames():
    code = _code(_ADDON / "cache_loader" / "_ops.py")
    assert "if n == 0:" in code, (
        "round-61: attach_cache must branch on the 0-frame case")
    assert '"WARNING"' in code and "0 renderable frames" in code, (
        "round-61: attach_cache must WARNING (not INFO) when 0 frames load")


def test_render_uses_output_sanity():
    code = _code(_ADDON / "operators" / "render.py")
    assert "render_output_sanity" in code, (
        "round-61: render.py must run the shared output sanity check "
        "(both sync + modal) instead of reporting success blindly")


def test_mpm_inflow_velocity_not_silently_overridden():
    code = _code(_REPO / "src" / "gpufluid" / "cli" / "commands.py")
    assert "(0.0, 0.0, -1.0)" not in code, (
        "round-61 regressed: MPM inflow zero-velocity again rewritten to "
        "a downward (0,0,-1) kick (round-51 UX-default bug resurrected)")
    assert "velocity=tuple(inf.velocity)," in code, (
        "round-61: MPM inflow must forward inf.velocity verbatim")
