"""Pre-public-test audit (2026-06-01) — fixes for the five should-fix
findings from the adversarially-verified multi-agent audit.

Two layers, same as the round-61 suite:
  * UNIT — the bpy-free ``cache_sanity`` whitewater honesty helpers
    (§9.2: every new branch gets a real test).
  * CONTRACT — source-grep guards (§9.12) for the source changes that
    can't be unit-tested without bpy (operator wiring, property defaults).
    Comment lines are stripped before each assertion so the fix history
    in comments can't satisfy or trip a grep.

Findings closed here:
  FU-001  whitewater attach reports 0-frame WARNING (mirror of surface)
  FU-002b pre-bake cleanup wipes stale whitewater/ + whitewater_kinds/
  FU-003  animated inflow frame_end clamp warns + writes the clamped end
  FU-004  inflow temperature default/units aligned to Kelvin (293)
  FU-005  degenerate-domain ValueError caught → clean ERROR (not traceback)
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
    name = "_test_cache_sanity_audit"
    spec = importlib.util.spec_from_file_location(
        name, _ADDON / "cache_sanity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _code(path: Path) -> str:
    """Source with comment lines stripped (§9.12)."""
    src = path.read_text(encoding="utf-8")
    return "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _load_scene_dict():
    """Load the bpy-free scene_dict module directly (package __init__ pulls bpy)."""
    import importlib.util
    name = "_test_scene_dict_audit"
    spec = importlib.util.spec_from_file_location(
        name, _ADDON / "scene_dict.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── FU-001 UNIT: whitewater attach honesty classifier ────────────────

def test_count_ww_frames_missing_dir(tmp_path):
    cs = _load_cache_sanity()
    assert cs.count_ww_frames(str(tmp_path)) == 0  # no whitewater/ subdir


def test_count_ww_frames_counts_npy(tmp_path):
    cs = _load_cache_sanity()
    ww = tmp_path / "whitewater"
    ww.mkdir()
    for i in range(4):
        (ww / f"frame_{i:04d}.npy").write_bytes(b"\x00")
    (ww / "manifest.json").write_text("{}")  # non-frame file ignored
    assert cs.count_ww_frames(str(tmp_path)) == 4


def test_attach_ww_zero_frames_is_warning():
    cs = _load_cache_sanity()
    level, msg = cs.attach_ww_sanity(0)
    assert level == "WARNING", "empty whitewater attach must WARNING, not pass"
    assert "0 frames" in msg


def test_attach_ww_nonzero_is_silent():
    cs = _load_cache_sanity()
    assert cs.attach_ww_sanity(7) == (None, "")


# ── FU-001 CONTRACT: the operator actually wires the helper ───────────

def test_attach_ww_operator_uses_sanity():
    code = _code(_ADDON / "cache_loader" / "_ops.py")
    assert "count_ww_frames" in code and "attach_ww_sanity" in code, (
        "FU-001: attach_ww_cache must run the shared whitewater sanity "
        "check instead of reporting INFO success blindly")
    # The old unconditional 'attached to' INFO without a frame count must
    # be gone (it now lives under the else-branch with an (N frames) suffix).
    assert 'self.report({"INFO"}, f"whitewater cache attached to' not in code \
        or "frames)" in code, (
        "FU-001: the 0-frame whitewater attach must not still report a bare "
        "INFO success")


# ── FU-002b CONTRACT: stale whitewater dirs cleaned pre-bake ──────────

def test_prebake_cleanup_includes_whitewater():
    code = _code(_ADDON / "operators" / "bake.py")
    assert '"whitewater"' in code and '"whitewater_kinds"' in code, (
        "FU-002b: pre-bake cleanup must wipe stale whitewater/ and "
        "whitewater_kinds/ dirs (a shorter re-bake otherwise serves the "
        "prior run's tail frames silently)")


# ── FU-003 CONTRACT: animated inflow frame_end clamp is honest ───────

def test_animated_inflow_frame_end_clamp_warns():
    code = _code(_ADDON / "operators" / "bake.py")
    assert "if fe_eff < fe:" in code, (
        "FU-003: collect_scene must warn when an animated inflow's "
        "frame_end overshoots the bake range")
    assert 'entry["frame_end"] = fe_eff' in code, (
        "FU-003: the clamped end must be written to the TOML so the solver "
        "stops emitting when the animated keyframes run out (else the "
        "emitter freezes at its last position and keeps emitting)")


# ── FU-004 CONTRACT: inflow temperature is Kelvin, default 293 ────────

def test_inflow_temperature_kelvin_default():
    code = _code(_ADDON / "properties.py")
    assert "default=20.0, min=0.0, max=5000.0, soft_max=2000.0" not in code, (
        "FU-004: inflow temperature must no longer default to 20 in a '°C' "
        "convention (it silently mixed against the Fluid source's Kelvin)")
    assert "default=293.0, min=0.0, max=5000.0, soft_max=2000.0" in code, (
        "FU-004: inflow temperature must default to 293 K (room-temp water) "
        "on the same absolute scale as a Fluid source")
    # The misleading '°C in default convention' wording must be gone.
    assert "C in default convention" not in code, (
        "FU-004: the '°C default convention' description must be replaced "
        "with the Kelvin convention")


# ── FU-016 CONTRACT: addon forwards real world extent to the CLI ─────

def test_addon_forwards_size_world():
    """FU-016: collect_scene must put the real metre extent in the domain
    dict (size_world) so the CLI scales gravity/velocity by metres, not the
    resolution aspect-ratio. config_builder must emit it into the TOML."""
    bake = _code(_ADDON / "operators" / "bake.py")
    assert '"size_world"' in bake, (
        "FU-016: collect_scene must forward domain size_world (metres)")
    cb = _code(_ADDON / "config_builder.py")
    assert "size_world" in cb, (
        "FU-016: config_builder must emit [domain] size_world into the TOML")


def test_gravity_no_longer_uses_aspect_ratio_domain_size():
    """FU-016: the CLI gravity scaling must read world_size (metres), not the
    aspect-ratio domain_size."""
    cmds = _code(_REPO / "src" / "gpufluid" / "cli" / "commands.py")
    assert "scene.world_size" in cmds, (
        "FU-016: gravity/velocity scaling must use scene.world_size (metres)")
    assert "dom_size = scene.domain_size" not in cmds, (
        "FU-016 regressed: gravity scaling is back on the aspect-ratio "
        "scene.domain_size instead of world_size")


# ── FU-017 CONTRACT: MPM is the default solver (FLIP is legacy) ──────

def test_mpm_is_default_solver():
    """2026-06-01: MPM is the production path for everything; FLIP is legacy
    water-only and (unlike MPM after FU-016) does NOT metre-scale gravity. The
    addon UI default must be 'mpm' so a public-test user's first bake lands on
    the correct path, not on legacy FLIP with the gravity bug."""
    code = _code(_ADDON / "properties.py")
    assert 'default="mpm"' in code, (
        "FU-017: the Domain solver EnumProperty must default to 'mpm'")
    assert 'default="flip"' not in code, (
        "FU-017 regressed: solver default is back to legacy FLIP")


# ── FU-019 CONTRACT: axis-aligned box uses per-axis (not averaged) scale ──

def test_axis_aligned_box_uses_per_axis_halfsize():
    """FU-019 (deep-dive): the unrotated MPM-OBB box must size its half-extents
    per-axis (hx/hy/hz = world half * inv_size[axis]), not via the averaged
    inv_avg — averaging mis-sizes the collider on a non-cubic domain. The
    averaged path is RETAINED only for the rotated case (per-axis would shear
    an oriented box)."""
    code = _code(_ADDON / "operators" / "bake.py")
    assert "rot_is_identity" in code, (
        "FU-019: the OBB box branch must split identity vs rotated and use "
        "per-axis half-size for the axis-aligned case")
    assert "hx_sim, hy_sim, hz_sim = hx, hy, hz" in code, (
        "FU-019: axis-aligned box must use the per-axis hx/hy/hz half-extents")


def test_mesh_obstacle_warns_on_noncubic_domain():
    """FU-019: MESH obstacle uses a single-scalar `scale`, so it can't scale
    per-axis. Since MESH is advertised as the exact-fit workaround for
    SPHERE/CYL, it must at least WARN on a non-cubic domain instead of silently
    squashing the collider."""
    code = _code(_ADDON / "operators" / "bake.py")
    # there must be a MESH-branch warning keyed on the same anisotropy ratio.
    assert code.count("max(inv_size) / min(inv_size) > 1.05") >= 3, (
        "FU-019: the explicit MESH obstacle branch must warn on non-cubic "
        "domains (SPHERE + CYLINDER already do; MESH was silent)")


# ── FU-019 UNIT: size_world validation gives a clear key-path error ──

def test_size_world_validation_accepts_good():
    sd = _load_scene_dict()
    sd.validate_scene_dict({"domain": {"resolution": [48, 48, 48],
                                       "size_world": [2.5, 2.5, 2.5]}})  # no raise


def test_size_world_validation_rejects_malformed():
    sd = _load_scene_dict()
    import pytest
    for bad in ("2.5", [2.5], [2.5, 2.5], [2.5, 2.5, 0.0], [2.5, "x", 2.5]):
        with pytest.raises(sd.SceneDictError):
            sd.validate_scene_dict({"domain": {"resolution": [48, 48, 48],
                                               "size_world": bad}})


# ── FU-005 CONTRACT: degenerate domain → clean ERROR, not traceback ──

def test_collect_scene_catches_value_error():
    code = _code(_ADDON / "operators" / "bake.py")
    assert "except (RuntimeError, ValueError) as e:" in code, (
        "FU-005: the collect_scene call must catch the ValueError that "
        "DomainTransform.from_world_aabb raises for a degenerate Domain "
        "(zero/inverted extent) so the user sees a clean ERROR report "
        "instead of a raw traceback popup")
    assert "except RuntimeError as e:\n            self.report" not in code, (
        "FU-005: the narrow RuntimeError-only catch around collect_scene "
        "must be widened")
