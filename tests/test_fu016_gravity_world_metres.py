"""FU-016 (2026-06-01 pre-public-test audit, live-smoke follow-up).

The MPM gravity (and the velocity / v_terminal / anti-splash terms that
share `inv_dom_z`) was scaled by `scene.domain_size[2]` = nz/max(res) — the
resolution ASPECT RATIO, not real metres. The real world extent was computed
in the addon (DomainTransform.dom_size) but never forwarded to the CLI, so:

  * cubic 2.5 m domain  -> g unscaled (-9.81)  -> water falls 1.59x too fast
  * flat 4x4x1 m domain -> g x4 (-39.24)       -> 2.0x too fast
  * only a true 1 m unit cube was correct.

Fix: forward `domain.size_world` (metres) from the addon; the CLI uses a new
`SceneCfg.world_size` (falls back to `domain_size` for hand-written TOMLs).
This file is the UNIT layer (the config module imports without warp/bpy);
the addon-side forwarding is grep-guarded in test_audit_20260601_fixes.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# The CLI config module is bpy-free and warp-free; make `src` importable
# without requiring the package to be pip-installed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpufluid.cli.config import DomainCfg, SceneCfg, load_scene  # noqa: E402


def _g_norm_z(scene: SceneCfg, g_world: float = -9.81) -> float:
    """TEST-LOCAL MIRROR of the commands.py gravity normalisation (the real
    code path needs warp). The mirror's formula direction is pinned to prod
    by test_prod_gravity_formula_divides_by_world_z below — without that
    pin, an inversion in commands.py would leave every mirror test green
    (audit-20260610)."""
    dom_z = scene.world_size[2] if scene.world_size[2] > 1e-9 else 1.0
    return g_world / dom_z


def test_prod_gravity_formula_divides_by_world_z():
    """audit-20260610: pin the REAL commands.py expression, not just 'reads
    world_size'. Invariant: g_norm's Z term DIVIDES real-world g by the
    metre extent (g_world / dom_z). A flip to multiplication (or swapping
    operands) would pass every mirror-based test above; this grep is the
    only thing that fails on an inversion short of a live GPU bake."""
    src = (Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli"
           / "commands.py").read_text(encoding="utf-8")
    # §9.12: strip docstrings + comment lines before asserting.
    src = re.sub(r'("""|\'\'\')[\s\S]*?\1', "", src)
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "dom_size = scene.world_size" in code, (
        "FU-016: gravity scaling must source the metre extent from "
        "scene.world_size (the addon-forwarded size_world)")
    assert re.search(
        r"g_norm\s*=\s*\(\s*0\.0\s*,\s*0\.0\s*,\s*g_world\s*/\s*dom_z\s*\)",
        code), (
        "FU-016: the gravity Z term must be `g_world / dom_z` (divide real "
        "g by the world-Z metres). If the names changed, keep a "
        "divide-by-world-extent form and update this pin")
    assert not re.search(r"g_world\s*\*\s*dom_z|dom_z\s*/\s*g_world", code), (
        "FU-016 INVERTED: gravity is multiplied by (or divided into) the "
        "domain extent — water falls wrong by dom_z² relative to the fix")


def test_world_size_uses_forwarded_metres():
    s = SceneCfg(domain=DomainCfg(resolution=(48, 48, 48), size_world=(2.52, 2.52, 2.52)))
    assert s.world_size == (2.52, 2.52, 2.52)
    # domain_size is the (different) normalised aspect ratio — must NOT change.
    assert s.domain_size == (1.0, 1.0, 1.0)


def test_world_size_falls_back_to_aspect_ratio_when_absent():
    """Hand-written standalone TOMLs without size_world keep the pre-FU-016
    behaviour (aspect ratio) — backward compatible."""
    s = SceneCfg(domain=DomainCfg(resolution=(64, 64, 16)))
    assert s.domain.size_world is None
    assert s.world_size == s.domain_size == (1.0, 1.0, 0.25)


def test_gravity_metre_correct_cubic_domain():
    """Cubic 2.5 m domain: gravity must scale by 2.5 m, not stay at -9.81
    (the 1.59x-too-fast bug)."""
    s = SceneCfg(domain=DomainCfg(resolution=(48, 48, 48), size_world=(2.52, 2.52, 2.52)))
    g = _g_norm_z(s)
    assert abs(g - (-9.81 / 2.52)) < 1e-6
    assert abs(g) < 9.81, "gravity must be SMALLER in normalised units for a >1 m domain"


def test_gravity_metre_correct_flat_domain():
    """Flat 4x4x1 m domain: with size_world the Z extent is 1.0 m -> g stays
    -9.81. Pre-FU-016 the aspect-ratio dom_z was 0.25 -> g=-39.24 (2x fast)."""
    s = SceneCfg(domain=DomainCfg(resolution=(64, 64, 16), size_world=(4.0, 4.0, 1.0)))
    assert abs(_g_norm_z(s) - (-9.81)) < 1e-6
    # And confirm the OLD (fallback) path would have been wrong by 4x.
    s_old = SceneCfg(domain=DomainCfg(resolution=(64, 64, 16)))
    assert abs(_g_norm_z(s_old) - (-9.81 / 0.25)) < 1e-6


def test_unit_cube_unchanged():
    """A true 1 m unit cube was already correct; must stay -9.81 either way."""
    s = SceneCfg(domain=DomainCfg(resolution=(48, 48, 48), size_world=(1.0, 1.0, 1.0)))
    assert abs(_g_norm_z(s) - (-9.81)) < 1e-6


def test_size_world_round_trips_through_toml(tmp_path):
    """load_scene must parse `domain.size_world` back into DomainCfg."""
    toml = tmp_path / "scene.toml"
    toml.write_text(
        "[domain]\n"
        "resolution = [64, 64, 16]\n"
        "size_world = [4.0, 4.0, 1.0]\n"
        "[fluid]\n"
        "type = \"box\"\n"
        "lo = [0.1, 0.1, 0.1]\n"
        "hi = [0.4, 0.4, 0.4]\n"
        "[simulation]\n"
        "gravity = -9.81\n"
        "frames = 10\n"
        "[output]\n"
        "cache_dir = \"out\"\n",
        encoding="utf-8")
    scene = load_scene(str(toml))
    assert scene.domain.size_world == (4.0, 4.0, 1.0)
    assert scene.world_size == (4.0, 4.0, 1.0)
