"""Prod-hardening (2026-06-21): MPM velocity-cap opt-out.

Independent audit (reviewer-velcaps) found the tap (terminal-velocity) +
anti-splash vz caps are faucet/splash heuristics that are ON by default and
clamp legitimate BULK motion in a released-block / dam-break scene (anti-splash
clamps every particle above the obstacle mid-height; tap brakes the initial-
column footprint). They can't regress the tuned faucet/cascade demos (default
stays True), but a scene that doesn't want them needs an off switch. This adds
`[simulation.mpm].velocity_caps` (+ exposes the previously-hardcoded vz_min as
`vz_min_splash`). Config-level tests (no GPU bake).
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from gpufluid.cli.config import load_scene

COMMANDS = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli" / "commands.py"

_BASE = """
[domain]
resolution = [32, 32, 32]
[[fluids]]
type = "box"
lo = [0.3, 0.3, 0.6]
hi = [0.5, 0.5, 0.8]
[simulation]
solver = "mpm"
{mpm}
[output]
cache_dir = "tmp/x"
"""


def _scene(tmp_path, mpm_block=""):
    p = tmp_path / "s.toml"
    p.write_text(textwrap.dedent(_BASE.format(mpm=mpm_block)), encoding="utf-8")
    return load_scene(p)


def test_velocity_caps_default_on(tmp_path):
    sc = _scene(tmp_path)
    assert sc.simulation.mpm_velocity_caps is True, (
        "default must stay True so the tuned faucet/cascade demos are unchanged")
    assert sc.simulation.mpm_vz_min_splash == -2.0


def test_velocity_caps_can_be_disabled(tmp_path):
    sc = _scene(tmp_path, "[simulation.mpm]\nvelocity_caps = false\nvz_min_splash = -3.5\n")
    assert sc.simulation.mpm_velocity_caps is False
    assert sc.simulation.mpm_vz_min_splash == -3.5


def test_commands_gates_caps_on_the_flag():
    """§9.12: the MpmConfig build must pass tap/anti_splash=None when the flag is
    off (the solver guards `is not None`), and vz_min must use the knob, not the
    old hardcoded -2.0 literal."""
    code = "\n".join(l for l in COMMANDS.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    assert "if sim.mpm_velocity_caps else None" in code, (
        "tap/anti_splash must be gated on mpm_velocity_caps")
    assert "vz_min=sim.mpm_vz_min_splash * inv_dom_z" in code, (
        "anti-splash vz_min must use the new knob")
    assert "vz_min=-2.0 * inv_dom_z" not in code, (
        "the hardcoded -2.0 vz_min literal must be gone")


def test_caps_gate_count():
    """Both tap AND anti_splash gated (not just one — §9.6 mirror)."""
    code = COMMANDS.read_text(encoding="utf-8")
    assert code.count("if sim.mpm_velocity_caps else None") == 2
