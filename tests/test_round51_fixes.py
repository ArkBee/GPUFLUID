"""Round-51 — live-test finding: MpmInflow default velocity confused user.

Pre-round-51 `GpufluidInflowProps.velocity` defaulted to (0, -3, 0).
A user marking an Empty/Sphere as inflow then expecting gravity-only
drop got a confusing diagonal stream — emitter silently shot
particles sideways at 3 m/s along -Y. Caught only by live MCP-driven
bake in Blender 5.1.1 on RTX 4080 SUPER, post-merge to main.

Source-grep contract test (lesson §9.12): the default tuple must be
all-zero. The previous tuple form must NOT appear in non-comment code.
"""
from __future__ import annotations

from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


def test_inflow_velocity_default_is_zero():
    src = (_ADDON_DIR / "gpufluid_blender" / "properties.py").read_text()
    code_lines = [
        ln for ln in src.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)

    # Pre-round-51 form must be gone from live (non-comment) code.
    assert "default=(0.0, -3.0, 0.0)" not in code, (
        "round-51 regressed: MpmInflow velocity default reintroduced "
        "the confusing (0, -3, 0) — live-test silently lost gravity-"
        "only intent")

    # And the new zero default must be present somewhere in the
    # `velocity:` block of GpufluidInflowProps.
    inflow_pos = code.find("class GpufluidInflowProps")
    next_class = code.find("\nclass ", inflow_pos + 1)
    body = code[inflow_pos:next_class] if next_class > 0 else code[inflow_pos:]
    assert "default=(0.0, 0.0, 0.0)" in body, (
        "round-51 contract: inflow velocity must default to all-zero "
        "so a fresh `mark_inflow` op falls under gravity only")
