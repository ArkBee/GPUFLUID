"""Round-58 — warn when an MPM inflow's emission window is shorter
than the bake length.

Pre-R58, this trap silently degraded long bakes: a TOML with e.g.
`[[inflow]] frame_end = 200` and `[simulation] frames = 600` would
emit for the first 200 frames then sit idle for 400 — fluid stream
just stopped mid-bake. The pool count looked right
(`rate * (frame_end - frame_start) / fps`) so neither logs nor pool
allocation flagged it. Live-found on the round-57 OBB scene when the
user bumped `frames` 80 → 600 without bumping the inflow window.

Contract (lesson §9.12 source-grep):
  * commands.py inflow assembly path must emit a `[mpm] WARN: inflow`
    line when `inf.frame_end < sim.frames` (and > 0).
  * The warning must reference both numeric values + the recovery
    ("set frame_end = simulation.frames" or "omit it") so the user
    knows what to change.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CMDS = _REPO / "src" / "gpufluid" / "cli" / "commands.py"


def _code() -> str:
    src = _CMDS.read_text(encoding="utf-8")
    return "\n".join(
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    )


def test_warning_emitted_when_inflow_window_shorter_than_bake():
    code = _code()
    assert "inflow[" in code and "frame_end=" in code, (
        "round-58 regressed: per-inflow window warning missing")
    # The trigger must be `0 < end < sim.frames` (clamping case stays
    # silent — InflowCfg default 1_000_000 always clamps).
    assert "< int(sim.frames)" in code, (
        "round-58: warning must trigger on frame_end < sim.frames")
    # Must explain the recovery path so the user knows what to do.
    assert "simulation.frames" in code, (
        "round-58: warning must reference simulation.frames as the fix")
    # Stderr so addon stdout-drain can keep it separate from progress.
    assert "file=sys.stderr" in code, (
        "round-58: warning must go to stderr to avoid mixing with "
        "the per-step progress on stdout")
