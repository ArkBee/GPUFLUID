"""S2.17.8 — MPM outflow (drain) despawn.

The MPM path pre-allocates a fixed particle array, so a drain can't compact;
it despawns by flipping particle_selection to 1 (mirror of the inflow gate).
These tests cover the config plumbing + the gate arithmetic without needing a
GPU (the kernel itself is exercised by the live cascade-drain bake).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import HAS_CUDA

pytest.importorskip("warp")  # outflow module imports warp at module load

_MPM = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "sim" / "mpm"

# audit-20260610: §9.12 strip docstrings (triple-quoted blocks) too, not
# only comment lines — a docstring retelling the fix could satisfy a grep.
_TRIPLE_RE = re.compile(r'("""|\'\'\')[\s\S]*?\1')


def _code(path: Path) -> str:
    src = _TRIPLE_RE.sub("", path.read_text(encoding="utf-8"))
    return "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))


def test_mpm_outflow_dataclass_defaults():
    from gpufluid.sim.mpm.outflow import MpmOutflow
    o = MpmOutflow(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 0.2))
    assert o.frame_start == 0
    assert o.frame_end == 1_000_000  # "whole bake" sentinel


def test_mpm_config_accepts_outflows():
    from gpufluid.sim.mpm.solver import MpmConfig
    from gpufluid.sim.mpm.outflow import MpmOutflow
    cfg = MpmConfig(outflows=(MpmOutflow(lo=(0, 0, 0), hi=(1, 1, 0.2)),))
    assert len(cfg.outflows) == 1


def test_outflow_params_convert_frames_to_steps():
    """The solver must convert output-frame gates to sim steps via dump_every,
    exactly like the inflow gate (so a drain active 'from frame 0' lines up
    with the same clock the inflow uses)."""
    from gpufluid.sim.mpm.solver import MpmConfig, MpmSolver
    from gpufluid.sim.mpm.outflow import MpmOutflow
    import numpy as np
    # tiny initial column so the solver builds without GPU work at import
    col = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    cfg = MpmConfig(
        initial_column=col,
        dump_every=5,
        outflows=(MpmOutflow(lo=(0.1, 0.1, 0.1), hi=(0.9, 0.9, 0.2),
                             frame_start=2, frame_end=10),),
    )
    try:
        solver = MpmSolver(cfg)
    except Exception:
        # audit-20260610: skip ONLY when CUDA is genuinely absent. On a GPU
        # box a constructor exception is a real regression and must FAIL —
        # the old blanket `except → skip` turned it into a green skip.
        if HAS_CUDA:
            raise
        pytest.skip("no CUDA device for MpmSolver construction")
    assert len(solver._outflow_params) == 1
    step_start, step_end = solver._outflow_params[0][0], solver._outflow_params[0][1]
    assert step_start == 2 * 5
    assert step_end == 10 * 5


def test_predicate_in_box_cpu():
    """CPU coverage of the despawn gate+containment logic via
    ``predicate_in_box`` — HONESTY NOTE (audit-20260610): this function is a
    hand-maintained CPU MIRROR shipped next to the kernel; the warp kernel
    re-implements the predicate inline and never calls it. This test
    therefore locks the intended semantics, NOT the kernel — the kernel's
    inline form is pinned by test_outflow_kernel_semantics_contract below,
    and the dynamic behaviour by the live GPU cascade-drain bake."""
    from gpufluid.sim.mpm.outflow import predicate_in_box
    lo, hi = (0.1, 0.1, 0.1), (0.9, 0.9, 0.2)
    # inside box, within frame window -> True
    assert predicate_in_box((0.5, 0.5, 0.15), lo, hi, 5, 0, 10) is True
    # outside box (z above) -> False
    assert predicate_in_box((0.5, 0.5, 0.5), lo, hi, 5, 0, 10) is False
    # inside box but before the window -> False
    assert predicate_in_box((0.5, 0.5, 0.15), lo, hi, 0, 2, 10) is False
    # inside box but after the window -> False
    assert predicate_in_box((0.5, 0.5, 0.15), lo, hi, 11, 0, 10) is False
    # boundary inclusive
    assert predicate_in_box((0.1, 0.1, 0.2), lo, hi, 10, 0, 10) is True


def test_outflow_kernel_semantics_contract():
    """§9.12 source-grep (audit-20260610): keep the KERNEL's inline predicate
    in step with the CPU mirror tested above — inclusive AABB bounds and an
    inclusive [step_start, step_end] frame gate. The mirror can't catch a
    kernel-side drift (it is never called by the kernel), so pin the inline
    comparison forms here."""
    of = _code(_MPM / "outflow.py")
    # Inclusive containment on every axis (the kernel spells it per-axis).
    for required in ("pos[0] >= lo_x and pos[0] <= hi_x",
                     "pos[1] >= lo_y and pos[1] <= hi_y",
                     "pos[2] >= lo_z and pos[2] <= hi_z"):
        assert required in of, (
            f"S2.17.8 kernel drift: inclusive containment `{required}` is "
            f"gone from k_outflow_despawn — boundary particles (e.g. resting "
            f"exactly on the drain face) would stop draining, diverging from "
            f"predicate_in_box (the tested CPU mirror)")
    # Inclusive frame gate (reject-outside form, same as the mirror).
    assert "current_step < step_start or current_step > step_end" in of, (
        "S2.17.8 kernel drift: the inclusive [step_start, step_end] gate "
        "form is gone — an exclusive gate would skip the boundary frames "
        "the CPU mirror (and the inflow clock) treat as active")
    # And the exclusive forms must not have crept in.
    for forbidden in ("pos[0] > lo_x", "pos[2] < hi_z",
                      "current_step <= step_start",
                      "current_step >= step_end"):
        assert forbidden not in of, (
            f"S2.17.8 kernel drift: exclusive comparison `{forbidden}` "
            f"appeared in outflow.py — kernel no longer matches the "
            f"inclusive semantics predicate_in_box locks")


def test_outflow_despawn_is_one_way_contract():
    """§9.12 source-grep: the drain must mark a dedicated despawned flag and
    the inflow gate must skip despawned particles — otherwise the gate
    re-releases a drained inflow particle every frame (workflow-found
    2026-06-02). Guards against a refactor dropping the flag."""
    of = _code(_MPM / "outflow.py")
    inf = _code(_MPM / "inflow.py")
    assert "despawned[p] = 1" in of, "drain must set the despawned flag"
    # audit-20260610: the old check (`"despawned" in inf and "return" in
    # inf`) was satisfied by ANY return anywhere; require the actual
    # early-return — a `pass` body under the despawned check would
    # re-release the drained particle and still grep green.
    assert re.search(r"if\s+despawned\[idx\]\s*==\s*1\s*:\s*\n\s*return\b",
                     inf), (
        "inflow gate must EARLY-RETURN when despawned==1 (the release "
        "branch below it would otherwise resurrect a drained particle)")


def test_cli_threads_outflow_into_mpm(tmp_path):
    """Contract (§9.6): the MPM CLI branch must build MpmOutflow from
    scene.outflow. Pre-this-fix only the FLIP path consumed outflow."""
    code = _code(Path(__file__).resolve().parents[1] / "src" / "gpufluid"
                 / "cli" / "commands.py")
    assert "MpmOutflow(" in code, "MPM branch must construct MpmOutflow"
    assert "outflows=outflows" in code, "MpmConfig must receive outflows"
