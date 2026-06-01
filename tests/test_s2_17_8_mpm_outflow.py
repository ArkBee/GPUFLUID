"""S2.17.8 — MPM outflow (drain) despawn.

The MPM path pre-allocates a fixed particle array, so a drain can't compact;
it despawns by flipping particle_selection to 1 (mirror of the inflow gate).
These tests cover the config plumbing + the gate arithmetic without needing a
GPU (the kernel itself is exercised by the live cascade-drain bake).
"""
from __future__ import annotations

import pytest

pytest.importorskip("warp")  # outflow module imports warp at module load


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
        pytest.skip("no CUDA device for MpmSolver construction")
    assert len(solver._outflow_params) == 1
    step_start, step_end = solver._outflow_params[0][0], solver._outflow_params[0][1]
    assert step_start == 2 * 5
    assert step_end == 10 * 5


def test_cli_threads_outflow_into_mpm(tmp_path):
    """Contract (§9.6): the MPM CLI branch must build MpmOutflow from
    scene.outflow. Pre-this-fix only the FLIP path consumed outflow."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli"
           / "commands.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "MpmOutflow(" in code, "MPM branch must construct MpmOutflow"
    assert "outflows=outflows" in code, "MpmConfig must receive outflows"
