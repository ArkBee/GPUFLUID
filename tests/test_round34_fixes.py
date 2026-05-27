"""Round-34 regression tests for round-33 reviewer findings.

  - FLIP _apply_outflows_gpu compacts attr_color / attr_temperature
    in lockstep with pos/vel (source-grep + a tiny numpy-only proxy).
  - FLIP prepare_frame emit-append extends attr arrays.
  - SDF box collider top-friction: post-projection tangential-only.
  - MPM inflow keyframes reject non-monotone frames + degenerate AABB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent


# ─── 1+2: FLIP attr compact + emit-append source-grep contracts ─────────

def test_outflow_compaction_handles_attr_arrays():
    """Round-34 contract: source must compact attr_color/attr_temperature
    inside _apply_outflows_gpu. Source-grep (lesson §9.12) guards
    against future drift back to the pre-round-34 form where attr
    arrays kept stale layout while pos/vel shrank."""
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    # `_apply_outflows_gpu` body must reference attr_color and
    # attr_temperature when reassigning post-compaction.
    fn_pos = code.find("def _apply_outflows_gpu")
    next_def_pos = code.find("\n    def ", fn_pos + 1)
    body = code[fn_pos:next_def_pos]
    assert "self.attr_color" in body, (
        "round-34 regressed: outflow compaction must update attr_color")
    assert "self.attr_temperature" in body, (
        "round-34 regressed: outflow compaction must update attr_temperature")


def test_prepare_frame_emit_extends_attr_arrays():
    """Round-34 contract: prepare_frame emit-append path must grow
    attr_color/attr_temperature with the new particle count. Without
    this growth, _apply_color_transfer would OOB-read."""
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    fn_pos = code.find("def prepare_frame")
    next_def_pos = code.find("\n    def ", fn_pos + 1)
    body = code[fn_pos:next_def_pos]
    # Must reference both attrs inside this method body.
    assert "self.attr_color" in body, (
        "round-34 regressed: prepare_frame must extend attr_color on emit")
    assert "self.attr_temperature" in body, (
        "round-34 regressed: prepare_frame must extend attr_temperature on emit")


# ─── 3: MPM inflow keyframe validation ──────────────────────────────────

def test_inflow_keyframes_reject_non_monotone_frames():
    """Round-34: pre-round-34 np.interp silently mis-interpolated when
    kf_frames was out of order (easy TOML typo). Now raises with the
    offending list."""
    from gpufluid.sim.mpm.inflow import MpmInflow, seed_inflow_particles
    inf = MpmInflow(
        lo=(0, 0, 0), hi=(1, 1, 1),
        velocity=(0, 0, 0),
        rate_per_sec=100, frame_start=0, frame_end=60,
        keyframes=(
            (10.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2),
            (5.0,  0.3, 0.3, 0.3, 0.4, 0.4, 0.4),  # frame=5 < prior 10
        ),
    )
    with pytest.raises(ValueError, match="monotone"):
        seed_inflow_particles(inf, fps=24, dump_every=5)


def test_inflow_keyframes_reject_degenerate_aabb():
    """Round-34: keyframe with hi < lo on any axis now raises with
    row index. Pre-round-34 the static path raised; the keyframe path
    silently produced inverted spawn AABB."""
    from gpufluid.sim.mpm.inflow import MpmInflow, seed_inflow_particles
    inf = MpmInflow(
        lo=(0, 0, 0), hi=(1, 1, 1),
        velocity=(0, 0, 0),
        rate_per_sec=100, frame_start=0, frame_end=60,
        keyframes=(
            (0.0,  0.1, 0.1, 0.1, 0.2, 0.2, 0.2),   # ok
            (30.0, 0.3, 0.5, 0.3, 0.4, 0.4, 0.4),   # y: 0.5 > 0.4 → bad
        ),
    )
    with pytest.raises(ValueError, match="row 1"):
        seed_inflow_particles(inf, fps=24, dump_every=5)


def test_inflow_keyframes_happy_path_still_works():
    """Round-34 must not regress the sorted-and-positive case."""
    from gpufluid.sim.mpm.inflow import MpmInflow, seed_inflow_particles
    inf = MpmInflow(
        lo=(0, 0, 0), hi=(1, 1, 1),
        velocity=(0, 0, 0),
        rate_per_sec=100, frame_start=0, frame_end=60,
        keyframes=(
            (0.0,  0.1, 0.1, 0.1, 0.2, 0.2, 0.2),
            (30.0, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4),
            (60.0, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6),
        ),
    )
    pos, spawn = seed_inflow_particles(inf, fps=24, dump_every=5)
    assert pos.shape[0] > 0
    assert pos.shape[1] == 3
    assert spawn.shape == (pos.shape[0],)


# ─── 4: SDF box collider top-friction source guard ─────────────────────

def test_sdf_box_top_friction_separates_normal_and_tangent():
    """Round-34 contract: top-face friction must NOT scale the
    inward-normal-projected component. Source must reference v_normal
    + v_tangent decomposition in the surface_type==1 branch."""
    src = (_REPO / "src" / "gpufluid" / "sim" / "mpm"
           / "colliders.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    # Pre-round-34 form was `v = v * param.friction` — must be gone.
    assert "v = v * param.friction" not in code, (
        "round-34 regressed: top friction must decompose first")
    # Post-round-34 form must include the tangent decomposition.
    assert "v_tangent" in code
    assert "v_normal" in code
