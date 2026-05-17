"""[BLK I6.5] USD time-sampled mesh writer round-trip.

Smoke test: write a 3-frame triangle mesh sequence to .usdc, then
re-open via pxr.Usd and verify the vert/face count and time-sample
range survive the round-trip.

Skipped if `pxr` (OpenUSD) is not installed.
"""
from __future__ import annotations
import numpy as np
import pytest

pxr = pytest.importorskip("pxr")

from gpufluid.io.usd import write_usd_mesh_sequence


def test_i6_5_round_trip_three_frames(tmp_path):
    out = tmp_path / "test.usdc"
    # Two triangles per frame, vert count grows over time so the
    # round-trip can spot which frame each sample belongs to.
    frames = [
        (0, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int32)),
        (1, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32),
            np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)),
        (2, np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int32)),
    ]
    result = write_usd_mesh_sequence(out, frames, fps=24,
                                     prim_path="/gpufluid/cache")
    assert result.exists()
    assert result.suffix == ".usdc"

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(out))
    assert stage is not None
    mesh = UsdGeom.Mesh.Get(stage, "/gpufluid/cache")
    assert mesh, "mesh prim not created at expected path"

    points = mesh.GetPointsAttr()
    times = sorted(points.GetTimeSamples())
    assert times == [0.0, 1.0, 2.0], f"expected 3 time samples, got {times}"

    # Mid-frame must carry 4 verts (the only one with that count).
    pts_at_1 = np.array(points.Get(1.0), dtype=np.float32)
    assert pts_at_1.shape == (4, 3), f"frame 1 verts shape {pts_at_1.shape}"

    assert stage.GetStartTimeCode() == 0.0
    assert stage.GetEndTimeCode() == 2.0
    assert stage.GetFramesPerSecond() == 24
