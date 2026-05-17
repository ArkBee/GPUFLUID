"""Tests for layer C7 CLI: config parsing, end-to-end simulate, info.

Covers [BLK C7.1] TOML config schema, [BLK C7.2] simulate command,
[BLK C7.3] bench command, [BLK C7.4] info command."""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from gpufluid.cli.config import load_scene, SceneCfg, ObstacleSphereCfg, ObstacleCylinderYCfg
from gpufluid.cli.commands import main, cmd_info
from gpufluid.blocks import BlockError


def _write_minimal(tmp_path: Path, frames: int = 2) -> Path:
    cfg = tmp_path / "scene.toml"
    cfg.write_text(f"""
[domain]
resolution = [16, 16, 16]

[fluid]
type = "box"
lo = [0.10, 0.30, 0.10]
hi = [0.50, 0.90, 0.50]
ppc = 8

[[obstacle]]
type = "sphere"
center = [0.5, 0.20, 0.5]
radius = 0.10

[[obstacle]]
type = "cylinder_y"
center = [0.30, 0.30, 0.30]
radius = 0.05
half_height = 0.20

[simulation]
dt = 0.005
pressure_iters = 20
frames = {frames}
fps = 24

[output]
cache_dir = "cache"
mesh = true
particles = true
iso_level = 0.5
smooth_passes = 1
""")
    return cfg


def test_c7_1_parse_minimal(tmp_path: Path):
    cfg_path = _write_minimal(tmp_path)
    scene = load_scene(cfg_path)
    assert isinstance(scene, SceneCfg)
    assert scene.domain.resolution == (16, 16, 16)
    assert len(scene.obstacle) == 2
    assert isinstance(scene.obstacle[0], ObstacleSphereCfg)
    assert isinstance(scene.obstacle[1], ObstacleCylinderYCfg)
    assert scene.simulation.frames == 2


def test_c7_1_unknown_obstacle_raises(tmp_path: Path):
    cfg = tmp_path / "bad.toml"
    cfg.write_text('''[[obstacle]]\ntype = "trapezohedron"\ncenter = [0,0,0]\n''')
    with pytest.raises(BlockError) as ei:
        load_scene(cfg)
    assert ei.value.block_id == "C7.1"


def test_c7_1_missing_file_raises(tmp_path: Path):
    with pytest.raises(BlockError) as ei:
        load_scene(tmp_path / "nope.toml")
    assert ei.value.block_id == "C7.1"


@pytest.mark.gpu
def test_c7_2_simulate_end_to_end_creates_cache(tmp_path: Path):
    cfg_path = _write_minimal(tmp_path, frames=2)
    rc = main(["simulate", str(cfg_path)])
    assert rc == 0
    cache_dir = tmp_path / "cache"
    assert (cache_dir / "cache.json").exists()
    assert (cache_dir / "mesh" / "frame_0000.ply").exists()
    assert (cache_dir / "mesh" / "frame_0001.ply").exists()
    assert (cache_dir / "particles" / "frame_0000.npy").exists()
    manifest = json.loads((cache_dir / "cache.json").read_text())
    assert manifest["frame_count"] == 2
    assert manifest["fps"] == 24
    assert manifest["resolution"] == [16, 16, 16]


def test_c7_4_info_runs():
    buf = io.StringIO()
    with redirect_stdout(buf):
        import argparse
        rc = cmd_info(argparse.Namespace())
    out = buf.getvalue()
    assert rc == 0
    assert "gpufluid version" in out
    assert "Registered blocks" in out
    assert "G1:" in out
