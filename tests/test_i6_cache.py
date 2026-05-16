"""Tests for I6.2 cache manifest."""
import json
import pytest
from pathlib import Path

from gpufluid.io.cache import (
    CacheManifest, write_cache_manifest, read_cache_manifest, CACHE_SCHEMA_VERSION,
)
from gpufluid.blocks import BlockError


def test_i6_2_write_then_read(tmp_path: Path):
    m = CacheManifest(fps=30, frame_count=42, resolution=[32, 32, 32], dx=1.0 / 32)
    written = write_cache_manifest(tmp_path, m)
    assert written.exists()
    raw = json.loads(written.read_text())
    assert raw["fps"] == 30
    assert raw["frame_count"] == 42
    assert raw["schema_version"] == CACHE_SCHEMA_VERSION
    assert raw["created_unix"] > 0.0
    assert raw["gpufluid_version"]

    m2 = read_cache_manifest(tmp_path)
    assert m2.fps == 30
    assert m2.frame_count == 42


def test_i6_2_missing_raises(tmp_path: Path):
    with pytest.raises(BlockError) as ei:
        read_cache_manifest(tmp_path)
    assert ei.value.block_id == "I6.2"


def test_i6_2_unsupported_schema_raises(tmp_path: Path):
    (tmp_path / "cache.json").write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(BlockError) as ei:
        read_cache_manifest(tmp_path)
    assert ei.value.block_id == "I6.2"


def test_i6_2_mesh_path_format(tmp_path: Path):
    m = CacheManifest(fps=24, frame_count=10)
    assert m.mesh_path(tmp_path, 5).name == "frame_0005.ply"
    assert m.particles_path(tmp_path, 5) is None
    m.particles_pattern = "particles/frame_{:04d}.npy"
    assert m.particles_path(tmp_path, 5).name == "frame_0005.npy"
