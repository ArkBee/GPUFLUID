"""Layer I6 tests — PLY writer + roundtrip via trimesh."""
import numpy as np
import pytest
import trimesh
from pathlib import Path

from gpufluid.io.ply import write_ply, read_ply, write_particles_npy
from gpufluid.blocks import BlockError


def _tri_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return verts, faces


def test_i6_1_ply_roundtrip(tmp_path: Path):
    verts, faces = _tri_mesh()
    p = tmp_path / "tet.ply"
    write_ply(p, verts, faces)
    mesh = trimesh.load(str(p), process=False)
    assert mesh.vertices.shape == verts.shape
    assert mesh.faces.shape == faces.shape
    assert np.allclose(mesh.vertices, verts)
    assert (mesh.faces == faces).all()


def test_i6_1_rejects_bad_shape(tmp_path: Path):
    with pytest.raises(BlockError) as ei:
        write_ply(tmp_path / "x.ply", np.zeros((4, 2), dtype=np.float32), np.zeros((1, 3), dtype=np.int32))
    assert ei.value.block_id == "I6.1"


def test_i6_1_ply_read_matches_write(tmp_path: Path):
    verts, faces = _tri_mesh()
    p = tmp_path / "tet.ply"
    write_ply(p, verts, faces)
    v2, f2 = read_ply(p)
    assert v2.shape == verts.shape and (v2 == verts).all()
    assert f2.shape == faces.shape and (f2 == faces).all()


def test_i6_3_particles_npy_roundtrip(tmp_path: Path):
    pos = np.random.RandomState(0).rand(100, 3).astype(np.float32)
    p = tmp_path / "p.npy"
    write_particles_npy(p, pos)
    loaded = np.load(p)
    assert loaded.shape == pos.shape
    assert np.allclose(loaded, pos)
