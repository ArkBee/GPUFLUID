"""Regression tests for M5.6 mesh decimation."""
import numpy as np
import pytest
import trimesh

from gpufluid.meshing.decimate import decimate_mesh


def test_m5_6_decimation_reduces_face_count_by_ratio():
    sphere = trimesh.creation.icosphere(subdivisions=4)   # 5120 faces
    v = np.asarray(sphere.vertices, dtype=np.float32)
    f = np.asarray(sphere.faces, dtype=np.int32)
    v2, f2 = decimate_mesh(v, f, target_ratio=0.25)
    # within 20% of target
    target = len(f) * 0.25
    assert abs(len(f2) - target) < target * 0.20, f"target≈{target} got {len(f2)}"
    # bbox should be ≈ preserved (radius 1 sphere)
    rb_before = float(np.linalg.norm(v.max(0) - v.min(0)))
    rb_after = float(np.linalg.norm(v2.max(0) - v2.min(0)))
    assert abs(rb_before - rb_after) < 0.05


def test_m5_6_ratio_one_is_passthrough():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    v = np.asarray(sphere.vertices, dtype=np.float32)
    f = np.asarray(sphere.faces, dtype=np.int32)
    v2, f2 = decimate_mesh(v, f, target_ratio=1.0)
    assert v2 is v and f2 is f


def test_m5_6_empty_mesh_no_op():
    v = np.zeros((0, 3), dtype=np.float32)
    f = np.zeros((0, 3), dtype=np.int32)
    v2, f2 = decimate_mesh(v, f, target_ratio=0.5)
    assert v2 is v and f2 is f
