"""Round-32 regression tests for round-31 reviewer findings.

  - PLY reader rejects non-triangle faces explicitly.
  - FlipSolver3D.has_diverged() detects NaN; FlipDivergenceError typed.
  - CacheManifest carries solver/truncated_at_frame/truncation_reason.
  - DomainTransform per-axis degeneracy guard.
  - OT_clear_cache selective preload invalidate (source-grep guard).
  - pushback snap_eps source-grep contract.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_ADDON_DIR = _REPO / "addon"


# ─── 1. PLY non-triangle rejected ───────────────────────────────────────

def _make_quad_ply(verts: np.ndarray) -> bytes:
    """Binary little-endian PLY with a single quad face (count byte = 4)."""
    n = verts.shape[0]
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"element face 1\n"
        f"property list uchar int vertex_indices\n"
        f"end_header\n"
    ).encode("ascii")
    body = verts.astype(np.float32).tobytes()
    # 1-byte count=4 + 4× int32 indices
    body += b"\x04" + np.array([0, 1, 2, 3], dtype=np.int32).tobytes()
    return header + body


def test_ply_reader_rejects_non_triangle_face(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "_test_ply32",
        _ADDON_DIR / "gpufluid_blender" / "cache_loader" / "_ply.py",
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                     dtype=np.float32)
    p = tmp_path / "quad.ply"
    p.write_bytes(_make_quad_ply(verts))
    with pytest.raises(ValueError, match="non-triangle"):
        mod._read_ply_minimal(str(p))


# ─── 2. FLIP divergence detector ────────────────────────────────────────

def test_flip_divergence_error_typed():
    """Round-32 contract: FlipDivergenceError mirrors the MPM analogue
    (typed, carries truthful frame fields)."""
    from gpufluid.solvers.solver3d import FlipDivergenceError
    err = FlipDivergenceError(frame=42, last_dumped_frame=40,
                              requested_frames=100)
    assert isinstance(err, RuntimeError)
    assert err.frame == 42
    assert err.last_dumped_frame == 40
    assert err.requested_frames == 100
    assert "42" in str(err)


def test_flip_has_diverged_detects_nan():
    """`has_diverged()` returns True iff self.pos contains NaN."""
    from gpufluid.solvers.solver3d import FlipSolver3D
    sol = object.__new__(FlipSolver3D)
    sol.pos = None
    sol.n_particles = 0
    assert sol.has_diverged() is False
    # Stand-in for wp.array — has `.numpy()`
    sol.pos = types.SimpleNamespace(
        numpy=lambda: np.array([[0, 0, 0], [np.nan, 0, 0]],
                                dtype=np.float32))
    sol.n_particles = 2
    assert sol.has_diverged() is True
    sol.pos = types.SimpleNamespace(
        numpy=lambda: np.array([[0, 0, 0]], dtype=np.float32))
    assert sol.has_diverged() is False


# ─── 3. Cache manifest schema extends with solver + truncation ──────────

def test_cache_manifest_has_truncation_fields(tmp_path):
    """Round-32: round-31 reviewer found that MPM wrote
    truncated_at_frame via raw JSON while FLIP used the typed
    dataclass — read_cache_manifest dropped the field on the typed
    path. Now the dataclass carries it consistently."""
    from gpufluid.io.cache import (
        CacheManifest, write_cache_manifest, read_cache_manifest)
    m = CacheManifest(
        frame_count=10,
        solver="flip",
        truncated_at_frame=7,
        truncation_reason="flip_divergence",
    )
    write_cache_manifest(tmp_path, m)
    back = read_cache_manifest(tmp_path)
    assert back.solver == "flip"
    assert back.truncated_at_frame == 7
    assert back.truncation_reason == "flip_divergence"


def test_cache_manifest_defaults_back_compat():
    """Pre-round-32 cache.json files without solver/truncation keys
    must still parse (default `solver="flip"`, others None)."""
    from gpufluid.io.cache import CacheManifest
    m = CacheManifest()
    assert m.solver == "flip"
    assert m.truncated_at_frame is None
    assert m.truncation_reason is None


# ─── 4. DomainTransform per-axis guard ──────────────────────────────────

def test_domain_transform_rejects_negative_extent_axis():
    """Round-32: pre-round-32 only checked `max(dom_size) <= 0`. A
    Mixed-sign extent (mirror on one axis) passed the guard and
    produced negative inv_size — particles seeded outside [0,1]³."""
    spec = importlib.util.spec_from_file_location(
        "_test_dt",
        _ADDON_DIR / "gpufluid_blender" / "domain_transform.py",
    )
    dt = importlib.util.module_from_spec(spec)
    sys.modules["_test_dt"] = dt   # dataclass needs cls.__module__ lookup
    spec.loader.exec_module(dt)
    # world_lo[1] > world_hi[1] → dom_size[1] negative
    with pytest.raises(ValueError, match="degenerate"):
        dt.DomainTransform.from_world_aabb(
            world_lo=(0.0, 1.0, 0.0),
            world_hi=(2.0, -0.5, 1.0),
            resolution=64,
        )


# ─── 5. OT_clear_cache source-grep contract ─────────────────────────────

def test_clear_cache_does_not_drop_other_domains_preload():
    """Round-32: pre-round-32 OT_clear_cache called `_PRELOAD.clear()`
    on a single-domain clear, dropping every OTHER domain's preload
    until re-attach. Source-grep contract: no live (non-comment) line
    calls the full clear API in helpers.py."""
    src = (_ADDON_DIR / "gpufluid_blender" / "operators"
           / "helpers.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    assert "_PRELOAD.clear()" not in code, (
        "round-32 regressed: must invalidate per-domain, not clear all")
    assert "_PRELOAD.invalidate" in code, (
        "round-32: expected selective .invalidate(name) path")


# ─── 6. pushback snap_eps now parametric ────────────────────────────────

def test_pushback_snap_eps_no_longer_hardcoded():
    """Round-32: 0.0001 literal removed from non-comment code; snap_eps
    is passed as a kernel parameter scaled by dx."""
    src = (_REPO / "src" / "gpufluid" / "sim" / "mpm"
           / "pushback.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    assert "0.0001" not in code, "round-32 regressed: snap-eps re-hardcoded"
    assert "snap_eps" in code
