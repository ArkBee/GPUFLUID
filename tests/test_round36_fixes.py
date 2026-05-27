"""Round-36 regression tests for round-35 reviewer findings.

  - CPU reseed_particles carries attr_color / attr_temperature
    in lockstep with pos/vel (fixes round-34 mirror-drift §9.6).
  - CLI FLIP reseed branch propagates attrs (source-grep guard).
  - MeshExtractor.extract returns (None, None) on empty pos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parent.parent


# ─── 1. CPU reseed attr lockstep ────────────────────────────────────────

def test_cpu_reseed_keeps_attr_color_in_sync():
    """Pre-round-36 a CPU reseed dropped attr arrays — colours went
    white silently because the downstream guard found shape mismatch."""
    from gpufluid.sim.reseed import reseed_particles, ReseedConfig
    N = 16
    dx = 1.0 / N
    cfg = ReseedConfig(min_per_cell=2, max_per_cell=8, jitter=0.4)
    # 20 particles all in cell (8, 8, 8) → over capacity → some culled
    pos = np.full((20, 3), 0.5, dtype=np.float32)
    vel = np.zeros((20, 3), dtype=np.float32)
    col = np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (20, 1))
    marker = np.zeros((N, N, N), dtype=np.int32)
    marker[8, 8, 8] = 1
    rng = np.random.default_rng(0)
    new_pos, new_vel, n_emit, n_cull, new_col, new_t = reseed_particles(
        pos, vel, marker, dx, cfg, rng, attr_color=col)
    assert n_cull > 0, "fixture must cull to exercise the keep_mask path"
    assert new_col is not None
    assert new_col.shape[0] == new_pos.shape[0], (
        f"colour rows must match pos rows after reseed; got "
        f"{new_col.shape[0]} vs {new_pos.shape[0]}")
    assert new_t is None


def test_cpu_reseed_keeps_attr_temperature_in_sync():
    from gpufluid.sim.reseed import reseed_particles, ReseedConfig
    N = 16
    dx = 1.0 / N
    cfg = ReseedConfig(min_per_cell=2, max_per_cell=8, jitter=0.4)
    pos = np.full((20, 3), 0.5, dtype=np.float32)
    vel = np.zeros((20, 3), dtype=np.float32)
    t = np.full((20,), 25.0, dtype=np.float32)
    marker = np.zeros((N, N, N), dtype=np.int32)
    marker[8, 8, 8] = 1
    rng = np.random.default_rng(0)
    new_pos, new_vel, n_emit, n_cull, new_col, new_t = reseed_particles(
        pos, vel, marker, dx, cfg, rng, attr_temperature=t)
    assert new_t is not None
    assert new_t.shape[0] == new_pos.shape[0]
    assert new_col is None


def test_cpu_reseed_emits_get_default_attrs():
    """When the reseed EMITS new particles, the appended attr rows must
    default to (0,0,0) colour / 20.0 temperature — mirrors round-34's
    prepare_frame convention."""
    from gpufluid.sim.reseed import reseed_particles, ReseedConfig
    N = 16
    dx = 1.0 / N
    # Force emission: empty fluid cell needs reseeding.
    cfg = ReseedConfig(min_per_cell=4, max_per_cell=16, jitter=0.4)
    pos = np.zeros((0, 3), dtype=np.float32)
    vel = np.zeros((0, 3), dtype=np.float32)
    col = np.zeros((0, 3), dtype=np.float32)
    t = np.zeros((0,), dtype=np.float32)
    marker = np.zeros((N, N, N), dtype=np.int32)
    marker[8, 8, 8] = 1
    rng = np.random.default_rng(0)
    new_pos, new_vel, n_emit, n_cull, new_col, new_t = reseed_particles(
        pos, vel, marker, dx, cfg, rng,
        attr_color=col, attr_temperature=t)
    assert n_emit > 0
    assert new_col.shape == (n_emit, 3)
    assert np.allclose(new_col, 0.0)
    assert new_t.shape == (n_emit,)
    assert np.allclose(new_t, 20.0)


def test_cpu_reseed_back_compat_4tuple_when_no_attrs():
    """Existing callers (tests/, examples/) unpack 4 values. Variant
    return preserves that signature when no attrs are passed."""
    from gpufluid.sim.reseed import reseed_particles, ReseedConfig
    N = 16
    dx = 1.0 / N
    cfg = ReseedConfig(min_per_cell=2, max_per_cell=8, jitter=0.4)
    pos = np.full((20, 3), 0.5, dtype=np.float32)
    vel = np.zeros((20, 3), dtype=np.float32)
    marker = np.zeros((N, N, N), dtype=np.int32)
    marker[8, 8, 8] = 1
    out = reseed_particles(pos, vel, marker, dx, cfg,
                            np.random.default_rng(0))
    assert len(out) == 4, f"4-tuple back-compat broken: got {len(out)}"
    new_pos, new_vel, n_emit, n_cull = out
    assert new_pos.shape[1] == 3


# ─── 2. CLI source-grep — FLIP reseed branch propagates attrs ──────────

def test_cli_reseed_passes_attrs_for_cpu_path():
    """Source-grep contract (lesson §9.12): the CPU FLIP-reseed branch
    in commands.py must pass `attr_color=` and `attr_temperature=` to
    `reseed_particles`, mirroring the GPU branch."""
    src = (_REPO / "src" / "gpufluid" / "cli"
           / "commands.py").read_text()
    code = "\n".join(ln for ln in src.splitlines()
                    if not ln.lstrip().startswith("#"))
    # Locate the CPU branch (the `else` after `RESEED_GPU_THRESHOLD`).
    cpu_branch_start = code.find("RESEED_GPU_THRESHOLD")
    cpu_branch_end = code.find("def ", cpu_branch_start)
    body = code[cpu_branch_start:cpu_branch_end]
    assert "attr_color=" in body, (
        "round-36 regressed: CPU reseed must pass attr_color")
    assert "attr_temperature=" in body, (
        "round-36 regressed: CPU reseed must pass attr_temperature")


# ─── 3. MeshExtractor empty-pos guard ───────────────────────────────────

def test_mesh_extractor_empty_returns_none():
    """Round-36: zero-particle extract returns (None, None) cleanly
    instead of crashing in HashGrid.build OR returning the previous
    frame's ghost surface."""
    # Build a tiny extractor without GPU work (skip if warp/CUDA
    # missing — this test exercises only the guard at the top).
    pytest.importorskip("warp")
    import warp as wp
    from gpufluid.meshing.surface import MeshExtractor
    try:
        ex = MeshExtractor(8, 8, 8, dx=1 / 8, use_gpu_mc=False)
    except Exception:
        pytest.skip("MeshExtractor init requires CUDA device")
    empty = wp.array(np.zeros((0, 3), dtype=np.float32),
                     dtype=wp.vec3, device=ex.device)
    verts, faces = ex.extract(empty)
    assert verts is None and faces is None
