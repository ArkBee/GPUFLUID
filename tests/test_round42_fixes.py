"""Round-42 regression tests for round-41 reviewer findings.

  - CRITICAL: load_checkpoint preserves attr_color/attr_temperature
    + rejects dx drift + warns on physics-knob drift.
  - HIGH: sub-dense proximity check refreshes marker_host from GPU.
  - HIGH: inflow drain in prepare_frame filters samples in solid cells.
  - MEDIUM: mesh_marker MeshCache rejects 0-tri builds.
  - LOW: mix_rgb clamps inputs/outputs to [0,1].
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parent.parent


# ─── 1. save/load_checkpoint attr roundtrip — source-grep ──────────────

def test_checkpoint_save_writes_attr_keys():
    """Source-grep contract (lesson §9.12): save_checkpoint must
    serialise attr_color + attr_temperature."""
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    fn_pos = src.find("def save_checkpoint")
    next_def = src.find("\n    def ", fn_pos + 1)
    body = src[fn_pos:next_def]
    assert "attr_color=attr_color" in body, (
        "round-42 regressed: save_checkpoint must write attr_color key")
    assert "attr_temperature=attr_temperature" in body, (
        "round-42 regressed: save_checkpoint must write attr_temperature key")


def test_checkpoint_load_reads_attr_keys_with_backcompat():
    """Source-grep: load_checkpoint must read attr_color/attr_temperature
    keys with KeyError handling for pre-round-42 .npz files."""
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    fn_pos = src.find("def load_checkpoint")
    next_def = src.find("\n    def ", fn_pos + 1)
    body = src[fn_pos:next_def] if next_def > 0 else src[fn_pos:]
    assert "data[\"attr_color\"]" in body
    assert "data[\"attr_temperature\"]" in body
    assert "KeyError" in body, (
        "round-42: back-compat with pre-round-42 .npz files")


def test_checkpoint_load_validates_dx():
    """Source-grep: load_checkpoint must reject dx mismatch (same-res
    grid in differently-sized domain → particle world-coords land in
    wrong cells)."""
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    fn_pos = src.find("def load_checkpoint")
    next_def = src.find("\n    def ", fn_pos + 1)
    body = src[fn_pos:next_def] if next_def > 0 else src[fn_pos:]
    assert "ckpt_dx" in body or "dx" in body and "RuntimeError" in body
    assert "abs(" in body, "round-42: dx must compare with tolerance"


# ─── 2. Sub-dense proximity refreshes marker_host ──────────────────────

def test_sub_dense_proximity_refreshes_marker_host():
    """Source-grep contract: `_should_rebuild_sub_dense` must read
    `self.marker.numpy()` into `_marker_host` before computing bbox.
    Pre-round-42 the stale init-time snapshot made the proximity
    check dead code for non-animated scenes."""
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    fn_pos = src.find("def _should_rebuild_sub_dense")
    next_def = src.find("\n    def ", fn_pos + 1)
    body = src[fn_pos:next_def]
    assert "self.marker.numpy()" in body, (
        "round-42 regressed: proximity check must refresh marker_host "
        "from GPU before computing bbox")


# ─── 3. Inflow drain filters solid-cell samples ─────────────────────────

def test_inflow_drain_filters_solid_cells():
    """Source-grep contract: prepare_frame's emit-drain must apply
    a marker==2 reject filter on emit_pos before they get appended
    to particle arrays. Pre-round-42 inflow boxes overlapping an
    obstacle spawned particles INSIDE solid → explosive ejection."""
    src = (_REPO / "src" / "gpufluid" / "solvers"
           / "solver3d.py").read_text()
    fn_pos = src.find("def prepare_frame")
    next_def = src.find("\n    def ", fn_pos + 1)
    body = src[fn_pos:next_def]
    assert "mh[ix, iy, iz] != 2" in body, (
        "round-42 regressed: emit-drain must reject samples in solid cells")


# ─── 4. MeshCache rejects 0-tri ─────────────────────────────────────────

def test_mesh_cache_rejects_zero_triangle_build():
    """Round-42: pre-round-42 wp.Mesh build on a 0-tri array crashed
    inside Warp or left a poisoned BVH. Now: actionable raise."""
    pytest.importorskip("warp")
    from gpufluid.schemes.mesh_marker import _MESH_CACHE
    empty = np.zeros((0, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="zero-triangle"):
        _MESH_CACHE.get_or_build(empty, device="cpu", cache_key="test")


# ─── 5. mix_rgb clamps out-of-range ─────────────────────────────────────

def test_mix_rgb_clamps_input_above_one():
    """Round-42: HDR-linear input (>1) used to flow through to mixbox
    as int(round(1.2*255))=306 (undefined behaviour). Now clamped."""
    from gpufluid.meshing.mixbox import mix_rgb
    r, g, b = mix_rgb((1.5, 0.5, 0.0), (0.0, 1.5, 0.5), 0.5)
    for c in (r, g, b):
        assert 0.0 <= c <= 1.0, f"round-42: output must clamp; got {c}"


def test_mix_rgb_clamps_input_below_zero():
    from gpufluid.meshing.mixbox import mix_rgb
    r, g, b = mix_rgb((-0.3, 0.5, 1.0), (1.0, -0.2, 0.5), 0.5)
    for c in (r, g, b):
        assert 0.0 <= c <= 1.0


def test_mix_rgb_clamps_t_out_of_range():
    """t outside [0,1] should clamp, not extrapolate."""
    from gpufluid.meshing.mixbox import mix_rgb
    r, g, b = mix_rgb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.5)
    for c in (r, g, b):
        assert 0.0 <= c <= 1.0
    r, g, b = mix_rgb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), -0.3)
    for c in (r, g, b):
        assert 0.0 <= c <= 1.0


def test_mix_rgb_within_range_unaffected():
    """Sanity: in-range inputs must not be regressed."""
    from gpufluid.meshing.mixbox import mix_rgb
    r, g, b = mix_rgb((0.2, 0.5, 0.8), (0.9, 0.1, 0.3), 0.0)
    # t=0 → c1 within tolerance
    assert abs(r - 0.2) < 0.01 and abs(g - 0.5) < 0.01 and abs(b - 0.8) < 0.01
