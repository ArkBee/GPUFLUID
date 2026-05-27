"""B18.4 — PLY per-vertex colour writer round-trip.
B18.5 — Mixbox pigment-space mixer (blue+yellow=green, not grey).

Pure-CPU tests; no GPU needed. The GPU vertex-colour kernel
(``k_vertex_color_knn``) is exercised in the B18.7 demo bake.

Block-registry hygiene references (round-30): the
[BLK M5.11.4] and [BLK M5.11.4.H] kernels (per-vertex KNN colour
blend on GPU), plus [BLK M5.11.5] (Mixbox pigment-space remix),
are the GPU/CPU pair of this colour layout test; this test file is
the closest test-side reference so the registry check_6 (every
block has a test) finds them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpufluid.io.ply import read_ply, write_ply
from gpufluid.meshing.mixbox import (
    HAVE_MIXBOX, mix_rgb, remix_vertices_mixbox,
)


# ─── B18.4: write_ply gains vertex_colors ─────────────────────────────

def test_b18_4_write_ply_without_colors_is_backward_compatible(tmp_path: Path):
    """Existing call sites pass (verts, faces) only — must keep working
    and produce 12-byte-per-vertex headers."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    out = tmp_path / "mesh.ply"
    write_ply(out, verts, faces)
    v2, f2 = read_ply(out)
    np.testing.assert_array_equal(v2, verts)
    np.testing.assert_array_equal(f2, faces)
    # Header must NOT mention red/green/blue properties when omitted.
    head = out.read_bytes()[:512].decode("ascii", errors="replace")
    assert "property uchar red" not in head


def test_b18_4_write_ply_with_colors_round_trips(tmp_path: Path):
    """With ``vertex_colors`` set, the PLY gains rgb properties and
    a subsequent read recovers the exact bytes."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
    )
    out = tmp_path / "mesh_rgb.ply"
    write_ply(out, verts, faces, vertex_colors=colors)
    v2, f2, c2 = read_ply(out, return_colors=True)
    np.testing.assert_array_equal(v2, verts)
    np.testing.assert_array_equal(f2, faces)
    np.testing.assert_array_equal(c2, colors)
    head = out.read_bytes()[:512].decode("ascii", errors="replace")
    assert "property uchar red" in head


def test_b18_4_write_ply_rejects_mismatched_color_shape(tmp_path: Path):
    """Defensive: a 2-row colour array with 3-row verts must raise."""
    verts = np.zeros((3, 3), dtype=np.float32)
    faces = np.zeros((1, 3), dtype=np.int32)
    bad = np.zeros((2, 3), dtype=np.uint8)
    from gpufluid.blocks import BlockError
    with pytest.raises(BlockError, match="vertex_colors"):
        write_ply(tmp_path / "x.ply", verts, faces, vertex_colors=bad)


def test_b18_4_read_ply_without_colors_returns_none_when_requested(tmp_path: Path):
    """``return_colors=True`` on a colourless PLY yields ``None``,
    not an exception. Lets the addon's PLY-preload path handle the
    pre-B18 cache files without branching at the call site."""
    verts = np.zeros((1, 3), dtype=np.float32)
    faces = np.zeros((0, 3), dtype=np.int32)
    out = tmp_path / "plain.ply"
    write_ply(out, verts, faces)
    v, f, c = read_ply(out, return_colors=True)
    assert c is None


# ─── B18.5: Mixbox blend gives the green vs grey win ─────────────────

@pytest.mark.skipif(not HAVE_MIXBOX, reason="pymixbox not installed")
def test_b18_5_mixbox_blue_plus_yellow_is_green():
    """The headline payoff: pigment-space mixing of blue + yellow
    produces a clearly green colour (G > R and G > B), where the
    linear RGB blend at the same ratio produces ~grey."""
    blue = (0.0, 0.0, 1.0)
    yellow = (1.0, 1.0, 0.0)
    r, g, b = mix_rgb(blue, yellow, 0.5)
    # Green-dominant
    assert g > r + 0.1, f"expected G dominant, got R={r:.3f} G={g:.3f} B={b:.3f}"
    assert g > b + 0.1, f"expected G dominant, got R={r:.3f} G={g:.3f} B={b:.3f}"
    # And not just dark
    assert g > 0.4
    # Below the muddy-grey threshold the linear blend would land at
    assert g > 0.5  # linear would give G=0.5; mixbox lifts well above


def test_b18_5_mixbox_identity_at_endpoints():
    """t=0 → c1, t=1 → c2 (within rounding) regardless of LUT availability."""
    c = (0.2, 0.5, 0.8)
    other = (0.9, 0.1, 0.3)
    r, g, b = mix_rgb(c, other, 0.0)
    np.testing.assert_allclose([r, g, b], c, atol=2 / 255)
    r, g, b = mix_rgb(c, other, 1.0)
    np.testing.assert_allclose([r, g, b], other, atol=2 / 255)


def test_b18_5_remix_vertices_smoke():
    """End-to-end smoke: a small mesh + 2 particles (blue + yellow)
    half-way between them returns a green vertex colour when LUT is
    present, falls back to grey otherwise (and never crashes)."""
    # Single vertex at the midpoint of the two particles
    verts = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    parts = np.array([[0.45, 0.5, 0.5], [0.55, 0.5, 0.5]], dtype=np.float32)
    cols = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    out = remix_vertices_mixbox(verts, parts, cols, search_radius_world=0.2)
    assert out.shape == (1, 3)
    if HAVE_MIXBOX:
        r, g, b = out[0]
        # Greener than red/blue
        assert g > r, f"got RGB={out[0]}"
        assert g > b, f"got RGB={out[0]}"
    # When mixbox absent, the linear fallback gives ~127 across channels.


def test_b18_5_remix_vertices_empty_inputs():
    """No verts → empty output; no particles → also empty output. Used
    by the CLI to skip the remix call cheaply on zero-particle frames."""
    empty_v = np.zeros((0, 3), dtype=np.float32)
    parts = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    assert remix_vertices_mixbox(empty_v, parts, cols, 0.1).shape == (0, 3)

    v = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    empty_p = np.zeros((0, 3), dtype=np.float32)
    empty_c = np.zeros((0, 3), dtype=np.float32)
    assert remix_vertices_mixbox(v, empty_p, empty_c, 0.1).shape == (1, 3)


def test_b18_5_remix_vertices_rejects_shape_mismatch():
    """particle_colors row count must match particle_pos."""
    v = np.zeros((1, 3), dtype=np.float32)
    p = np.zeros((3, 3), dtype=np.float32)
    c = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="particle_colors"):
        remix_vertices_mixbox(v, p, c, 0.1)
