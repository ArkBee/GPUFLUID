"""Audit 2026-06-14 — MC vertices off by half a cell vs the cell-centred grid (#12).

The density/SDF grid is cell-centred (scatter: i = p/dx - 0.5), but MC verts were
mapped to world via g*dx instead of (g+0.5)*dx, shifting the fluid mesh -0.5*dx
relative to the raw particles and the cell-centred obstacle SDFs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_CUDA

SURFACE = (Path(__file__).resolve().parents[1] / "src" / "gpufluid"
           / "meshing" / "surface.py")


def test_both_mc_paths_add_the_half_cell():
    code = "\n".join(l for l in SURFACE.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    assert "(verts + 0.5) * self.dx" in code, \
        "audit #12: GPU MC must map grid-index g -> world (g+0.5)*dx"
    assert "+ 0.5 * self.dx" in code, \
        "audit #12: CPU (skimage) MC must add the half-cell too"


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_mesh_centre_aligns_with_particle_centre():
    """The meshed surface must sit ON the particles, not half a cell off."""
    import warp as wp
    from gpufluid.meshing.surface import MeshExtractor

    N, dx = 64, 1.0 / 64.0
    rng = np.random.default_rng(0)
    P = rng.uniform([0.30, 0.30, 0.30], [0.70, 0.70, 0.70],
                    size=(20000, 3)).astype(np.float32)
    p_centre = (P.min(0) + P.max(0)) / 2
    pos = wp.array(P, dtype=wp.vec3)
    for use_gpu in (True, False):
        ex = MeshExtractor(N, N, N, dx, use_gpu_mc=use_gpu)
        v, _ = ex.extract(pos, iso_level=0.6, smooth_passes=2)
        v_centre = (v.min(0) + v.max(0)) / 2
        off_cells = np.abs(v_centre - p_centre) / dx
        # before the fix this was ~0.5 cell; after it must be well under
        assert (off_cells < 0.25).all(), (
            f"{'GPU' if use_gpu else 'CPU'} mesh centre off particle centre by "
            f"{off_cells.round(2)} cells (audit #12: half-cell shift back?)")
