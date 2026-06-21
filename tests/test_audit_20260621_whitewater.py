"""Prod-hardening (2026-06-21, reviewer-whitewater H2): density calibration.

The M5 trilinear density grid is the raw scattered particle COUNT per cell
(k_density_scatter deposits weight 1/particle -> ~ppc for a packed cell), NOT the
[0,1] field the classifier's 0.2/0.8 thresholds expect. Passed raw, every
submerged sample read > 0.8 -> everything was misclassified BUBBLE and foam
vanished. calibrate_density_field normalises by the bulk density.

The old unit test masked this with an idealized [0,1] synthetic grid (§9.7
mock-fidelity failure). These tests use a PRODUCTION-FAITHFUL ppc-scaled grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gpufluid.sim.whitewater import (
    WhitewaterSystem, calibrate_density_field, KIND_FOAM, KIND_SPRAY, KIND_BUBBLE,
)

COMMANDS = Path(__file__).resolve().parents[1] / "src" / "gpufluid" / "cli" / "commands.py"
PPC = 8.0
DX = 0.05


def _ppc_grid():
    """A realistic count grid: bulk≈ppc, surface≈0.3·ppc, sparse≈0.1·ppc."""
    g = np.zeros((20, 20, 20), dtype=np.float32)
    g[5:15, 5:15, 2:8] = PPC            # submerged bulk
    g[5:15, 5:15, 8:10] = 0.3 * PPC     # surface band
    g[5:15, 5:15, 10:12] = 0.1 * PPC    # spray/sparse above
    return g


def _cell_centre_pos(cells):
    # pos that _world_to_cell (floor(p/dx-0.5)) maps back to `cells`
    return (np.asarray(cells, float) + 1.0) * DX


def test_calibrate_normalises_bulk_to_one():
    cal = calibrate_density_field(_ppc_grid())
    occ = cal[cal > 1e-6]
    assert 0.9 <= float(np.percentile(occ, 90)) <= 1.1, "bulk must map to ~1.0"
    assert occ.max() <= 1.5, "calibrated field must be ~[0,1], not ~ppc"


def test_empty_grid_is_passthrough():
    z = np.zeros((4, 4, 4), dtype=np.float32)
    assert calibrate_density_field(z) is z  # all-zero -> unchanged
    assert calibrate_density_field(None) is None


def test_raw_grid_misclassifies_then_calibration_recovers_foam():
    ww = WhitewaterSystem()
    grid = _ppc_grid()
    # sample one particle per region (bulk z=4, surface z=8, spray z=10)
    cells = [(9, 9, 4), (9, 9, 8), (9, 9, 10)]
    pos = _cell_centre_pos(cells)

    raw = ww.classify_kinds(pos, grid, DX, vel=None)
    # BUG (classes shifted "denser" on the ~ppc grid): the SURFACE band (2.4 >
    # 0.8) is wrongly BUBBLE, and the sparse cell (0.8, not > 0.8) is wrongly
    # FOAM instead of SPRAY. Neither reads as what it physically is.
    assert raw[1] == KIND_BUBBLE, "raw: surface band wrongly classified BUBBLE"
    assert raw[2] != KIND_SPRAY, "raw: sparse cell wrongly NOT spray"

    cal = ww.classify_kinds(pos, calibrate_density_field(grid), DX, vel=None)
    # FIXED: bulk(1.0)->bubble, surface(0.3)->foam, sparse(0.1)->spray.
    assert cal[0] == KIND_BUBBLE, "bulk must stay BUBBLE"
    assert cal[1] == KIND_FOAM, "surface band must classify as FOAM after calibration"
    assert cal[2] == KIND_SPRAY, "sparse cell must classify as SPRAY after calibration"


def test_commands_calibrates_before_classify():
    code = "\n".join(l for l in COMMANDS.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#"))
    assert "calibrate_density_field(extractor.dens.numpy())" in code, (
        "the whitewater density grid must be calibrated before the classifier")
