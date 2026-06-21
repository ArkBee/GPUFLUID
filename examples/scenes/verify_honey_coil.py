"""Honest verdict for demo_honey_coil.toml — does the honey ROPE-COIL?

Reads the raw particle dump (the honest arbiter — a smoothed mesh render can
hide the truth, see memory feedback-particle-dump-verification). Measures the
DEPOSITION point: the centroid of the freshly-laid top shell of the pile, per
frame, relative to the nozzle axis.

  * A weakly-compressible FLUID rope deposits an axisymmetric heap -> the contact
    sits on-axis (offset ~0.1 cells), no matter the viscosity.
  * The viscoELASTIC (viscoplastic) rope BUCKLES -> the contact sits clearly
    off-axis (~1-2 cells) and WINDS around the axis (the rope coils). That
    off-axis, winding deposition is the coiling signature.

Asserts (drive exit code):
  * heap-builds-up / heap-stays-put — honey cohesion (a contained pile, not a
    water-like flat sheet).
  * COILS — the deposition is genuinely off-axis (mean late-phase contact
    offset > 0.8 cells; the fluid heap is < 0.4) i.e. the rope buckled/coiled
    rather than stacking centrally. Net winding is reported alongside.

Usage:  python verify_honey_coil.py [cache_dir]
Exit 0 = all criteria passed, 1 = a criterion failed.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
from gpufluid.io.ply import read_points_ply  # noqa: E402

PLATE_Z = 0.16
CX = CY = 0.5
PLATE_HALF = 0.30
COIL_OFFSET_CELLS = 0.8   # late-phase contact offset above this = buckled/coiled


def _report(name, passed, detail):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def main() -> int:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        _REPO / "examples" / "scenes" / "tmp" / "demo_honey_coil")
    dx = float(json.load(open(cache / "cache.json")).get("dx", 1.0 / 128))
    files = sorted(glob.glob(str(cache / "particles_raw" / "*.ply")))
    if not files:
        raise SystemExit(f"no particle dump in {cache} - bake the scene first")

    frames = [p for p in (read_points_ply(f) for f in files) if len(p)]
    last = frames[-1]
    x, y, z = last[:, 0], last[:, 1], last[:, 2]
    pile = z > PLATE_Z + 0.5 * dx
    r = np.hypot(x[pile] - CX, y[pile] - CY)
    print(f"honey_coil verdict - {len(frames)} non-empty frames, dx={dx:.4f}")

    # ── honey cohesion ──────────────────────────────────────────────────
    height = float(z[pile].max() - PLATE_Z) if pile.any() else 0.0
    ok_up = _report("heap-builds-up", height > 0.05,
                    f"pile rises {height:.3f} above plate (>0.05)")
    foot = float(np.percentile(r, 95)) if pile.any() else 0.0
    ok_put = _report("heap-stays-put", foot < PLATE_HALF,
                     f"95th-pct footprint r={foot:.3f} (<{PLATE_HALF})")

    # ── coiling: off-axis, winding deposition point ─────────────────────
    offs = []
    for p in frames:
        zz = p[:, 2]
        pl = zz > PLATE_Z + 0.5 * dx
        if pl.sum() < 50:
            continue
        shell = pl & (zz > zz[pl].max() - 6 * dx)
        if shell.sum() < 20:
            continue
        offs.append((p[shell, 0].mean() - CX, p[shell, 1].mean() - CY))
    offs = np.array(offs)
    late = offs[len(offs) // 2:]
    late_off = float(np.hypot(late[:, 0], late[:, 1]).mean() / dx) if len(late) else 0.0
    if len(late) > 4:
        ang = np.unwrap(np.arctan2(late[:, 1], late[:, 0]))
        turns = float((ang[-1] - ang[0]) / (2 * np.pi))
    else:
        turns = 0.0
    ok_coil = _report(
        "COILS", late_off > COIL_OFFSET_CELLS,
        f"late deposition is {late_off:.2f} cells off-axis "
        f"(>{COIL_OFFSET_CELLS} = buckled/coiled; a fluid heap is ~0.1), "
        f"winding {turns:+.2f} turns")

    passed = ok_up and ok_put and ok_coil
    print(f"\n{'ALL CHECKS PASSED - honey rope-coils' if passed else 'CHECK FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
