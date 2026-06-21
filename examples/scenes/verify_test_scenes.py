"""Strict physical-property checks for the 2026-06-21 MPM test scenes.

Each scene encodes ONE clear physics property with a hard, quantitative
pass/fail criterion read from the raw particle dump (the honest arbiter — the
beauty render can mislead, see memory feedback-particle-dump-verification):

  tank   — a box-walled tank must CONTAIN a deep pool: mass conserved, ZERO
           particles leak outside the tank / below the floor on any frame, and
           the settled pool is deep + roughly level.
  sphere — water poured on a sphere apex must NOT penetrate the collider (0
           particles inside on any frame), sheds SYMMETRICALLY in X and Y, and
           drains to the floor.

Usage:
    python verify_test_scenes.py tank
    python verify_test_scenes.py sphere
    python verify_test_scenes.py all      # both; exit 1 if ANY criterion fails

Run AFTER baking each scene (`gpufluid simulate examples/scenes/demo_<name>.toml`).
As of 2026-06-21 the bake STRIPS stale per-frame files from a reused cache_dir
itself (cli/commands.py _clear_stale_frame_outputs), so re-baking is safe; the
mass-conservation check below still guards against any stale-file mix.
Exit code 0 = all checked criteria passed, 1 = at least one failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# repo src on path so `gpufluid.io.ply` imports when run directly
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
from gpufluid.io.ply import read_ply  # noqa: E402

DX96 = 1.0 / 96.0  # these scenes are 96³


def _load_frames(cache_dir: Path):
    files = sorted((cache_dir / "particles_raw").glob("sim_*.ply"))
    if not files:
        raise SystemExit(f"no particle dump in {cache_dir} — bake the scene first")
    out = []
    for f in files:
        d = read_ply(f)
        v = d[0] if isinstance(d, tuple) else d
        out.append(np.asarray(v, dtype=float))
    return out


def _report(name, passed, detail):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    return passed


# ── tank: containment / zero-leak / mass ─────────────────────────────────

def check_tank(cache_dir: Path) -> bool:
    frames = _load_frames(cache_dir)
    n0 = len(frames[0])
    ok = True

    counts = [len(f) for f in frames]
    mass_ok = all(c == n0 for c in counts)
    ok &= _report("mass conserved", mass_ok,
                  f"count constant = {n0}" if mass_ok
                  else f"count drifted: min={min(counts)} max={max(counts)}")

    # ZERO leak: tank outer faces at x/y = 0.23/0.77; a 2-cell margin (0.21/0.79)
    # cleanly separates wall-snap jitter from a real leak (which slips to the
    # domain wall ≈0.02–0.20). Floor slip plane at z=0.10.
    worst = 0
    worst_f = -1
    for i, p in enumerate(frames):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        esc = (x < 0.21) | (x > 0.79) | (y < 0.21) | (y > 0.79) | (z < 0.095)
        c = int(esc.sum())
        if c > worst:
            worst, worst_f = c, i
    leak_ok = worst == 0
    ok &= _report("zero leak (every frame)", leak_ok,
                  "0 particles ever outside tank / below floor" if leak_ok
                  else f"{worst} particles escaped at frame {worst_f}")

    # deep + level settled pool (final frame), measured inside the tank interior.
    p = frames[-1]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    inside = (x > 0.27) & (x < 0.73) & (y > 0.27) & (y < 0.73)
    pin = p[inside]
    depth = float(pin[:, 2].max() - 0.10)
    # > 0.06 sim ≈ 6 cells: unambiguously a contained POOL, ~12× the ~0.5-cell
    # slip-floor film a soft EOS collapses to (this scene uses bulk_modulus=10000
    # to stay ~incompressible; achieved depth ≈0.088).
    deep_ok = depth > 0.06
    ok &= _report("deep contained pool", deep_ok,
                  f"settled depth = {depth:.3f} sim (> 0.06 required)")

    # level: per-XY-column top-z spread should be small for a flat surface.
    ix = np.clip(((pin[:, 0] - 0.27) / 0.46 * 20).astype(int), 0, 19)
    iy = np.clip(((pin[:, 1] - 0.27) / 0.46 * 20).astype(int), 0, 19)
    tops = {}
    for cx, cy, cz in zip(ix, iy, pin[:, 2]):
        k = (cx, cy)
        tops[k] = max(tops.get(k, -1e9), cz)
    col_tops = np.array(list(tops.values()))
    spread = float(col_tops.std()) if len(col_tops) > 4 else 0.0
    level_ok = spread < 0.05
    ok &= _report("level surface", level_ok,
                  f"column-top std = {spread:.3f} sim (< 0.05 required)")
    return ok


# ── sphere: non-penetration / symmetry / drainage ────────────────────────

def check_sphere(cache_dir: Path) -> bool:
    frames = _load_frames(cache_dir)
    c = np.array([0.50, 0.50, 0.30])
    r = 0.16
    ok = True

    # NON-PENETRATION: nothing strictly inside (radius minus a half-cell tol).
    r_in = r - 0.5 * DX96
    worst = 0
    worst_f = -1
    for i, p in enumerate(frames):
        d = np.linalg.norm(p - c, axis=1)
        cnt = int((d < r_in).sum())
        if cnt > worst:
            worst, worst_f = cnt, i
    pen_ok = worst == 0
    ok &= _report("non-penetration (every frame)", pen_ok,
                  f"0 particles inside r={r_in:.3f}" if pen_ok
                  else f"{worst} particles inside the sphere at frame {worst_f}")

    # SYMMETRY at peak shedding: pick the frame with the most particles in the
    # shell around the sphere (radial_xy < 0.35, z in [0.15, 0.52]).
    best_f, best_n = -1, -1
    for i, p in enumerate(frames):
        rad = np.hypot(p[:, 0] - 0.5, p[:, 1] - 0.5)
        shell = (rad < 0.35) & (p[:, 2] > 0.15) & (p[:, 2] < 0.52)
        if int(shell.sum()) > best_n:
            best_n, best_f = int(shell.sum()), i
    p = frames[best_f]
    rad = np.hypot(p[:, 0] - 0.5, p[:, 1] - 0.5)
    shell = p[(rad < 0.35) & (p[:, 2] > 0.15) & (p[:, 2] < 0.52)]
    nx_p = int((shell[:, 0] > 0.5).sum()); nx_m = int((shell[:, 0] < 0.5).sum())
    ny_p = int((shell[:, 1] > 0.5).sum()); ny_m = int((shell[:, 1] < 0.5).sum())
    ax = abs(nx_p - nx_m) / max(1, nx_p + nx_m)
    ay = abs(ny_p - ny_m) / max(1, ny_p + ny_m)
    sym_ok = ax < 0.15 and ay < 0.15
    ok &= _report("symmetric shedding", sym_ok,
                  f"frame {best_f}: asym_X={ax:.2%}, asym_Y={ay:.2%} "
                  f"(both < 15% required; n_shell={best_n})")

    # DRAINAGE: final frame, almost everything reached the floor (z < 0.20).
    p = frames[-1]
    frac = float((p[:, 2] < 0.20).mean())
    drain_ok = frac > 0.90
    ok &= _report("drains to floor", drain_ok,
                  f"{frac:.1%} on floor at settle (> 90% required)")
    return ok


SCENES = {
    "tank":   ("tmp/demo_tank", check_tank),
    "sphere": ("tmp/demo_sphere_pour", check_sphere),
}


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    names = list(SCENES) if which == "all" else [which]
    all_ok = True
    base = Path(__file__).resolve().parent
    for name in names:
        if name not in SCENES:
            raise SystemExit(f"unknown scene {name!r}; pick from {list(SCENES)} or 'all'")
        rel, fn = SCENES[name]
        print(f"== {name} ({rel}) ==")
        all_ok &= fn(base / rel)
    print(f"\nOVERALL: {'ALL PASS' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
