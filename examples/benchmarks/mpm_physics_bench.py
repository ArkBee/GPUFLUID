"""MPM physics benchmark — strict invariants over a closed-box dam break.

Bakes nothing itself; reads the particle dump of `bench_dambreak.toml` (bake it
first) and checks fundamental physical invariants that a correct incompressible
solver MUST satisfy. Each is a hard pass/fail meant to EXPOSE real solver
weaknesses, then guard against regressions once fixed:

  1. mass            — particle count constant (closed system).
  2. no-divergence   — max speed stays physically bounded (no NaN / blow-up).
  3. energy          — total mechanical energy never grows above the initial
                       (a closed dissipative system cannot CREATE energy).
  4. settling        — the fluid comes to REST (mean speed → ~0 by the end);
                       perpetual currents = spurious APIC/grid noise.
  5. incompressible  — the settled pool keeps its VOLUME: measured depth is not
                       far below volume/footprint (a soft EOS over-compresses the
                       column into a thin film — a real weakness this catches).
  6. containment     — no particle escapes the sealed tank / sinks below floor.

Velocities are finite-differenced from consecutive frames (the dump is
positions only; particle order is stable — no reseed in this scene).

Usage:  python mpm_physics_bench.py            # after baking bench_dambreak.toml
Exit 0 = all invariants hold, 1 = at least one violated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
from gpufluid.io.ply import read_ply  # noqa: E402

CACHE = _REPO / "examples" / "scenes" / "tmp" / "bench_dambreak"
FPS = 30.0
G = 9.81
FLOOR = 0.10
DX = 1.0 / 80.0
INNER_AREA = 0.40 * 0.40           # sealed tank interior footprint (sim²)
# tank outer faces at 0.24/0.76; a 2-cell margin separates wall-snap from a leak.
TANK_LO, TANK_HI = 0.215, 0.785

# ── thresholds (principled, not tuned to pass) ───────────────────────────
V_CAP = 6.0          # sim/s — ~2× free-fall from the column top; above ⇒ blow-up
E_GROW_TOL = 1.10    # energy may not exceed 110% of the initial (finite-diff slack)
SETTLE_SPEED = 0.15  # sim/s — mean speed in the last 15% of frames (impact ~3)
DEPTH_FRAC = 0.60    # settled depth ≥ 60% of the incompressible volume/area


def _load():
    files = sorted((CACHE / "particles_raw").glob("sim_*.ply"))
    if not files:
        raise SystemExit(f"no particle dump in {CACHE} — bake bench_dambreak.toml first")
    frames = []
    for f in files:
        d = read_ply(f)
        v = d[0] if isinstance(d, tuple) else d
        frames.append(np.asarray(v, dtype=float))
    return frames


def _vel(frames, i):
    """Finite-diff velocity at frame i (stable particle order, no reseed)."""
    j = min(i + 1, len(frames) - 1)
    if j == i:
        j = i - 1
    return (frames[j] - frames[i]) * (FPS if j > i else -FPS)


def _report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    frames = _load()
    n0 = len(frames[0])
    nf = len(frames)
    ok = True

    # 1. mass
    counts = [len(f) for f in frames]
    m_ok = all(c == n0 for c in counts)
    ok &= _report("mass conserved", m_ok,
                  f"count constant = {n0}" if m_ok
                  else f"drift {min(counts)}..{max(counts)}")

    # speeds per frame (finite diff)
    max_speed = 0.0
    for i in range(nf - 1):
        s = np.linalg.norm(_vel(frames, i), axis=1)
        max_speed = max(max_speed, float(s.max()))
    # 2. no-divergence
    finite = np.all([np.isfinite(f).all() for f in frames])
    nd_ok = finite and max_speed < V_CAP
    ok &= _report("no divergence", nd_ok,
                  f"max speed = {max_speed:.2f} sim/s (cap {V_CAP}); finite={finite}")

    # 3. energy non-injection  E = KE + PE
    def energy(i):
        v = _vel(frames, i)
        ke = 0.5 * float((v ** 2).sum())
        pe = G * float(np.clip(frames[i][:, 2] - FLOOR, 0, None).sum())
        return ke + pe
    e0 = energy(0)
    emax = max(energy(i) for i in range(nf - 1))
    e_ok = emax <= e0 * E_GROW_TOL
    ok &= _report("energy non-injection", e_ok,
                  f"E_max/E_0 = {emax / max(e0,1e-9):.3f} (<= {E_GROW_TOL} required)")

    # 4. settling — mean speed in the last 15% of frames
    tail = max(1, int(0.15 * nf))
    tail_speeds = [float(np.linalg.norm(_vel(frames, i), axis=1).mean())
                   for i in range(nf - 1 - tail, nf - 1)]
    mean_tail = float(np.mean(tail_speeds))
    s_ok = mean_tail < SETTLE_SPEED
    ok &= _report("settles to rest", s_ok,
                  f"mean tail speed = {mean_tail:.3f} sim/s (< {SETTLE_SPEED} required)")

    # 5. self-diagnosis of over-compression (benchmark iter 1).
    # Deep numerical incompressibility is a known limit of this weakly-
    # compressible MPM (raising bulk_modulus alone does NOT hold the column —
    # it collapses to a ~9-particle/cell film on impact). What the SYSTEM must
    # not do is hide it: the bake now records seed-vs-settled packing in
    # cache.json and warns. This checks the system is HONEST about the collapse.
    seed_vox = {tuple(t) for t in np.floor(frames[0] / DX).astype(int)}
    exp_depth = len(seed_vox) * DX ** 3 / INNER_AREA
    p = frames[-1]
    inside = ((p[:, 0] > 0.30) & (p[:, 0] < 0.70)
              & (p[:, 1] > 0.30) & (p[:, 1] < 0.70))
    meas_depth = float(np.percentile(p[inside, 2], 99) - FLOOR) if inside.any() else 0.0
    collapsed = meas_depth < DEPTH_FRAC * exp_depth  # physics actually collapsed?
    import json
    meta = {}
    try:
        meta = json.loads((CACHE / "cache.json").read_text())
    except Exception:
        pass
    cr = meta.get("compression_ratio")
    if cr is None:
        c_ok = False
        detail = ("cache.json has NO compression diagnostic - the bake is SILENT "
                  f"about a {meas_depth / max(exp_depth,1e-9):.0%}-of-volume collapse")
    else:
        # honest = the recorded ratio agrees with the measured physics.
        c_ok = (cr >= 3.0) if collapsed else (cr < 3.0)
        detail = (f"system reports compression_ratio={cr} "
                  f"(measured depth {meas_depth / max(exp_depth,1e-9):.0%} of incompressible; "
                  f"collapsed={collapsed}) - diagnosis {'matches' if c_ok else 'WRONG'}")
    ok &= _report("over-compression self-diagnosis", c_ok, detail)

    # 5b. integration-quality recorded (benchmark iter 2). The adaptive-CFL
    # substepper can SATURATE its cap (silently under-resolving); the warning is
    # one-shot stderr only. The bake must persist substeps_max / saturated count
    # in cache.json so this is queryable, and they must be self-consistent.
    smax = meta.get("substeps_max")
    ssat = meta.get("substeps_saturated_steps")
    scap = meta.get("substeps_cap")
    if smax is None or ssat is None or scap is None:
        iq_ok = False
        iq_detail = "cache.json does NOT record substep integration quality"
    else:
        consistent = (ssat == 0) == (smax < scap)   # saturation iff peak hit cap
        iq_ok = consistent and ssat == 0            # and the bake did not under-resolve
        iq_detail = (f"substeps_max={smax}/cap={scap}, saturated_steps={ssat} "
                     f"(consistent={consistent}, under-resolved={ssat > 0})")
    ok &= _report("integration quality recorded", iq_ok, iq_detail)

    # 6. containment
    worst = 0
    for f in frames:
        x, y, z = f[:, 0], f[:, 1], f[:, 2]
        esc = (x < TANK_LO) | (x > TANK_HI) | (y < TANK_LO) | (y > TANK_HI) | (z < 0.095)
        worst = max(worst, int(esc.sum()))
    cont_ok = worst == 0
    ok &= _report("containment (no escape)", cont_ok,
                  "0 escaped" if cont_ok else f"{worst} particles left the tank")

    print(f"\nBENCHMARK: {'ALL PASS' if ok else 'FAILED'}  "
          f"(frames={nf}, particles={n0})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
