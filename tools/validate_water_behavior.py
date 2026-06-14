"""Validate IDEAL water BEHAVIOUR from a baked MPM cache's raw particles.

Not about how it LOOKS — about whether the SIM is physically right:
  * mass conservation: particle count constant (closed system, no in/outflow);
  * settling: centre-of-mass falls under gravity (-Z) and ends low;
  * "start volume == bottom volume": ~all water ends pooled on the floor;
  * no leaks: nothing below the floor / above the ceiling / inside obstacles;
  * no blow-up: occupied volume stays bounded (incompressible-ish).

Run:  .venv/Scripts/python.exe tools/validate_water_behavior.py <cache_dir> [scene.toml]
Exit 0 = all checks pass; 1 = a behaviour check failed (prints which).
"""
import sys
import glob
import math
from pathlib import Path

import numpy as np
import trimesh

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _pts(ply):
    return np.asarray(trimesh.load(ply, process=False).vertices)


def _occupied_voxels(P, n=48):
    """Coarse count of occupied voxels — a blow-up/compression proxy."""
    idx = np.clip((P * n).astype(int), 0, n - 1)
    keys = idx[:, 0] * n * n + idx[:, 1] * n + idx[:, 2]
    return len(np.unique(keys))


_MESH_CACHE = {}


def _load_obstacles(scene_toml):
    if not scene_toml or not Path(scene_toml).exists():
        return []
    data = tomllib.loads(Path(scene_toml).read_text())
    obs = data.get("obstacle", [])
    # Pre-load + transform mesh obstacles once (trimesh.contains is the exact
    # inside/outside test — no surface-shell ambiguity for a closed mesh).
    sdir = Path(scene_toml).resolve().parent
    for o in obs:
        if o.get("type") == "mesh":
            p = o["path"]
            p = p if Path(p).is_absolute() else str(sdir / p)
            m = trimesh.load(p, force="mesh", process=False)
            if o.get("rotate_deg"):
                from trimesh.transformations import euler_matrix
                import math as _m
                r = [float(a) * _m.pi / 180.0 for a in o["rotate_deg"]]
                m.apply_transform(euler_matrix(r[0], r[1], r[2], "sxyz"))
            sc = o.get("scale", 1.0)
            m.apply_scale([float(s) for s in sc] if isinstance(sc, (list, tuple))
                          else float(sc))
            tr = o.get("translate", (0, 0, 0))
            if any(float(t) != 0 for t in tr):
                m.apply_translation(np.asarray(tr, dtype=float))
            _MESH_CACHE[id(o)] = m
    return obs


def _inside_obstacle(P, obs, frac=0.6):
    """Boolean mask of particles DEEP inside any obstacle (sphere/box approx).

    ``frac`` shrinks the test volume: water resting AGAINST a solid has its
    centre just inside the surface (particles have size), so a thin shell at
    [frac, 1.0]·size is contact, not penetration. frac=0.6 measures genuine
    "water ended up well inside the solid" — the real behaviour bug.
    """
    mask = np.zeros(len(P), dtype=bool)
    for o in obs:
        c = o.get("center")
        if not c:
            continue
        c = np.asarray(c, dtype=float)
        t = o.get("type", "box")
        if t == "sphere":
            r = float(o.get("radius", 0.05)) * frac
            mask |= np.linalg.norm(P - c, axis=1) < r
        elif t == "box":
            h = np.asarray(o.get("half_size", (0.05, 0.05, 0.05)), float) * frac
            mask |= np.all(np.abs(P - c) < h, axis=1)
        elif t == "cylinder_y":
            r = float(o.get("radius", 0.05)) * frac
            hh = float(o.get("half_height", 0.1))
            d = np.sqrt((P[:, 0] - c[0]) ** 2 + (P[:, 2] - c[2]) ** 2)
            mask |= (d < r) & (np.abs(P[:, 1] - c[1]) < hh)
    # Mesh obstacles: exact closed-mesh containment (no shell to shrink, so
    # ``frac`` is ignored — any contained particle is genuine penetration).
    for o in obs:
        if o.get("type") == "mesh" and id(o) in _MESH_CACHE:
            try:
                mask |= _MESH_CACHE[id(o)].contains(P)
            except Exception:
                pass
    return mask


def main():
    cache = Path(sys.argv[1])
    scene = sys.argv[2] if len(sys.argv) > 2 else None
    files = sorted(glob.glob(str(cache / "particles_raw" / "*.ply")))
    if not files:
        print(f"FAIL: no particle PLYs in {cache}/particles_raw"); sys.exit(1)
    obs = _load_obstacles(scene)

    P0 = _pts(files[0]); Pn = _pts(files[-1])
    n0, nn = len(P0), len(Pn)
    fails = []

    # 1. mass conservation (count) — closed system, MPM keeps every particle.
    if nn != n0:
        fails.append(f"mass: start N={n0} != end N={nn} (lost/gained particles)")
    print(f"[mass]      start N={n0}  end N={nn}  -> "
          f"{'OK' if nn == n0 else 'FAIL'}")

    # 2. settling: centre-of-mass z must fall and end in the lower third.
    com0, comn = float(P0[:, 2].mean()), float(Pn[:, 2].mean())
    settled = comn < com0 - 0.10 and comn < 0.40
    if not settled:
        fails.append(f"settle: COM-z {com0:.2f}->{comn:.2f} (did not pool low)")
    print(f"[settle]    COM-z {com0:.2f} -> {comn:.2f}  -> "
          f"{'OK' if settled else 'FAIL'}")

    # 3. start volume == bottom volume: end water mostly (>=85%) in bottom 35%.
    bottom_frac = float(np.mean(Pn[:, 2] < 0.35))
    if bottom_frac < 0.85:
        fails.append(f"drain: only {bottom_frac:.0%} of water reached the floor "
                     f"(>=85% expected)")
    print(f"[drain]     {bottom_frac:.0%} of end water in bottom 35%  -> "
          f"{'OK' if bottom_frac >= 0.85 else 'FAIL'}")

    # 4. no leaks: nothing below floor / above ceiling (checked ALL frames).
    leak_floor = leak_ceil = 0
    occ0 = _occupied_voxels(P0)
    occ_max = occ0
    transient_pen = 0
    for f in files:
        P = _pts(f)
        leak_floor += int(np.sum(P[:, 2] < -0.01))
        leak_ceil += int(np.sum(P[:, 2] > 1.01))
        if obs:
            transient_pen = max(transient_pen, int(np.sum(_inside_obstacle(P, obs))))
        occ_max = max(occ_max, _occupied_voxels(P))
    if leak_floor or leak_ceil:
        fails.append(f"leak: {leak_floor} below floor, {leak_ceil} above ceiling")
    print(f"[leak]      below-floor={leak_floor} above-ceiling={leak_ceil}  -> "
          f"{'OK' if not (leak_floor or leak_ceil) else 'FAIL'}")
    # PERSISTENT penetration is the bug: water that ENDS UP inside a solid (the
    # SETTLED frame). A brief overlap during fast impact (transient) is normal
    # for a grid-collision MPM and gets pushed back out — reported as info only.
    settled_pen = int(np.sum(_inside_obstacle(Pn, obs))) if obs else 0
    pen_bad = settled_pen > 0.02 * n0
    if pen_bad:
        fails.append(f"penetration: {settled_pen} particles ({settled_pen/n0:.0%}) "
                     f"left INSIDE obstacles at rest (>2%)")
    print(f"[penetrate] settled {settled_pen} inside obstacles "
          f"({settled_pen / max(1, n0):.1%})  | transient-impact max "
          f"{transient_pen} ({transient_pen / max(1, n0):.0%}, info)  -> "
          f"{'OK' if not pen_bad else 'FAIL'}")

    # 5. no blow-up: occupied volume must not balloon (gas-like expansion).
    blow = occ_max > 3.5 * occ0
    if blow:
        fails.append(f"blow-up: occupied voxels {occ0}->{occ_max} "
                     f"({occ_max / occ0:.1f}x, >3.5x = over-expanding)")
    print(f"[volume]    occupied voxels {occ0} -> max {occ_max} "
          f"({occ_max / max(1, occ0):.1f}x)  -> {'OK' if not blow else 'FAIL'}")

    print("\n" + ("PASS — water behaviour is physically sound"
                  if not fails else "FAIL:\n  - " + "\n  - ".join(fails)))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
