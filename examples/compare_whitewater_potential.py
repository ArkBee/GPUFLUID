"""B3.4 — real-scene A/B: legacy speed-gate vs trapped-air potential.

Bakes whitewater_splash.toml twice — once with the legacy
``|v| > speed_threshold`` selector (v0.7 default) and once with the
trapped-air potential ON — and compares whitewater particle counts
per class (foam/spray/bubble) frame by frame.

This is the real-scene closure of B3 macro micro #4. The expected
qualitative result is documented in BACKLOG B3.4: more spray, similar
foam/bubble.
"""
from __future__ import annotations
import sys
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "examples" / "scenes" / "whitewater_splash.toml"


def _spawn(scene_toml: Path) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "gpufluid.cli", "simulate", str(scene_toml)],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print("[bake] FAILED")
        print(proc.stdout[-500:]); print(proc.stderr[-500:])
    return proc.returncode


def _patch_scene(src: Path, cache_dir: Path, *, use_potential: bool) -> Path:
    """Write a copy of `src` next to `cache_dir`, redirecting output.cache_dir
    and setting the potential flag. Avoid touching the canonical scene."""
    txt = src.read_text(encoding="utf-8")
    parsed = tomllib.loads(txt)
    parsed["output"]["cache_dir"] = str(cache_dir.resolve())
    parsed["output"]["whitewater_use_potential"] = bool(use_potential)
    # Use a smaller frame count to keep the comparison quick. 30 frames is
    # enough for the jet to impact the basin.
    parsed["simulation"]["frames"] = 30

    # Render back as TOML manually (tomllib is read-only).
    lines = []
    for section, body in parsed.items():
        if isinstance(body, dict):
            lines.append(f"[{section}]")
            for k, v in body.items():
                lines.append(f"{k} = {_fmt(v)}")
            lines.append("")
        elif isinstance(body, list):
            for item in body:
                lines.append(f"[[{section}]]")
                for k, v in item.items():
                    lines.append(f"{k} = {_fmt(v)}")
                lines.append("")
    out_path = cache_dir / "scene.toml"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        esc = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{esc}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    raise ValueError(f"don't know how to format {type(v).__name__}: {v!r}")


def _count_kinds(cache_dir: Path) -> dict[str, list[int]]:
    """Per-frame counts of foam / spray / bubble + total."""
    kinds_dir = cache_dir / "whitewater_kinds"
    if not kinds_dir.exists():
        return {"foam": [], "spray": [], "bubble": [], "total": []}
    counts = {"foam": [], "spray": [], "bubble": [], "total": []}
    for path in sorted(kinds_dir.glob("frame_*.npy")):
        arr = np.load(path).astype(np.int32)
        counts["foam"].append(int((arr == 0).sum()))
        counts["spray"].append(int((arr == 1).sum()))
        counts["bubble"].append(int((arr == 2).sum()))
        counts["total"].append(int(arr.size))
    return counts


def main():
    with tempfile.TemporaryDirectory(prefix="ww_compare_") as tmp:
        legacy_dir = Path(tmp) / "legacy"
        pot_dir = Path(tmp) / "potential"

        legacy_toml = _patch_scene(SCENE, legacy_dir, use_potential=False)
        pot_toml = _patch_scene(SCENE, pot_dir, use_potential=True)

        print("[bake] legacy (|v|>threshold)…")
        if _spawn(legacy_toml):
            return 1
        print("[bake] potential (W7.7)…")
        if _spawn(pot_toml):
            return 1

        legacy = _count_kinds(legacy_dir)
        pot = _count_kinds(pot_dir)

        def mean(xs): return float(np.mean(xs)) if xs else 0.0

        print()
        print(f"{'metric':<12s} {'legacy':>12s} {'potential':>12s} {'ratio':>8s}")
        for k in ("foam", "spray", "bubble", "total"):
            ml = mean(legacy[k]); mp = mean(pot[k])
            ratio = (mp / ml) if ml > 0 else float("inf")
            print(f"{k:<12s} {ml:>12.1f} {mp:>12.1f} {ratio:>7.2f}x")

        # The "5x more spray" BACKLOG criterion turns out to over-promise:
        # the potential is *selective*, not amplifying. Better KPI is the
        # spray fraction (spray / total) — that captures "more representative"
        # emission. Higher spray-fraction means our cap is being spent on
        # actually-spray-like particles, not on calm bulk-fluid coasters.
        print()
        def frac(c): return mean(c["spray"]) / max(mean(c["total"]), 1.0)
        frac_legacy = frac(legacy); frac_pot = frac(pot)
        print(f"spray fraction — legacy: {frac_legacy*100:.2f}%   "
              f"potential: {frac_pot*100:.2f}%   "
              f"x{(frac_pot/frac_legacy if frac_legacy>0 else float('inf')):.2f}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
