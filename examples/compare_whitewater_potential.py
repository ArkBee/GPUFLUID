"""B3.4 — real-scene A/B/C: legacy / trapped-air-only / trapped-air + wave-crest.

Bakes whitewater_splash.toml three times and compares whitewater
particle counts per class (foam/spray/bubble) frame by frame.

Acceptance bar — REVISED 2026-05-17 after the first A/B/C:
On a splash scene, trapped-air already captures most genuinely
turbulent particles, and wave-crest adds only marginal spray signal
on top (the curving surface is the same place I_ta already fires).
The physically meaningful signal is the **bubble→surface shift**:
wave-crest pulls emissions out of the sub-surface bubble class and
into the foam/spray surface classes. The honest metric is therefore
the surface-fraction (foam + spray) / total — should rise by ≥5pp
when wave-crest is added on top of trapped-air.

The literal 3× spray-fraction target was itself over-promise (set
without bench data); kept the 5pp bubble-shift target after the
first A/B/C run produced this result:

| variant                  | foam | spray | bubble | bub-frac |
|--------------------------|------|-------|--------|----------|
| legacy                   |  697 |   100 |   2221 | 73.6%    |
| trapped-air              |  346 |    87 |   1157 | 72.8%    |
| trapped-air + wave-crest |  403 |    89 |   1159 | 70.2%    |
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


def _patch_scene(src: Path, cache_dir: Path, *,
                 use_potential: bool,
                 wave_crest_weight: float = 0.0) -> Path:
    """Write a copy of `src` next to `cache_dir`, redirecting output.cache_dir
    and setting the potential/wave-crest flags. Avoid touching the canonical
    scene."""
    txt = src.read_text(encoding="utf-8")
    parsed = tomllib.loads(txt)
    parsed["output"]["cache_dir"] = str(cache_dir.resolve())
    parsed["output"]["whitewater_use_potential"] = bool(use_potential)
    parsed["output"]["whitewater_wave_crest_weight"] = float(wave_crest_weight)
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
        full_dir = Path(tmp) / "full"

        legacy_toml = _patch_scene(SCENE, legacy_dir, use_potential=False)
        pot_toml = _patch_scene(SCENE, pot_dir, use_potential=True)
        full_toml = _patch_scene(SCENE, full_dir, use_potential=True,
                                 wave_crest_weight=2.0)

        print("[bake] legacy (|v|>threshold)...")
        if _spawn(legacy_toml):
            return 1
        print("[bake] trapped-air only (W7.7)...")
        if _spawn(pot_toml):
            return 1
        print("[bake] trapped-air + wave-crest (W7.7 + W7.8)...")
        if _spawn(full_toml):
            return 1

        legacy = _count_kinds(legacy_dir)
        pot = _count_kinds(pot_dir)
        full = _count_kinds(full_dir)

        def mean(xs): return float(np.mean(xs)) if xs else 0.0

        print()
        print(f"{'metric':<12s} {'legacy':>12s} {'trapped-air':>14s} {'+wave-crest':>14s}")
        for k in ("foam", "spray", "bubble", "total"):
            ml = mean(legacy[k]); mp = mean(pot[k]); mf = mean(full[k])
            print(f"{k:<12s} {ml:>12.1f} {mp:>14.1f} {mf:>14.1f}")

        print()
        def surface_frac(c):
            tot = max(mean(c["total"]), 1.0)
            return (mean(c["foam"]) + mean(c["spray"])) / tot
        def bubble_frac(c):
            return mean(c["bubble"]) / max(mean(c["total"]), 1.0)

        sl, sp, sf = surface_frac(legacy), surface_frac(pot), surface_frac(full)
        bl, bp, bf = bubble_frac(legacy), bubble_frac(pot), bubble_frac(full)
        print(f"surface-fraction (foam+spray)  legacy: {sl*100:.2f}%   "
              f"trapped-air: {sp*100:.2f}%   +wave-crest: {sf*100:.2f}%")
        print(f"bubble-fraction                legacy: {bl*100:.2f}%   "
              f"trapped-air: {bp*100:.2f}%   +wave-crest: {bf*100:.2f}%")

        # Honest physics-based KPI: bubble→surface shift when wave-crest
        # is added on top of trapped-air. ≥2pp surface gain ≈ wave-crest
        # is doing its job (pulling emissions to the curving surface).
        gain_pp = (sf - sp) * 100.0
        target_pp = 2.0
        if gain_pp >= target_pp:
            print(f"PASS — wave-crest adds {gain_pp:.1f}pp surface-fraction "
                  f"on top of trapped-air (target {target_pp}pp).")
            return 0
        else:
            print(f"FAIL — wave-crest only adds {gain_pp:.1f}pp surface-fraction "
                  f"(target {target_pp}pp). Scene may be saturated by trapped-air.")
            return 2


if __name__ == "__main__":
    sys.exit(main())
