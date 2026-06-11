"""Visual regression benchmark — bake 3 canonical scenes, render 5 key
frames each, assemble one labelled contact-sheet PNG for eyeball diffing
between runs.

Motivation: 594 unit tests cannot catch "the water looks wrong"; demo
quality stalls on visual issues no assertion sees. This tool gives a
single command that produces a scenes x key-frames grid you compare by
eye against the previous run's sheet.

Usage (from repo root, venv python):

    .venv\\Scripts\\python.exe tools\\visual_bench.py
    .venv\\Scripts\\python.exe tools\\visual_bench.py --scenes dam --reuse-cache
    .venv\\Scripts\\python.exe tools\\visual_bench.py --skip-render   # scatter fallback

Pipeline per scene:
  1. simulate  — `python -m gpufluid.cli simulate examples/scenes/bench_<n>.toml`
                 (skipped when cache.json exists and --reuse-cache is given)
  2. render    — headless Blender Eevee via examples/render_fluid_on_cube_eevee.py
                 in --test-frames mode: exactly N_KEY evenly-spaced frames
                 (0%, 25%, 50%, 75%, 100% of the baked range), NOT the full
                 animation — keeps a bench run cheap.
                 --skip-render instead scatter-plots the raw particle .npy
                 dumps (cross-check per feedback_particle_dump_verification:
                 meshes can hide a pathological pile the particles reveal).
  3. sheet     — PIL grid (preferred), else matplotlib, else a folder of
                 per-cell PNGs + printed paths (graceful degradation).

Known limitations / upstream follow-ups in the prod render path
(`gpufluid render` -> addon render_bridge + render_fluid_on_cube_eevee):
  * (FIXED — FU-032) `--label` used to silently double as the MATERIAL
    selector with a honey catch-all that rendered BLACK headlessly. The
    renderer now takes an explicit `--material`; the bench passes the
    real scene name as the label + `--material water`.
  * Lighting is hardcoded dim (SUN 4.0, AREA 70, world 0.6) with no
    light/world/exposure CLI knob — only --color and --samples reach
    the scene. Bench compensation: bright default --color + per-tile
    auto-levels/brightness normalisation in the PIL sheet (_tile_pop).

Exit code is non-zero if any selected scene fails to bake or render.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCENES_DIR = REPO / "examples" / "scenes"
BENCH_TMP = REPO / "tmp" / "bench"
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
RENDERER = REPO / "examples" / "render_fluid_on_cube_eevee.py"
N_KEY = 5  # key frames per scene: 0/25/50/75/100% of baked range

# scene name -> TOML (cache_dir inside each TOML already points at tmp/bench/<name>)
SCENES = {
    "dam_break":     SCENES_DIR / "bench_dam_break.toml",
    "waterfall":     SCENES_DIR / "bench_waterfall.toml",
    "step_overflow": SCENES_DIR / "bench_step_overflow.toml",
}


def log(scene: str, stage: str, **kv) -> None:
    extra = "  ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[bench] scene={scene} stage={stage} {extra}".rstrip(), flush=True)


def cache_dir_for(name: str) -> Path:
    return BENCH_TMP / name


def key_frame_indices(frame_count: int, n: int = N_KEY) -> list[int]:
    """Evenly spaced frame indices including first and last."""
    last = max(frame_count - 1, 0)
    return sorted({round(i * last / max(n - 1, 1)) for i in range(n)})


# ---------------------------------------------------------------------------
# stage 1 — simulate
# ---------------------------------------------------------------------------

def simulate(name: str, toml: Path, reuse: bool) -> bool:
    cache = cache_dir_for(name)
    if reuse and (cache / "cache.json").exists():
        log(name, "simulate", action="skip", reason="cache exists (--reuse-cache)")
        return True
    t0 = time.time()
    log(name, "simulate", action="run", toml=toml.name)
    r = subprocess.run([sys.executable, "-m", "gpufluid.cli", "simulate", str(toml)],
                       cwd=str(REPO))
    ok = r.returncode == 0 and (cache / "cache.json").exists()
    log(name, "simulate", result="ok" if ok else "FAIL",
        rc=r.returncode, secs=f"{time.time() - t0:.1f}")
    return ok


# ---------------------------------------------------------------------------
# stage 2a — render key frames via headless Blender (--test-frames mode)
# ---------------------------------------------------------------------------

def render_keyframes(name: str, toml: Path, blender: str, samples: int) -> list[Path]:
    cache = cache_dir_for(name)
    out_dir = cache / "bench_render"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # stale PNGs from a previous run must not leak in
    out_dir.mkdir(parents=True)
    cmd = [blender, "--background", "--python", str(RENDERER), "--",
           "--cache", str(cache), "--scene", str(toml),
           # FU-032: material is now explicit — the scene name can be the
           # label again (it used to trigger the honey catch-all and
           # render near-black; the bench worked around with --label water).
           "--out", str(out_dir), "--label", name, "--material", "water",
           # Bright cyan-blue: the renderer's hardcoded lighting is dim
           # (see module docstring) — a "physically nice" dark water color
           # renders near-black. Pop > realism for a regression sheet.
           "--color", "0.35", "0.80", "1.00",
           "--fps", "24", "--samples", str(samples),
           "--test-frames", str(N_KEY)]
    t0 = time.time()
    log(name, "render", action="run", blender=Path(blender).name, samples=samples)
    r = subprocess.run(cmd, cwd=str(REPO))
    pngs = sorted(out_dir.glob("test_f*.png"))
    ok = r.returncode == 0 and len(pngs) == N_KEY
    log(name, "render", result="ok" if ok else "FAIL",
        rc=r.returncode, pngs=len(pngs), secs=f"{time.time() - t0:.1f}")
    return pngs if ok else []


# ---------------------------------------------------------------------------
# stage 2b — scatter fallback from particle dumps (--skip-render)
# ---------------------------------------------------------------------------

def scatter_keyframes(name: str) -> list[Path]:
    import json
    import numpy as np
    cache = cache_dir_for(name)
    try:
        frame_count = int(json.loads((cache / "cache.json").read_text())["frame_count"])
    except Exception as exc:
        log(name, "scatter", result="FAIL", reason=f"cache.json unreadable: {exc}")
        return []
    out_dir = cache / "bench_scatter"
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs: list[Path] = []
    for idx in key_frame_indices(frame_count):
        npy = cache / "particles" / f"frame_{idx:04d}.npy"
        if not npy.exists():
            log(name, "scatter", result="FAIL", reason=f"missing {npy.name} "
                "(bake the scene with [output] particles = true)")
            return []
        pos = np.load(npy)
        out = out_dir / f"scatter_f{idx:04d}.png"
        _scatter_png(pos, out, f"{name} f{idx}")
        pngs.append(out)
    log(name, "scatter", result="ok", pngs=len(pngs))
    return pngs


def _scatter_png(pos, out: Path, title: str) -> None:
    """Orthographic XY scatter (gravity is -Y in bench scenes). PIL-first."""
    try:
        from PIL import Image
        w = h = 480
        img = Image.new("RGB", (w, h), (18, 18, 24))
        px = img.load()
        for x, y in zip(pos[:, 0], pos[:, 1]):
            ix, iy = int(x * (w - 1)), int((1.0 - y) * (h - 1))
            if 0 <= ix < w and 0 <= iy < h:
                px[ix, iy] = (90, 170, 230)
        img.save(out)
        return
    except ImportError:
        pass
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pos[:, 0], pos[:, 1], s=0.2, c="#5aaae6")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(title)
    fig.savefig(out, dpi=96); plt.close(fig)


# ---------------------------------------------------------------------------
# stage 3 — contact sheet
# ---------------------------------------------------------------------------

def build_contact_sheet(rows: dict[str, list[Path]], out: Path) -> str:
    """rows: scene -> N_KEY PNG paths. Returns the mode used."""
    try:
        return _sheet_pil(rows, out)
    except ImportError:
        pass
    try:
        return _sheet_matplotlib(rows, out)
    except ImportError:
        pass
    # degraded: copy frames into a grid-named folder, report paths
    grid = out.with_suffix("")  # folder named after the sheet
    grid.mkdir(parents=True, exist_ok=True)
    for scene, pngs in rows.items():
        for j, p in enumerate(pngs):
            shutil.copy2(p, grid / f"{scene}__col{j}{p.suffix}")
    print(f"[bench] sheet mode=folder (no PIL/matplotlib) -> {grid}")
    return "folder"


def _tile_pop(im):
    """Per-tile auto-levels + brightness floor so near-black water (the
    prod renderer's dim hardcoded lighting, see module docstring) becomes
    judgeable at thumbnail size. autocontrast stretches the histogram;
    the ImageEnhance pass then lifts tiles whose mean luminance is still
    below a readable floor (dark frames stay relatively darker — we
    normalise readability, not absolute exposure)."""
    from PIL import ImageEnhance, ImageOps, ImageStat
    # preserve_tone: per-channel stretch shifts hues (grey floor turned
    # pink in testing) — hue drift is poison for a regression sheet.
    im = ImageOps.autocontrast(im, cutoff=1, preserve_tone=True)
    mean = sum(ImageStat.Stat(im.convert("L")).mean) or 1.0
    floor = 90.0  # target mean luminance for a readable thumbnail
    if mean < floor:
        im = ImageEnhance.Brightness(im).enhance(min(floor / mean, 2.5))
    return im


def _sheet_pil(rows: dict[str, list[Path]], out: Path) -> str:
    from PIL import Image, ImageDraw
    cell_w, label_h = 360, 22
    tiles, cell_h = {}, 0
    for scene, pngs in rows.items():
        imgs = []
        for p in pngs:
            im = _tile_pop(Image.open(p).convert("RGB"))
            im = im.resize((cell_w, max(1, round(im.height * cell_w / im.width))))
            imgs.append((p.stem, im))
        tiles[scene] = imgs
        cell_h = max(cell_h, max(im.height for _, im in imgs))
    n_cols = max(len(v) for v in tiles.values())
    sheet = Image.new("RGB", (n_cols * cell_w, len(tiles) * (cell_h + label_h)),
                      (10, 10, 12))
    draw = ImageDraw.Draw(sheet)
    for i, (scene, imgs) in enumerate(tiles.items()):
        y0 = i * (cell_h + label_h)
        for j, (stem, im) in enumerate(imgs):
            sheet.paste(im, (j * cell_w, y0 + label_h))
            draw.text((j * cell_w + 4, y0 + 4), f"{scene}  {stem}",
                      fill=(235, 235, 235))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return "pil"


def _sheet_matplotlib(rows: dict[str, list[Path]], out: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    n_rows, n_cols = len(rows), max(len(v) for v in rows.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows),
                             squeeze=False)
    for i, (scene, pngs) in enumerate(rows.items()):
        for j in range(n_cols):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(pngs):
                ax.imshow(mpimg.imread(str(pngs[j])))
                ax.set_title(f"{scene} {pngs[j].stem}", fontsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return "matplotlib"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Visual regression bench: bake canonical scenes -> contact sheet")
    ap.add_argument("--scenes", default="",
                    help="comma-separated substring filter on scene names "
                         f"(available: {', '.join(SCENES)})")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="skip simulate when cache.json already exists")
    ap.add_argument("--blender", default=DEFAULT_BLENDER,
                    help="Blender executable for headless Eevee renders")
    ap.add_argument("--samples", type=int, default=8,
                    help="Eevee samples per key frame (bench default low)")
    ap.add_argument("--skip-render", action="store_true",
                    help="no Blender: scatter-plot particle dumps instead")
    args = ap.parse_args(argv)

    wanted = [s.strip() for s in args.scenes.split(",") if s.strip()]
    selected = {n: p for n, p in SCENES.items()
                if not wanted or any(w in n for w in wanted)}
    if not selected:
        print(f"[bench] no scene matches filter {args.scenes!r}; "
              f"available: {', '.join(SCENES)}")
        return 2
    missing = [p.name for p in selected.values() if not p.exists()]
    if missing:
        print(f"[bench] missing scene TOMLs: {missing}")
        return 2
    if not args.skip_render and not Path(args.blender).exists():
        print(f"[bench] blender not found at {args.blender!r}; "
              "pass --blender or use --skip-render")
        return 2

    rows: dict[str, list[Path]] = {}
    failed: list[str] = []
    for name, toml in selected.items():
        if not simulate(name, toml, args.reuse_cache):
            failed.append(name)
            continue
        pngs = (scatter_keyframes(name) if args.skip_render
                else render_keyframes(name, toml, args.blender, args.samples))
        if pngs:
            rows[name] = pngs
        else:
            failed.append(name)

    if rows:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sheet = BENCH_TMP / f"contact_sheet_{stamp}.png"
        mode = build_contact_sheet(rows, sheet)
        print(f"[bench] sheet mode={mode} scenes={len(rows)} -> {sheet}")

    if failed:
        print(f"[bench] FAILED scenes: {', '.join(failed)}")
        return 1
    print(f"[bench] done. all {len(rows)} scenes ok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
