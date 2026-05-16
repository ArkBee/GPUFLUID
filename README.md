# gpufluid

GPU FLIP/PIC fluid simulator on NVIDIA Warp, with a Blender 4.5/5.x addon.

> **Status:** v0.8 in development. Core solver is feature-complete (v0.7
> closed); the Blender addon now exposes most of those features through the
> N-panel. **Not production-ready** — the Blender install path has not been
> independently verified after the v0.8 refactor of the bake operator.

## What it is

* **Engine.** A FLIP/PIC fluid solver built on [NVIDIA Warp](https://github.com/NVIDIA/warp).
  All hot paths (P2G, pressure projection, G2P, marching cubes, mesh SDF,
  whitewater) run on the GPU. The Python API and the `gpufluid` CLI are the
  stable surface — they're designed to be driven from any 3D DCC, not just
  Blender.
* **Blender addon.** A bridge that lets a Blender user mark objects as
  Domain / Fluid Source / Obstacle / Inflow / Outflow, configure simulation
  parameters in the N-panel, click *Bake*, and get an animated mesh
  imported via the cache loader.

## Feature highlights (v0.7)

| Capability | Block IDs | Notes |
|------------|-----------|-------|
| FLIP / PIC / APIC transfer | S2.1, S2.8, S2.12 | switchable per scene |
| Pressure projection: Jacobi, GS-RB, PCG | S2.6.1/2/3 | + block-sparse Jacobi (S2.6.4) |
| Surface tension (CSF) with capillary CFL | S2.14 | auto-substep, force-balance |
| Per-particle RGB color | S2.15 | linear blend; per-source `color = [r,g,b]` |
| Mesh obstacles with GPU BVH inside-test | D4.3.GPU.BVH | uses `wp.Mesh`, scales to 80k+ tris |
| Animated obstacles (linear, keyframes) | D4.6 | full moving-BC, real wake/bow waves |
| Marching cubes on GPU | M5.4 | `wp.MarchingCubes`, ~8× faster than skimage at 128³ |
| Reseed | S2.11 + S2.11.GPU | bounds per-cell density, fights voids on long sims |
| Whitewater: foam / spray / bubble | W7.* | density-grid classifier + per-class dynamics |
| USD output | I6.5 | native Blender `MeshSequenceCache` import |
| Checkpoint / resume | F3.5 | per-frame checkpoint with `--resume` |

The full block index is in [docs/BLOCKS.md](docs/BLOCKS.md); the
architectural design contract is [docs/DESIGN.md](docs/DESIGN.md).

## Hardware target

* NVIDIA RTX 4080 Super (sm_89) verified
* Warp 1.13, CUDA 12.9
* Should run on any sm_70+ Ampere/Ada/Hopper card; not tested on older

## Quickstart — CLI only

```bash
git clone https://github.com/ArkBee/GPUFLUID.git
cd GPUFLUID
python -m venv .venv
.venv\Scripts\activate          # or `source .venv/bin/activate` on Linux
pip install -e .
gpufluid info                   # confirms Warp + CUDA + lists ~74 registered blocks
gpufluid simulate examples/scenes/two_color_drop.toml
```

Bake output lands in the path given by `[output] cache_dir = "..."`. A
typical cache directory contains:

```
<cache_dir>/
├── cache.json              # manifest: which streams exist + per-frame counts
├── mesh/frame_NNNN.ply     # surface mesh per frame
├── particles/frame_NNNN.npy  # (optional) particle positions
├── colors/frame_NNNN.npy     # (optional) per-particle RGB
└── whitewater/frame_NNNN.npy # (optional) foam/spray/bubble particles
```

See [docs/SCENE_SCHEMA.md](docs/SCENE_SCHEMA.md) for the TOML schema.

## Quickstart — Blender addon

1. Build the zip:
   ```bash
   cd addon
   python -c "import shutil; shutil.make_archive('gpufluid_blender', 'zip', '.', 'gpufluid_blender')"
   ```
2. In Blender: *Edit → Preferences → Add-ons → Install from Disk* →
   pick `addon/gpufluid_blender.zip`.
3. In the addon preferences, set **Python interpreter** to the
   `.venv` you used above (must have `gpufluid` installed).
4. In the 3D viewport's N-panel ("GpuFluid" tab):
   * Click *Add Domain* → an Empty appears; configure resolution / frames.
   * Mark an existing object as *Fluid Source*. Set particles-per-cell,
     optionally tint particles a colour.
   * Mark another object as *Obstacle*. Choose its type (bounding box,
     sphere, cylinder, mesh, plane).
   * Optionally enable Surface Tension (S2.14) or Whitewater (W7) on the
     Domain panel.
   * Click *Bake*. The bake runs in a subprocess; progress shows in
     the status bar. On completion the cache is auto-attached to the
     target object (or a new mesh is created).

## Project layout

```
gpufluid/
├── src/gpufluid/         # engine: G1, S2, F3, D4, M5, I6, C7 layers
│   ├── primitives/       # G1 — Warp runtime + grid math
│   ├── solvers/          # F3 — FlipSolver3D, the orchestrator
│   ├── domain/           # D4 — SDFs, mesh SDF (CPU + GPU + BVH), animation
│   ├── meshing/          # M5 — surface mesh, smoothing, decimation
│   ├── sim/              # extra — whitewater, reseed
│   ├── io/               # I6 — PLY, cache.json, USD
│   └── cli/              # C7 — TOML config + simulate/bench/info
├── addon/gpufluid_blender/  # A8 — Blender addon
├── tests/                # pytest, 129 tests, all green
├── examples/             # demo scenes + render scripts
└── docs/                 # DESIGN, BLOCKS, BACKLOG, HANDOFF, SCENE_SCHEMA
```

## Roadmap

Current and upcoming work lives in [docs/BACKLOG.md](docs/BACKLOG.md).
Milestones:

* **v0.7** Solver feature complete — closed 2026-05-16.
* **v0.8** *Reachable* — Blender exposes the v0.7 features. **In progress.**
* **v0.9** *Production-fast* — block-sparse for GS-RB/PCG, CUDA graphs.
* **v1.0** *Scale* — sparse storage (NanoVDB), 256³+ scenes.

## Not promised

We are not building painted-fluid art (Mixbox is the closest thing in the
backlog and has a non-commercial-license problem); phase-change
(melt/freeze); FX-grade caustics. Those would require new architectural
layers that aren't on the roadmap.

## License

Not yet declared. The repo is under active development; if you want to
use any of this in a project, open an issue first.
