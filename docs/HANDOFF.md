# gpufluid — Session Handoff

> Read this file FIRST in a new session. Everything important is here.
> Then read `docs/DESIGN.md` for the architecture contract and `docs/BLOCKS.md`
> for the block index.
>
> **Picking the next task:** the v0.7 roadmap is closed. New macro tasks live
> in `docs/BACKLOG.md` (ordered queue, not FIFO — risk-aware). Open that and
> grab a Tier 1 micro if you don't have user input. Don't pick a Tier 3 macro
> without running its spike micro first (those exist specifically to abort
> the macro early if reality disagrees with the plan).

## 1. Identity

- **Project**: `gpufluid` — GPU FLIP/PIC fluid simulator on NVIDIA Warp + a Blender 4.5/5.x addon.
- **Repo root**: `E:\projects\gpu_flip\gpufluid\`
- **Goal**: a Blender plugin competitive with or better than the paid FLIP Fluids addon.
- **Hardware target**: NVIDIA RTX 4080 Super (sm_89), Warp 1.13, CUDA 12.9.
- **OS**: Windows 11. Bash via Git Bash. Python 3.11.9.

## 2. Workflow Principles (user-set, MUST follow)

1. **Architecture first, code second.** Layered design (G1 → A8). DESIGN.md is the contract.
2. **Block IDs.** Every callable carries a stable `[BLK X.Y.Z]` tag + `@block(...)` decorator. Errors raise `BlockError` with the ID. Sortable / cross-referenceable across DESIGN.md / BLOCKS.md / source.
3. **Tests before "done".** Each feature needs a regression test that programmatically verifies the feature *changes the answer* (not just "code runs"). Lesson learned: claiming "passed" because pipeline didn't crash is wrong — see step8 fix history.
4. **Docs first, then code per docs.** Update DESIGN.md/BLOCKS.md when adding a feature, then implement.
5. **User has gone to sleep style**: keep going through roadmap autonomously, don't ask, deviate from order if a step is clearly faster/cheaper at hand.
6. **Per-step videos.** Each user-visible step produces `out/videos/step{N}.mp4` so user can scrub progress.
7. **Be honest.** If a feature can't be verified end-to-end, say so; don't claim "passed" when it's "pipeline didn't error".

## 3. Directory layout

```
gpufluid/
├── docs/
│   ├── DESIGN.md          # architectural contract (16 sections)
│   ├── BLOCKS.md          # block index + status table
│   └── HANDOFF.md         # this file
├── src/gpufluid/
│   ├── __init__.py        # public re-exports, also triggers @block registrations
│   ├── blocks.py          # registry + decorator + BlockError
│   ├── primitives/        # G1 — Warp runtime + grid math
│   ├── schemes/           # S2 — (reserved, Warp kernels currently live in solvers/)
│   ├── solvers/
│   │   ├── solver2d.py    # legacy 2D solver (still works)
│   │   └── solver3d.py    # main FlipSolver3D — most of the action
│   ├── domain/            # D4 — sdf, mesh_sdf, mesh_sdf_gpu, animation, regions
│   ├── meshing/           # M5 — surface (MC), smoothing, decimate
│   ├── sim/               # extra — whitewater, reseed
│   ├── io/                # I6 — ply r/w, cache.json manifest, usd writer
│   └── cli/               # C7 — config (TOML), commands (simulate/bench/info)
├── addon/gpufluid_blender/   # A8 — Blender addon, ships as zip
│   ├── __init__.py        # bl_info + register/unregister
│   ├── blender_manifest.toml
│   ├── preferences.py     # interpreter path
│   ├── properties.py      # Domain/Fluid/Obstacle/Inflow/Outflow property groups
│   ├── operators/
│   │   ├── bake.py        # modal subprocess + scene collection
│   │   └── helpers.py     # add domain / mark fluid / mark obstacle / etc.
│   ├── cache_loader.py    # frame_change handler + PLY reader for non-USD imports
│   ├── config_builder.py  # bpy-free scene dict → TOML translator
│   └── panels.py          # N-sidebar UI
├── tests/                 # pytest, ≈70 tests, all green
├── examples/
│   ├── scenes/            # TOML demo scenes for each step
│   ├── render_ply_sequence.py        # mp4 renderer (matplotlib)
│   └── render_side_by_side.py        # ablation mp4 (left | right)
├── out/                   # bake outputs (cache/, videos/)
├── pyproject.toml         # editable install, pytest config, console script
└── examples/scenes/*.obj  # generated obstacle/fluid meshes
```

## 4. Environment setup (for fresh checkout)

```bash
cd E:\projects\gpu_flip\gpufluid
python -m venv .venv
.venv\Scripts\activate
pip install -e .
gpufluid info     # confirms Warp + CUDA + lists registered blocks
pytest -q         # ~70 tests; takes <10s with cached Warp kernels
```

Blender addon install:

```bash
cd addon
python -c "import shutil; shutil.make_archive('gpufluid_blender', 'zip', '.', 'gpufluid_blender')"
# install gpufluid_blender.zip via Blender Edit → Preferences → Add-ons → Install from Disk
# in addon prefs set "Python interpreter" to E:\projects\gpu_flip\gpufluid\.venv\Scripts\python.exe
```

## 5. Block registry — current status

| Layer | Implemented | Notes |
|-------|-------------|-------|
| **G1** primitives | G1.1 init, G1.2 zeros, G1.3 clamp, G1.5 sample3, G1.6 scatter_face, G1.7 box-blur, G1.8 cell_centers | rock solid |
| **S2** schemes | S2.1 P2G, S2.2 normalize, S2.3 gravity, S2.4 enforce_solid_bc (**moving-BC ready: uses solid_u/v/w**), S2.5 divergence, S2.6.1 Jacobi pressure, S2.6.2 GS-RB, S2.6.3 PCG (GPU-resident scalars), **S2.6.4 per-tile sparse Jacobi (opt-in `pressure_block_sparse=True`)**, S2.7 grad subtract, S2.8 G2P+FLIP/PIC, S2.10 cfl_substep_count, S2.10.GPU GPU vmax reduction, S2.11 reseed (CPU), S2.11.GPU reseed, S2.12 APIC (QA'd), S2.13 viscosity, S2.14 surface tension (6 sub-blocks), S2.15 per-particle RGB color (3 sub-blocks), **S2.16 active-block bitmask + compaction** | all real |
| **F3** solver | F3.2 FlipSolver3D, F3.3 step pipeline, F3.4 step_cfl, F3.5 save/load checkpoint, F3.6 prepare_frame | |
| **D4** domain | D4.1 wall shell, D4.2.1 sdf_sphere, D4.2.2 sdf_box, D4.2.3 sdf_cylinder_y, D4.2.4 sdf_plane, D4.2.5 sdf_union, D4.3 mesh→SDF (CPU/trimesh), D4.3.GPU GPU triangle ray-cast, **D4.3.GPU.BVH BVH-accelerated inside-test (wp.Mesh+winding query, auto-engaged ≥256 tris)**, D4.4 mark_solid_from_sdf, D4.5.1 box seeder, D4.5.2 mesh seeder, D4.6 animated obstacles (linear+keyframes, ALL kinds incl. mesh) + write_solid_face_vel, D4.7 inflow+outflow, D4.7.GPU stream compaction | |
| **M5** meshing | M5.1 density scatter, M5.2 density blur, M5.3 marching cubes (skimage CPU) + wall mask, **M5.4 GPU MC via wp.MarchingCubes (auto-engages ≥64³)**, M5.5 Taubin/Laplacian smoothing, M5.6 quadric decimation, M5.7 wall margin (now GPU kernel) | |
| **I6** io | I6.1 PLY r/w, I6.2 cache manifest (cache.json), I6.3 particle .npy, I6.5 USD writer (time-sampled mesh) | I6.4 Alembic not implemented (USD wins for Blender's MeshSequenceCache) |
| **C7** cli | C7.1 TOML schema, C7.2 simulate (with --resume / --start-frame / --checkpoint-every), C7.3 bench, C7.4 info | console script `gpufluid` |
| **A8** addon | A8.1–A8.8 all impl. v0.6 UI exposes reseed, decimate, fill_mesh, PLANE obstacle, motion, inflow, outflow, USD, wall_margin | |
| **W7** whitewater | W7.1 system, W7.2 emit, W7.3 ballistic step | first-cut Ihmsen-lite, no foam/spray differentiation yet |

Run `gpufluid info` to see live count. **Verified 2026-05-16 (post-S2.14/15/D4.3.GPU.BVH):** `gpufluid info` now force-loads `gpufluid.sim.whitewater`, `gpufluid.sim.reseed`, and `gpufluid.domain.mesh_sdf_gpu` so they always appear (was a footgun before). Importing `gpufluid` alone in a script still misses those — call the same `import` trio in user scripts if you need the full registry programmatically.

## 6. Demos / videos (in `out/videos/`)

| # | What | Status |
|---|------|--------|
| 1 | M5.7 wall mask | ✅ Eevee render |
| 2 | I6.5 USD + native MeshSequenceCache | ✅ Eevee |
| 3 | GPU-resident PCG perf | ✅ Eevee |
| 4 | Addon UI bake | ✅ Eevee |
| 5 | Whitewater foam emit (W7.x) | ✅ Eevee, peak 7350 ww particles |
| 6 | Viscosity (35× slowdown ν=0 vs ν=2) | ✅ Eevee |
| 7 | APIC vs FLIP | ✅ Eevee, measurably different |
| 8 | Moving mesh obstacle (D4.3.GPU + moving-BC) — **fixed twice**: marker overwrite (round 1), then no-impulse-from-static-BC (round 2). Now real wake (+30 mm bow, −11 mm trough, top-down view). | ✅ matplotlib top-down |
| 9 | 1M-particle waterfall + GPU compaction (D4.7.GPU) | ✅ matplotlib |
| 10 | Reseed on/off ablation (S2.11) | ✅ matplotlib SBS |
| 11 | Mesh seeder (D4.5.2) — bbox confirmed oblong | ✅ matplotlib |
| 12 | Decimation ×4 file shrink (M5.6) | ✅ matplotlib SBS |
| 13 | SDF plane ramp (D4.2.4) ablation | ✅ matplotlib SBS, 36 fr |
| 14 | Checkpoint/resume (F3.5) ablation | ✅ matplotlib SBS |
| 15 | GS-RB 2× convergence vs Jacobi (S2.6.2) | ✅ matplotlib SBS |
| 16 | Kitchen-sink: mesh seed + plane + reseed + decimate + GS-RB + USD | ✅ matplotlib, 30 fr |
| 17 | **Surface tension (S2.14)**: zero-G cube ablation σ=0 vs σ=1 — right pane contracts cube→sphere (35% shrink, 0.7% COM drift), left pane stays put. Required follow-on work in same session: S2.14.5 capillary-wave CFL substepping + S2.14.6 force-balance + 48³ resolution + light viscosity. Initial 32³ render hit parasitic-current instability and smeared against a wall — documented as known-limit of explicit Brackbill-Kothe on coarse grids. | ✅ matplotlib SBS, 60 fr |
| 18 | **Per-particle color (S2.15)**: red cube (left) + blue cube (right) fall under gravity, merge in basin, particles in contact zone develop purple via grid-mean blending. By frame 50 ~99% of particles carry intermediate RGB; bulk mean RGB conserved to <5%. Linear RGB (additive), not pigment — Mixbox LUT is the follow-up. | ✅ matplotlib colored-particle scatter, 90 fr |
| 19 | **BVH mesh obstacle (D4.3.GPU.BVH)**: 20k-tri torus + waterfall. Pre-BVH this scene OOMed at startup (CPU `mesh_to_sdf` allocated 17.9 GB for 20k tris × 64³ cells). With BVH: sim total 0.76s for 90 frames. Bench: 2.3× speedup vs brute-force at 20k tris, 4.4× at 80k tris. CLI auto-routes ≥256-tri obstacles through GPU BVH + indicator-SDF synth. | ✅ matplotlib mesh isosurface, 90 fr |
| 20 | **GPU marching cubes (M5.4)**: 128³ dam-break with per-frame meshing. Bench: 7.9× speedup vs skimage CPU MC at 128³ (39.5ms → 5.0ms). Identical vertex/face counts (lockstep with CPU path at iso-level=0.6). 90-frame bake: sim 5.15s + mesh 1.89s; pre-M5.4 the mesh step alone would have been ~3.6s. Resolved an architectural ceiling — 128³ scenes are now realtime-feasible. | ✅ matplotlib mesh, 90 fr |
| 21 | **GPU particle reseed (S2.11.GPU)**: 80³ scene, initial water block + inflow frames 0-60 + corner outflow, reseed every 15 frames (re-tuned after first take). Bench (out-of-band): 7.2× speedup vs CPU at 500k particles (151ms → 21ms); 6.5× at 1M. Identical emit/cull counts to the CPU path on the same seed (test passes). Per-particle colour compacts in lockstep. **First-take lesson**: with `min_per_cell=4` + `every_n_frames=3` reseed emitted 145k/pass and the render looked cyclic — see trap #12 below. Cured by sane reseed params. | ✅ matplotlib mesh, 90 fr |
| 22 | **Whitewater quality (W7.4/W7.5)**: 80³ waterfall splash with three differentiated classes — spray (cyan dots, low drag, full gravity), foam (white, near-surface), bubble (blue, submerged, pops on reaching surface). Classification by `pos+vel·0.02` density-grid lookup at emit, then re-classifies at every step so particles evolve through classes (spray→foam→bubble→pop). Per-frame counts ~30 spray + ~800 foam + ~2100 bubble. New renderer `examples/render_whitewater.py` shows mesh + colour-coded ww overlay with legend. | ✅ matplotlib mesh+scatter, 90 fr |
| 23 | **Whitewater selector A/B (B3.1/B3.3 demo)**: side-by-side 80³ waterfall + basin splash, identical scene baked twice — left pane uses legacy `\|v\|>threshold` selector, right pane uses the new W7.7 trapped-air potential. Live per-class counters overlay each frame. Peak-splash (frame 30) snapshot: legacy emits 829 foam + 29 spray + 2116 bubble = 2974 total; potential emits 379 foam + 27 spray + 969 bubble = 1375 total. Mesh is identical between panes (same solver state), so the visible whitewater difference is entirely the selector. Demonstrates the v0.8 thesis: trapped-air potential redirects the emit budget toward genuinely turbulent regions instead of treating all fast-moving fluid as ww-eligible. Renderer `examples/render_step23.py` builds the side-by-side directly from the two cache dirs. | ✅ matplotlib mesh+scatter SBS, 60 fr |
| 23b | **Same scene, Eevee renderer**: parallel mp4 produced via `examples/render_step23_eevee.py` (Blender 5.1, headless Eevee). 60 frames bake **27.3 s wall / 456 ms/frame**, ~9× faster than the matplotlib pipeline used for the canonical step23 (4 min). Establishes the Eevee path for future demos (step24+) as the default — matplotlib stays only when per-frame per-vertex colour overlays are required. **Polish gap:** Eevee version currently renders all whitewater as a single emissive sphere; the per-class colour and the live count overlays from step23.mp4 are lost. Closing micro: drive material colour from the `gpufluid_kind` INT attribute via a Geometry Nodes graph + 2D compositor text. | ✅ Blender Eevee, 60 fr |
| 24 | **Kitchen-sink v0.8 demo** — every major v0.8 feature in one bake: `[[fluids]]` multi-source colour (red + blue), surface tension σ=0.1, APIC transfer mode (B6: known safe with σ), sphere obstacle, W7.7 trapped-air potential whitewater, Eevee-rendered single pane. Sim 3.6s for 90 frames @ 64³ + 32 CFL substeps/frame (σ-bound). Renderer `examples/render_step24_eevee.py`: builds a fresh scene in Blender, attaches a custom per-frame loader that (a) reads the surface mesh PLY, (b) per mesh vertex finds the nearest fluid particle and copies its RGB into a FLOAT_COLOR `fluidcol` attribute, (c) updates a vertex-only whitewater mesh that instances a small emissive sphere on every point. Result: the colour-mixing story is visible directly on the smooth surface — red drop + blue drop fall, get rounded by σ, hit the cream obstacle, splash and mix into purple in the basin. Render wall: ~150 s for 90 frames @ 1600×900 (most of it is the per-vertex cdist nearest-particle pass; render alone is ~50 ms/frame). | ✅ Blender Eevee, 90 fr |

**Renderer notes:** Blender MCP hung around step8 first attempt; switched to matplotlib (`examples/render_ply_sequence.py` and `render_side_by_side.py`). Both renderers work, Blender Eevee gives prettier output, matplotlib is reliable. For animation visibility — keep frame count to active-physics phase only (typically 30–60 frames). Otherwise water settles and the rest looks frozen.

## 7. Known traps / lessons learned

1. **`prepare_frame` order matters.** GPU mesh ray-cast writes to `self.marker` directly; if `wp.copy` rebuilds marker AFTER, the GPU marks get wiped. Fix: apply mesh marks AFTER `wp.copy`. (Step 8 round 1)
2. **Static-BC ≠ moving obstacle.** Marking a cell solid is not enough; need `solid_u/v/w` per-face arrays and obstacle velocity written into them, otherwise the obstacle pushes nothing. (Step 8 round 2 — current state)
3. **Reseed marker source.** `_marker_host` only has walls and static obstacles; per-step P2G fluid marks live only on GPU. Reseed must call `solver.marker.numpy()` to get the current marker, or it'll think there are no fluid cells and emit nothing.
4. **Blender 5.x extensions** use `bl_idname = __package__` (resolves to `bl_ext.user_default.gpufluid_blender`) — NOT the literal addon name.
5. **`cache_dir` default.** Blender 5.x StringProperty rejects `"//"` relative prefix at default-value time; set it from the Add Domain operator at runtime instead.
6. **Cache loader handler must filter `obj.type == "MESH"`** — the Domain Empty also carries `gpufluid_cache_dir` (for the bake operator's own bookkeeping) and writing to `.data` on an Empty crashes.
7. **Video looks static? Physics settled.** Check `centroid_y` per frame; if delta < 0.001 from some frame on, that's the equilibrium. Shorten the bake or add continuous inflow.
8. **`@block` ID regex** allows `[A-Z0-9]+` in sub-IDs — so `D4.3.GPU` is legal. Numeric sort order uses tuple `(0, int)` for digits and `(1, str)` for letters.
9. **PCG is correct but slow on small grids.** GPU-resident scalars cut sync overhead 3×, but per-iter kernel launches still dominate vs Jacobi. PCG wins at very large grids; default to Jacobi or GS-RB.
10. **matplotlib renderer is slow** (~1–3s/frame). For 90-frame video budget 2–3 min. For production renders use Blender Eevee/Cycles via the addon.
11. **CSF (S2.14) has its own CFL.** Explicit Brackbill-Kothe is stable only when `dt ≤ √(ρ·dx³/(2π·σ))`. `step_cfl` (F3.4) auto-substeps to enforce this when `surface_tension > 0`. If the user sets `cfl_max_substeps` too low for the chosen σ, step_cfl prints a one-shot stderr warning. Lesson: a discrete-grid CSF *also* needs explicit force-balancing (S2.14.6, subtract mean per-axis impulse) AND a touch of viscosity (`viscosity=0.05`) AND ≥48³ resolution to suppress parasitic currents — otherwise a contracted blob slowly drifts and smears against a wall (this was the user-visible regression on the first step17 render; root cause = O(dx) noise in finite-difference κ on a coarse 32³ grid). The current tested-stable knobs are in `examples/scenes/surftens_on.toml`.
12. **Reseed (S2.11/S2.11.GPU) tuning matters — aggressive defaults thrash the sim.** Setting `reseed_min_per_cell=4` + `reseed_every_n_frames=3` on a scene with mean per-cell density ~1.5 causes reseed to emit 100k+ particles per pass while culling almost as many, recycling the *entire fluid population* every 6-9 frames. Visual symptom: the render looks "cyclic" — same voids reappear in the same places each cycle and get re-filled with zero-velocity emit particles. Logged per-frame emit/cull counts to diagnose this (step21 first take). Cure: use `min_per_cell=1..2` and `every_n_frames=10..15` for steady-state scenes, OR keep aggressive params only for very dense ppc=8 scenes. The reseed kernel itself is correct — this is a scene-config trap.

## 8. Roadmap — v0.7 closed, v0.8+ lives in `docs/BACKLOG.md`

The original v0.7 roadmap is now closed (one-liner summary table below).
**Pick the next task from `docs/BACKLOG.md`** — it's a risk-ordered queue
of macro tasks, each broken into 3–10 micros sized for a single session.
Macros are tagged with `risk:low|med|high`, `value:user|infra|research`, and
dependency edges so a new session can pick safely.

### 8.0 v0.8 partial (2026-05-16) — Blender-side smoke verified

Status of v0.8 milestone macros after this session:

| # | Macro                                                | Where to look |
|---|-------------------------------------------------------|---------------|
| B1 | Addon UI exposes v0.7 params (B1.1–B1.5 done; B1.6 verified via MCP) | step §8.0.B1 below |
| B2 | Mixbox LUT — **skipped**: CC BY-NC 4.0 license conflicts with "compete with paid FLIP Fluids" goal | see §8.0.B2 |
| B3 | Better whitewater classifier — **partial**: B3.1 trapped-air done + emit fold-in. B3.2 / B3.4 real bake / B3.5 video pending. | step §8.0.B3 |

**§8.0.B1 verification record (Blender 5.1.1, MCP, 2026-05-16):**
Reinstalled `addon/gpufluid_blender.zip` (now v0.8.0) into Blender's extensions
dir. Property groups load (`Domain.surface_tension_group` with σ=0 / passes=2,
`Domain.whitewater_group` with enable=False / speed_thr=4.0 / show_foam=True,
`Fluid.use_color` + `Fluid.color`). Reset interpreter preference (extension
re-install nukes it — note for future repackaging). Configured Domain
(32³, 4 frames, σ=0.05, ww enabled) + Fluid (color=red, use_color=True).
`collect_scene()` returned a dict with `cfl=True auto-on from σ>0`, 2 fluid
sources, color RGB carried through, whitewater knobs threaded, 1 in / 1 out
/ 2 obs. Subprocess bake via configured interpreter returncode=0 in 2.13s,
all 4 cache streams written (mesh / particles / colors / whitewater = 4 ply +
3 × 4 npy). `bpy.ops.gpufluid.attach_cache` returned `{'FINISHED'}`,
`frame_change_post` handler registered, scrubbing frames 1→4 swapped
target-mesh vertex counts 1084 → 1196 → 1424 → 1782 (exact match to solver
log). No exceptions from any panel draw during property-access tests.
**Latent bug found:** addon zip reinstall resets the
`prefs.interpreter_path` to empty, so the first bake after upgrade fails
with WinError 87 until the user re-points it. Document or add a default
sniff in preferences.py (future micro).

**§8.0.B2 — Mixbox license:** the published LUT is CC BY-NC 4.0
(scrtwpns/mixbox). Either pay for a commercial licence or write our own
2-pigment K-M solver (~50 lines, less painterly). Deferred until product
direction decided.

**§8.0.B3 status:** GPU trapped-air potential ships (`W7.7` in
`sim/whitewater_potentials.py`); `emit_from_fluid` accepts `potential=`
kwarg for weighted selection. NOT yet wired into `cli/commands.py`, so
the real whitewater_splash bake still uses the legacy speed-threshold
selector. Closing micro: thread `potential` through `commands.py:cmd_simulate`
under a flag, then refresh step22.mp4.

### 8.1 v0.7 done (2026-05-16) — one-line summary

| # | Macro                                                  | Where to look for details |
|---|--------------------------------------------------------|---------------------------|
| 1 | Surface tension (S2.14 + capillary CFL + force balance) | step17.mp4, `test_s2_14_*`, DESIGN.md §5.1 |
| 2 | Per-particle colour (S2.15 P2G/G2P + `[[fluids]]` TOML) | step18.mp4, `test_s2_15_color.py`, DESIGN.md §5.2 |
| 3 | BVH mesh SDF (D4.3.GPU.BVH via `wp.Mesh`)              | step19.mp4, `test_d4_3_gpu_bvh.py`, DESIGN.md §5.3 |
| 4 | GPU marching cubes (M5.4 via `wp.MarchingCubes`)       | step20.mp4, `test_m5_4_gpu_mc.py`, DESIGN.md §8.1 |
| 5 | APIC obstacle QA — no fix needed                       | `test_s2_12_apic_obstacle.py` |
| 6 | Alembic writer (deferred to BACKLOG B10)               | — |
| 7 | GPU particle reseed (S2.11.GPU)                        | step21.mp4, `test_s2_11_gpu_reseed.py`, DESIGN.md §5.4 |
| 8 | GPU CFL `vmax` reduction (S2.10.GPU)                   | `test_s2_10_gpu_cfl.py` |
| 9 | Whitewater quality — foam/spray/bubble                 | step22.mp4, `test_w7_quality.py`, DESIGN.md §6.1 |
| 10 | Sparse v1 (block-sparse pressure)                     | `test_s2_16_sparse_jacobi.py`, DESIGN.md §5.5 — Sparse v2 is BACKLOG B7 |

### 8.2 Live queue (`docs/BACKLOG.md`) — current milestone: **v0.8**

The next-task queue is in `docs/BACKLOG.md`, with release milestones at the
top. We're currently in the **v0.8 "Reachable"** window — Tier 1 macros that
expose v0.7 features through the Blender addon UI.

**Default next task:** B1 (addon UI for v0.7 params) → start with B1.1
(`SurfaceTensionGroup` PropertyGroup in `properties.py`). Then B1.2..B1.6 in
order. After B1 lands → pick B2 (Mixbox LUT) or B3 (better whitewater
classifier) — both Tier 1, both safe in any order. Once all three are done,
v0.8 ships and we move into the v0.9 perf-squeeze window.

**Picking rules:**

1. If user has a concrete request → do that first.
2. Else: take the next unstarted micro from the current milestone (see table in BACKLOG.md §"Version milestones").
3. **Never** start a Tier 3 macro (v1.0+) without doing its `*.1` spike micro first — those exist exactly to abort the macro if reality disagrees with the plan.
4. Tier 4 macros (B10/B11/B12) are opportunistic — slot them in when no clearer priority exists, but don't pre-empt the current milestone.

When you finish a macro:

1. Add a one-liner to **§8.1 table above** (this file) with links to test/demo/DESIGN section.
2. Tick all its micros in BACKLOG.md, then move the macro entry under the `## Completed` heading there with the date and a summary paragraph.
3. If the macro was the last in its milestone, bump the milestone status in BACKLOG.md (`▶ next` → `✅ closed YYYY-MM-DD`) and mark the next milestone as `▶ next`.

## 9. Conventions when adding a new block

1. Add to DESIGN.md §4–11 (the layer table) — declare the ID.
2. Add to BLOCKS.md (status `plan` → `impl` → `test`).
3. Write the code with `# [BLK X.Y]` comment and `@block("X.Y", "...")` decorator.
4. Write a pytest in `tests/test_<layer><id>_<short>.py`:
   - At least one test asserts the feature **changes the answer**, not just "doesn't crash".
   - For solver features: ablation (with vs without) and check observable diff.
5. Add an example TOML scene if user-visible. Bake it.
6. Render `out/videos/step{N}.mp4` (matplotlib via `render_side_by_side.py` for ablation, `render_ply_sequence.py` for single scene).
7. If addon-relevant: extend `properties.py`, `panels.py`, `operators/bake.py:collect_scene`, `config_builder.py`. Rebuild zip.

## 10. Repro of last verified state

```bash
cd E:\projects\gpu_flip\gpufluid
source .venv/Scripts/activate
pytest -q                          # 70 passed
gpufluid info                      # 50 block IDs across 7 layers (G1/S2/F3/D4/M5/I6/C7); A8/W7 separate
gpufluid simulate examples/scenes/waterfall.toml   # waterfall + ball drop (anim obstacle)
gpufluid simulate examples/scenes/v06_kitchen_sink.toml   # all v0.6 features
```

## 11. Quick reference: key files when continuing

| Want to | Open |
|---------|------|
| Understand architecture | `docs/DESIGN.md` |
| Find a block by ID | `docs/BLOCKS.md` |
| Add a new pressure solver / scheme | `src/gpufluid/solvers/solver3d.py` |
| Add a new obstacle type | `src/gpufluid/domain/sdf.py` (+ wire in `cli/commands.py:_build_obstacle_sdf` + `cli/config.py:_parse_obstacle`) |
| Extend the addon UI | `addon/gpufluid_blender/{properties,panels,operators/bake,config_builder}.py` |
| Bake a scene | `gpufluid simulate <scene.toml>` |
| Render an mp4 from a cache | `python examples/render_ply_sequence.py --cache out/<dir> --out out/videos/<name>.mp4` |
| Render ablation | `python examples/render_side_by_side.py --left-cache A --right-cache B --out out/videos/<name>.mp4` |
| Manually test the addon in your Blender | Use the MCP server (`mcp__Blender__execute_blender_code`); the addon is installed at `C:\Users\timof\AppData\Roaming\Blender Foundation\Blender\5.1\extensions\user_default\gpufluid_blender\` |

## 12. Open TODOs at handoff

- Step 17 candidate: **Surface tension (S2.14)** — most visible classic FLIP feature still missing
- Step 18 candidate: **Per-particle color attribute** — enables Mixbox-style two-fluid demos
- Step 19 candidate: **BVH for GPU mesh SDF** — unblocks 10k+ tri obstacle meshes per frame
- All "FUTURE" items in `docs/BLOCKS.md` are valid targets.

---

**State at handoff (verified 2026-05-16, end of v0.7 sprint):** 109/109 tests green, **72 unique block IDs / 90 callables** via `gpufluid info`, 22 step videos in `out/videos/`, new colored-particle renderer at `examples/render_colored_particles.py`, addon zip 24 KB at `addon/gpufluid_blender.zip` (addon UI not yet exposing `surface_tension`/`csf_smoothing_passes`/`color` — CLI/TOML only). TOML now supports `[[fluids]]` array for multi-source scenes (each with optional `color = [r,g,b]`). Cache layout: when `output.particles=true` and any seed has a colour, a `colors/frame_NNNN.npy` sidecar (Nx3 float32) is written next to `particles/frame_NNNN.npy`. Moving-boundary BC works (numerically: 30 mm bow wave, 11 mm wake trough). USD cache pipeline tested in Blender. Reseed/decimate/checkpoint/GSRB/CSF all wired through CLI; CSF auto-engages `step_cfl` substepping (S2.14.5) and force balance (S2.14.6).

Drive-by fix included in this session: viscosity branch of step() was calling `k3_enforce_solid_bc` with 7 args instead of 9 (missing `solid_u/v/w`). Path only fires when `viscosity>0`, so it was latent. Fixed alongside CSF integration.

## 13. Repo hygiene notes (2026-05-16)

- **Not a git repo.** `git status` fails. If you want history/branching, run `git init && git add -A && git commit -m "snapshot at v0.6 handoff"` early in the session. All workflow rules in §9 (block IDs, tests-before-done, per-step videos) still apply without git, but rollback is manual until then.
- **`docs/BLOCKS.md` is stale.** It still marks several blocks as `plan` that are actually `impl` in source (S2.6.2 GS-RB, S2.11 reseed, S2.12 APIC, D4.2.4 plane, D4.5.2 mesh seeder, F3.5 checkpoint, M5.6 decimate, I6.5 USD, A8.* addon, W7.* whitewater). The implementations exist and are tested — BLOCKS.md just hasn't caught up. Refresh it before/when claiming a new feature is "added"; otherwise trust `gpufluid info` output and DESIGN.md for the source of truth.
- **Two D4.3 rows in BLOCKS.md** (one `plan`, one `impl,test`) — dedupe on next BLOCKS edit.
