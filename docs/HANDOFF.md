# gpufluid — Session Handoff

> Read this file FIRST in a new session. Everything important is here.
> Then read `docs/DESIGN.md` for the architecture contract and `docs/BLOCKS.md`
> for the block index.
>
> **End-of-session state (2026-05-16, post-v0.9-closure + v1.0 spike):**
> 171 tests total (170 green + 1 documented-flaky `test_gpu_mc_speedup_at_128`),
> 80 unique IDs / 103 callables (`gpufluid info`), v0.7 + v0.8 + **v0.9 all
> closed**. CUDA-graph eligibility matrix is at **9/9** (every shipped
> pressure-solver path captures cleanly: jacobi/gs-rb/PCG × dense/sparse,
> with or without CSF). **v1.0 unblocked**: B7-alt.1 spike GREEN
> (18.96× memory drop on connected-blob 128³/5%, bit-exact coord-translation
> on Jacobi); macro micros B7-alt.2…B7-alt.8 spelled out in BACKLOG. The
> next session inherits a known-good repo with a clear macro queue.
>
> **What this session added (5 commits, all on `origin/main`):**
> ```
> adfb06c  B7-alt.1 spike GREEN — deferred-dense feasible, v1.0 macro greenlit
> 797a952  v0.9 polish — B3.5 step22 Eevee refresh + B11.3 reseed-temp + step26 feature video
> ce5c7d1  B5 Option B — sparse pressure graph-eligibility via n_active_dev (8/9 → 9/9)
> 91e35b4  B5 Option A — PCG dense graph-eligibility via on-device stop flag (6/9 → 8/9; 4.17× on big_pcg)
> de18bfe  B11.3 lava demo + v0.9 closure
> ```
>
> **Picking the next task:** §12 of this file lists open items in
> priority order. Default is **B7-alt.2** (start the v1.0 macro by
> refactoring `FlipSolver3D` field storage for sub-dense backing).
> See `BACKLOG.md` B7-alt.2…B7-alt.8 for the full micro chain. Don't
> tear out the dense path — B7-alt is a parallel implementation behind
> a flag, dense stays as the default until 256³ scenes prove out.

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
| 22 | **Whitewater quality (W7.4/W7.5) — refreshed 2026-05-16 (B3.5).** 80³ waterfall splash with three differentiated classes — spray (cyan dots, low drag, full gravity), foam (white, near-surface), bubble (blue, submerged, pops on reaching surface). Classification by `pos+vel·0.02` density-grid lookup at emit; particles evolve through classes (spray→foam→bubble→pop) at every step. **B3.5 refresh**: scene now uses the W7.7 trapped-air potential selector (`whitewater_use_potential = true`) instead of the legacy `\|v\|>thr`. Eevee Next renderer `examples/render_step22_eevee.py` — mesh + three vertex-instanced point buckets, one per class with its own emissive material (foam white, spray cyan, bubble blue). 44 s wall for 90 frames vs ~3 min matplotlib. The mp4 in `out/videos/step22.mp4` is the Eevee version; the original matplotlib renderer `render_whitewater.py` is kept for ablation/legacy. | ✅ Blender Eevee Next, 90 fr |
| 23 | **Whitewater selector A/B (B3.1/B3.3 demo)**: side-by-side 80³ waterfall + basin splash, identical scene baked twice — left pane uses legacy `\|v\|>threshold` selector, right pane uses the new W7.7 trapped-air potential. Live per-class counters overlay each frame. Peak-splash (frame 30) snapshot: legacy emits 829 foam + 29 spray + 2116 bubble = 2974 total; potential emits 379 foam + 27 spray + 969 bubble = 1375 total. Mesh is identical between panes (same solver state), so the visible whitewater difference is entirely the selector. Demonstrates the v0.8 thesis: trapped-air potential redirects the emit budget toward genuinely turbulent regions instead of treating all fast-moving fluid as ww-eligible. Renderer `examples/render_step23.py` builds the side-by-side directly from the two cache dirs. | ✅ matplotlib mesh+scatter SBS, 60 fr |
| 23b | **Same scene, Eevee renderer**: parallel mp4 produced via `examples/render_step23_eevee.py` (Blender 5.1, headless Eevee). 60 frames bake **27.3 s wall / 456 ms/frame**, ~9× faster than the matplotlib pipeline used for the canonical step23 (4 min). Establishes the Eevee path for future demos (step24+) as the default — matplotlib stays only when per-frame per-vertex colour overlays are required. **Polish gap:** Eevee version currently renders all whitewater as a single emissive sphere; the per-class colour and the live count overlays from step23.mp4 are lost. Closing micro: drive material colour from the `gpufluid_kind` INT attribute via a Geometry Nodes graph + 2D compositor text. | ✅ Blender Eevee, 60 fr |
| 24 | **Kitchen-sink v0.8 demo** — every major v0.8 feature in one bake: `[[fluids]]` multi-source colour (red + blue), surface tension σ=0.1, APIC transfer mode (B6: known safe with σ), sphere obstacle, W7.7 trapped-air potential whitewater, Eevee-rendered single pane. Sim 3.6s for 90 frames @ 64³ + 32 CFL substeps/frame (σ-bound). Renderer `examples/render_step24_eevee.py`: builds a fresh scene in Blender, attaches a custom per-frame loader that (a) reads the surface mesh PLY, (b) per mesh vertex finds the nearest fluid particle and copies its RGB into a FLOAT_COLOR `fluidcol` attribute, (c) updates a vertex-only whitewater mesh that instances a small emissive sphere on every point. Result: the colour-mixing story is visible directly on the smooth surface — red drop + blue drop fall, get rounded by σ, hit the cream obstacle, splash and mix into purple in the basin. Render wall: ~150 s for 90 frames @ 1600×900 (most of it is the per-vertex cdist nearest-particle pass; render alone is ~50 ms/frame). | ✅ Blender Eevee, 90 fr |
| 25 | **Lava demo (B11.3)** — closed the last v0.9 micro. Hot lava drop (`temperature = 1500.0`) splashes from above into a cool basin (`temperature = 300.0`); the *visible colour* is driven entirely by the per-particle temperature scalar S2.18, not by per-source RGB. Wiring: `[[fluids]] temperature = X` lands in `FluidBoxCfg.temperature` (cli/config.py), threads through `cmd_simulate._seed_one` into `solver.seed_box(temperature=...)` / `solver.seed_mesh(temperature=...)` (mesh seeder was extended in this session for parity). Per-frame sidecar `<cache>/temperatures/frame_NNNN.npy` written next to `colors/`. Renderer `examples/render_step25_eevee.py` clones step24's nearest-particle cdist transfer but maps T → RGB through a 5-anchor blackbody-ish colormap and ALSO drives a per-vertex emission-strength attribute (`lavaemit`), so the splash contact zone visibly cools as the P2G→G2P pass mixes hot bulk with cold basin. Eevee bloom on for hot-peak glow. Scene `examples/scenes/lava_drop.toml` (64³, σ=0.1, APIC, sphere obstacle, viscosity=0.02, 90 fr). Regression `tests/test_b11_3_temperature_toml.py` — 2 pure-config tests + 1 GPU end-to-end. Bake 0.38 s sim + 2.16 s mesh; render ~5 min (most of it is the per-vertex cdist on 218 k particles). | ✅ Blender Eevee, 90 fr |
| 26 | **Options A + B feature showcase (perf-fishka via text overlay)** — first demo in the project for a *visually-invisible* feature. Bakes `big_pcg.toml` (96³, PCG 60-iter, 90 frames) with `--enable-cuda-graphs`, renders a single-pane Eevee Next scene with four camera-parented emissive text lines that carry the entire story: feature name (white), headline metric (yellow: `sim 4.29s → 1.05s (4.09×, 88% hit rate)`), dependent feature (grey), closure status (cyan: `eligibility matrix 9/9`). Renderer `examples/render_step26_eevee.py`; render wall ~28 s for 90 frames @ 1600×900. Establishes the project's "text-overlay" demo convention for perf / API / eligibility wins that don't change pixels — distinct from the "visible-change" convention used by steps 17, 18, 22, 24, 25. See memory `demo_video_overlay_pattern.md` for when each convention applies. | ✅ Blender Eevee Next, 90 fr |

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

### 8.0b v0.9 progress (2026-05-16) — Tier 2 macros closed, v1.0 pivoted

Status of v0.9 milestone macros (the "Production-fast" window) and the
v1.0 gating spike after this session:

| # | Macro                                                  | Status |
|---|--------------------------------------------------------|--------|
| B4 | Block-sparse iteration for GS-RB + PCG                 | ✅ closed |
| B5 | CUDA-graphs capture for the per-step kernel sequence   | ✅ closed (6 of 9 configs) |
| B6 | APIC + CSF interaction QA                              | ✅ closed (no fix needed) |
| B7 | Sparse v2 — NanoVDB-backed storage                     | ❌ **ABORTED** by spike — pivot to B7-alt |
| B11 | Per-particle scalar attributes (temperature)          | ✅ closed (B11.3 lava demo shipped 2026-05-16) |
| B12 | Per-frame timing instrumentation (`--timings`)        | ✅ closed |

Key per-macro detail (everything else is in BACKLOG):

**§8.0b.B4 — block-sparse stack complete.** S2.6.5 per-tile GS-RB and
S2.6.6 per-tile PCG kernels added (plus shared `_build_active_blocks`
helper). Bench at 128³ / ~10% fill:
* Jacobi sparse (S2.16, v0.7) — 2.4× kernel-only
* GS-RB sparse (S2.6.5)        — **2.1× kernel-only** (80 iters)
* PCG sparse (S2.6.6)          — **1.1× full-step** (30 iters)
PCG's modest speedup is the macro-level honest finding: its 8 device
ops/iter vs GS-RB's 2 mean per-tile dispatch overhead + one extra
device→host sync (n_active) eats into the per-iter saving. Documented
follow-up: cache `n_active` on-device.

**§8.0b.B5 — CUDA graphs ship for 9 of 9 configurations (post Options A + B).**
Eligibility matrix:
```
                       no-CSF    CSF (σ>0)
jacobi/gsrb dense       ✅          ✅       ← 4 (B5)
PCG dense               ✅          ✅       ← +2 (Option A)
sparse (block_sparse=1) ✅          ✅       ← +3 (Option B, this session) — 9/9 closed
```
* B5.1 spike (Jacobi, no CSF): direct 0.43 ms → graph 0.20 ms = **2.17×**
* Real bake `two_color_drop` (90 frames, jacobi, 8 substeps): sim 0.72s
  → 0.15s = **4.8×**, 88% hit rate
* Real bake `surftens_on` (60 frames, σ=1, 38 substeps): sim 3.35s
  → 0.19s = **17.6×**, 97% hit rate ← CSF made eligible by moving
  S2.14.6 force-balance fully device-side
* **Option A real bake `big_pcg` (96³, PCG 60-iter, 90 frames):** sim
  **4.21s → 1.01s = 4.17×**, 88% hit rate. Pre-A this scene was the
  documented regression (4.8s → 9.3s); the on-device stop-flag pattern
  flips it into a 4.17× win. Same scene, same bench harness.
* **Option B real bake `big_pcg` (sparse PCG, 96³, 60-iter, 90 frames):**
  sparse-graph-off 4.19s → sparse-graph-on **1.00s = 4.19×**, 88% hit
  rate. Identical to dense-graph at this fill (~10%). The eligibility
  flip is the win — actual perf comes from the same kernel-launch
  reduction. At higher 256³+ resolutions with low-fill scenes, sparse
  starts to beat dense thanks to bandwidth — Option B unlocks the
  combined sparse+graph speedup that v1.0 needs.

CLI flag: `gpufluid simulate --enable-cuda-graphs` (off by default).
All shipped pressure-solver combinations are now graph-eligible.

**§8.0b.B6 — APIC + CSF stable, no fix needed.** `tests/test_b6_apic_csf_interaction.py`:
COM drift < 2%, max|v| < 35 m/s on the surftens_on parameter set with
`transfer_mode="apic"`. Existing knobs from `surftens_on.toml`
(48³+, 3 smoothing passes, mild viscosity, CFL substep cap 64) work
unchanged.

**§8.0b.B7 — sparse v2 ABORTED, pivot documented.** Spike found two
fatal problems with the BACKLOG plan:
1. **`wp.Volume` is read-only from kernels in Warp 1.13.** No
   `volume_store_*` API. The plan to atomic_add into a sparse Volume
   is physically impossible.
2. Even read-only, `volume_lookup_i` is 1.3-2.3× slower than dense
   indexing — borderline at the BACKLOG abort threshold.

`tests/test_b7_1_volume_is_read_only` is the trip-wire: when Warp ships
the missing store API, that test starts failing and forces re-eval.

**§8.0b.B7-alt.1 — deferred-dense spike GREEN (2026-05-16, same session).**
Pivot from B7 (NanoVDB → deferred-dense `wp.array3d` over bbox of
active 8³ tiles). The B7-alt.1 spike (`tests/test_b7_alt_1_spike_deferred_dense.py`)
answers both killers:

* **Memory bbox ratio @ 128³ / ~5% fill:**
  * connected-blob topology: bbox=(48,48,48), **18.96× memory drop**
    (way above the 5× greenlight bar).
  * scattered-droplets (200 droplets): bbox=(128,128,128), **1.00×
    drop** — bbox covers the whole domain. Pathological case; macro
    must be documented as "not a silver bullet for dispersed scenes."
* **Coord-translation correctness:** a Jacobi-like kernel re-launched
  on a sub-dense `wp.array3d` of size bbox+dilation=1 with an
  `offset_xyz` parameter produces **bit-exact** results vs full-dense
  on every active fluid cell after 40 iterations (rel err = 0.000e+00).

Verdict: GREEN on realistic blob-shaped scenes. Macro micros
B7-alt.2 … B7-alt.8 are spelled out in BACKLOG (refactor field
storage; thread `offset_xyz` through ~20 dense kernels; on-device
resize; 256³ dam-break acceptance bench; user-facing extent-ratio
warning for scattered scenes).

**§8.0b.B11 — scalar attribute pipeline (temperature), closed.** S2.18.1/2/3
kernels parallel the S2.15 colour pipeline but for one float channel.
`attr_temperature` slot on FlipSolver3D; `seed_box(temperature=...)`
kwarg with append-in-lockstep semantics. Drive-by: fixed `seed_box`
to concatenate positions whenever ANY attribute is in play (was
colour-only). **B11.3 (this session)** wired `[[fluids]] temperature = X`
through cli/config.py + cmd_simulate, extended `seed_mesh` for parity,
added per-frame `<cache>/temperatures/frame_NNNN.npy` sidecar (mirrors
`colors/`), shipped scene `examples/scenes/lava_drop.toml`, renderer
`examples/render_step25_eevee.py` (5-anchor blackbody-ish colormap T→RGB
+ per-vertex emission-strength attribute, Eevee bloom), regression
`tests/test_b11_3_temperature_toml.py`. v0.9 milestone now fully closed.

**§8.0b.B12 — per-section profiler shipped.** `StepProfiler` wraps
`wp.ScopedTimer(synchronize=True)`; opt-in via
`FlipSolver3D(enable_timing=True)`. CLI flag `--timings` prints
per-section totals at end and dumps `<cache>/timings.json`. Sample on
two_color_drop (40³, 90 frames): pressure 49.9%, color 11%, p2g 7.2%,
g2p_advect 4.7%, rest <5% each.

### 8.0c v0.9 fully closed (2026-05-16, B11.3 lava demo)

| # | Macro                                                  | Where to look |
|---|--------------------------------------------------------|---------------|
| B11.3 | Lava demo — per-particle temperature drives colour    | step25 row in §6, `test_b11_3_temperature_toml.py`, `examples/scenes/lava_drop.toml`, `examples/render_step25_eevee.py` |

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

### 8.2 Live queue (`docs/BACKLOG.md`) — current milestone: **v0.9 essentially closed, v1.0 pivot pending**

The next-task queue is in `docs/BACKLOG.md`, with release milestones at the
top. After this session:

* **v0.8** essentially closed (B1 ✅, B2 skipped license, B3 partial).
* **v0.9** fully closed: B4 ✅, B5 ✅ (6 of 9 configs), B6 ✅, B11 ✅
  (incl. B11.3 lava demo this session), B12 ✅.
* **v1.0** pivoted: original B7 (NanoVDB) aborted; replacement "B7-alt"
  (deferred dense allocation, ~70-80% of the win) needs its own spike.

**Default next task** (no user direction):
* If you want to push into v1.0 → **B7-alt spike**: prototype deferred-dense
  allocation. Allocate `wp.array3d` lazily for the bounding box of active
  8³ tiles, rebuild every N frames with a dilation margin. Risk: needs a
  coord-translation layer in every kernel (offset_x/y/z). Run the spike at
  128³ with 5% fill before committing to the macro.
* If you want to close out the B5 follow-ups → port PCG and block-sparse to
  the graph-eligible path. PCG needs an on-device convergence-check
  (stop-flag idiom). Block-sparse needs `n_active_dev` device buffer + cap
  in 6 per-tile kernels. Both unblock the remaining 3 of 9 configs.
* If you want a v0.8 polish closeout → **B3.5** refresh step22.mp4 with the
  W7.7 selector turned ON (CLI flag already exists since B3.3).

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
pytest -q                          # 170 green + 1 deselected flaky (test_gpu_mc_speedup_at_128 — see §12 hygiene)
gpufluid info                      # 80 unique block IDs / 103 callables across G1/S2/F3/D4/M5/I6/C7/A8/W7
gpufluid simulate examples/scenes/two_color_drop.toml --enable-cuda-graphs --timings
                                   # canonical demo of B5 + B12 wins (jacobi dense + graphs)
gpufluid simulate examples/scenes/surftens_on.toml --enable-cuda-graphs
                                   # CSF + graphs (17.6× sim speedup)
gpufluid simulate examples/scenes/big_pcg.toml --enable-cuda-graphs
                                   # PCG dense + graphs (Option A — 4.17× sim speedup on this scene)
gpufluid simulate examples/scenes/lava_drop.toml --enable-cuda-graphs
                                   # B11.3 lava demo — temperatures sidecar populated
pytest -q tests/test_b7_alt_1_spike_deferred_dense.py -s
                                   # B7-alt.1 spike verdict — prints bbox ratio + correctness on stdout
```

**Demo videos in `out/videos/` (all tracked in git):**
* `step23.mp4`, `step23_eevee.mp4` — whitewater selector A/B (legacy vs W7.7).
* `step24.mp4` — kitchen-sink v0.8 demo (multi-source colour + σ + APIC + W7.7).
* `step25.mp4` — **B11.3 lava demo**: per-particle temperature → blackbody-ish
  colormap on surface + per-vertex emission. Eevee bloom.
* `step22.mp4` — refreshed 2026-05-16: now Eevee Next with three vertex-instance
  WW buckets (foam/spray/bubble), driven by W7.7 trapped-air potential.
* `step26.mp4` — **Options A + B feature showcase** via text-overlay-on-Eevee
  (the project's first "visually-invisible feature" demo convention). Carries
  the 4.09× + 9/9 closure story in four camera-parented emissive text lines.

## 11. Quick reference: key files when continuing

| Want to | Open |
|---------|------|
| Understand architecture | `docs/DESIGN.md` |
| Find a block by ID | `docs/BLOCKS.md` |
| Add a new pressure solver / scheme | `src/gpufluid/solvers/solver3d.py` |
| Add a new obstacle type | `src/gpufluid/domain/sdf.py` (+ wire in `cli/commands.py:_build_obstacle_sdf` + `cli/config.py:_parse_obstacle`) |
| Extend the addon UI | `addon/gpufluid_blender/{properties,panels,operators/bake,config_builder}.py` |
| Bake a scene | `gpufluid simulate <scene.toml>` |
| Render a "visible feature" demo (Eevee Next default) | Clone `examples/render_step24_eevee.py` or `_step25_eevee.py`; the per-vertex cdist colour-transfer pattern works for any per-particle attribute. Stitch via `imageio.get_writer(fps=24, codec='libx264')`. |
| Render a "perf / API / matrix-closure" demo (text overlay) | Clone `examples/render_step26_eevee.py`. Camera-parented emissive text in 4 colour-coded lines (white name, **yellow metric**, grey dependent, cyan closure). See memory `demo_video_overlay_pattern.md` for the decision rule. |
| Legacy matplotlib renderers (kept for ablation only) | `render_ply_sequence.py`, `render_side_by_side.py`, `render_whitewater.py`. ~3-10× slower than Eevee Next; only use when per-frame per-vertex overlays Eevee can't express. |
| Manually test the addon in your Blender | Use the MCP server (`mcp__Blender__execute_blender_code`); the addon is installed at `C:\Users\timof\AppData\Roaming\Blender Foundation\Blender\5.1\extensions\user_default\gpufluid_blender\` |

## 12. Open TODOs at handoff (2026-05-16 session end — v1.0 macro phase)

Sized roughly by session-count and grouped by where they unblock.

**Default next task** (no user direction): **B7-alt.2** — refactor
`FlipSolver3D` field storage to support a sub-dense `wp.array3d`
backing + `self._sub_offset = (ox, oy, oz)`. See `BACKLOG.md` B7-alt.2
… B7-alt.8 for the full v1.0 micro chain. The spike (B7-alt.1) is
GREEN and committed (`tests/test_b7_alt_1_spike_deferred_dense.py`);
the macro is greenlit.

**v1.0 macro queue (B7-alt.2 … B7-alt.8 from BACKLOG, in order):**

1. **B7-alt.2 — sub-dense field storage + rebuild trigger.** New
   `_sub_offset` attribute on `FlipSolver3D`; `prepare_frame` rebuilds
   the sub-dense bbox every `sub_rebuild_every` frames or when active
   tiles get within `dilation` cells of the edge. Per-field re-alloc
   copies old → new at the offset delta.
2. **B7-alt.3 — thread `offset_xyz` through dense per-step kernels.**
   ~20 kernels, each gets a 4-line addition: `gi = li + off_x` for
   marker/div cross-lookups; neighbour pressure reads stay in
   sub-dense coordinates. The spike kernel `k3_jacobi_spike_offset`
   in `tests/test_b7_alt_1_spike_deferred_dense.py` is the prototype.
3. **B7-alt.4 — particle ↔ sub-dense mapping.** Add `pos_to_sub(p)`
   helper. Out-of-bbox particles should already be caught by D4.7
   outflow at the wall margin, but bounds-check explicitly so a
   misconfigured rebuild doesn't silently drop fluid.
4. **B7-alt.5 — CUDA-graph interaction.** `_cuda_graph_invalidate()`
   already fires on `prepare_frame` (B5.3). Verify hit rate ≥ 80%
   across rebuilds via a flowing-fluid bench scene.
5. **B7-alt.6 — 256³ dam-break acceptance bench.** The v1.0 headline:
   the dense path OOMs (~512 MB/snapshot on 16 GB GPU); B7-alt should
   fit comfortably at 10-20% fill (4-25 MB/field).
6. **B7-alt.7 — user-facing scattered-droplet warning.** Compute
   spatial extent ratio at first rebuild; if > 0.8, print one-shot
   stderr warning that B7-alt isn't helping (the spike's
   scattered-droplets test showed 1.00× drop on that topology).
7. **B7-alt.8 — on-device resize kernel.** Pure CPU resize would
   dominate per-frame cost; copy old → new at `offset_delta` on GPU.

Estimated total: 3-5 sessions to ship B7-alt.2 … B7-alt.8 + 256³ bench.

**Tier 4 — opportunistic / hygiene:**
- **Flaky `tests/test_m5_4_gpu_mc.py::test_gpu_mc_speedup_at_128`** —
  pre-existing perf-bench sensitive to GPU thermal/concurrent load.
  Passes in isolation; occasionally fails in the full suite under
  thermal pressure. Fix: relax threshold from 3× to 2.5× or warm GPU
  more aggressively. Run-time invariant: deselect it via
  `--deselect tests/test_m5_4_gpu_mc.py::test_gpu_mc_speedup_at_128`
  in CI.
- **B3.2 (v0.8) — Mixbox pigment LUT** — still skipped on the
  CC BY-NC license concern. Re-evaluate if commercial licence cost
  becomes acceptable, or write our own 2-pigment K-M solver (~50
  lines, less painterly). No urgency.
- **Addon UI: per-source `temperature` field** — B11.3 wired TOML +
  CLI but not the Blender N-panel. Mirror the existing per-source
  `color` field in `addon/gpufluid_blender/properties.py` +
  `config_builder.py`. Trivial.

**Tier 1 (closed this session — for context, do NOT re-open):**
- ~~PCG graph-eligibility~~ — **Option A shipped** (`91e35b4`, 4.17× on big_pcg).
- ~~Block-sparse + CUDA graphs~~ — **Option B shipped** (`ce5c7d1`, matrix 9/9).
- ~~B3.5 step22 refresh~~ — **shipped** as Eevee Next (`797a952`).
- ~~B11.3 reseed propagates `attr_temperature`~~ — **shipped** (`797a952`,
  new `k_scatter_alive_scalar` + 6-tuple return shape).
- ~~B7-alt.1 spike~~ — **GREEN** (`adfb06c`), macro greenlit.

**Hygiene (carried forward):**
- Flaky `tests/test_m5_4_gpu_mc.py::test_gpu_mc_speedup_at_128` —
  pre-existing perf bench sensitive to GPU thermal/concurrent load.
  Passes in isolation, occasionally fails in the full suite. Fix: relax
  threshold from 3× to 2.5× or warm GPU more aggressively.

---

**State at handoff (2026-05-16, end of session — v0.9 closed, v1.0 macro greenlit):**
* **Tests:** 171 total, 170 green + 1 documented-flaky
  (`test_gpu_mc_speedup_at_128`). Verified via `pytest -q
  --deselect tests/test_m5_4_gpu_mc.py::test_gpu_mc_speedup_at_128`
  on the user's Windows box (Warp 1.13 + CUDA 12.9 + RTX 4080 SUPER).
* **Registry:** 80 unique block IDs / **103** callables via
  `gpufluid info` (+1 vs prior session: the new `k3_check_converged`
  helper at S2.6.3 for Option A).
* **Demo videos:** 26 mp4 in `out/videos/` (steps 1-26). New this
  session: step25 (B11.3 lava), step22 refreshed (Eevee Next),
  step26 (Options A+B perf-overlay).
* **Addon zip:** 37 KB at `addon/gpufluid_blender.zip` (v0.8.0). UI
  still does NOT expose per-source `temperature` — opportunistic
  follow-up (§12 hygiene), trivial mirror of existing `color` field.
* **CLI features:** `gpufluid simulate <scene.toml>` accepts
  `--enable-cuda-graphs` (B5 — now opens **9/9** configs after
  Options A + B), `--timings` (B12), `--resume`, `--start-frame`,
  `--checkpoint-every`. TOML accepts `[[fluids]] temperature = X`
  (B11.3) + `pressure_block_sparse` (B4 + Option B).
* **Repo:** `https://github.com/ArkBee/GPUFLUID` (branch `main`).
  All work this session pushed; `origin/main` ends at `adfb06c`
  (B7-alt.1 spike GREEN). The 5-commit stack from this session
  (newest first): `adfb06c`, `797a952`, `ce5c7d1`, `91e35b4`,
  `de18bfe`.

## 13. Repo hygiene notes (2026-05-16, end of session)

- **`origin/main` at `adfb06c`** (B7-alt.1 spike GREEN). All work
  pushed via one-shot PAT this session; new sessions need fresh auth
  if pushing more commits.
- **`docs/BLOCKS.md` is in sync as of session end** — S2.6.5, S2.6.6,
  S2.14.6_dev, S2.18.1/2/3, G1.9 all have rows; S2.6.3 and S2.16
  descriptions bumped to mention the Options A / B helpers
  (`_pcg_done`, `_n_active_dev` + `k_store_n_active`). `gpufluid info`
  remains the source of truth (80 unique IDs / 103 callables after
  this session — +1 callable from `k3_check_converged`).
- **CSF S2.14.6 has TWO kernel variants now:**
  `k3_csf_subtract_bias_{u,v,w}` (legacy, takes host-computed `bias`)
  vs `_dev` siblings (read sum/count from device). `_apply_surface_tension`
  uses the `_dev` ones. Legacy kernels are still registered + tested
  via `test_s2_14_*`; nothing calls them in the live solver. Safe to
  delete in a future cleanup, but they're zero-cost dead weight.
- **PCG on-device convergence-check ACTIVE** (this session, Option A).
  `_pcg_done` device int + `k3_check_converged` set the flag mid-loop;
  all 7 PCG iter kernels gate on `done[0] != 0` for first-line return.
  `_cuda_graph_eligible(...)` returns True for `pressure_solver="pcg"`
  (dense and sparse). The previous "dormant `_no_host_sync` plumbing"
  note is obsolete — that path now drives the eligible-PCG branch.
- **Sparse paths read `_n_active_dev`** (Option B). Six per-tile
  kernels take an `n_active_dev: wp.array(dtype=int)` parameter and
  early-return when `blk >= n_active_dev[0]`. `_build_active_blocks`
  writes the device-side mirror via `k_store_n_active` (S2.16); the
  host `.numpy()[0]` read is now optional (`_no_host_sync=True` skips
  it, returning n_active=-1 — graph-capture path uses this).
- **`render_step26_eevee.py` is the reference impl** for "perf-fishka"
  demos — visually-invisible features get a text-overlay-on-Eevee
  rendering instead of fluid-rendering. Convention details in the
  memory file `demo_video_overlay_pattern.md` (linked from the
  session's memory index). When adding a new perf demo, clone
  `render_step26_eevee.py`, swap the four overlay lines, change the
  cache path; keep camera-parented text + emissive material recipe
  unchanged.
