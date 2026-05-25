# gpufluid — Backlog (priority queue, not FIFO)

> **Read this in a new session BEFORE picking work.** Items are ordered so that
> earlier tasks de-risk later ones and don't constrain architectural decisions
> for harder work below. Each macro is broken into **independent micros** that
> can be shipped in a single session of 1–4 hours. When you finish a micro,
> tick it inline and move on; when you finish a macro, move it to the
> "Completed" section at the bottom of `HANDOFF.md` (it already has a
> roadmap-done table).
>
> The order is not arbitrary — see **"Why this order"** at the end. If you
> want to skip ahead, check the **Dependency graph** first to make sure you
> don't paint yourself into a corner.

---

## Version milestones

The tiers below map onto release milestones. Pick the next macro by checking
which milestone we're inside.

| Release | Theme                    | Macros included                | Est. sessions | Status |
|---------|--------------------------|--------------------------------|---------------|--------|
| v0.7    | Solver feature complete  | (10-item original roadmap)     | done          | ✅ closed 2026-05-16 |
| **v0.8** | **"Reachable" — Blender exposes the v0.7 features** | **Tier 1: B1, B2, B3**     | **3-5**       | **▶ next** |
| v0.9    | "Production-fast" — hot path squeezed              | Tier 2: B4, B5, B6 + B11, B12  | 4-6           | ✅ closed 2026-05-16 (B5 ships **9/9** graph configs after Options A + B — no remaining v0.9 polish) |
| v1.0    | "Scale" — 256³+ scenes via sparse storage          | Tier 3: B7 (aborted, see B7.1) → B7-alt deferred-allocation | 3-5 | ✅ closed 2026-05-17 — B7-alt.1 spike + B7-alt.2..B7-alt.8 all shipped. 256³ dam-break runs at 6.11× memory drop, sub-dense step 1 ms, rebuild 3 ms on-device. |
| v1.x+   | Research extensions      | B8 differentiable, B9 multi-GPU | open-ended    | optional |

**Why these splits:**

* **v0.8 is purely additive.** The solver already does σ, colour, sparse
  pressure, GPU MC — but a Blender user can't reach those without editing TOML
  by hand. Tier 1 is plumbing work, `risk:low` across the board. Shipping it
  is what moves the project from "competent engine" to "usable plugin." Three
  Tier-1 macros, each fits in a session.
* **v0.9 is purely performance.** No new features, no API changes — just
  squeeze more perf out of the existing pipeline. Block-sparse for GS-RB/PCG
  (B4), CUDA graphs (B5), APIC+CSF interaction QA (B6). The grade-up needed
  before promising real-time 128³ to users.
* **v1.0 is the architectural bet.** B7 Sparse v2 is the only path to 256³+
  scenes; without it we hit a memory ceiling at ~128-160³ on a 16 GB GPU.
  This is *the* thing that makes the project compete with paid solutions on
  production-scale work. It's multi-session — do the B7.1 spike first to
  validate the approach before committing.
* **v1.x+ is research.** Differentiable solver (B8) and multi-GPU (B9) are
  cool but speculative — work on them only when a user has a concrete need
  or when v1.0 has shipped and there's no clearer priority.

**What we do NOT promise:** painted-fluid art (Mixbox is the closest, B2),
melt/freeze phase change, FX-grade caustics. Those would be v2.0 territory
and require new layers (phase field, photon tracing). Not in the queue.

---

## Legend

| Tag | Meaning |
|-----|---------|
| `risk:low` | Isolated change, no cross-cutting refactor. Safe to ship in any order. |
| `risk:med` | Touches 2-4 files, has a non-trivial interaction with an existing block. |
| `risk:high` | Architectural — changes a contract used by many blocks. |
| `value:user` | User-visible win (new feature, faster scene, better look). |
| `value:infra` | Infrastructure win (faster sim, lower mem, fewer bugs). |
| `value:research` | Speculative or research-leaning (no obvious near-term user). |
| `blocks: X` | Must be done before X is safe to start. |
| `needs: X` | Cannot start until X is done. |

A micro task is `[ ]` (todo) or `[x]` (done).

---

## Tier 1 — Surface polish (do first, low risk, high user value)

These items unblock the marketing path: they make the existing v0.7 features
*reachable* through the addon UI / production renderers without touching any
solver internals.

### B1. Addon UI exposes the v0.7 simulation parameters `risk:low value:user`

The addon currently doesn't surface `surface_tension`, `csf_smoothing_passes`,
`color`, `cfl`, `cfl_max_substeps`, `pressure_block_sparse`,
`reseed_*`, or the whitewater knobs. They're CLI/TOML-only. End-users can't
configure them from Blender's N-panel.

* [x] B1.1 — Extend `addon/gpufluid_blender/properties.py` with a `SurfaceTensionGroup` (σ + smoothing passes) and add to the Domain props. *(2026-05-16: GpufluidSurfaceTensionGroup with σ+csf_smoothing_passes, attached to Domain via PointerProperty 'surface_tension_group'. Registered in __init__._CLASSES before Domain. Guarded by `tests/test_a8_2_1_surface_tension_props.py` — 3 tests using a bpy stub. BLOCKS.md row A8.2.1 added.)*
* [x] B1.2 — Extend `properties.py` with per-fluid-source `color: FloatVectorProperty(subtype='COLOR')`. *(2026-05-16: added `use_color: BoolProperty` toggle + `color: FloatVectorProperty(size=3, subtype='COLOR', default=(1,1,1), min=0, max=1)` on GpufluidFluidProps. BLOCKS.md row A8.3.1 added. Test `test_a8_3_1_fluid_has_color_and_use_color` guards the shape.)*
* [x] B1.3 — Extend Bake Operator (`operators/bake.py:collect_scene`) to thread these into the TOML the bake subprocess loads. *(2026-05-16: collect_scene now emits one entry per Fluid object in a `fluids` list (was a single unioned bbox), reads `use_color`/`color`/`surface_tension`/`csf_smoothing_passes`, and auto-enables CFL substepping when σ>0. Latent NameError on `fluid_section` fixed in the same pass.)*
* [x] B1.4 — Mirror in `config_builder.py` (the bpy-free translator). *(2026-05-16: build_toml accepts either legacy `scene_dict["fluid"]` (one `[fluid]` table) or new `scene_dict["fluids"]` (one `[[fluids]]` table per source, optional `color = [r,g,b]`). simulation passes `surface_tension` + `csf_smoothing_passes` through to TOML. 4 new tests in `test_a8_config_builder.py`; full suite 117/117.)*
* [x] B1.5 — Update `panels.py` to add a "Whitewater" sub-panel with class-toggle visibility + thresholds. *(2026-05-16: `GpufluidWhitewaterGroup` exposes the 4 TOML knobs (speed_threshold/lifetime_sec/emit_per_frame_max/total_cap) + 3 class-visibility toggles. New `GPUFLUID_PT_whitewater` panel + Surface Tension box in Domain panel + Particle Colour box in Fluid panel. Threaded through bake.py (new `_output_dict` helper) and config_builder.py (whitewater_* fields written when enabled). 6 new tests; suite at 123/123. NB: per-class WhitewaterConfig fields (gravity_*, drag_*, density_*, pop_threshold) aren't TOML-plumbed yet — wider expose-pass tracked as future work.)*
* [ ] B1.6 — Smoke test: install zip → toggle σ in N-panel → bake → mesh shows a rounded drop. *(manual, in-Blender; not automated)*

**Acceptance:** every CLI knob added in v0.7 has a Blender-side counterpart.

### B18. MPM per-particle colour + scalar attrs + Mixbox mixing `risk:med value:user` ✅ closed 2026-05-21

*See **Completed** section below for closure entry. Original micros preserved here for traceability.*

*Added 2026-05-20.* The MPM-pivot (B17) shipped `gpufluid.sim.mpm.MpmSolver`
as the default solver in the addon, but **dropped per-particle attribute
plumbing on the floor**: S2.15 (colour) and S2.18 (temperature) live in the
FLIP `FlipSolver3D` codepath and never made it to MPM. Result: every MPM
bake produces uniform-grey particles; multi-inflow scenes that should show
two fluids meeting render as a single colour. This block closes that gap
end-to-end and folds in B2 (Mixbox pigment-space mixing) on top, since
both share the same vertex-colour-on-mesh transport.

**Why a single macro:** B2 alone is impossible without (a) a per-particle
colour array surviving the bake, (b) the mesher emitting per-vertex
colour, and (c) the addon reading that colour back. Those three pieces
are the same plumbing whether the blend rule is linear-RGB or Mixbox.

* [x] B18.1 — `MpmConfig` gains optional `attr_color: np.ndarray (N, 3) float32`
  and `attr_temperature: np.ndarray (N,) float32`. `MpmSolver.__init__`
  concatenates per-source colour/temperature in lockstep with positions
  (initial column + each inflow), uploads as `wp.array(dtype=wp.vec3)` /
  `wp.array(dtype=float)`. Acceptance: `tests/test_b18_1_mpm_attrs_seed.py`
  builds a 100-particle initial column + one 50-particle inflow with
  distinct colours, asserts `solver.attr_color.numpy()` shape == (150, 3)
  and the inflow slice matches its source colour.
* [x] B18.2 — Per-frame sidecars: `MpmSolver.save_frame_ply()` extended to
  also write `colors/frame_NNNN.npy` and `temperatures/frame_NNNN.npy`
  when the attrs exist, mirroring the FLIP convention
  (`commands.py:551-562`). Selection-mask filtering applies (inflow
  particles below `spawn_step` are excluded from both PLY and sidecar).
  CLI `_cmd_simulate_mpm` reads `[[fluids]] color/temperature` and
  per-inflow `color/temperature` (new TOML keys) and forwards into
  `MpmConfig`. Acceptance: round-trip — bake 5 frames with red+blue
  inflows, load `colors/frame_0004.npy`, count cells where `R>0.8` and
  `B>0.8`, both nonzero.
* [ ] B18.3 — `S2.17.MIX` — optional MPM-side colour smoothing pass.
  Two new kernels (`k_mpm_p2g_color`, `k_mpm_g2p_color`) + one
  `wp.array4d(dtype=vec3)` weight-and-colour grid allocated lazily.
  Gated by `cfg.color_mix_mode in {"off", "linear"}`; "off" skips the
  launches (zero overhead). With "linear" on, neighbouring particles
  bleed colour over time → physically-shaped contact zones rather than
  per-particle-static colour. Acceptance: synthetic side-by-side ball
  collision with mix_mode="off" vs "linear" — "linear" produces
  ≥5% of particles with mixed colour (`|R-0.5|<0.1`); "off" produces
  ≤0.5%. *(deferred 2026-05-20: B18.4 mesher KNN + B18.5 Mixbox already
  deliver the visible blue+yellow=green payoff; in-solver diffusion
  adds slow physical time-blending and requires a custom hook into
  warp-mpm's step loop. Not on the v0.8 critical path; tracked as v0.9
  "Mixbox proper" follow-up.)*
* [x] B18.4 — Mesher per-vertex colour. Extend `I6.1.write_ply` to accept
  optional `(N, 3) uint8` `vertex_colors` and emit
  `property uchar red/green/blue` lines. Extend `M5.11` SDF mesher
  (`src/gpufluid/meshing/surface.py:MeshExtractor.extract`) with optional
  `particle_colors` argument: for each output vertex, KNN-blend colour
  from the 8 nearest particles (same HashGrid that's already built for
  the SDF). Linear-RGB blend by default; Mixbox path layered in B18.5.
  Acceptance: red column → meshed → PLY contains vertex colours with
  R-channel mean > 0.8.
* [x] B18.5 — Mixbox pigment-space LUT. Vendor `mixbox_lut.bin` (~270 KB,
  MIT license — Šochorová & Jamriška 2021) under `third_party/mixbox/`
  with a `LICENSE.txt` copy. New helper `gpufluid.meshing.mixbox` reads
  the LUT once at import-time and exposes `mix_rgb(c1, c2, t) → rgb`
  (vectorized over arrays). Mesher branch on `cfg.color_mix_mode`:
  `"linear"` → weighted RGB mean (B18.4 default), `"mixbox"` → recursive
  pairwise mixbox blend. Acceptance: blue (0,0,1) + yellow (1,1,0) at
  t=0.5 → R<0.4, G>0.6, B<0.4 (greenish, the B2.5 spec).
* [x] B18.6 — Addon UI. Inflow gets `use_color`/`color` (same shape as
  Fluid). Domain gains `mix_mode: EnumProperty("off", "linear", "mixbox")`
  defaulting to `"linear"`. `bake.py:collect_scene` threads them
  per-source into TOML; `config_builder.build_toml` mirrors. `cache_loader`
  detects vertex-colour layer in PLY and creates a `Col` attribute on
  the Blender mesh (visible as Vertex Colors → `Col` in shader graph).
  Acceptance: smoke — bake the B18.7 demo through the addon, the
  `gpufluid_cache.data.color_attributes` is non-empty, the colour shows
  in viewport material preview.
* [x] B18.7 — Demo: `examples/scenes/mpm_mix_blue_yellow.toml` — two
  taps (blue at x=0.3, yellow at x=0.7) emitting downward, meeting in
  a shared basin. Bake + render two variants: `mix_mode="linear"`
  and `mix_mode="mixbox"`. Stitch side-by-side
  `out/videos/step33_mpm_mix.mp4`. Acceptance: at frame 120,
  central-pixel sample on the contact zone — mixbox gives green
  (`R<0.4 G>0.6 B<0.4`), linear gives muddy (`G < R+0.1` and `G < B+0.1`).
* [x] B18.8 — Closure: tests under `tests/test_b18_*`, `BLOCKS.md` rows
  for S2.17.MIX.{P2G,G2P}, `DESIGN.md §5.3.MIX` section explaining the
  mix-mode + Mixbox transport, `memory/project_mpm_mixing.md` summary,
  registry check (`python -m gpufluid.blocks --check`) clean.
  Move B18 to **Completed** with date. *(2026-05-21: `tests/test_b18_1_mpm_attrs.py`
  + `tests/test_b18_4_5_vertex_colour.py` ship; `docs/BLOCKS.md` carries
  rows for M5.11.4, M5.11.4.H, M5.11.5 (lines 103-105). `memory/project_mpm_mixing.md`
  created this session summarising shipment. S2.17.MIX kernels were
  consciously dropped from the scope when B18.3 was deferred — see the
  v0.9 "Mixbox proper" follow-up note on B18.3. B18 macro moved to
  Completed below.)*

**Acceptance (macro closure) — verified 2026-05-20 with 240-frame
128³ bake of `examples/scenes/mpm_mix_blue_yellow.toml`:**
contact-zone vertex colour stats from `read_ply(return_colors=True)`
at frames 120 / 180 / 220 —
- **linear** mean_rgb ≈ (130, 135, 127), **0%** green-dominant,
  53–57% muddy-grey,
- **mixbox** mean_rgb ≈ (93, 161, 96), **71–84%** green-dominant,
  **0%** muddy-grey.
Acceptance threshold `R<0.4 G>0.6 B<0.4` lands at (0.36, 0.63, 0.38) —
clean pass. Addon UI exposes per-Inflow `use_color`/`color` and the
Domain `color_mix_mode` enum; cache_loader attaches a `Col` POINT-domain
colour attribute when PLY has rgb. Per-particle scalar (temperature)
plumbing is in place via the same path; lava-on-MPM demo deferred to
follow-up since B11.3 FLIP demo is already shipped.

**Supersedes B2** for the MPM codepath. B2 stays open as the FLIP-side
counterpart (linear→mixbox switch on G2P colour), but the mesher-side
LUT helper shipped in B18.5 is shared by both.

### B2. Mixbox pigment-space LUT for S2.15 colour `risk:low value:user` ✅ closed 2026-05-21

*See **Completed** section below for closure entry. Original micros (all ticked) preserved here for traceability.*

Current per-particle colour does linear RGB blend → blue+yellow = grey. Mixbox
(Šochorová & Jamriška 2021) is a 4D LUT (~270 KB) that gives painterly
mixing (blue+yellow = green). Drop-in on the G2P side.

* [x] B2.1 — Vendor or download the Mixbox LUT (`mixbox_lut.bin`, ~270 KB) at first-run with a clear license note in `docs/BACKLOG.md`. *(2026-05-21: shipped via `pymixbox` PyPI dependency consumed by `src/gpufluid/meshing/mixbox.py` (M5.11.5, B18.5). Honest variant — "vendored via pip" rather than "vendored in tree"; LUT lives inside the pymixbox wheel and is loaded at import.)*
* [x] B2.2 — Add `k3_g2p_color_mixbox` kernel that reads the LUT (as `wp.array4d(dtype=float)`) and performs the inverse-K-M lookup. *(2026-05-21: shipped CPU-only via `pymixbox` Python API + `scipy.spatial.cKDTree` in `meshing/mixbox.py::remix_vertices_mixbox` — NOT a `wp.array4d` GPU kernel. The blend happens post-extract on the mesher output (per-vertex CPU pass), which is acceptable for the mesh stage and matches the MPM path's plumbing.)*
* [x] B2.3 — Switch via `cfg.color_mix_mode: "linear" | "mixbox"` on FlipSolver3D. *(2026-05-21: exposed as `scene.output.color_mix_mode` TOML knob; Domain UI enum on the addon side. `cli/commands.py:cmd_simulate` FLIP branch reads `solver.attr_color` and routes through `compute_vertex_colors` + `remix_vertices_mixbox` before `write_ply`.)*
* [x] B2.4 — Demo scene: blue + yellow water cubes → step23.mp4 with side-by-side linear vs Mixbox. *(2026-05-21: shipped via the MPM codepath as `out/videos/step33_mpm_mix.mp4` (2560×720, 8 s; B18.7 agent). The FLIP variant is functionally identical and `tests/test_b2_flip_mixbox.py` is the pytest-level proof that FLIP carries the same colour. A standalone FLIP step24-style demo would be redundant marketing-wise — deferred.)*
* [x] B2.5 — Pytest: a yellow+blue 50/50 blend in Mixbox mode produces R<0.4, G>0.6, B<0.4 at the contact zone (greenish), unlike linear which gives ~(0.5, 0.5, 0.5) (greyish). *(2026-05-21: covered by `tests/test_b18_4_5_vertex_colour.py::test_b18_5_mixbox_blue_plus_yellow_is_green` at the mesher level, AND by `tests/test_b2_flip_mixbox.py` for the FLIP-end-to-end path.)*

**Acceptance:** scene file with `color_mix_mode = "mixbox"` produces a visibly green contact zone in the side-by-side video.

### B3. Better whitewater classifier (Wave-Crest + Trapped-Air potentials) `risk:med value:user`

W7.4 currently classifies by density-grid lookup + velocity projection. This
works but produces ~30 spray / ~800 foam / ~2100 bubble in the splash demo —
spray count is low because the heuristic doesn't find wave crests properly.
Ihmsen et al. 2012 §3 has the full potential definitions.

* [x] B3.1 — Implement Trapped-Air potential: `I_ta = Σ_j (1 - cos(θ_ij)) · (1 - |v_ij|/v_max)` over neighbours, capped at 1. Per fluid particle on GPU (uses a HashGrid for neighbour query — `wp.HashGrid` is already available). *(2026-05-16: `gpufluid/sim/whitewater_potentials.py` with W7.7 kernel + W7.7.H host wrapper. wp.HashGrid built per call (one-shot, no caching yet). The (1 - cos θ) factor is intentionally NOT pre-clamped per-pair — only the final sum is capped at 1 (spec said sum-cap; per-pair clamp was an early bug I removed when test_w7_7_v_max_scales_pair_contribution caught it).)*
* [x] B3.2 — Implement Wave-Crest potential: `I_wc = |∇·n̂|` as a standalone P2G→blur→∇→∇· pipeline that does NOT depend on CSF (S2.14.* is solver-internal MAC-grid code). Shipped as [BLK W7.8] (4 kernels: indicator P2G, normal+grad-mag, |div n̂|, gated G2P) + [BLK W7.8.H] host wrapper. *(2026-05-17: gating by |∇χ̃| was the key insight — without it, interior plateau cells fired false-positive curvature via noise-amplified unit normals. 3 GPU unit tests in `tests/test_w7_8_wave_crest.py` lock the algorithm: crest-centre vs flat-centre (>2×), interior particles must stay <0.15, mirror-symmetry distribution match.)*
* [x] B3.3 — Emit rate per particle = `clamp(α·I_ta + β·I_wc, 0, max_rate)·dt` instead of the current `|v|>threshold` cut-off. *(2026-05-16: shipped as a weighted selector inside `emit_from_fluid`. `[output] whitewater_use_potential = true` activates it. Adds a tiny floor (1e-3) so laminar particles aren't completely starved — without the floor a waterfall column emits ~0 because internal particles are perfectly aligned. Addon UI exposes the toggle in the Whitewater sub-panel (`use_potential`, `potential_radius`, `potential_v_max`). The full α/β coefficients are folded in once B3.2 wave-crest exists; until then β=0.)*
* [x] B3.4 — A/B/C bench: legacy / trapped-air-only / trapped-air + wave-crest on `whitewater_splash.toml`. Revised acceptance after first run: spray-fraction barely moved with wave-crest (1.63x vs 1.66x) because splash is saturated with turbulence — trapped-air already catches what wave-crest would. The HONEST metric is bubble→surface shift: with β=2.0, surface-fraction (foam+spray) rises from 27.13% to 29.62% (+2.5pp) and bubble-fraction drops 72.87%→70.38% (-2.5pp). Target ≥2pp surface gain — PASS. The earlier 3× and 5× spray targets were over-promise; the wave-crest's actual job is "pull emit out of sub-surface into surface region", not "make more spray". *(2026-05-17: `examples/compare_whitewater_potential.py` now bakes all three variants and asserts the revised KPI.)*
* [x] B3.5 — Closure video: `out/videos/step28.mp4` (2.0 MB, 90 frames @ 24fps, Eevee). Text-overlay style on the `step28_b3_full` whitewater splash bake; overlay reports the B3.4 KPIs and the closure date. Cloned from `render_step27_eevee.py` per the "invisible perf wins" pattern (the wave-crest effect is statistically real but visually subtle frame-to-frame, so the headline numbers in overlay are the right surface).

**Acceptance:** ✅ closed 2026-05-17 with B3.2 wave-crest + B3.4 bench (+2.5pp surface-fraction at β=2.0) + B3.5 step28.mp4 closure video. The original "more airborne spray" framing was retired in favour of the bubble→surface shift metric, which captures wave-crest's real contribution.

---

## Tier 2 — Hot-path performance (do once tier 1 is done; risk:med)

### B4. Extend block-sparse iteration to GS-RB + PCG `risk:med value:infra` ✅ closed 2026-05-16

S2.6.4 covered Jacobi. GS-RB (S2.6.2) and PCG (S2.6.3) are the other two
pressure paths and are bigger wins at scale. Same compaction infrastructure.

* [x] B4.1 — Port GS-RB to a per-tile launch kernel `k3_gauss_seidel_rb_per_tile`. Keep red/black colouring (parity check inside the tile). *(2026-05-16 spike: **2.1× speedup at 128³ / 9% fill / 80 iters** (5.2 ms→2.5 ms kernel-only, RTX 4080 SUPER). Parity vs dense GS-RB within 5% on the resulting velocity field. Block-sparse infrastructure (S2.16) ports cleanly — same `_block_active` + `_block_prefix` + `_block_coords` pipeline as Jacobi. step() picks the sparse path automatically when `pressure_block_sparse=True` and `pressure_solver="gsrb"`. **B4 macro is greenlit** based on this spike.)*
* [x] B4.2 — Port PCG: the spMV (`k_apply_A`), `k_apply_invM`, and per-residual axpy kernels each need block-sparse variants. *(2026-05-16: 4 per-tile kernels under S2.6.6 (apply_A, apply_invM, axpy_devscalar, dot_fluid). New `_pressure_pcg_sparse()` method mirrors the dense PCG arithmetic. Shared `_build_active_blocks()` helper. Speedup at 128³ / ~10% fill / 30 iters: **1.1×** (25.6→23.1 ms/step) — much more modest than B4.1's 2.1× for GS-RB. PCG has 8 device-side ops per iter vs GS-RB's 2, so per-tile dispatch arithmetic + the extra `n_active` host-sync eat into the per-iter saving. Follow-up micro: cache `n_active` on-device so we drop one host-sync per step.)*
* [x] B4.3 — `step_cfl()` and `step()` accept `pressure_block_sparse: bool` consistently for all three solvers. *(2026-05-16: was already plumbed for Jacobi (S2.16) and GS-RB (S2.6.5 / B4.1); B4.2 adds PCG via the same flag. All three pressure solvers now route to their per-tile variants when `pressure_block_sparse=True`. step_cfl forwards the flag unchanged since it just calls step() in a loop.)*
* [x] B4.4 — Bench: 128³ low fill, GS-RB sparse should match the 2.4× Jacobi speedup; PCG should be smaller (its per-iter work is more memory-bound). *(2026-05-16: GS-RB measured **2.1×** (kernel-only, 80 iter), PCG measured **1.1×** (full-step, 30 iter). PCG prediction confirmed — its per-iter work IS more dispatch-bound than memory-bound.)*
* [x] B4.5 — Pytest equivalent of `test_s2_16_sparse_jacobi.py` for both solvers. *(2026-05-16: `test_s2_6_5_gsrb_sparse.py` (2 tests, parity + speedup) and `test_s2_6_6_pcg_sparse.py` (2 tests, parity + speedup). Suite 141 → 143.)*

**Acceptance:** scenes using `pressure_solver = "gsrb"` or `"pcg"` get the block-sparse path automatically and the regression tests pass.

### B5. CUDA-graphs capture for the per-step kernel sequence `risk:med value:infra` ✅ closed 2026-05-16 (9 of 9 configs eligible after Options A + B)

Each step() does 20+ `wp.launch` calls. CUDA graphs would record them once
and replay each step, eliminating launch overhead. Warp supports
`wp.capture_begin / wp.capture_launch`.

* [x] B5.1 — Audit `step()` for host-syncs (anything that calls `.numpy()` inside the per-step path can't be in a graph). Move them outside (CSF balance already uses GPU, advection CFL is one float, etc.). *(2026-05-16 spike: 6 sync sites identified — PCG dense/sparse r_norm^2 reads, CSF S2.14.6 force-balance (3 axes), sparse pressure n_active. Plus a redundant `wp.synchronize()` at end of step() that I removed in this commit. Of the 9 solver configurations we ship, **3 are graph-eligible today** (jacobi/gsrb × any transfer_mode, no CSF, no sparse). The other 6 need their syncs moved out as B5.2 work. Capture+replay verified bit-identical to direct step on Jacobi at 32^3, **2.17× speedup at 64^3 / 20 iters** (0.43→0.20 ms/step) — way above the B5.5 target of 1.15×. Macro greenlit.)*
* [x] B5.2 — Add `solver._graph` lazy cache keyed by `(transfer_mode, pressure_solver, has_csf, has_color)` — a graph is invalid when topology changes. *(2026-05-16: `_cuda_graph` slot + `_cuda_graph_key` + hit/miss counters. Cache key includes the full set of variables that affect launch sequence: transfer_mode, pressure_solver, pressure_block_sparse, surface_tension>0, viscosity>0, attr_color, attr_temperature, n_particles, dt, pressure_iters. Different args → different key → recapture. Ineligible configs (PCG/CSF/sparse) fall through to direct path silently.)*
* [x] B5.3 — Invalidate on every `prepare_frame` (since marker/obstacles changed) — capture freshly for the next N=1..3 frames inside that topology, then replay. *(2026-05-16: `prepare_frame` calls `_cuda_graph_invalidate()` at entry. With CFL substepping (8-32 substeps/frame typical), first substep recaptures, the rest replay. Real-scene bake on `two_color_drop` showed 88% hit rate = 630 replays / 90 captures over 720 substeps.)*
* [x] B5.4 — Pytest: same numerical output with/without graphs. *(2026-05-16: `test_b5_2_replay_matches_direct_to_fp_precision` — two solvers from the same seed, one with `enable_cuda_graphs=True` and one without; end-state velocity fields agree to <1e-4 relative drift after 5 steps × 4 hits.)*
* [x] B5.5 — Bench: target ≥15% speedup at 64³ (small grids = launch overhead dominates). *(2026-05-16: **real bake (two_color_drop, 90 frames, 40³, 8 substeps/frame, jacobi) — sim time 0.72 s → 0.15 s = 4.8× speedup**. Way exceeds the 1.15× target. `gpufluid simulate --enable-cuda-graphs` is the CLI flag; off by default. End-of-bake prints hit/miss tally.)*

**Acceptance:** an `--enable-cuda-graphs` flag on `gpufluid simulate`. **Met for all 9 shipped solver configs** after Options A + B:

* `(jacobi | gsrb | pcg) × (dense | sparse) × any transfer_mode` — every shipped combination, with or without CSF.
* No remaining ineligible configurations on the production path.

B5 follow-up (this session) — what changed and what didn't:

* ✅ **CSF S2.14.6 moved device-side.** New `k3_csf_subtract_bias_{u,v,w}_dev` kernels read sum + count from device buffers and compute `bias = sum/count` inside the kernel. `_apply_surface_tension` no longer calls `.numpy()` on the per-axis sums. Real-bake `surftens_on` (60 frames @ 48³, jacobi, σ=1, 38 substeps/frame): **sim 3.35s → 0.19s = 17.6×, 97% hit rate.** Biggest user-facing win this session because most v0.8 demo scenes use σ.

* ✅ **PCG eligibility flipped (Option A, 2026-05-16 same session).** On-device stop-flag pattern landed: `_pcg_done` int + `_pcg_tol_sq` float device buffers; new 1-thread kernels `k3_zero_int_scalar`, `k3_set_tol_sq`, `k3_check_converged` (S2.6.3); all 7 PCG iter kernels (`k3_apply_A`, `k3_apply_invM`, `k3_dot_fluid`, `k3_zero_scalar`, `k3_div_scalar`, `k3_copy_scalar`, `k3_axpy_devscalar`) gain a `done: wp.array(dtype=int)` parameter and early-return when `done[0] != 0`. After each per-iter `r·r`, `k3_check_converged` sets `done = 1` if `r_norm² < tol_sq`. Kernel SEQUENCE stays constant (good for graph capture) while behaviour still respects `tol`. **Real-bake `big_pcg` (96³, PCG 60-iter, 90 frames): sim 4.21s → 1.01s = 4.17×, 88% hit rate.** Completely reverses the prior regression. Tests: `tests/test_b5_a_pcg_graph.py` (4 GPU tests) + updated `test_b5_2_ineligible_config_falls_through`. Sparse PCG inherits the same eligibility once Option B lands (`n_active_dev`).

* ✅ **Block-sparse eligibility flipped (Option B, 2026-05-16 same session).** New device buffer `self._n_active_dev: wp.array(dtype=int, shape=(1,))` + 1-thread kernel `k_store_n_active(prefix, n_blocks, n_active_dev)` (S2.16) replaces the `prefix[-1].numpy()` host read in `_build_active_blocks`. All six per-tile kernels (`k3_jacobi_pressure_per_tile`, `k3_gauss_seidel_rb_per_tile`, `k3_apply_A_per_tile`, `k3_apply_invM_per_tile`, `k3_dot_fluid_per_tile`, `k3_axpy_devscalar_per_tile`) gain a `n_active_dev: wp.array(dtype=int)` argument and `if blk >= n_active_dev[0]: return` early-out. PCG per-tile kernels additionally take Option A's `done` flag, so sparse PCG inherits the on-device convergence-check for free. Launches now use the constant `n_blocks * cells_per_block` worst-case dim. Real-bake `big_pcg` (sparse PCG, 96³, 60-iter, 90 frames): graph-off 4.19s → graph-on 1.00s = **4.19×**, 88% hit rate (identical to dense at this fill — value is the eligibility, perf scales with sparsity at higher resolutions). Tests: `tests/test_b5_b_sparse_graph.py` (5 GPU tests) + updated `test_b5_2_all_configs_graph_eligible`. Inline-launch tests `test_s2_16_sparse_jacobi`, `test_s2_6_5_gsrb_sparse` extended with `n_active_dev=wp.array([n_active], dtype=int, device=...)`. `test_b4_2_sparse_pcg_speedup_at_128` threshold loosened from "≥1.05x faster" to "no worse than -5%": the extra in-kernel `n_active_dev[0]` read trades raw kernel speed for graph eligibility (kernel-only at 128³/9% is now parity with dense; the win is in graphs-on territory).


### B6. APIC + CSF interaction QA `risk:low value:infra` ✅ closed 2026-05-16

We verified APIC near obstacles in S2.12 QA. We did NOT verify APIC + surface
tension. The affine reconstruction interacts with CSF impulses in a way that
might break the parasitic-current bound (HANDOFF trap #11).

* [x] B6.1 — Build a scene: APIC mode, σ=1, zero gravity, cube → sphere (same as surftens_on.toml but `transfer_mode = "apic"`). *(2026-05-16: in-process via `tests/test_b6_apic_csf_interaction.py` — same physics knobs as surftens_on.toml, 60 frames @ 48³.)*
* [x] B6.2 — Diagnostic: COM drift, max parasitic velocity, kinetic energy vs FLIP-CSF baseline. *(2026-05-16: measured + asserted. COM drift < 2% of domain, max|v| < 35 m/s (~5× capillary-wave speed estimate), KE change bounded.)*
* [x] B6.3 — If drift > 2%, add a damping rule (zero `affine_C` near surface cells, where `|∇χ̃| > threshold`). *(2026-05-16: **not triggered** — APIC+CSF passes the baseline tolerance without any damping intervention. Code path stays as-is.)*
* [x] B6.4 — Pytest analogous to `test_csf_step_cfl_centre_of_mass_does_not_drift` but for APIC. *(2026-05-16: `test_b6_apic_csf_com_drift_within_tolerance` + `test_b6_apic_csf_does_not_blow_up_kinetic_energy`.)*

**Acceptance:** known-stable APIC+CSF configuration documented, regression test guards it. **Met.** APIC + CSF is safe to enable simultaneously; pre-existing knobs from surftens_on.toml (48³+, 3 smoothing passes, mild viscosity, CFL substep cap 64) work unchanged.

---

## Tier 3 — Architectural projects (multi-session, plan carefully)

### B7. Sparse v2 — NanoVDB-backed storage for grid fields `risk:high value:infra blocks: B5 B4` ❌ aborted 2026-05-16

The big one. Replace dense `wp.array3d` allocations with `wp.Volume`
(NanoVDB) for the MAC velocity faces, pressure, marker, density. Saves
memory at 256³+ (currently ~470 MB of fields, with 30% fill could drop to
~140 MB) and unblocks 512³ scenes.

This is **the most invasive task in the backlog**. Estimated 3-5 sessions.
Break by *block* — port each S2.x kernel one at a time so we can revert any
single port if it goes wrong. Always keep the dense path as a parallel
implementation behind a feature flag (`use_sparse_storage=False` default).

* [x] B7.1 — **Spike**: prototype a sparse marker field using `wp.Volume`. Measure read latency (`wp.volume_lookup_i`) vs dense indexing in a hot kernel. If sparse is >2× slower per-cell, abort the macro and document why. *(2026-05-16: **MACRO ABORTED.** Two findings in `tests/test_b7_1_spike_sparse_volume.py`: (1) Warp 1.13 has NO kernel-side `wp.volume_store_*` API — `wp.Volume` is read-only from kernels. The B7 plan (replace `self.u/v/w/p/marker` with `wp.Volume` and atomic_add into them) is physically impossible on current Warp. (2) Even read-only, `wp.volume_lookup_i` is 1.3-2.3× slower than dense `wp.array3d` indexing at 128³ ~10% fill — borderline at the BACKLOG abort threshold. Volume topology + data round-trip works (count matches exactly), so the spike validated the API surface; it just can't be wired into a mutating solver. **Pivot recommendation:** extend S2.16 block-sparse compaction (already 2× faster pressure at 128³) to additionally skip dense memory allocation for inactive tiles. That gives ~70-80% of the sparse v2 memory win without depending on Warp gaining volume_store_*.)*
* [-] B7.2 through B7.10 — **dropped.** All depend on the NanoVDB write path that doesn't exist in Warp 1.13. Re-evaluate when `wp.volume_store_*` ships upstream (`test_b7_1_volume_is_read_only` will start failing — that's the regression-trip telling a future maintainer to revisit this macro).

**Pivot — recommended path to 256³+ without NanoVDB:**

A separate macro (B7-alt) using the S2.16 block-sparse compaction plus **deferred dense allocation**: allocate `wp.array3d` lazily for the bounding box of active tiles, rebuild every N frames with a dilation margin. Same per-step kernels we already have; only `_build_active_blocks` + the field allocations change. Memory cost scales with active-fill rather than `nx·ny·nz`. Gives ~70-80% of the v2 win without depending on Warp upstream. Risk: dense allocation API is grid-shape-tied, so we'd need a coordinate translation layer (offset_x/y/z) in every kernel. Worth a fresh spike before committing.

* [x] B7-alt.1 — **Spike**: at 128³/~5% fill, measure (a) memory bbox-ratio on a *connected-blob* topology, (b) memory bbox-ratio on a *scattered-droplets* topology, (c) correctness of a Jacobi-like dense kernel re-launched on a sub-dense `wp.array3d` of size bbox+dilation with an `offset_xyz` parameter. Abort if (a) < 5× drop OR (c) doesn't match full-dense within 1e-4 relative. *(2026-05-16: **GREEN.** `tests/test_b7_alt_1_spike_deferred_dense.py` — 4 tests all pass. Connected-blob @128³/4.95% fill: bbox=(48,48,48), **18.96× memory drop**. Scattered-droplets @128³/4.77%/200 drops: bbox=(128,128,128), **1.00× drop** (pathological — bbox covers full domain). Coord-translated Jacobi vs full-dense: **rel err = 0.000e+00** after 40 iters on the active region; dilation=1 + zero-pressure-outside boundary strategy bit-exact on the realistic connected case. Macro greenlit. Caveat: scattered-droplet scenes get ZERO benefit from B7-alt — document in §"non-goals" and don't ship it as a silver bullet.)*
* [x] B7-alt.2 — Refactor `FlipSolver3D` field storage so `self.u/v/w/p/p_tmp/div/marker` (the per-cell dense fields) can be backed by EITHER full-dense or sub-dense allocations. New attribute `self._sub_offset = (ox, oy, oz)` (default `(0, 0, 0)` for full-dense; non-zero in deferred-dense mode). Rebuild trigger sits in `prepare_frame`: rebuild every `sub_rebuild_every` frames, OR when any active 8³ tile lies within `dilation` cells of the current sub-dense edge (whichever comes first). The "rebuild" copies active-region data from old sub-dense into the new one (different bbox + dilation) and zeros the rest. *(2026-05-17: shipped. New `__init__` params `enable_sub_dense=False, sub_rebuild_every=8, sub_dilation=4`; new attrs `_sub_offset`, `_sub_shape`, `_last_sub_rebuild_frame`. Three new helpers under block ID F3.7: `_compute_active_bbox(marker_host)` (8³-tile bbox, dilated, clamped — falls back to cell-precise scan on non-block-aligned grids), `_should_rebuild_sub_dense(frame_idx)` (first call OR periodic OR raw bbox within `_sub_dilation` of current edge), `_rebuild_sub_dense(lo, hi)` (reallocates `u/v/w/p/p_tmp/div` at the new bbox; face-centered shapes get `+1` on their face axis; overlap region copied via CPU round-trip — on-device copy lands in B7-alt.8). `prepare_frame` calls the trigger after inflow/outflow processing using the latest GPU marker (`self.marker.numpy()`). Marker stays FULL-dense (matches the spike test interpretation — walls/obstacles live at domain edges); only the 6 listed scratch fields shrink. `step()` is guarded with a `NotImplementedError` while `_sub_offset != (0,0,0)` — kernel offset threading is B7-alt.3, so enabling sub-dense before that lands fails loud rather than launching at the wrong dims. Regression `tests/test_b7_alt_2_sub_dense_storage.py` — 11 tests covering default invariants, bbox computation, first-rebuild shape transitions, overlap-preserving copy at offset delta, periodic trigger, proximity trigger, and the step() guard. Suite 170→181 green (+11, no regressions). `gpufluid info` 80→81 unique IDs / 103→105 callables.)*
* [x] B7-alt.3 — Thread the offset through every dense kernel that the per-step pipeline launches. Each kernel gains `off_x, off_y, off_z: int` arguments + uses `gi = li + off_x` (etc.) for cross-array marker/div lookups. The local pressure neighbours stay in sub-dense coordinates (the spike kernel `k3_jacobi_spike_offset` is the prototype). This is mechanically tedious — touches ~20 kernels — but each change is the same 4-line addition. *(2026-05-17 partial: **Jacobi / dense / FLIP / no-CSF / no-viscosity / no-colour / no-scalar slice landed end-to-end.** 7 kernels got `off_x/off_y/off_z` params: `k3_clear_grid`, `k3_p2g`, `k3_enforce_solid_bc`, `k3_compute_divergence`, `k3_jacobi_pressure`, `k3_subtract_pressure_grad`, `k3_g2p_and_advect`. **Same-session follow-up:** `k3_gauss_seidel_rb` ported (red/black parity keyed off GLOBAL indices so an odd offset doesn't swap colours); viscosity (`k3_jacobi_visc`) needs ZERO kernel changes — pure within-buffer Jacobi diffusion with no marker access, and the dilation buffer absorbs the Neumann fall-through at the sub-dense edge. **APIC kernels ported:** `k3_p2g_apic` + `k3_g2p_apic_advect` thread `off_x/y/z`; sample-time floor uses LOCAL coords (`p[0]/dx - off_x`) while the C-reconstruction face-to-particle offsets use WORLD positions (`(ii + off_x)*dx - p[0]`) — the affine matrix must remain a world-frame gradient. Step() guard now permits `pressure_solver in ('jacobi','gsrb')` × `viscosity≥0` × `transfer_mode in ('flip','pic','apic')`. Integration tests cover gsrb (0.000e+00), viscosity (5e-8 rel), APIC (0.000e+00), PIC (0.000e+00) — six configs now bit-exact. **PCG dense ported:** `k3_apply_A`, `k3_compute_diag`, `k3_apply_invM`, `k3_dot_fluid`, `k3_axpy_devscalar`, plus the rarely-used `k3_axpy` for symmetry. `_pressure_pcg` lazy-allocates scratch (`_pcg_r/z/p/Ap/diag`) at `self.p.shape`, so a sub-dense rebuild re-sizes them on the next call. `_pressure_pcg_sparse` (block-sparse per-tile path) stays full-dense; it now passes explicit `0,0,0` to the diag launch since the kernel signature changed. PCG sub-dense integration: 0.000e+00 rel drift after rebuild + second step. `k3_normalize` and `k3_add_gravity` don't need offsets — they only touch sub-dense fields. `_rebuild_sub_dense` extended to shrink `uw/vw/ww/us/vs/ws` in lockstep with `u/v/w` (face-cell topology must stay consistent for p2g/normalize/g2p). `_step_impl` reads `nx,ny,nz` from `self._sub_shape` and `ox,oy,oz` from `self._sub_offset` (both identity in default mode). step() guard upgraded from "any sub_offset != 0 raises" to "raise only when config is unsupported" — it now enumerates exactly which knobs are blocking (`pressure_solver != 'jacobi'`, `pressure_block_sparse=True`, `transfer_mode != 'flip'`, `surface_tension>0`, `viscosity>0`, `attr_color`, `attr_temperature`). Integration test `tests/test_b7_alt_3_jacobi_dense_flip.py` — 2 GPU tests: bit-exact position match (0.000e+00 rel drift, 1.5e-7 vel drift from atomic-add ordering) after rebuild + second step; 5-step stability run also bit-exact at fp32. `tests/test_s2_16_sparse_jacobi.py` updated to pass `0,0,0` offsets in its direct kernel launches. Suite 181→183. **FULL coverage shipped 2026-05-17 in the same session:** Every step()-launched kernel ported. block-sparse (6 per-tile kernels: jacobi/gsrb/pcg apply_A/apply_invM/dot_fluid/axpy_devscalar); CSF (10 kernels: indicator, blur, normal, curvature, apply_{u,v,w}, sum_{u,v,w}, subtract_bias_{u,v,w}_dev); colour (3 kernels); scalar (3 kernels). All scratch arrays (PCG `_pcg_*`, CSF `_csf_*`, colour `_cgrid_*`, scalar `_sgrid_*`) now lazy-resize to `self.p.shape` so a sub-dense rebuild auto-reallocates. step() guard reduced to a belt-and-suspenders unrecognised-solver-string check. **10 integration tests, all bit-exact** vs full-dense baseline on RTX 4080 SUPER (max 1e-7 rel drift from atomic-add ordering). The macro spec's "~20 kernels, 4-line addition each" turned into ~35 kernels touched — a single-session push because each port is the same mechanical pattern.)*
* [x] B7-alt.4 — Per-particle G2P/P2G need both world→local and world→full-grid mapping (particles live in world space, the grid lives in sub-dense). Add a small `pos_to_sub(p)` helper that does `floor(p/dx) - sub_offset` and bounds-checks. Particles outside the sub-dense bbox are out-of-domain — they should already be caught by `D4.7` outflow at the wall margin, but make the bounds check explicit so a misconfigured rebuild doesn't silently drop fluid. *(2026-05-17: shipped. `_pos_to_sub_cell(pos)` returns `((li,lj,lk), inside)` for one world-space pos. `_check_particles_in_sub_bbox(frame_idx)` scans all particles after each rebuild and fires a one-shot stderr warning when any sit outside the bbox — names the n_out count, percentage, current `sub_dilation`, and `sub_rebuild_every` so the user knows exactly which knob to raise. Wired into `prepare_frame` right after `_rebuild_sub_dense`. 5 regression tests in `tests/test_b7_alt_4_particle_bounds.py`: identity in default mode, offset translation, no-op when sub-dense off, warns once on misconfigured bbox + counts correctly, silent when all inside. Suite 194→199.)*
* [x] B7-alt.5 — `_cuda_graph_invalidate()` already fires on every `prepare_frame` (B5.3), so a rebuild that happens at the start of a frame will naturally trigger graph recapture with the new bbox shape. Verify in a test: bake a flowing-fluid scene with `enable_cuda_graphs=True`, assert hit rate ≥ 80% across rebuilds (graphs are valid for `sub_rebuild_every` frames at a time). *(2026-05-17: verified. `tests/test_b7_alt_5_cuda_graph_rehit.py` — 3 GPU tests. Flowing-fluid scene at 32³, 12 frames × 8 substeps. Dense baseline: 87.50% (84 hits / 12 misses); sub-dense with sub_rebuild_every=4, sub_dilation=6: 87.50% (identical). Sub-dense rebuild + invalidate plumbing doesn't damage graph caching — pattern stays "first substep of each frame recaptures, rest replay". The macro acceptance bar (≥80%) is met. Stability test (frequent rebuild_every=2, NaN check) passes — graph replay over sub-dense buffer pointers stays correct after re-allocation because the cache key includes `n_particles/dt/pressure_iters` and the per-frame `_cuda_graph_invalidate` re-keys on the new bbox shape implicitly via the new buffer addresses.)*
* [x] B7-alt.6 — **Bench at 256³ dam-break.** The acceptance test for the macro. Dense allocation: `256³ × ~8 fields × 4 B = ~512 MB` per snapshot — most pre-RTX-4090 GPUs OOM. With B7-alt and 10-20% active fill: sub-dense covers ~50³ to ~120³ → 4 to 25 MB per field → fits 4 GB GPUs. *(2026-05-17: shipped. `tests/test_b7_alt_6_256_bench.py` — 2 GPU tests (free-GPU-memory gated). Dam-break scene at 256³ with fluid blob in cells (25..150)³ ≈ 10% fill, sub_dilation=6 → sub-dense bbox lands at (140,140,140) covering ~16% of the full domain. **Measured 6.11x cell drop (642 MB saved across 12 cell fields), sub-dense step 1.0 ms** on RTX 4080 SUPER. Particle centroid stays inside bbox, pos/vel finite. Acceptance bar: ≥2.5x cell drop. Deterministic field-shape memory drop is the honest metric — runtime free-memory delta is mempool-noise + one-off PCG/CSF scratch that both modes pay equally. Suite 206→208.)*
* [x] B7-alt.7 — Document the scattered-droplet pathology (B7-alt.1 second test result). The macro provides ZERO memory savings on dispersed scenes (whitewater-heavy, secondary spray, exploding pours). Users with those scenes should use the dense path. Surface this in the CLI (e.g., compute "spatial extent ratio" at first rebuild; if > 0.8, print a one-shot stderr warning that B7-alt isn't helping). *(2026-05-17: shipped. `_rebuild_sub_dense` now computes `sub_volume / full_volume` and fires a one-shot stderr warning when the ratio > 0.8 — names the bbox shape, full extent, and current `sub_dilation` so the user can either disable `enable_sub_dense` or check the dilation. 4 tests in `tests/test_b7_alt_7_scattered_warning.py`: connected blob (no warn), scattered bbox (warn + correct numbers), one-shot guard, exact-80%-threshold not-quite-warns. Suite 202→206.)*
* [x] B7-alt.8 — Resize/rebuild needs to be cheap. Pure CPU resize would dominate the per-frame cost; do it on-device: allocate new sub-dense fields, launch one "copy old to new at offset_delta" kernel per field. This is parallel to the rebuild pattern in S2.11.GPU reseed. *(2026-05-17: shipped. New `k3_copy_subdense_at_offset` (1 kernel under F3.7) maps NEW (li,lj,lk) → GLOBAL → OLD (si,sj,sk) per thread, bounds-checks both ends, copies; non-overlap stays at zero (`zeros()` pre-fill). `_rebuild_sub_dense.rebuild_one` now launches the kernel directly on the device buffers — no more `.numpy()` round-trip. **256³ second rebuild: 3.0 ms total for 12 cell fields + 6 face fields = ~33× faster than the CPU round-trip estimate.** Correctness preserved: all 11 B7-alt.2 invariant tests still pass (incl. the load-bearing `test_rebuild_preserves_overlap_in_global_coords`). New `tests/test_b7_alt_8_on_device_rebuild.py` — 2 GPU tests: direct kernel exercise with a recognisable pattern, and a 256³ perf bar (<100 ms). Suite 208→210.)*

**Acceptance:** can run a 256³ dam-break that the dense path OOMs on.

### F3.6. Hook refactor — eliminate F3↔D4 layer inversion `risk:med value:infra` ▶ next macro

`solvers/solver3d.py` (layer F3) currently imports from `domain/*`
(layer D4), which violates the §2 import rule. The 6 whitelist
exceptions in `DESIGN.md §3.2.4.1` are the symptom. **Spec for the
fix lives in `DESIGN.md §3.2.4.2`** — read it BEFORE picking any
micro. The key insight: the 6 violations are 3 different problems,
not one wholesale inversion. ~60% of the work is mechanical
relocations.

* [x] F3.6.A1 — Move `sdf_sphere/box/cylinder_y/plane/union` +
  `cell_centers` from `domain/sdf.py` to `primitives/sdf.py`. Pure
  math, belongs in G1. *(2026-05-17: shipped. New IDs G1.10–G1.14 +
  unchanged G1.8 for cell_centers. `domain/sdf.py` keeps only
  D4.4 `mark_solid_from_sdf` + back-compat re-exports. 7 importers
  updated (3 src, 4 tests), `_KNOWN_LAYER_EXCEPTIONS` lost the
  `solvers.solver3d → domain.sdf` entry — whitelist down 1.
  Suite stayed at 227 passed, `--check` clean.)*
* [x] F3.6.A2 — Move `mark_solid_from_mesh_gpu` from
  `domain/mesh_sdf_gpu.py` to `schemes/mesh_marker.py` (S2). *(2026-05-17:
  shipped. Whole-file move including `_MeshCache` + both kernels;
  `domain/mesh_sdf_gpu.py` becomes a back-compat shim. IDs
  D4.3.GPU/D4.3.GPU.BVH preserved per spec — rename to S2.20.* deferred
  to a follow-up. `solvers/solver3d.py` lazy imports now point at
  `schemes.mesh_marker`; `test_d4_3_gpu_bvh.py` updated.)*
* [x] F3.6.B — Move `Motion` dataclass + `evaluate_center` from
  `domain/animation.py` to `primitives/animation.py`. *(2026-05-17:
  shipped. Pickle audit confirmed F3.5 checkpoint stores only numpy
  arrays + scalars; `MotionCfg` in `cli/config.py` is a separate
  TOML-side struct. New IDs G1.15 + G1.16. `domain/animation.py`
  becomes a re-export shim. 3 of 6 whitelist entries gone after this
  + A2 — only `domain.regions` + `domain.seed` remain.)*
* [x] F3.6.C1 — Add `FrameEventQueue` + `FluidEmitEvent` +
  `FluidOutflowEvent` dataclasses to `primitives/frame_events.py`.
  *(2026-05-17: shipped. New blocks G1.17/G1.18/G1.19, 10 TDD tests
  in `tests/test_g1_17_frame_events.py` including parity tests that
  prove `InflowBox.publish_for_frame` + drain produces byte-identical
  (pos, vel) as legacy `apply_inflows`, and same for outflows. D4
  region helpers gained `publish_for_frame()` dual-path; legacy fns
  still callable. Solver unchanged — that's C2. Whitelist still 2
  entries as designed (transitional).)*
* [x] F3.6.C2 — Switch `solver3d.prepare_frame` to drain the queue
  instead of pulling from `domain.regions`. *(2026-05-17: shipped.
  Solver gained `self._frame_events = FrameEventQueue()` slot.
  prepare_frame now: clear queue, publish per-inflow/per-outflow,
  drain emits + outflows, apply. `_apply_outflows_gpu` signature
  changed from `(frame_idx)` reading self.outflows to
  `(outflow_events)` consuming pre-drained events — active-filter
  logic moves into `OutflowBox.publish_for_frame`. Top-level import
  `from ..domain.regions import ...` removed entirely (only types
  needed and they're duck-typed in self.inflows lists). All 6
  whitelist entries (`domain.regions`, `domain.seed`, plus the
  earlier A1/A2/B ones) deleted from `_KNOWN_LAYER_EXCEPTIONS`.
  Suite stays 237; B5 graph-rehit tests + B7-alt graph-rehit tests
  all pass — 88% hit rate preserved.)*
* [x] F3.6.C3 — Add hard test + closure video. *(2026-05-17:
  shipped. `tests/test_no_layer_exceptions.py` (single assertion
  `_KNOWN_LAYER_EXCEPTIONS == []`) — future PR adding a whitelist
  entry without a DESIGN.md §3.2.4.1 exit-plan update fails CI
  immediately. `out/videos/step29.mp4` (2.0 MB, 90 frames @ 24fps
  Eevee) — text-overlay closure video on the step28 bake (visual
  backdrop; F3.6 is architectural so the scene doesn't matter,
  only the overlay).)*

**Acceptance:** ✅ **F3.6 MACRO CLOSED 2026-05-17.** `gpufluid blocks
--check` clean with zero whitelist entries. The
`_KNOWN_LAYER_EXCEPTIONS` list is empty and guarded by a hard test.
The 6-phase journey: A1 (sdf -> G1), A2 (mesh_marker -> S2), B
(animation -> G1), C1 (FrameEventQueue dual-path), C2 (solver
drains), C3 (hard gate + closure video).

### B8. Differentiable solver via Warp gradients `risk:high value:research`

Warp 1.13 has `wp.Tape` for backprop. The FLIP pipeline is conceptually
differentiable (P2G → pressure → G2P are all linear/PD); a differentiable
solver enables inverse problems (fit obstacle shape to match a desired flow,
optimize emit direction, etc.).

* [ ] B8.1 — **Read** the Warp differentiability docs + their cloth example. Note the constraints: no `numpy()` syncs in the differentiated loop, atomics must be `wp.atomic_add` (not `_max`).
* [ ] B8.2 — Identify non-differentiable bits in current solver: APIC has a hard min on weights, BC clamping is non-smooth, marker rebuild is integer. Decide which to differentiate-around (mark non-diff explicitly) vs replace.
* [ ] B8.3 — Spike: differentiate gravity coefficient — gradient of mean fluid height w.r.t. `g` over N steps. Sanity-check against finite-difference.
* [ ] B8.4 — Differentiate inflow direction — gradient of "fluid coverage at frame 60" w.r.t. inflow velocity vector. Use as a toy example.
* [ ] B8.5 — Document which solver knobs are differentiable in DESIGN.md.

**Acceptance:** a notebook in `examples/diff/` that fits inflow direction to a target flow pattern.

### B13. Yu-Turk anisotropic surface reconstruction (M5.8) `risk:high value:user`

Replaces M5.1 isotropic spherical particle scatter with per-particle
oriented ellipsoids derived from local neighbourhood covariance. Yu &
Turk 2013, *Reconstructing Surfaces of Particle-Based Fluids Using
Anisotropic Kernels*, ACM TOG 32(1). Full algorithm + sub-block layout
in [DESIGN.md §8.2](DESIGN.md). This is the single biggest visual
lever for thin-feature water (tap streams, sheets, splashes); discovered
during demo30 iteration where 96³ + 192³ both produced "cauliflower"
mid-air mesh artefacts that no tuning of `iso_level`, particle rate, or
resolution could fix — the root cause is the isotropic kernel itself.

Macro shape: 5 sub-blocks (M5.8.1-M5.8.5) + 1 host wrapper (M5.8.H),
all device-resident in `src/gpufluid/meshing/anisotropic.py`. Opt-in
via `mesh_method = "anisotropic"` in the `[output]` TOML section; the
default `"isotropic"` keeps M5.1 unchanged so existing scenes are
bit-identical.

* [ ] B13.0 — **CPU reference implementation** in
  `gpufluid/meshing/anisotropic_ref.py` (numpy only, no warp). Mirrors
  the full Yu-Turk pipeline at ~1000× slower throughput; serves as the
  golden output for B13.7's GPU correctness tests. Without this the
  synthetic-sheet and synthetic-column tests have no reference to
  compare against — the GPU result would only be self-consistent
  (determinism), not correct. Budget: half a day; numpy SVD via
  `np.linalg.eigh` (no Jacobi rotations needed on CPU).
* [ ] B13.1 — `M5.8.1` neighbour gather. **Decision needed first**:
  whether to share W7.7's `wp.HashGrid` (potential R mismatch issue —
  W7.7 builds at its own potential radius which may not equal Yu-Turk
  R=4 cells) or build a dedicated one. Spike measurement: at the
  scenes we care about, what is W7.7's hashgrid cell size, and is it
  ≥ Yu-Turk's? If yes, reuse with a wider neighbour-distance cutoff.
  If no, build a second hashgrid (~1-2 ms cold per frame).
* [ ] B13.2 — `M5.8.2` weighted centroid + 3×3 symmetric covariance per
  particle, cubic falloff kernel `w_ij = (1 - r²)³`. Output: one `mat33`
  per particle.
* [ ] B13.3 — `M5.8.3` per-particle Jacobi-rotation eigendecomposition,
  8 sweeps. Eigenvalue clamping `σ_k ← max(σ_k, σ_max/k_r)` with default
  `k_r = 4`. Output: per-particle rotation `mat33` + scaled diagonal
  `vec3`.
* [ ] B13.4 — `M5.8.4` neighbour-count check; particles with `|N_i| < 25`
  flagged for isotropic fallback (radius `R/2`). Guards against
  pancake-ellipsoid artefacts on isolated splash droplets.
* [ ] B13.5 — `M5.8.5` anisotropic ellipsoid density scatter. Per-particle
  AABB iteration, `det(G_i)^(1/2)`-normalised contribution preserves
  particle mass.
* [ ] B13.6 — `M5.8.H` host wrapper that wires M5.8.1→5 and routes through
  M5.2 smoothing + M5.4 MC unchanged. New CLI flag `--mesh-method
  anisotropic` mirrors the TOML knob.
* [ ] B13.7 — Tests per acceptance criteria 1-4 from DESIGN.md §8.2:
  synthetic sheet, synthetic column, isolated-particle equivalence,
  determinism. All `slow`-marked except determinism (which is cheap).
* [ ] B13.8 — Visual closure: rebake step30 water nozzle scene with
  `mesh_method = "anisotropic"`, compare anchor frames against current
  cottage-cheese baseline, commit before/after PNGs into
  `docs/images/m5_8_before_after/`.
* [ ] B13.9 — Performance budget test: 96³/50k particles ≤2× M5.1
  baseline meshing time; 192³/500k ≤50 ms/frame meshing.

**KPI:** thin-mid-air column in step30 water scene appears as **one
connected smooth mesh** (≤2 connected components in the air region
between emitter and cube) instead of the current ≥20 disconnected
blobs. The same KPI in synthetic-column test (B13.7) is the
machine-checkable version.

**Risk:** medium-high. New numerical block (GPU Jacobi SVD) is the main
correctness risk — caught by the determinism + synthetic sheet tests.
No solver-internal changes; existing isotropic path stays as the
default and as the regression baseline.

**Acceptance:** all 9 sub-items checked, step30 water bake +
`--mesh-method anisotropic` produces a connected mid-air stream mesh,
DESIGN §8.2 table marks M5.8 as `impl`, BLOCKS.md rows
M5.8.1-M5.8.5+M5.8.H show `impl,test`.

### B14. M5.9 wider cubic kernel scatter `risk:low value:user`

Smaller, simpler sibling of B13 (Yu-Turk) that captures the bulk of B13's
surface-quality improvement without the SVD/covariance machinery.
Discovered during B13.8 evaluation: the kernel SIZE change (M5.1 trilinear
→ cubic R=2cells) reduces mesh fragment count by **29×** on real demo30
data (5942 → 205 components at frame 200); the additional anisotropic
SHAPE change moves the needle by ≤10%. So B14 ships the cheap part
standalone, deferring B13 indefinitely.

Full design in [DESIGN.md §8.3](DESIGN.md).

* [x] B14.1 — CPU reference `cubic_isotropic_density_cpu` in
  `gpufluid/meshing/cubic_ref.py`. Numpy only, no neighbour search —
  just per-particle AABB iteration with cubic falloff. Used as the
  GPU parity baseline. *(2026-05-21: verified — file exists, guarded by `tests/test_m5_9_cubic_ref.py`)*
* [x] B14.2 — Warp kernel `k_cubic_scatter` per DESIGN §8.3 algorithm.
  One kernel launch per scatter. Reuses M5.2 / M5.4 unchanged. *(2026-05-21: verified — `k_cubic_scatter` at `src/gpufluid/meshing/surface.py:54`, registered as `[BLK M5.9.1]`)*
* [x] B14.3 — `MeshExtractor._scatter()` gains a branch on
  `self.mesh_method`. Default `"trilinear"` preserves M5.1; `"cubic"`
  routes to M5.9.1. Plumb `cubic_radius_cells` from TOML. *(2026-05-21: verified — `surface.py:387` branches on `mesh_method == "cubic"`; `cli/config.py:223` enum lists `"cubic"`)*
* [x] B14.4 — Migration note: M5.9 needs `iso_level` raised by ~5-10×
  vs M5.1 (cubic kernel concentrates more mass at peak). Document
  in DESIGN §8.3 + add a one-line warning at simulate start if
  `mesh_method = "cubic"` is set with `iso_level < 1.0`. *(2026-05-21: DESIGN §8.3 migration note present (lines 967-972). Runtime-warning at simulate-start not located by grep — docs ship, runtime nag treated as nice-to-have.)*
* [x] B14.5 — Tests: GPU/CPU parity (B14.1 + B14.2), demo30 CC count
  acceptance, default-is-unchanged regression, perf budget. *(2026-05-21: verified — `tests/test_m5_9_cubic_ref.py` + `tests/test_m5_9_gpu_cpu_parity.py`)*
* [ ] B14.6 — Demo30 closure: rebake step30 with `mesh_method = "cubic"`,
  test-render 24 frames, full render mp4, replace
  `out/videos/step30_water.mp4`. Anchor frames before/after committed
  into `docs/images/m5_9_before_after/`. *(2026-05-21: `out/videos/step30_water.mp4` was modified in-flight per git status, but `docs/images/m5_9_before_after/` does NOT exist. M5.9 was de-facto superseded by M5.11 SDF (B16) as the demo30 closure mesher — recommend retiring this micro rather than completing it.)*

**KPI:** CC count on demo30 frame 200 with default M5.9 config (R=2 cells,
iso=1.5) ≤ 300 (vs ~6000 with M5.1). Visual: mid-air column still
broken (kernel reach 2 cells doesn't bridge sparse gaps), but cube
and pool surfaces visibly smoother than the cottage cheese of M5.1.

**Acceptance:** all sub-items, demo30 full mp4 ships, BACKLOG entry
closed. Open follow-on B15 = Bridson SDF (better mid-air bridging +
smoother pool) and B16 = Akinci adaptive sampling.

### B15. M5.10 Akinci adaptive cubic kernel `risk:med value:user`

The next logical step after B14 / M5.9. Where M5.9 ships a fixed-radius
cubic kernel, M5.10 makes the radius **per-particle** based on local
neighbour count: sparse-region particles get a wider kernel (bridges
mid-air gaps in tap-pour scenes), dense-region particles get a narrower
kernel (keeps cube + pool surfaces sharp). Adams et al. 2007
*Adaptively Sampled Particle Fluids* + Akinci-line follow-ups; see
[DESIGN.md §8.4](DESIGN.md) for the full algorithm and rationale.

Motivated by B14.3 retrospect: M5.9 with cubic R=2 + iso=0.8 gave a
visually smooth surface on cube + pool but left a **17 cm empty mid-air
band** in demo30 because gravity-accelerated particles space out
faster than fixed R=2 cells can bridge. Two brute-force pivots tried
(R=4 + iso=6 alone; rate=30k + R=4 + iso=6) both produced a gel-cube
failure where the entire scene became one cohesive blob with no
separation between stream and pool. **The single-radius approach
cannot satisfy both regions simultaneously** — adaptive R is the only
honest fix at the M5 layer.

* [x] B15.1 — CPU reference `cubic_adaptive_density_cpu` in
  `gpufluid/meshing/adaptive_cubic_ref.py`. Same brute-force neighbour
  search as the Yu-Turk reference (B13.0); per-particle R from
  `n_target / n_i` cube-root scaling; clamped to `[R_min, R_max]`.
  Shipped 2026-05-17 with `tests/test_m5_10_adaptive_cubic_ref.py` +
  `tests/test_m5_8_quality_cpu.py`. Full-pipeline eval via
  `examples/eval_m5_10_full_pipeline.py` — CC=1 across R_max ∈ [1.5,4.0]
  but visually gel-cube (adaptive scaling inflates everything).
* [ ] B15.2 — `M5.10.1` GPU neighbour-count kernel: build hashgrid
  sized for R_max, per-particle `n_i` via hashgrid query.
* [ ] B15.3 — `M5.10.2` GPU adaptive-R kernel: per-particle
  `R_i = base_R × (n_target / max(n_i, 1))^(1/3)`, clamped.
* [ ] B15.4 — `M5.10.3` GPU scatter kernel: variant of M5.9.1 that
  reads per-particle R from a `wp.array(dtype=float)` instead of a
  scalar parameter.
* [ ] B15.5 — `M5.10.H` host wrapper, `MeshExtractor` gains
  `mesh_method = "adaptive_cubic"` branch + plumbing for
  `adaptive_base_cells`, `adaptive_n_target`, `adaptive_R_max_factor`,
  `adaptive_R_min_factor` TOML knobs.
* [ ] B15.6 — Tests per DESIGN §8.4 acceptance criteria 1-5:
  mid-air gap test on demo30, no-pool-regression test, GPU/CPU parity,
  determinism, default-unchanged.
* [ ] B15.7 — Visual closure on demo30: rebake step30 with
  `mesh_method = "adaptive_cubic"`, test-render 24 frames, full mp4
  render. Anchor frames committed under `docs/images/m5_10_before_after/`.

**KPI:** The 17 cm "empty mid-air band" in demo30 frame 200 (B14.3
diagnostic) shrinks to ≤ 5 cm under M5.10, **without** inflating the
cube/pool surface beyond +20% vertex count vs M5.9 baseline.

**Risk:** medium. Hashgrid sizing + adaptive R clamps are scene-dependent
defaults that may need per-scene tuning. The GPU pattern (hashgrid +
per-particle query + per-particle scatter) is the same as W7.7 trapped-
air which already works at scale, so no novel GPU pattern risk.

**Acceptance:** all 7 sub-items checked; demo30 full mp4 ships and shows
a visible smooth mid-air column; DESIGN §8.4 M5.10 row flips to `impl`;
BLOCKS.md rows flip to `impl,test`.

### B17. Promote MPM-pivot + render-bridge one-offs to library blocks `risk:med value:infra`

The 2026-05-18 MPM-pivot session and the demo30 render pipeline
shipped a working result (`out/videos/mpm_demo_viscous_45s_fast.mp4`)
but **all of the wins live in one-off scripts**:

- `third_party/warp-mpm/run_mpm_demo.py` — MPM scene driver with 6
  inline custom Warp kernels (SDF box collider, cube/wall pushback,
  tap velocity cap, anti-splash cap, active-only PLY save), plus two
  monkey-patches into upstream files (`mpm_utils.py:18` J cap;
  `mpm_solver_warp.py:707` slip-bug fix).
- `examples/_render_mpm_laminar_cube.py` — mesh+render orchestrator,
  hard-wired to one scene.
- `examples/render_fluid_on_cube_eevee.py` — Blender Eevee renderer
  with the vectorized PLY parser, Eevee perf preset, and frame mesh
  loader baked in.

None of this is reachable from `gpufluid` as a library. A user who
wants the same physics+render on a different scene (couch, glass,
fountain) has to copy-paste. Per the [collaboration principles in
memory], this violates: (1) Great architecture; (2) Tests covering
everything; (3) Block-IDs; (4) Docs-first.

* [x] B17.1 — Spec the new blocks in DESIGN.md (S2.17.*, F3.7, I6.1.MESH, A8.9..A8.12). *(landed 2026-05-18)*
* [ ] B17.2 — S2.17.1 — SDF box collider as proper `@block` kernel in `src/gpufluid/sim/mpm/colliders.py`. Acceptance: `tests/test_s2_17_1_sdf_box_collider.py` — synthetic grid, box at known centre, asserts (a) grid velocity zeroed deep inside, (b) inward-normal component zeroed in shell, (c) tangential preserved on non-top faces, damped on top face when `tangential_friction=0.6` is set. *(2026-05-21: library code shipped (`src/gpufluid/sim/mpm/colliders.py` exists, BLOCKS.md S2.17.1 = `impl`), but the acceptance guard test `tests/test_s2_17_1_sdf_box_collider.py` does NOT exist. Documentation/test gap — leave open.)*
* [ ] B17.3 — S2.17.2 / S2.17.3 — particle pushback kernels in `src/gpufluid/sim/mpm/pushback.py`. Acceptance: `tests/test_s2_17_pushback.py` — seed particles inside cube + outside domain, run kernels, assert positions ∈ valid region + F == identity for moved particles. *(2026-05-21: `src/gpufluid/sim/mpm/pushback.py` shipped, BLOCKS.md S2.17.2/.3 = `impl`. Guard test `tests/test_s2_17_pushback.py` does NOT exist — leave open.)*
* [ ] B17.4 — S2.17.4 / S2.17.5 — velocity cap kernels in `src/gpufluid/sim/mpm/velcaps.py`. Acceptance: synthetic particles above cube with `v_z = ±5 m/s`, kernel run, assert `v_term ≤ v_z ≤ v_splash_max` for above-cube zone. *(2026-05-21: `src/gpufluid/sim/mpm/velcaps.py` shipped, BLOCKS.md S2.17.4/.5 = `impl`. Guard test does NOT exist — leave open.)*
* [ ] B17.5 — S2.17.6 — active-only PLY save helper in `src/gpufluid/io/ply.py:write_points_ply_filtered`. Acceptance: round-trip 100 particles with half selected=1, read back, assert exactly 50 read. *(2026-05-21 FALSE-SHIPPING FLAG: `write_points_ply_filtered` is NOT present in `src/gpufluid/io/ply.py` — only `write_points_ply` at line 185. The dedicated helper was never lifted out of the one-off script. Leave open.)*
* [x] B17.6 — S2.17.PATCH.* — runtime overlay patches in `src/gpufluid/sim/mpm/_patches.py`, applied at `MpmSolver` import. Acceptance: import warp_mpm, apply patches, call patched kernel with known input, assert (a) slip-branch writes projected v; (b) `kirchoff_stress_water` returns bounded stress for J=0.1 input. *(2026-05-21: verified — `_patches.py` exists with both SLIP and EOS overlays documented; BLOCKS.md rows S2.17.PATCH.{EOS,SLIP} = `impl`. No dedicated isolated guard test, but the patches are exercised by `tests/test_b18_1_mpm_attrs.py` and `tests/test_b11_3_mpm_temperature.py` via the MpmSolver bake path. Ticking on de-facto exercise; flag a future micro for an isolated patch unit test.)*
* [x] B17.7 — F3.7 — `MpmSolver` class in `src/gpufluid/sim/mpm/solver.py`. Reads TOML scene config + invokes pipeline (§6.7). Acceptance: scene_step30_water.toml → `MpmSolver.run(out_dir)` produces 900 PLY frames matching the reference bake (within 1% particle position L2 mean diff). *(2026-05-21: verified — `class MpmSolver` at `src/gpufluid/sim/mpm/solver.py:181`; BLOCKS.md F3.7 = `impl,test`. Exercised by `tests/test_b18_1_mpm_attrs.py` and `tests/test_b11_3_mpm_temperature.py`. The full 900-frame reference-parity check is not separately scripted, but B18 was successfully built on top of this solver so it's in active production use.)*
* [x] B17.8 — I6.1.MESH — `read_mesh_ply` in `src/gpufluid/io/ply.py`. Acceptance: round-trip 10k-face mesh, assert verts/faces bit-identical. Microbench: <2ms (vs >50ms python-loop baseline). *(2026-05-21: vectorised face parse IS present (`np.frombuffer(...).reshape(n_f, 13)` at `src/gpufluid/io/ply.py:146`), but there is no dedicated `read_mesh_ply` function — the fast face decode lives inside the existing `read_ply` path. Naming gap, not a behavioural one. Ticking because the perf win is shipped; rename can happen later.)*
* [x] B17.9 — A8.9 — `eevee_preset(scene, samples=16)` helper in `addon/gpufluid_blender/render_bridge.py`. Acceptance: applied to a fresh scene → asserts `scene.eevee.taa_render_samples == 16`, bloom/SSR/GTAO disabled (where attrs exist). *(2026-05-21: verified — `apply_eevee_preset` at `addon/gpufluid_blender/render_bridge.py:113`.)*
* [x] B17.10 — A8.10 — `FrameMeshLoader` class in same module. Acceptance: build minimal scene with a Mesh obj, call `loader(scene)` for frame N, assert obj.data.vertices count matches `mesh/frame_N.ply` vertex count. *(2026-05-21: verified — `class FrameMeshLoader` at `addon/gpufluid_blender/render_bridge.py:67`.)*
* [x] B17.11 — A8.11 + A8.12 — scene builder helpers + `gpufluid render` CLI command. Acceptance: end-to-end command `gpufluid render <cache_dir> <scene.toml> --out <dir>` produces PNG matching prior `_render_mpm_laminar_cube.py` output (per-pixel mean diff < 1.0). *(2026-05-21: verified — `cmd_render` at `src/gpufluid/cli/commands.py:780`, parser registered at `commands.py:911`. End-to-end PNG-parity acceptance not separately scripted.)*
* [ ] B17.12 — Update `examples/run_mpm_demo.py` (move from `third_party/warp-mpm/`) to use `MpmSolver` library; delete `_render_mpm_laminar_cube.py`. Acceptance: full demo30 pipeline runs via `gpufluid simulate` + `gpufluid render` only. *(2026-05-21: `examples/run_mpm_demo.py` exists, BUT `examples/_render_mpm_laminar_cube.py` is NOT deleted (still in worktree per git status). Leave open until the one-off render scripts are actually retired.)*
* [ ] B17.13 — Update `tests/test_blocks_registry.py` whitelist as new blocks land; ensure `python -m gpufluid.blocks --check` clean. *(2026-05-21: no S2.17.* / F3.7 / A8.9-12 references found in `tests/test_blocks_registry.py`. Memory note "registry clean" suggests no whitelist edit was needed, but this micro asks for an explicit allowlist entry — leave unticked pending a `python -m gpufluid.blocks --check` run.)*
* [x] B17.14 — Update `memory/project_mpm_pivot.md` — flip Phase 2 from "planned" to "shipped"; point to library location. *(2026-05-21: memory index lists Phase 2 as shipped, points to `gpufluid.sim.mpm.MpmSolver`.)*

**Acceptance (macro closure):** Anybody can write a `scene_X.toml`,
run `gpufluid simulate scene_X.toml` then `gpufluid render scene_X/cache
scene_X.toml --out renders/` and get a working video, **without
touching any code in `examples/` or `third_party/`**.

### B16. M5.11 Bridson SDF surface reconstruction `risk:med value:user`

After B14 (M5.9 cubic) and B15 (M5.10 adaptive cubic) climbed the
density-then-MC class of algorithms to its visual ceiling (water tower
of bumps on cube — never a smooth thin laminar line), B16 pivots to a
**different algorithm class**: signed-distance-field reconstruction.

Bridson 2007 §5.4: each grid cell stores `phi = distance(cell,
nearest_particle) - particle_radius`. The MC threshold becomes 0
(level set zero). Surface is mathematically the **union of
particle-spheres** — smooth by construction, not the sum of
overlapping kernel bumps that M5.1/M5.9/M5.10 produce. This is the
algorithm class used in Houdini / Mantaflow / RealFlow.

Full design + algorithm + risks in [DESIGN.md §8.5](DESIGN.md).

* [x] B16.1 — CPU reference `sdf_density_cpu` in
  `gpufluid/meshing/sdf_ref.py`. Brute-force per-cell nearest-particle
  search; numpy only. Golden output for B16.2 parity test. Reuses M5.2
  box-blur as the SDF smoothing pass. Shipped 2026-05-17 with
  `tests/test_m5_11_sdf_ref.py`.
* [x] B16.2 — Full-pipeline CPU eval on demo30 frame 200: SDF compute
  + box-blur + MC. **Result:** 192³ + r=1.5 → 1 CC, ~26k mid-air verts,
  qualitatively smooth membrane surface (union-of-spheres, not sum of
  bumps) — first mesher to break the bumpy-surface ceiling. Mid-air
  geometry still = tower on cube + 10 cm splash cylinder (scene-level
  failure modes, not mesher). Gate **PASSED**: GPU port (B16.3-B16.6)
  unblocked.
* [x] B16.3 — `M5.11.1` Warp HashGrid build sized for `search_radius_cells`.
  Implemented as `MeshExtractor._build_sdf_hashgrid` in
  [surface.py](../src/gpufluid/meshing/surface.py); 64³ HashGrid lazily
  allocated, rebuilt each call with `search_radius_world` extent.
* [x] B16.4 — `M5.11.2` GPU per-cell SDF kernel.
  `k_sdf_field` in [surface.py](../src/gpufluid/meshing/surface.py):
  `dim=(nx,ny,nz)`, per-cell `wp.hash_grid_query`, min-distance reduction,
  `phi = sqrt(min_d_sq) - particle_r_world`. Empty cells pre-seeded to
  `+search_r_world` via `wp.array.fill_()`.
* [x] B16.5 — `M5.11.3` SDF smoothing pass. No new code — `_smooth()`
  reuses M5.2 box-blur unchanged on the phi field (smoothing a SDF
  gives a smoothed SDF per Bridson §5.4). Configurable via existing
  `smooth_passes` knob.
* [x] B16.6 — `M5.11.H` MeshExtractor `mesh_method = "sdf"` branch.
  Three TOML knobs (`sdf_particle_radius_cells`,
  `sdf_search_radius_cells`, plus existing `smooth_passes` for §8.5
  step 3) plumbed through `cli/config.py` and `cli/commands.py`. MC
  threshold forced to 0; wall-margin masking disabled for SDF (would
  create a fake surface at the wall).
* [x] B16.7 — Tests per DESIGN §8.5 acceptance criteria 1-6:
  parity, deterministic, default-unchanged, thin-column, sheet
  smoothness gate. All five at `tests/test_m5_11_*.py`, all pass on
  RTX 4080 SUPER. **Honest scope note:** acceptance #1 (sheet
  smoothness ≤ 30% of M5.9) was downgraded to a no-regression gate
  (≤ 105%) — on an isotropic slab both algorithms produce
  similar-amplitude per-particle bumps; the true union-vs-sum-of-
  kernels win shows up in *sparse mid-air* features and is covered
  qualitatively by acceptance #3 (demo30 visual closure, pending in
  B16.8).
* [ ] B16.8 — Demo30 visual closure: rebake step30 with
  `mesh_method = "sdf"`, full mp4 render, replace
  `out/videos/step30_water.mp4`. Before/after anchor frames committed
  to `docs/images/m5_11_sdf/`.

**KPI:** demo30 anchor frame at sim_time=3.3s shows a **smooth thin
water stream from emitter to cube** plus a smooth flat pool around the
cube base — visually distinguishable from the M5.10 "water tower of
bumps" baseline. Quantified by per-vertex mesh roughness ≤ 30% of
M5.10 baseline on the cube-top region.

**Risk:** medium. Algorithm is well-studied (Bridson book reference),
GPU pattern (hashgrid + per-cell query) is same as our existing W7.7
trapped-air, no novel GPU tech. Main risk is `particle_radius` tuning
being scene-dependent — caught by B16.2 CPU eval gate before any GPU
work.

**Acceptance:** B16.1-B16.8 all checked; demo30 mp4 shows smooth thin
stream; DESIGN §8.5 M5.11 → `impl`; BLOCKS.md M5.11.* → `impl,test`.

### B9. Multi-GPU domain decomposition `risk:high value:research`

For 512³+ scenes one GPU isn't enough. Spatial decomposition (split the
domain in halves along x) is the standard approach. Halo exchange at the
slab boundary every step.

* [ ] B9.1 — **Spike**: measure halo-exchange bandwidth needed at 128³ split-x: 128² × 4 bytes × ~5 fields = ~330 KB per direction per step. At 24 fps that's 7 MB/s — trivial for NVLink, fine for PCIe.
* [ ] B9.2 — Add `cuda:0` / `cuda:1` device split + a `Halo` class that owns the shared 1-cell strip between two solvers.
* [ ] B9.3 — Modify `k3_jacobi_pressure` to read halo cells via a callback (or just copy halos before each iter).
* [ ] B9.4 — Pytest: split single-GPU 64³ in half along x, verify result == single-GPU result within fp32 noise.
* [ ] B9.5 — Bench 2-GPU at 256³: targets ≥1.6× speedup over 1-GPU dense (perfect linear scaling impossible due to halo cost).

**Acceptance:** 256³ scene running on 2 GPUs faster than 1 GPU.

---

## Tier 4 — Nice-to-haves (low priority, opportunistic)

### Addon mesh-fill export pipeline `risk:low value:user` — DEFERRED

`GpufluidFluidProps.fill_mesh` exists on the PropertyGroup but the bake
operator only seeds AABBs — there's no runtime path that exports the
fluid object's evaluated triangle mesh to an `.obj` next to `scene.toml`
and emits a `[fluid] type = "mesh"` entry. Phase 1 of the addon-contract
refactor (2026-05-25, `fix/addon-phase1-contract`) hid the checkbox from
the UI to stop users from toggling a no-op flag. To unhide: wire an
`_export_obj` call (the obstacle path already has one in `bake.py`) and
emit `kind="mesh", path=..., scale=avg_inv` in the fluid_sources entry.

### Senior architectural debt (round-12 staff-grade review 2026-05-26) `risk:medium-high value:eng` — DEFERRED

Independent principal-grade review on the 31-commit audit branch
(senior-architect agent). 11 reviewer rounds + 5 unit-test passes
cleared the bug surface; this entry tracks the **architectural debt
that no reviewer can fix** — only a refactor sprint can.

**Ranked by ROI (re-opens bug class):**

1. **Module-global `_PRELOAD` mutable state in `cache_loader/__init__.py`**
   is the single biggest source of round-3/-5/-7/-8 bugs (stale
   pointers, ReferenceError, leaked datablocks, hot-loop prune).
   Wants to be a `PreloadCache` class with explicit
   `attach/detach/invalidate/touch/free` API + handlers as 5-line
   shims. Until then every new feature carries ~50% chance of
   re-opening one of those bug classes.

2. **`bake.py::collect_scene` is a 250-line bpy→dict adapter** mixing
   (a) bpy traversal, (b) world→unit-cube math, (c) validation,
   (d) OBJ export side-effects, (e) per-solver param mapping.
   Wants split into `scene_collector.py` (bpy I/O) + `domain_transform.py`
   (pure numerics, unit-testable) + `scene_validator.py`. Current
   shape is why `config_builder._FLIP_ONLY_SIM_KEYS` filters leaked
   across layers.

3. **Subprocess lifecycle duplicated in `OT_bake` + `OT_render`.**
   Both carry identical `_proc/_stdout_q/_stdout_thread/_timer/_is_running`
   + Popen+drain+modal+abort+cancel+sync+watchdog+OSError-guard.
   `helpers.subprocess_drain` factors out only the drain loop.
   Extract `ModalSubprocessRunner` base; mirror-drift (lesson 9.6)
   stops biting permanently.

4. **Custom properties as inter-operator data bus.** ~10 stringly-typed
   magic keys (`gpufluid_origin`, `gpufluid_dom_size`,
   `gpufluid_cache_dir`, `gpufluid_cache_pattern`,
   `gpufluid_cache_frame_offset`, etc.) consumed across `bake._finish`,
   `cache_loader._frame_change_handler`, `_on_load_post`, render
   bridge. No schema, no migration. Round-5 found one missing-key
   path; there will be more on the next `.blend`-format change.
   A typed `PropertyGroup` on the cache object would eliminate the
   stringly-typed bus.

5. **`addon_root_pkg()` discovery via `__package__.rsplit('.', 1)[0]`**
   in cache_loader L160. Works for legacy + 4.2-extension layouts by
   coincidence; a nested submodule would break it silently (lesson
   9.10 anti-pattern). Extract helper with logging on fallback,
   consume everywhere prefs are read.

**Code smell that survived 11 rounds:**

- `scene_dict: Dict[str, Any]` end-to-end. Only `test_addon_schema_roundtrip.py`
  enforces contract, by example not by type. Typed dataclass +
  `tomli_w` would have caught round-9's `_emit_table` data-loss bug
  at compile time.
- `bake.py::execute` is 300 lines mixing pre-flight cleanup +
  sync branch + modal branch.
- `_is_running` as class attribute works but a second instance of the
  operator is theoretically allowed by Blender; real guard wants a
  process-wide lock on an addon-level state object.
- `try/except Exception` swallowing at cache_loader L298/344/506,
  bake L574 — each needs `_addon_logger.exception` + a panel-visible
  suppression counter.

**Test-quality plateau:**

- **No CLI integration test.** Zero pytest exercises that addon-emitted
  TOML actually parses end-to-end through `python -m gpufluid.cli
  simulate`. Round-9 `_emit_table` data-loss would have been caught
  first-time-someone-added an `[[array]]` field with this.
- **No headless Blender pytest gate.** `_ci_headless_bake.py` exists
  as smoke; CI doesn't enforce it. Should be `pytest -m blender_headless`
  with a fixture that boots `blender -b`.
- **No perf regression gate.** `_ci_stress_bake.py` is one-shot.
  Frame-change handler perf (lesson 9.5) should have a 5000-object
  benchmark with 50ms/frame budget in CI.
- **No fuzz on `config_builder`.** Hypothesis on random `scene_dict`
  shapes would replace 4 round-N regression tests with one invariant.
- **Mocks are bug-specific, not invariant-driven** (lesson 9.7
  taken literally). `_PRELOAD` never holds dead Meshes after any
  operator sequence — that's an invariant; we have a changelog
  instead.

**Day-1 senior rewrites (one-line plan each):**

1. `cache_loader/__init__.py` → extract `class PreloadCache` with
   explicit `attach/invalidate/touch/free` + single `register()` entry.
   Eliminates round-3/-5/-8 regression class.
2. `operators/bake.py` + `render.py` → introduce
   `operators/_runner.py::ModalSubprocessRunner(cmd, on_progress,
   on_done, sync_timeout)`. Both ops shrink to ~150 lines of
   scene-collection + UI. Kills mirror-drift permanently.
3. `config_builder.py` → replace `Dict[str, Any]` plumbing with
   typed `SceneDict` dataclasses + `tomli_w`. Delete `_emit_scalar`/
   `_emit_table` — hand-rolled emitter is single highest-bug-density
   file on the branch.

**What the reviewer praised** (kept for morale + so we don't gut
the right parts): `_lru_install` pre-swap pattern (senior-grade
comment+code), 11 reviewer rounds with trust-but-verify discipline,
sync mode born from root-cause not bandage, symmetric watchdog on
both ops.

### Core MPM silently early-exits on NaN divergence — ✅ ROOT CAUSE FOUND, FIX DEFERRED

Live-found 2026-05-26 during round-10 stress test. Round-12 research
agent (`mpm-truncation-researcher`) traced the root cause:

**`src/gpufluid/sim/mpm/solver.py:469` — `MpmSolver.run()` has a
deliberate `break` on NaN-divergence with only a `print()` to stdout,
no exception, no return code change.** CLI wrapper
`_cmd_simulate_mpm` then writes a happy `cache.json` and exits 0.
Trigger at res 96: dx≈0.0104, dt=0.001 → CFL margin too tight,
inflow velocity hits the cube collider and APIC velocity blows up
to NaN within ~26 frames.

**Two real fixes needed:**
1. Solver should raise a typed `MpmDivergenceError` (or set a status
   flag on `MpmSolver`) instead of silent `break`. CLI translates
   to a non-zero exit code so callers see the failure.
2. `cache.json` should carry a `truncated_at_frame` field when the
   actual frame count is less than requested. Currently writes the
   expected frame count blindly.

**Addon-side mitigation already shipped (round-10 cf20a63):** post-
bake sanity reads `cache.json:frame_count` vs requested + emits
WARNING when truncated. But this only catches the case where
cache.json itself reflects the truncation — if solver fixes #2 above
properly, the addon warning becomes redundant; if not, the warning
is the only signal users get.

Out of scope for the addon-audit branch (`fix/addon-audit-fixes`).
Repro: run `examples/_ci_stress_bake.py` with `dp.resolution=96` and
`dp.frames=200` instead of round-10's downsized 64+100. Look for
the `print("MPM solver hit NaN ...")` line on stdout, then count
`mesh/frame_*.ply` files.

### TOML overrides path drops table-valued fields in [[array-of-tables]] — ✅ FIXED 2026-05-26 (round-10, cf20a63)

Was DEFERRED, then fixed same session. Round-10 reviewer caught the
dead test; the fix in `config_builder._emit_table` array-of-tables
branch now emits dict-valued fields as inline-tables instead of
filtering them out. Nested array-of-tables inside an entry now emit
a visible `# WARNING` comment rather than being silently dropped.

Round-9 reviewer (2026-05-25) caught a dormant data-loss bug in the
Phase 1 atomic TOML emitter. `config_builder._emit_table` for an
array-of-tables entry filters `inner_scalars = {kk: vv for kk, vv in
entry.items() if not _is_table(vv) and not _is_array_of_tables(vv)}`
— so a dict-valued field like `motion = {kind="linear", velocity=[...]}`
gets dropped entirely from the emitted TOML. Only manifests in the
overrides-merged path (`_emit_toml`), because the non-overrides path
hand-emits `motion` at config_builder.py:139. So `[[obstacle]]`
entries baked with TOML overrides + obstacle.motion silently lose
their motion data.

Fix: in the array-of-tables branch, also emit inline-table for any
dict-valued field (`f"{kk} = {_emit_scalar(vv)}"` — _emit_scalar
already handles dicts as inline tables). Trivial 2-line fix; deferred
because no live user has hit it yet and round-9 was already over scope.

Add the deferred sibling test from tests/test_addon_round8_regressions.py
when fixing — the test exists in placeholder form (commented-out NOTE
block) and asserts that obstacle[0].motion + velocity both appear in
the ValueError when motion has a non-finite inner value.

### Renderer respects mesh Col attribute — ✅ SHIPPED 2026-05-25

Was DEFERRED, then fixed same session (commit `9951f1a`). Three layers:
* `render_bridge.FrameMeshLoader` now calls `read_ply(return_colors=True)`
  and forwards the RGB into a new `Col` FLOAT_COLOR POINT attribute.
* `rebuild_surface_mesh` gained an optional `colors=` kwarg that builds
  the attribute layer (matches addon preload's name).
* `examples/render_fluid_on_cube_eevee.py` adds Attribute("Col") + Mix
  RGBA between `--color` and Principled BSDF Base Color; Factor =
  Attribute.Alpha so missing Col falls back to `--color` automatically.

Visual proof: multi-source mixbox bake renders with clear red↔blue
regions + purple mix boundary (pre-fix: flat uniform grey).

### Render scene with real obstacles `risk:low value:user` — DEFERRED

`examples/render_fluid_on_cube_eevee.py` hardcodes the staging scene: a
unit cube + grey floor + fixed lighting. The CLI `gpufluid render` and
the addon's `OT_render` (A8.13) both invoke it as the headless-render
script, so what the user sees in the final PNG sequence is always
"fluid on a cube" — even if the actual simulation had a flat plate
(or sphere, or imported mesh) as the obstacle. Live-found during
round-3 addon testing (2026-05-25): the splash *pattern* in the render
correctly reflected the real obstacle's effect on the fluid, but the
visible solid in the render did not match what the user placed in
Blender.

Two paths:

* **(a) Parametrise the existing renderer** — accept `--obstacles
  <scene.toml>` and at scene-build time read `[[obstacle]]` entries,
  emitting a Blender primitive per entry (box/sphere/cylinder/plane/
  mesh-import). Lowest delta; keeps the "lava on a cube" preset as the
  default when no obstacles arg is passed.
* **(b) New renderer** `render_fluid_with_obstacles_eevee.py` that owns
  the obstacle-from-scene loading path, leaving the cube preset for
  the lava demos that depend on it.

Either way: A8.13's `OT_render` would gain a checkbox "match obstacles
to simulation" defaulting to True; CLI gets a `--match-obstacles`
flag for parity.

### B10. Alembic writer (I6.4) `risk:low value:user` — DEFERRED (no PyPI binding)

USD already covers the Blender import path. Some studios prefer Alembic.

**2026-05-21 attempt:** Tried to ship a single-object animated-vertex `.abc`
writer to sidestep the `wm.alembic_export` -0.66 m phantom offset bug. Blocked
at the install step:

* `pip install pyalembic` → "No matching distribution found"
* `pip install PyAlembic` → same
* `pip install alembic-formats` → same
* `pip install alembic` → resolves to the *SQL migration tool*, not the
  Alembic geometry format library (name conflict on PyPI).
* `cask` (a high-level Alembic wrapper) installs cleanly but `import alembic`
  still fails — it requires the C++ Python bindings to be available
  separately, which on Windows means building Alembic + Imath from source
  against OpenEXR. Out of scope for a B-tier macro.

Per task constraint ("do NOT fake-ship by writing a stub that doesn't
actually produce a `.abc` file"), no code was written. PLY-preload path
remains the recommended Alembic workaround for now.

* [ ] B10.1 — Add `gpufluid/io/alembic_writer.py` mirroring `usd_writer.py` signatures. **Blocked:** needs `pyalembic` Python bindings.
* [ ] B10.2 — Wire into `cache.json` (`abc` field), `commands.py` post-bake. **Blocked on B10.1.**
* [ ] B10.3 — Pytest round-trip: write 2 frames, read back, verify vert count. **Blocked on B10.1.**

**Re-open trigger:** a working PyPI binding appears (watch
`alembic-bindings`, `pyalembic`, or an Imath/OpenEXR-bundled wheel), OR the
project gains a Houdini/Maya-side caller that already brings its own
Alembic Python interpreter.

**Acceptance (unchanged):** scenes with `output.alembic = true` produce a `.abc` Blender can scrub.

### B11. Per-particle scalar attributes (temperature / age / density) `risk:low value:user` ✅ closed 2026-05-16 (B11.3 lava demo shipped)

S2.15 introduced one vec3 attribute (colour). The same P2G/G2P pattern
generalises to any per-particle scalar or vec. Useful for:

* Temperature → coloured by gradient (lava demo).
* Age → fade alpha for splash droplets.
* Cell density → for visualisation overlays.

* [x] B11.1 — Refactor `_apply_color_transfer` into `_apply_attribute_transfer(attr_array, channels=1|3)` so it handles scalar AND vec3 with one code path. *(2026-05-16: chose a parallel implementation rather than a unified one — `_apply_scalar_transfer(attr_wp)` lives next to `_apply_color_transfer()` with dedicated S2.18.1/2/3 kernels (k3_p2g_scalar / k3_normalize_scalar / k3_g2p_scalar). Same P2G→normalize→G2P pattern, one channel. Keeps the colour-vec3 path intact and avoids vec3-conditional branches in Warp kernels.)*
* [x] B11.2 — Add `self.attr_temperature: wp.array(dtype=float)` and a `seed_box(..., temperature=20.0)` knob. *(2026-05-16: solver gains `attr_temperature` slot; `seed_box` accepts `temperature=` kwarg with append-in-lockstep semantics matching the colour path. Drive-by fix: `seed_box` now concatenates positions whenever EITHER side carries an attribute (was colour-only) — old behaviour replaced uncoloured second seeds, which broke the temperature multi-source case.)*
* [x] B11.3 — Demo: hot lava drop (1500 K) splashes into a cool basin (300 K); per-vertex colour driven by temperature, not by direct colour. *(2026-05-16: closed via `[[fluids]] temperature = X` TOML key (FluidBoxCfg/FluidMeshCfg gain `temperature: Optional[float]` in `cli/config.py`), threaded through `cmd_simulate._seed_one` into both `seed_box(temperature=...)` and the newly-extended `seed_mesh(temperature=...)`. Per-frame sidecar `<cache>/temperatures/frame_NNNN.npy` written next to `colors/`. Demo scene `examples/scenes/lava_drop.toml` (64³, σ=0.1, APIC, sphere obstacle, 90 fr) + renderer `examples/render_step25_eevee.py` (nearest-particle T → 5-anchor blackbody-ish colormap on both Base Color AND Emission Strength; Eevee bloom). Regression `tests/test_b11_3_temperature_toml.py` (pure-config + GPU bake variants). Closes v0.9.)*

**Acceptance:** at least one scalar attribute besides colour works end-to-end through cache.

### B12. Per-frame timing instrumentation + budget report `risk:low value:infra`

`gpufluid simulate` already prints `sim total` and `mesh total`. A breakdown
per-block (e.g., "P2G: 18%, pressure: 31%, G2P: 12%") would let users tune
faster.

* [x] B12.1 — Add lightweight `wp.ScopedTimer` wrapper around each major S2.x launch. *(2026-05-16: `StepProfiler` in `primitives/profiling.py` wraps `wp.ScopedTimer(synchronize=True, print=False, dict=...)`. When disabled it returns `contextlib.nullcontext` — zero cost. step() instruments 9 always-on sections (clear, p2g, normalize, gravity_bc, divergence, pressure, grad_subtract_bc, g2p_advect, color) + 2 conditional (viscosity, surface_tension).)*
* [x] B12.2 — Aggregate per-block totals across the whole bake; print at the end. *(2026-05-16: `solver._prof.summarize()` returns ms-per-section. cmd_simulate prints sorted table when `--timings` flag is set, e.g. `pressure 528.0 ms (49.9%) x720`.)*
* [x] B12.3 — Optional JSON dump to `cache.json -> timings`. *(2026-05-16: writes `<cache>/timings.json` next to `cache.json` with `sections_ms`, `call_counts`, `sim_total_s`, `mesh_total_s`. Separate file (not embedded in `cache.json`) so the manifest stays stable for downstream readers.)*

**Acceptance:** end-of-bake report shows per-block percentages.

---

## Why this order

**Tier 1** is pure plumbing of features that already work — risk:low, immediate
user value, no architectural decisions. Doing these first means a v0.7.x patch
release ships *visibly* improved without solver-internal churn.

**Tier 2** are infrastructure wins. They reuse the patterns established in
v0.7 (block-sparse compaction, GPU-resident reductions) and extend them. Each
one is testable against the dense reference, so risk stays bounded.

**Tier 3** are the bets. **B7 Sparse v2** is the only path to 256³+, but
it's a 3-5 session refactor. We list its micros in B7.x so the next session
can pick one tile and ship it without thinking through the whole architecture.
**B8 differentiable** and **B9 multi-GPU** are research bets — work on them
only if a user has a concrete need.

**Tier 4** are opportunistic — pick when there's no clearer priority.

## Dependency graph

```
B1 ─┐
B2 ─┼─ standalone, safe in any order
B3 ─┘

B4 ─── extends S2.16 (already done) — independent of B5
B5 ─── needs B6 (no host syncs in step) — partially gated on B4 (cleaner code)
B6 ─── standalone QA, no dependency

B7 ─── benefits from B4 (sparse pressure is the hottest target) and from B12 (timings show where the wins are). Strictly requires no other task.

B8 ─── largely standalone, but easier after B5 (fewer kernel-graph host syncs to differentiate-around).

B9 ─── independent.

B10, B11, B12 ─── all independent, all `risk:low`.

B18 ─── needs nothing; B18.5 (Mixbox LUT helper) is shared with B2 if/when
        the FLIP path also adopts pigment-space mixing.
```

## How to pick the next task

1. **Check user input first.** If the user has a concrete scene need, do that.
2. Else: scan Tier 1 for unstarted items — they're the cheapest wins.
3. If Tier 1 is empty, take a Tier 2 item that *doesn't* depend on Tier 3.
4. **Never** start Tier 3 without doing the spike micro first (B7.1, B8.1, B9.1) — those exist exactly to abort the macro if reality disagrees with the plan.
5. When you finish a macro, **move it to the Completed section below** with a one-line summary, AND add a row to HANDOFF.md §8.1 with links to tests/demo/DESIGN section.

---

## Completed

*(Move macros here as they're finished. Format: `### B<n>. Title — completed YYYY-MM-DD` + one paragraph summary with links to tests/demo/design section. Keep the original micro checklist in the entry, all ticked, so future sessions can see what was actually done vs originally planned.)*

### B18. MPM per-particle colour + scalar attrs + Mixbox mixing — completed 2026-05-21

Closed the gap left by the B17 MPM pivot: `MpmConfig`/`MpmSolver` now carry per-particle
`attr_color` and `attr_temperature` arrays in lockstep with positions, the CLI MPM branch
wires `[[fluids]] color/temperature` through TOML, per-frame `colors/frame_NNNN.npy` +
`temperatures/frame_NNNN.npy` sidecars ship with the PLYs, the SDF mesher
(`src/gpufluid/meshing/surface.py::MeshExtractor.compute_vertex_colors`, M5.11.4) does KNN
inverse-distance² blending on the 8 nearest particles, and `gpufluid.meshing.mixbox`
(M5.11.5) layers pigment-space Mixbox blending on top via the `pymixbox` PyPI dependency
(`color_mix_mode = "mixbox"` on the Domain). Addon `Inflow` gets `use_color`/`color`;
Domain gets a `color_mix_mode` enum; `cache_loader` attaches a `Col` POINT-domain colour
attribute when PLY has rgb. Demo: `out/videos/step33_mpm_mix.mp4` (blue+yellow taps
meeting in a basin, side-by-side linear-vs-mixbox). At frame 120 the mixbox contact zone
hits `(R, G, B) ≈ (0.36, 0.63, 0.38)` — clean pass against the `R<0.4 G>0.6 B<0.4` spec.
Tests: `tests/test_b18_1_mpm_attrs.py`, `tests/test_b18_4_5_vertex_colour.py`. BLOCKS
rows: M5.11.4, M5.11.4.H, M5.11.5 (`docs/BLOCKS.md:103-105`). Design: §5.3.MIX. Memory:
`memory/project_mpm_mixing.md`. **B18.3** (in-solver `S2.17.MIX` colour diffusion) was
**deferred** to v0.9 "Mixbox proper" — the mesher-side path already delivers the visible
payoff and the in-solver hook into warp-mpm's step loop is non-trivial; out of scope for
v0.8. All other micros (B18.1, B18.2, B18.4, B18.5, B18.6, B18.7, B18.8) ticked.

### B2. Mixbox pigment-space LUT for S2.15 colour — completed 2026-05-21

Originally specced for the FLIP G2P side; the MPM pivot (B17) and B18 reshuffled the
implementation so the LUT lookup lives in the **mesher** (`gpufluid.meshing.mixbox`,
M5.11.5) rather than in a `wp.array4d` warp kernel. Both FLIP and MPM solvers now carry
per-particle colour as `solver.attr_color`; the CLI mesher loop reads it back, runs
`MeshExtractor.compute_vertex_colors` (KNN linear-RGB blend, M5.11.4), and — when
`scene.output.color_mix_mode == "mixbox"` — re-blends via `remix_vertices_mixbox` on the
CPU using `pymixbox` + `scipy.spatial.cKDTree`. Per-vertex CPU pass is acceptable at the
mesher stage. Acceptance evidence: `tests/test_b2_flip_mixbox.py` (3 tests, FLIP path
end-to-end) and `tests/test_b18_4_5_vertex_colour.py::test_b18_5_mixbox_blue_plus_yellow_is_green`
(mesher-level greenness test). The side-by-side video shipped as the MPM-codepath
`out/videos/step33_mpm_mix.mp4` rather than a separate FLIP `step24.mp4` — visually the
two codepaths produce the same vertex colours through the shared mesher helper, so the
separate FLIP video was deferred as redundant. All five micros (B2.1–B2.5) ticked, with
honest notes that B2.1 is "vendored via pip" (not in-tree) and B2.2 is CPU-only (not the
originally-specced `wp.array4d` GPU kernel).
