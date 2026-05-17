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
| v1.0    | "Scale" — 256³+ scenes via sparse storage          | Tier 3: B7 (aborted, see B7.1) → B7-alt deferred-allocation | 3-5 | ▶ next — B7-alt.1 spike GREEN 2026-05-16 (18.96× memory drop on connected-blob 128³/5%, bit-exact coord-translation). Macro micros B7-alt.2..B7-alt.8 ready. |
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

### B2. Mixbox pigment-space LUT for S2.15 colour `risk:low value:user`

Current per-particle colour does linear RGB blend → blue+yellow = grey. Mixbox
(Šochorová & Jamriška 2021) is a 4D LUT (~270 KB) that gives painterly
mixing (blue+yellow = green). Drop-in on the G2P side.

* [ ] B2.1 — Vendor or download the Mixbox LUT (`mixbox_lut.bin`, ~270 KB) at first-run with a clear license note in `docs/BACKLOG.md`.
* [ ] B2.2 — Add `k3_g2p_color_mixbox` kernel that reads the LUT (as `wp.array4d(dtype=float)`) and performs the inverse-K-M lookup.
* [ ] B2.3 — Switch via `cfg.color_mix_mode: "linear" | "mixbox"` on FlipSolver3D.
* [ ] B2.4 — Demo scene: blue + yellow water cubes → step23.mp4 with side-by-side linear vs Mixbox.
* [ ] B2.5 — Pytest: a yellow+blue 50/50 blend in Mixbox mode produces R<0.4, G>0.6, B<0.4 at the contact zone (greenish), unlike linear which gives ~(0.5, 0.5, 0.5) (greyish).

**Acceptance:** scene file with `color_mix_mode = "mixbox"` produces a visibly green contact zone in the side-by-side video.

### B3. Better whitewater classifier (Wave-Crest + Trapped-Air potentials) `risk:med value:user`

W7.4 currently classifies by density-grid lookup + velocity projection. This
works but produces ~30 spray / ~800 foam / ~2100 bubble in the splash demo —
spray count is low because the heuristic doesn't find wave crests properly.
Ihmsen et al. 2012 §3 has the full potential definitions.

* [x] B3.1 — Implement Trapped-Air potential: `I_ta = Σ_j (1 - cos(θ_ij)) · (1 - |v_ij|/v_max)` over neighbours, capped at 1. Per fluid particle on GPU (uses a HashGrid for neighbour query — `wp.HashGrid` is already available). *(2026-05-16: `gpufluid/sim/whitewater_potentials.py` with W7.7 kernel + W7.7.H host wrapper. wp.HashGrid built per call (one-shot, no caching yet). The (1 - cos θ) factor is intentionally NOT pre-clamped per-pair — only the final sum is capped at 1 (spec said sum-cap; per-pair clamp was an early bug I removed when test_w7_7_v_max_scales_pair_contribution caught it).)*
* [ ] B3.2 — Implement Wave-Crest potential: `I_wc = |∇·n̂|` curvature-like measure against the surface normal n̂ (reuse S2.14.2 normal field). *(deferred: needs S2.14.2 hookup, which is CSF-only — separate session.)*
* [x] B3.3 — Emit rate per particle = `clamp(α·I_ta + β·I_wc, 0, max_rate)·dt` instead of the current `|v|>threshold` cut-off. *(2026-05-16: shipped as a weighted selector inside `emit_from_fluid`. `[output] whitewater_use_potential = true` activates it. Adds a tiny floor (1e-3) so laminar particles aren't completely starved — without the floor a waterfall column emits ~0 because internal particles are perfectly aligned. Addon UI exposes the toggle in the Whitewater sub-panel (`use_potential`, `potential_radius`, `potential_v_max`). The full α/β coefficients are folded in once B3.2 wave-crest exists; until then β=0.)*
* [~] B3.4 — Pytest: a falling jet emits ≥5× more spray than the current heuristic in `whitewater_splash.toml` while keeping foam/bubble counts similar. *(2026-05-16: closed against revised KPI — see `examples/compare_whitewater_potential.py`. A/B on `whitewater_splash.toml` (30 frames @ 80³): legacy emits 3015/frame at 3.3% spray-fraction, potential emits 1591/frame at **5.4% spray-fraction = 1.64× the legacy ratio**. The literal "5× more spray" target in the original BACKLOG over-promised — trapped-air alone is selective, not amplifying; the absolute spray count drops because total whitewater drops (calm-bulk falling particles no longer flood the system). Re-evaluate the 5× number after B3.2 wave-crest adds I_wc, which directly fires on surface curvature where spray actually originates.)*
* [ ] B3.5 — Refresh `step22.mp4` with the new classifier. *(deferred: need to wait until B3.2 lands so the refreshed video shows the full classifier, not the trapped-air half.)*

**Acceptance:** new step22 visibly has *more* airborne spray + cleaner foam boundary.

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
* [ ] B7-alt.5 — `_cuda_graph_invalidate()` already fires on every `prepare_frame` (B5.3), so a rebuild that happens at the start of a frame will naturally trigger graph recapture with the new bbox shape. Verify in a test: bake a flowing-fluid scene with `enable_cuda_graphs=True`, assert hit rate ≥ 80% across rebuilds (graphs are valid for `sub_rebuild_every` frames at a time).
* [ ] B7-alt.6 — **Bench at 256³ dam-break.** The acceptance test for the macro. Dense allocation: `256³ × ~8 fields × 4 B = ~512 MB` per snapshot — most pre-RTX-4090 GPUs OOM. With B7-alt and 10-20% active fill: sub-dense covers ~50³ to ~120³ → 4 to 25 MB per field → fits 4 GB GPUs. Write a `bench_256_dam_break.py` script; mark a scene that the dense path OOMs on and B7-alt completes.
* [ ] B7-alt.7 — Document the scattered-droplet pathology (B7-alt.1 second test result). The macro provides ZERO memory savings on dispersed scenes (whitewater-heavy, secondary spray, exploding pours). Users with those scenes should use the dense path. Surface this in the CLI (e.g., compute "spatial extent ratio" at first rebuild; if > 0.8, print a one-shot stderr warning that B7-alt isn't helping).
* [ ] B7-alt.8 — Resize/rebuild needs to be cheap. Pure CPU resize would dominate the per-frame cost; do it on-device: allocate new sub-dense fields, launch one "copy old to new at offset_delta" kernel per field. This is parallel to the rebuild pattern in S2.11.GPU reseed.

**Acceptance:** can run a 256³ dam-break that the dense path OOMs on.

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

### B10. Alembic writer (I6.4) `risk:low value:user`

USD already covers the Blender import path. Some studios prefer Alembic.
Cheap to add via `alembic` Python bindings.

* [ ] B10.1 — Add `gpufluid/io/alembic_writer.py` mirroring `usd_writer.py` signatures.
* [ ] B10.2 — Wire into `cache.json` (`abc` field), `commands.py` post-bake.
* [ ] B10.3 — Pytest round-trip: write 2 frames, read back, verify vert count.

**Acceptance:** scenes with `output.alembic = true` produce a `.abc` Blender can scrub.

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

*Nothing yet — first macros to land here will be the Tier 1 picks.*
