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
| v0.9    | "Production-fast" — hot path squeezed              | Tier 2: B4, B5, B6 + B11, B12  | 4-6           | queued |
| v1.0    | "Scale" — 256³+ scenes via sparse storage          | Tier 3: B7 (multi-session)     | 3-5           | queued |
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

* [ ] B3.1 — Implement Trapped-Air potential: `I_ta = Σ_j (1 - cos(θ_ij)) · (1 - |v_ij|/v_max)` over neighbours, capped at 1. Per fluid particle on GPU (uses a HashGrid for neighbour query — `wp.HashGrid` is already available).
* [ ] B3.2 — Implement Wave-Crest potential: `I_wc = |∇·n̂|` curvature-like measure against the surface normal n̂ (reuse S2.14.2 normal field).
* [ ] B3.3 — Emit rate per particle = `clamp(α·I_ta + β·I_wc, 0, max_rate)·dt` instead of the current `|v|>threshold` cut-off.
* [ ] B3.4 — Pytest: a falling jet emits ≥5× more spray than the current heuristic in `whitewater_splash.toml` while keeping foam/bubble counts similar.
* [ ] B3.5 — Refresh `step22.mp4` with the new classifier.

**Acceptance:** new step22 visibly has *more* airborne spray + cleaner foam boundary.

---

## Tier 2 — Hot-path performance (do once tier 1 is done; risk:med)

### B4. Extend block-sparse iteration to GS-RB + PCG `risk:med value:infra`

S2.6.4 covered Jacobi. GS-RB (S2.6.2) and PCG (S2.6.3) are the other two
pressure paths and are bigger wins at scale. Same compaction infrastructure.

* [ ] B4.1 — Port GS-RB to a per-tile launch kernel `k3_gauss_seidel_rb_per_tile`. Keep red/black colouring (parity check inside the tile).
* [ ] B4.2 — Port PCG: the spMV (`k_apply_A`), `k_apply_invM`, and per-residual axpy kernels each need block-sparse variants.
* [ ] B4.3 — `step_cfl()` and `step()` accept `pressure_block_sparse: bool` consistently for all three solvers.
* [ ] B4.4 — Bench: 128³ low fill, GS-RB sparse should match the 2.4× Jacobi speedup; PCG should be smaller (its per-iter work is more memory-bound).
* [ ] B4.5 — Pytest equivalent of `test_s2_16_sparse_jacobi.py` for both solvers.

**Acceptance:** scenes using `pressure_solver = "gsrb"` or `"pcg"` get the block-sparse path automatically and the regression tests pass.

### B5. CUDA-graphs capture for the per-step kernel sequence `risk:med value:infra`

Each step() does 20+ `wp.launch` calls. CUDA graphs would record them once
and replay each step, eliminating launch overhead. Warp supports
`wp.capture_begin / wp.capture_launch`.

* [ ] B5.1 — Audit `step()` for host-syncs (anything that calls `.numpy()` inside the per-step path can't be in a graph). Move them outside (CSF balance already uses GPU, advection CFL is one float, etc.).
* [ ] B5.2 — Add `solver._graph` lazy cache keyed by `(transfer_mode, pressure_solver, has_csf, has_color)` — a graph is invalid when topology changes.
* [ ] B5.3 — Invalidate on every `prepare_frame` (since marker/obstacles changed) — capture freshly for the next N=1..3 frames inside that topology, then replay.
* [ ] B5.4 — Pytest: same numerical output with/without graphs.
* [ ] B5.5 — Bench: target ≥15% speedup at 64³ (small grids = launch overhead dominates).

**Acceptance:** an `--enable-cuda-graphs` flag on `gpufluid simulate` (default off until proven robust).

### B6. APIC + CSF interaction QA `risk:low value:infra`

We verified APIC near obstacles in S2.12 QA. We did NOT verify APIC + surface
tension. The affine reconstruction interacts with CSF impulses in a way that
might break the parasitic-current bound (HANDOFF trap #11).

* [ ] B6.1 — Build a scene: APIC mode, σ=1, zero gravity, cube → sphere (same as surftens_on.toml but `transfer_mode = "apic"`).
* [ ] B6.2 — Diagnostic: COM drift, max parasitic velocity, kinetic energy vs FLIP-CSF baseline.
* [ ] B6.3 — If drift > 2%, add a damping rule (zero `affine_C` near surface cells, where `|∇χ̃| > threshold`).
* [ ] B6.4 — Pytest analogous to `test_csf_step_cfl_centre_of_mass_does_not_drift` but for APIC.

**Acceptance:** known-stable APIC+CSF configuration documented, regression test guards it.

---

## Tier 3 — Architectural projects (multi-session, plan carefully)

### B7. Sparse v2 — NanoVDB-backed storage for grid fields `risk:high value:infra blocks: B5 B4`

The big one. Replace dense `wp.array3d` allocations with `wp.Volume`
(NanoVDB) for the MAC velocity faces, pressure, marker, density. Saves
memory at 256³+ (currently ~470 MB of fields, with 30% fill could drop to
~140 MB) and unblocks 512³ scenes.

This is **the most invasive task in the backlog**. Estimated 3-5 sessions.
Break by *block* — port each S2.x kernel one at a time so we can revert any
single port if it goes wrong. Always keep the dense path as a parallel
implementation behind a feature flag (`use_sparse_storage=False` default).

* [ ] B7.1 — **Spike**: prototype a sparse marker field using `wp.Volume`. Measure read latency (`wp.volume_lookup_i`) vs dense indexing in a hot kernel. If sparse is >2× slower per-cell, abort the macro and document why.
* [ ] B7.2 — Add `SparseSolver3D` class (parallel to `FlipSolver3D`). Replaces `self.u/v/w/p/marker` with `wp.Volume` instances. Same public API.
* [ ] B7.3 — Port `k3_p2g` to `k3_p2g_sparse` — atomic_add into a sparse Volume. Compare numerically against dense.
* [ ] B7.4 — Port `k3_jacobi_pressure` to sparse. Reuse the block-sparse compaction (S2.16) for the active-tile list (`wp.Volume.allocate_by_tiles`).
* [ ] B7.5 — Port `k3_compute_divergence`, `k3_subtract_pressure_grad`, `k3_enforce_solid_bc`, `k3_g2p_*` one by one. Each port = its own micro task with its own test.
* [ ] B7.6 — Port S2.14 CSF kernels (chi, normal, curvature) to sparse — these have a wider stencil.
* [ ] B7.7 — Allocation strategy: per-frame `wp.Volume.allocate_by_tiles(active_tile_points)` is expensive. Investigate per-N-frames rebuild + dilation safety margin.
* [ ] B7.8 — Add `gpufluid simulate --sparse` flag. Pytest matrix: dense vs sparse at 64³ small + 128³ large, max abs error <1e-2 in pressure field.
* [ ] B7.9 — Bench: 256³ scene with 10% fill — memory used (target ≤200 MB grid fields) and per-step time (target ≤2× dense 128³).
* [ ] B7.10 — Demo: step24.mp4 — 256³ scene that *would not fit* on a 16 GB GPU in dense mode.

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

### B11. Per-particle scalar attributes (temperature / age / density) `risk:low value:user`

S2.15 introduced one vec3 attribute (colour). The same P2G/G2P pattern
generalises to any per-particle scalar or vec. Useful for:

* Temperature → coloured by gradient (lava demo).
* Age → fade alpha for splash droplets.
* Cell density → for visualisation overlays.

* [ ] B11.1 — Refactor `_apply_color_transfer` into `_apply_attribute_transfer(attr_array, channels=1|3)` so it handles scalar AND vec3 with one code path.
* [ ] B11.2 — Add `self.attr_temperature: wp.array(dtype=float)` and a `seed_box(..., temperature=20.0)` knob.
* [ ] B11.3 — Demo: hot water (red) poured into cold water (blue) — show colour blend driven by temperature, not by direct colour.

**Acceptance:** at least one scalar attribute besides colour works end-to-end through cache.

### B12. Per-frame timing instrumentation + budget report `risk:low value:infra`

`gpufluid simulate` already prints `sim total` and `mesh total`. A breakdown
per-block (e.g., "P2G: 18%, pressure: 31%, G2P: 12%") would let users tune
faster.

* [ ] B12.1 — Add lightweight `wp.ScopedTimer` wrapper around each major S2.x launch.
* [ ] B12.2 — Aggregate per-block totals across the whole bake; print at the end.
* [ ] B12.3 — Optional JSON dump to `cache.json -> timings`.

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
