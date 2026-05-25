# gpufluid — Architecture & Design

> GPU FLIP fluid simulator on NVIDIA Warp, with a Blender 4.5+ addon as the
> primary user-facing surface.
>
> Status: pre-v0.1 (alpha). This document is the **contract**. All code MUST
> conform to the block hierarchy below. Every function/kernel carries its
> block ID in source as `# [BLK X.Y.Z]`. Tests are named after blocks.

---

## 1. Vision & non-goals

### 1.1 In scope (v0.1 → v1.0)
- Single-domain FLIP/PIC solver on GPU (NVIDIA, sm_70+).
- Static and animated SDF obstacles (analytic primitives now, meshes next).
- Surface mesh extraction (marching cubes) per frame.
- Mesh cache export (PLY now, Alembic/USD later).
- Standalone CLI: `gpufluid simulate <config.toml>`.
- Blender addon: Domain / Fluid / Obstacle objects, Bake operator, mesh cache import.

### 1.2 Out of scope (for now)
- Multi-domain.
- Two-phase / multi-fluid mixing.
- Realtime viewport simulation (we are offline-bake).
- AMD/Intel GPU (Warp is NVIDIA-only).
- Whitewater, viscosity, surface tension, force fields — planned but not v0.1.

---

## 2. Layer architecture

The codebase is split into 8 strict layers. Lower layers MUST NOT import from
higher layers. Layer numbers correspond to the block prefix.

```
┌──────────────────────────────────────────────────────────────┐
│ A8  Blender Addon          (UI, registration, operators)     │
├──────────────────────────────────────────────────────────────┤
│ C7  CLI                    (gpufluid simulate / bench / info)│
├──────────────────────────────────────────────────────────────┤
│ I6  I/O                    (PLY, Alembic[future], cache idx) │
├──────────────────────────────────────────────────────────────┤
│ M5  Meshing                (density grid, marching cubes)    │
├──────────────────────────────────────────────────────────────┤
│ W7  Whitewater simulation  (foam/spray/bubble dynamics)      │
├──────────────────────────────────────────────────────────────┤
│ D4  Domain                 (SDF, seeders, walls, inflows)    │
├──────────────────────────────────────────────────────────────┤
│ F3  Solver orchestration   (FlipSolver2D, FlipSolver3D)      │
├──────────────────────────────────────────────────────────────┤
│ S2  Numerical schemes      (P2G, pressure, G2P, advect)      │
├──────────────────────────────────────────────────────────────┤
│ G1  GPU primitives         (Warp init, grid math, helpers)   │
└──────────────────────────────────────────────────────────────┘
```

Layer-number ordering: lower number = lower in the stack. Each layer
may only import from layers with a strictly smaller number. The number
7 in `W7` is historical — it landed before the strict naming
convention and slots in between D4 and M5 in dependency order (W7
consumes G1+S2 grid math but only the meshing layer M5 reads back its
output for surface foam rendering).

Allowed imports: layer N may import only layers `1..N-1`.

### 2.1 Folder mapping

```
src/gpufluid/
    __init__.py            # package surface (cross-cutting, no layer)
    blocks/                # block registry + --check tooling (cross-cutting, no layer)
    primitives/            # G1
    schemes/               # S2  (Warp kernels split by scheme)
    solvers/               # F3
    domain/                # D4
    sim/                   # W7  (whitewater: state, emit, classify, dynamics)
    meshing/               # M5
    io/                    # I6
    cli/                   # C7
addon/                     # A8 (separate top-level dir, packaged for Blender)
docs/                      # this directory
tests/                     # pytest mirror of src layout
examples/                  # runnable scripts using public API
```

---

## 3. Block numbering scheme

Every implementation unit (function, Warp kernel, class) has a stable
block ID. The ID is:
- declared in this document (sections 4–11),
- written in source as a `# [BLK X.Y.Z]` comment immediately above the def,
- registered at import time via `@block("X.Y.Z", "<description>")` decorator
  (or `block_kernel` for Warp kernels),
- referenced by test names: `def test_<x_y_z>_<aspect>():`,
- included in exception messages when a block fails.

The system is enumerated in `docs/BLOCKS.md` (auto-generated index +
status table). Run `python -m gpufluid.blocks --check` to verify that
every declared block has an implementation and every implementation has
a declared block. CI gate.

### 3.1 Error traceability

`BlockError` (in `gpufluid.blocks`) carries the block ID so any failure
prints e.g.:

```
BlockError [S2.6.1 Jacobi pressure iteration]: residual did not decrease
  at solvers/solver3d.py:142
```

This lets you say "пиздец в S2.6.1" and we both know exactly where.

### 3.2 `gpufluid blocks --check` — registry contract enforcement

The block system is only useful if the three sources of truth stay in
sync. They are:

1. **DESIGN.md sections 4–11** — the authoritative declaration. Each
   block ID appears here with a description and a layer assignment.
2. **`@block("X.Y.Z", "...")` decorators in `src/`** — the live runtime
   registry built at import time.
3. **`docs/BLOCKS.md`** — the human-readable index, generated FROM the
   registry; never hand-edit.

`gpufluid blocks --check` walks all three and refuses to exit 0 if they
disagree. It runs in CI (via `tests/test_blocks_registry.py::test_registry_is_clean`)
so a PR that adds a block without updating DESIGN.md fails the build.

#### 3.2.1 What gets checked

For each check below, "fail" means non-zero exit + a line-by-line diff
naming the offender. Checks 1–5 are hard errors; check 6 is a warning
that doesn't fail CI (aspirational hygiene).

| # | Check | Failure mode |
|---|-------|--------------|
| 1 | **ID format** — every registered ID matches `^[A-Z][0-9]+(\.[0-9A-Z]+){1,3}$` (already enforced at decoration; re-verified at check time so a future refactor can't silently weaken the regex). | Bad ID like `s2.6.1` or `S26.1` or `Q9.99` (no leading layer letter). |
| 2 | **Layer declared** — the layer prefix (e.g. `G1`, `W7`) of every registered ID appears as a `## N. Layer XN — ...` heading in DESIGN.md §§4–11. New layers must be declared in §2's diagram + get their own section first. | A `@block("Z9.1", ...)` decorator with no `## ... Layer Z9 ...` heading in DESIGN.md. |
| 3 | **BLOCKS.md mirrors the registry** — every (id, source_file) in the registry has exactly one row in BLOCKS.md with matching ID; every row with status `impl` or `impl,test` has ≥1 registry entry; every row with status `plan` has ZERO registry entries (a `plan` block that grew an impl must be flipped to `impl` and never both). | The current `F3.5` row appearing twice (one `plan`, one `impl,test`) is the canonical instance. |
| 4 | **Cross-layer imports** — AST-scan every `.py` under `src/gpufluid/` and `addon/gpufluid_blender/`; record `import gpufluid.<sub>` / `from gpufluid.<sub> import ...`; reject when `<sub>` resolves to a layer with a higher number than the importer's layer. The folder→layer map is parsed at check time from the DESIGN.md §2.1 code block (sole source of truth — `check.py` carries no parallel dict); a malformed §2.1 or an unmapped folder fails the check loudly. | A `src/gpufluid/schemes/foo.py` (S2) doing `from gpufluid.solvers import ...` (F3 > S2). |
| 5 | **Duplicate (id, qualname)** — multiple impls per ID are allowed and expected (e.g. `S2.6.6` covers 4 PCG per-tile kernels under one block), but two callables sharing both ID and qualname is a registration bug (module-reload double-decoration). | Decorator applied twice on the same function. |
| 6 | **Test coverage** (warning only) — declared blocks SHOULD have a guard test. Match by file `tests/test_<id_normalised>_*.py` OR test function name `test_<id_normalised>_*`. Missing coverage prints a list but doesn't fail CI. | A new `@block("S2.99", ...)` shipped without a test file or test function whose name carries `s2_99`. |

#### 3.2.2 ID normalisation rules

For test-name and BLOCKS.md row matching, the canonical normalisation is:

* `.` → `_`
* uppercase → lowercase
* hyphens (BACKLOG micro IDs like `B7-alt.3`) are NOT block IDs and are
  ignored by `--check`. Backlog micros live in `docs/BACKLOG.md` only.

So `S2.6.3` → `s2_6_3`, `D4.3.GPU.BVH` → `d4_3_gpu_bvh`.

#### 3.2.3 Multi-impl blocks

A block ID can map to N≥1 callables when it represents a small **family
of cooperating Warp kernels** that ship and break together. Current
examples: `F3.7` (4 sub-dense storage kernels), `S2.6.6` (4 per-tile
PCG kernels), `S2.18` (3 scalar-attribute kernels). The check accepts
this; BLOCKS.md collapses them into one row with a `(N kernels)` suffix
in the description.

#### 3.2.4 Edge cases the check explicitly handles

* **`blocks.py` is cross-cutting** — exempt from check 4 (layer-import).
  Any module may import it.
* **`cli/` (C7) is the top of the non-addon stack** — allowed to import
  G1…I6. Only A8 sits above C7.
* **Folder `sim/` hosts layer W7** — folder name doesn't match the
  layer letter (historical: whitewater landed before the strict naming
  convention). The folder→layer map in `check.py` records the
  exception. New layers must use matching folder names.
* **`addon/` lives outside `src/gpufluid/`** — its blocks (A8.*) are
  registered only when a `bpy` stub is importable. The check runs the
  addon scan only when `bpy` (real or stubbed) is on `sys.path`;
  otherwise A8 checks are skipped with a printed note.
* **`plan` rows in BLOCKS.md** are allowed for blocks declared in
  DESIGN.md but not yet implemented. They must have zero registry
  entries; the moment a `@block(...)` decorator with that ID lands, the
  row's status must flip to `impl` and the check enforces it.

#### 3.2.4.1 Known layer-import exceptions (technical debt)

The following imports violate the strict "layer N may only import from
1..N-1" rule but are whitelisted in `check.py`. Each one is a documented
debt with an exit plan. New exceptions require updating this list AND
the whitelist; silent additions break the build.

| Importer (layer) | Imports (layer) | Reason | Exit plan |
|------------------|-----------------|--------|-----------|
| (none) | F3.6 closed all 6 historical violations 2026-05-17: A1 sdf→G1, A2 mesh_sdf_gpu→S2, B animation→G1, C1 FrameEventQueue + dual-path, C2 solver drains queue + dropped legacy imports. C3 (hard `test_no_layer_exceptions` + step29.mp4 closure video) pending. | Macro F3.6 to close in C3. |
| `cli/commands.py` (C7) | `sim/whitewater*` (W7), `sim/reseed` (W7), `domain/*` (D4), `meshing/*` (M5), `io/*` (I6), `solvers/*` (F3), `schemes/*` (S2) | CLI is the top-of-stack orchestrator and is allowed to touch every lower layer by design. C7 > W7, M5, I6 — no exception needed. | n/a (not a violation; listed for clarity) |

If you find a NEW cross-layer import that the check flags:
1. Prefer fixing the dependency direction (the long-term right answer).
2. If a fix is genuinely out of scope, add a row above explaining what
   the import is and what would make it go away.
3. Add the (importer-module, imported-module) pair to
   `check.py:_KNOWN_LAYER_EXCEPTIONS`.

The exception list itself is asserted against in
`tests/test_blocks_registry.py` so a row that stops being needed (the
underlying import got fixed) starts failing the test — telling a future
maintainer to clean up the whitelist.

#### 3.2.4.2 F3.6 hook refactor — exit-plan spec

The 6 entries in §3.2.4.1 ALL trace back to `solvers/solver3d.py`
calling into `domain/*` per-frame. The original BACKLOG entry framed
this as "invert the direction via a hook", but a fresh audit of the
actual call sites reveals **three different problems sharing one
symptom**. Treating them as one is what makes the refactor look
overwhelming. Splitting them cuts the work by ~60%.

##### Audit of the 6 violating imports (as of 2026-05-17)

| Import | Call sites in solver3d.py | Real category |
|--------|--------------------------|---------------|
| `domain.sdf.{sdf_sphere, sdf_box, sdf_cylinder_y, sdf_union, cell_centers}` | analytic obstacle stamping (5+ places) | **A — mis-filed math primitives** |
| `domain.mesh_sdf_gpu.mark_solid_from_mesh_gpu` | initial mesh obstacle + animated mesh obstacles (2 places) | **A — mis-filed GPU kernel** |
| `domain.animation.{Motion, evaluate_center}` | obstacle position evaluation (4 places inside `_apply_animation_at_frame`) | **B — pure utility, no layer inversion needed** |
| `domain.regions.{InflowBox, OutflowBox, apply_inflows, apply_outflows}` | inflow emit + (legacy) outflow path in `prepare_frame` | **C — actually needs the hook** |

##### Category A: mis-filed (move, don't invert)

`sdf_*` are analytic distance functions. They take a point and a
shape primitive, return a scalar. Zero dependency on FlipSolver3D or
any D4-layer concept. They're pure math, same as `clamp_int` or
`sample3` in G1 today. Likewise `cell_centers` builds a host-side
cell-centre grid — already declared as `[BLK G1.8]` even though it
lives in `domain/sdf.py`. The file's location is the bug.

Resolution: **move `sdf_*` + `cell_centers` to `primitives/sdf.py`
under layer G1**, register under existing G1.8 + new G1.10..G1.14
slots. `domain/sdf.py` keeps the `apply_sdf_as_solid` host helper
(which TAKES a solver's marker array) but loses the math primitives.
Eliminates the `solvers/solver3d.py → domain.sdf` whitelist entry.

Similarly `mark_solid_from_mesh_gpu` is a Warp BVH kernel that takes
a marker array as output. The "domain" content of it is the input
triangle mesh — but that gets passed in by the caller. The kernel
itself is layer S2/F3 infrastructure.

Resolution: **move `mark_solid_from_mesh_gpu` to
`schemes/mesh_marker.py` under layer S2**, keep its current
`[BLK D4.3.GPU]` ID for now (rename to S2.x in a follow-up to avoid
breaking too many references in one PR). Eliminates the
`solvers/solver3d.py → domain.mesh_sdf_gpu` whitelist entry.

**Expected reduction: 2 of 6 whitelist entries (33%) for free.**

##### Category B: pure utility (one-line layer fix)

`Motion` and `evaluate_center` from `domain/animation.py` are a tiny
host-side state machine: given a base position and a motion spec,
return the current world position. The motion specs live in scene
config (D4), but the evaluator itself is pure data transformation.

Resolution: **move `Motion` dataclass + `evaluate_center` to
`primitives/animation.py` under layer G1**, register new
`[BLK G1.15] Motion spec` and `[BLK G1.16] evaluate_center`. Domain
config still PRODUCES `Motion` instances; F3 consumes them through
the G1 helper. Eliminates the `solvers/solver3d.py → domain.animation`
whitelist entry.

**Expected reduction: 3 of 6 whitelist entries (50%) after this step.**

##### Category C: real inversion (FrameEventQueue API)

After A + B, only ONE genuine D4→F3 coupling remains:
`apply_inflows` / `apply_outflows`. These really are domain logic —
"emit fluid at this region this frame" / "delete particles leaving
this box". F3 calling them per-step is the architectural
inversion.

Proposed API:

```python
# in primitives/frame_events.py (G1)
@dataclass
class FluidEmitEvent:
    """One inflow's worth of new fluid for the current frame."""
    positions: np.ndarray  # (N, 3) float32, in sim space
    velocities: np.ndarray  # (N, 3) float32

@dataclass
class FluidOutflowEvent:
    """Bounding box outside which particles must be culled."""
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    inside: bool  # True = "delete inside", False = "delete outside"

class FrameEventQueue:
    """One-shot per-frame event sink + drainer.

    Populated by D4 helpers (`inflow_box.publish_for_frame(...)`) at
    `prepare_frame` start; drained by F3 inside the same call. Empty
    once `prepare_frame` returns. Cleared between frames — events do
    not persist.
    """
    def push_emit(self, ev: FluidEmitEvent) -> None: ...
    def push_outflow(self, ev: FluidOutflowEvent) -> None: ...
    def drain_emits(self) -> list[FluidEmitEvent]: ...
    def drain_outflows(self) -> list[FluidOutflowEvent]: ...
```

Migration sketch:

1. Add `FrameEventQueue` + dataclasses to `primitives/frame_events.py`.
   Cross-layer-friendly (G1, foundational).
2. Add a new method to `InflowBox`: `publish_for_frame(queue, frame,
   dt, rng)`. It does the same work as today's `apply_inflows` but
   pushes to the queue instead of returning arrays.
3. In `solver3d.prepare_frame`, replace the direct call with a
   queue-drain loop. Solver owns a `FrameEventQueue` instance.
4. **Critical for CUDA-graph capture compat**: the queue MUST be
   fully drained before the per-step kernel sequence starts.
   Inflow/outflow change `n_particles`, which already invalidates
   the cached graph (see §B5.2). So drain at top of `prepare_frame`,
   then capture; the queue is empty for the entire substep loop.
5. Once both inflow + outflow are migrated, the import `from
   ..domain.regions import ...` is gone, and the 4 corresponding
   whitelist entries collapse to zero (regions is one file).

**Expected reduction: 6 of 6 whitelist entries → 0.**

##### Migration phasing (one micro per session)

| Phase | Scope | Acceptance |
|-------|-------|-----------|
| F3.6.A1 | Move `sdf_*` + `cell_centers` to G1; update all importers | suite green, --check shows 1 fewer whitelist entry |
| F3.6.A2 | Move `mark_solid_from_mesh_gpu` to S2 | suite green, 1 fewer entry |
| F3.6.B | Move `Motion` + `evaluate_center` to G1 | suite green, 1 fewer entry |
| F3.6.C1 | Add `FrameEventQueue` + tests; D4 helpers publish (but solver still pulls — dual path) | suite green, no whitelist change yet (transitional) |
| F3.6.C2 | Switch solver to drain queue, delete legacy pull path | suite green, 3 fewer entries — whitelist now empty |
| F3.6.C3 | Add `test_no_f3_to_d4_imports` assertion test (would fail today, passes after C2) | hard CI gate prevents future regression |

Each phase is independently shippable and reversible. C1+C2 cannot
land in the same session because C1's dual-path enables the parallel
test infrastructure that C2 then collapses.

##### Risks the spec must flag

* **CUDA-graph rehit rate.** Today's inflow/outflow code already
  invalidates the graph cache (changes `n_particles`). The queue
  drain happens at the SAME point, so rehit rate should be
  identical. Verify in C2 via `test_b5_3_invalidate_on_topology_change`.
* **Animation specs reference `Motion` by type.** Moving the
  dataclass changes `__module__`; if any pickle/checkpoint touches
  it, the restore will fail. Audit before B phase.
* **`mark_solid_from_mesh_gpu` block ID stays D4.3.GPU.** Renaming
  would touch 4 test files + BLOCKS.md (auto-regen handles the
  index; tests need slug rename). Defer to a follow-up to keep the
  A2 PR scope minimal.

##### Out of scope for the F3.6 macro

* Replacing the `cell_centers` host-side allocation with a Warp
  kernel (G1 perf concern, not architecture).
* Generalising `FrameEventQueue` to W7 whitewater events. W7 has
  no current cross-layer pulls; revisit if that changes.

#### 3.2.5 CLI surface

```
gpufluid blocks --check        # run all checks; exit 1 if any fail
gpufluid blocks --regen-index  # rewrite docs/BLOCKS.md from registry + DESIGN.md
gpufluid blocks --list [--layer S2]  # pretty print; defaults to all layers
```

`--regen-index` is the only way BLOCKS.md should ever change after this
spec lands. The regenerated file carries a `<!-- generated by
gpufluid blocks --regen-index; do not edit by hand -->` header; the
check refuses to run on a BLOCKS.md missing that header (prevents a
hand-edit from passing checks because it happened to match by accident).

#### 3.2.6 Pytest integration

`tests/test_blocks_registry.py` exposes one parametrised test per check
(1–5 are hard, 6 is `@pytest.mark.xfail(strict=False)` so warnings
surface without failing the suite). The test imports every module under
`src/gpufluid/` plus the addon (via the existing `tests/_bpy_stub.py`)
to populate the registry deterministically before checking.

A green `pytest` is a sufficient proxy for `gpufluid blocks --check`;
the CLI exists for ad-hoc local runs and IDE integration.

#### 3.2.7 Out of scope (intentionally)

* **Block ordering inside a layer** is not enforced — `S2.18.x` can land
  before `S2.17.x` is filled in. The numbers are stable IDs, not a
  chronological log.
* **Description string fidelity** between `@block(...)` and DESIGN.md is
  not strictly checked — only the ID's existence. Drift in prose
  descriptions is allowed (the decorator string is the source of truth
  for end-of-line error messages; DESIGN.md elaborates).
* **Test contents** — check 6 only verifies a test with the right name
  EXISTS, not that it actually exercises the block. Quality is the
  human reviewer's job.

---

## 4. Layer G1 — GPU primitives

Foundations: Warp init, array allocation, grid index/sample math.
Pure helpers. No domain logic.

| ID    | Block | Status |
|-------|-------|--------|
| G1.1  | Warp init & device selection                                | impl |
| G1.2  | Array allocation helpers (`zeros3`, `zeros`)                | impl |
| G1.3  | `clamp_int`, `clamp_float` helpers                          | impl |
| G1.4  | Trilinear weight computation                                 | impl |
| G1.5  | Trilinear sample (`sample3`)                                | impl |
| G1.6  | Trilinear scatter (`scatter_face`) with atomic_add          | impl |
| G1.7  | Box-filter 3D (`box_blur_3d`)                                | impl |
| G1.8  | Cell-center coordinate grid (numpy host helper)              | impl |
| G1.9  | Step profiler (per-section ScopedTimer wrapper)              | impl |
| G1.10 | SDF sphere (analytic, was D4.2.1 pre-F3.6.A1)                | impl |
| G1.11 | SDF axis-aligned box (was D4.2.2 pre-F3.6.A1)                | impl |
| G1.12 | SDF cylinder Y-axis (was D4.2.3 pre-F3.6.A1)                 | impl |
| G1.13 | SDF plane (was D4.2.4 pre-F3.6.A1)                           | impl |
| G1.14 | SDF union (min of components, was D4.2.5 pre-F3.6.A1)        | impl |
| G1.15 | Motion specs: LinearMotion + KeyframeMotion union (was D4.6) | impl |
| G1.16 | evaluate_center — animated centre at a frame (was D4.6)      | impl |
| G1.17 | FluidEmitEvent — per-frame inflow emission record (F3.6.C1)  | impl |
| G1.18 | FluidOutflowEvent — per-frame cull request record (F3.6.C1) | impl |
| G1.19 | FrameEventQueue — push/drain sink for D4→F3 events (F3.6.C1) | impl |

The G1.10–G1.14 analytic SDFs relocated from D4 on 2026-05-17 (F3.6.A1);
G1.15 + G1.16 motion utilities relocated 2026-05-17 (F3.6.B). G1.17–G1.19
event queue infrastructure landed 2026-05-17 (F3.6.C1) — D4 region
helpers now have a `publish_for_frame()` dual-path method alongside
the legacy `apply_inflows`/`apply_outflows` helpers. The solver
migration to drain-only is F3.6.C2.

---

## 5. Layer S2 — Numerical schemes

Per-step physical operations. Each block is one Warp kernel (or a
launcher wrapping one). Stateless w.r.t. the solver — takes arrays in,
mutates arrays out.

| ID      | Block | Status |
|---------|-------|--------|
| S2.1    | P2G transfer (particle vel → MAC faces, weighted)            | impl |
| S2.2    | Normalize faces (divide by accumulated weight, save old)     | impl |
| S2.3    | Apply body forces (gravity to v)                             | impl |
| S2.4    | Enforce solid boundary conditions on faces                   | impl |
| S2.5    | Compute divergence on fluid cells                            | impl |
| S2.6    | Pressure Poisson solve                                       | —    |
| S2.6.1  |   Jacobi iteration                                           | impl |
| S2.6.2  |   Gauss–Seidel red-black                                     | impl |
| S2.6.3  |   PCG                                                        | impl |
| S2.7    | Subtract pressure gradient from faces                        | impl |
| S2.8    | G2P + FLIP/PIC blend                                         | impl |
| S2.9    | Particle advection (semi-implicit Euler + clamp to domain)   | impl |
| S2.10   | CFL substep computation                                      | planned |
| S2.11   | Particle reseeding (fix voids / over-sampling)               | impl |
| S2.12   | APIC transfer (alternative to FLIP/PIC)                      | impl |
| S2.13   | Viscosity (semi-implicit Jacobi diffusion of face velocity)  | impl |
| S2.14   | Surface tension (Brackbill-Kothe CSF, face-applied)          | impl |
| S2.15   | Per-particle color attribute (RGB), P2G/G2P transfer         | impl |
| S2.18   | Per-particle scalar attribute (temperature), P2G/G2P transfer | impl |
| S2.11.GPU | Reseed particles fully on GPU (count → rank → compact → emit) | impl |
| S2.17   | MPM contact/inflow auxiliaries (rigid-body shell-out helpers) | —    |
| S2.17.1 |   SDF box collider on grid (separate-surface near boundary, zero deep interior, optional tangential friction on a chosen face) | impl |
| S2.17.2 |   Particle pushback inside cube body (snap to nearest face + F/C reset) | impl |
| S2.17.3 |   Particle pushback at domain walls (slip-clamp + F/C reset)  | impl |
| S2.17.4 |   Terminal-velocity cap in narrow tap zone (mimics air drag on laminar stream) | impl |
| S2.17.5 |   Anti-splash `|v_z|` cap above cube top (suppresses APIC stress-launch artifacts at rigid edges) | impl |
| S2.17.6 |   Active-only PLY save (filters `particle_selection==1`)      | impl |
| S2.17.PATCH.SLIP | Overlay fix for warp-mpm `add_surface_collider` slip-branch bug (was overwriting projected `v` with zero at end of kernel) | impl |
| S2.17.PATCH.EOS  | Overlay J floor on `kirchoff_stress_water` (J_safe = max(J, 0.5)) — caps unbounded pressure spike when fluid is numerically overcompressed at a rigid collider | impl |

### 5.3 S2.17 — MPM contact / inflow auxiliaries

Bundle of small kernels that make a third-party MPM solver (currently
[zeshunzong/warp-mpm](https://github.com/zeshunzong/warp-mpm), see F3.7)
behave well when fluid hits a rigid cube and forms a floor pool. These
are NOT part of any MPM scheme paper; they are numerical safety nets
for known failure modes at rigid-collider boundaries:

| Sub-block | Failure mode it fixes | Approach |
|-----------|----------------------|----------|
| S2.17.1 | Particles compressed in cube interior → stress explosion | Zero grid velocity deep inside; project only inward-normal component in a 1.5-cell shell; optional tangential friction on the chosen face (e.g. cube top → accumulation behavior) |
| S2.17.2 | P2G smoothing leaks particles into the cube body → mass accumulation → NaN | Pre- and post-P2G snap to nearest face; reset F/F_trial/C to identity (rigid contact history is ill-defined; standard production MPM trick) |
| S2.17.3 | Particle escapes [0, n_grid·dx]³ → CUDA OOB on next P2G | Slip-clamp at safe interior `[0.05, 0.95]³`; zero normal velocity, preserve tangential; F/C reset on contact |
| S2.17.4 | Free-falling tap arrives at cube top with 2-3 m/s → splash | Cap `v_z ≥ v_terminal` (e.g. −1.0 m/s) inside a narrow tap column above the cube; off the column, particles fall freely |
| S2.17.5 | APIC's velocity-gradient C-matrix at the discontinuous cube edge transfers downward momentum into upward "fountains" | Hard clamp `|v_z| ≤ v_splash_max` (e.g. 0.3 m/s) above the cube; physically: real water cannot bounce upward at >0.3 m/s without a hydraulic jump |
| S2.17.6 | Killed/inactive particles (selection=1) clutter the PLY dump and produce visible "ghost piles" in renders | Write only particles with `selection==0`. Mesh visualisation reads the filtered dump |
| S2.17.PATCH.SLIP | `add_surface_collider(surface="slip")` in upstream warp-mpm computes projected `v` correctly then overwrites with `wp.vec3(0,0,0)` on the last line of the kernel (`mpm_solver_warp.py:707` of `warp-mpm@v1.0`) | Replace the final write with `state.grid_v_out[...] = v` (the computed projection). Without this, "slip" behaves identical to "sticky" |
| S2.17.PATCH.EOS | `kirchoff_stress_water` uses `J^{-1.1}` which → +∞ when fluid is compressed (J → 0) at a rigid wall. One overcompressed particle launches the rest of the local pool. | Clamp `J_safe = max(J, 0.5)` before the pow. Compression beyond 50% is unphysical for weakly-compressible water; this just bounds the artifact |

All S2.17.* kernels are pure Warp launchers, no global state. The full
F3.7 step pipeline is responsible for invoking them in the correct
order (see §6.7 below). The two `PATCH.*` entries live as runtime
overlay patches applied at `F3.7` import time — we do not maintain a
fork of warp-mpm, just monkey-patch the two kernels via Warp's module
re-compilation hook. This way upgrading warp-mpm requires only
re-applying the two patches, not merging a fork.

### 5.4 D4.3.GPU.BVH — BVH-accelerated mesh-inside test

Replaces the O(cells × tris) brute-force ray-cast (`D4.3.GPU`) with a
single `wp.mesh_query_point_sign_winding_number(mesh.id, p, accuracy)`
per cell. Uses Warp's built-in `wp.Mesh` spatial structure (LBVH built
on construction) — no hand-rolled BVH builder. Tradeoff: we depend on
Warp's mesh primitive remaining stable across versions; in return the
acceleration is battle-tested, builds in O(N log N), and traverses on
GPU without manual stack management. Triangle threshold for auto-engage
is `>= 256` (below that the brute-force path's lower setup cost wins).
For animated mesh obstacles the `wp.Mesh` should be cached and only the
points buffer refreshed each frame (`mesh.refit()`).

### 5.1 S2.14 — Surface tension (Brackbill-Kothe CSF)

Continuum Surface Force model. The fluid fraction χ ∈ [0,1] is built
on cell centres from the per-step marker (fluid = 1, otherwise 0),
then box-blurred (G1.7) to a smooth indicator χ̃. Gradient ∇χ̃ is
the surface normal direction (un-normalised); the unit normal field
**n̂** = ∇χ̃ / (|∇χ̃| + ε). Curvature κ = −∇·**n̂** is computed on
cell centres by central differences of the normal field. The
per-face force is applied as a velocity impulse:

    Δv_face = (σ · κ_face · ∇χ̃_face) · dt / ρ

where σ is the surface tension coefficient, κ_face is averaged
across the face, and ∇χ̃_face is the directional gradient component
on that face. Applied between viscosity and divergence so it feeds
into the next pressure solve (Bridson §8.5, Brackbill et al. 1992).
Sub-blocks:

| Sub-ID    | Kernel                                       |
|-----------|----------------------------------------------|
| S2.14.1   | Build smoothed fluid indicator χ̃ from marker |
| S2.14.2   | Compute unit normal field n̂ on cell centres  |
| S2.14.3   | Compute curvature κ on cell centres           |
| S2.14.4   | Apply CSF impulse to MAC faces                |
| S2.14.5   | Capillary-wave CFL: `dt_max = 0.9·√(ρ·dx³/(2π·σ))` (host helper) |
| S2.14.6   | Force-balance: subtract per-axis mean impulse (kills parasitic drift) |

### 5.2 S2.15 — Per-particle color attribute

Each particle carries an RGB `vec3` in [0,1]. Scattered to a grid color
field by the same trilinear weights as velocity (S2.15.1), normalised by
the deposited weight (S2.15.2), then sampled back to particles each step
(S2.15.3). The net effect: where two fluids of different colour meet,
their grid cells average over both populations and particles drifting
across the interface inherit a blended colour. This is **linear RGB
blending** — physically equivalent to additive light mix, not pigment.
Sub-blocks:

| Sub-ID    | Kernel                                                  |
|-----------|---------------------------------------------------------|
| S2.15.1   | P2G scatter of `attr_color` (atomic add into grid vec3) |
| S2.15.2   | Normalize grid color by accumulated scalar weight       |
| S2.15.3   | G2P sample grid color back to particles                 |

**Future:** Mixbox pigment-space LUT (Šochorová & Jamriška 2021) for
realistic blue+yellow=green mixing instead of muddy gray. Drop-in
replacement on the G2P side: same color array, different blend rule.
Not in v0.6 (LUT is ~270 KB, needs a load-time download path).

### 5.6 S2.18 — Per-particle scalar attribute (temperature, B11)

Generalises the S2.15 pattern to a single-channel float. Each particle
carries an `attr_temperature` float; the field is scattered to a grid
scalar (S2.18.1, atomic add), normalised by the same weight grid used
by S2.15 (S2.18.2 reuses the S2.15.2 weight accumulator), and sampled
back to particles each step (S2.18.3). All three kernels are gated on
`self.attr_temperature is not None`, so scenes that don't use a scalar
attribute pay zero overhead (no allocations, no kernel launches, no
profiler section).

Sub-blocks:

| Sub-ID    | Kernel                                                       |
|-----------|--------------------------------------------------------------|
| S2.18.1   | P2G scatter of per-particle scalar (atomic add into grid)    |
| S2.18.2   | Normalize grid scalar by deposited weight                    |
| S2.18.3   | G2P gather grid scalar back to particle                      |

**Surface area:** `seed_box(..., temperature=X)` and `seed_mesh(...,
temperature=X)` accept a float that's broadcast to all particles in
that source, with append-in-lockstep semantics that mirror the colour
path (multi-source scenes work; uncoloured/un-tempered second seeds
get padded to a neutral value to stay aligned with existing arrays).

**TOML surface (B11.3, 2026-05-16):** `[[fluids]] temperature = X`
(any float, no implicit range) parses into `FluidBoxCfg.temperature`
or `FluidMeshCfg.temperature` (Optional[float]; None ⇒ no scalar
attribute allocated). The CLI threads this into the seeder via
`cmd_simulate._seed_one`. Per-frame the simulator dumps a
`<cache>/temperatures/frame_NNNN.npy` sidecar (mirrors the
`<cache>/colors/` sidecar from S2.15) for renderer consumption.

**Limitations:** the reseed paths (S2.11 CPU + S2.11.GPU) compact
`attr_color` in lockstep but DO NOT yet do the same for
`attr_temperature` — opportunistic follow-up. A scene that combines
`reseed=true` with per-source temperature will lose the scalar on the
first reseed pass.


---

## 6. Layer F3 — Solver orchestration

Owns state arrays, runs the per-step pipeline by calling S2.x in order.

| ID    | Block | Status |
|-------|-------|--------|
| F3.1  | `FlipSolver2D` class                                          | impl |
| F3.2  | `FlipSolver3D` class                                          | impl |
| F3.3  | `step(dt, pressure_iters)` pipeline (calls S2.1..S2.9)         | impl |
| F3.4  | `step_cfl(target_dt)` with substepping                         | planned |
| F3.5  | Restart / checkpoint state                                     | planned |
| F3.7  | `MpmSolver` shell-out adapter around third-party warp-mpm      | impl |
| F3.7.1 | Init: load TOML scene, build particle cloud, configure water EOS + colliders | impl |
| F3.7.2 | Per-step pipeline: pre-pushback → tap-cap → p2g2p → post-pushback → dump | impl |
| F3.7.3 | Patch overlay: applies S2.17.PATCH.SLIP and S2.17.PATCH.EOS at import time | impl |

### 6.7 F3.7 — MPM solver shell-out

Alternate solver path. The FLIP/APIC class (F3.1..F3.3) cannot resolve
lateral spread when a vertical water column hits a flat rigid plate —
pressure projection alone does not generate the necessary in-plane
forces (documented incident, see [memory: project-mpm-pivot]). F3.7
delegates the timestep to the GPU MLS-MPM solver in
`third_party/warp-mpm/`, wrapped in `gpufluid.sim.mpm.solver:MpmSolver`.

Pipeline per call to `MpmSolver.step()`:

```
pre_step:  S2.17.2 cube_pushback     (snap any particles already inside cube out)
           S2.17.3 wall_pushback     (slip-clamp escapees, if any)
           S2.17.4 tap_velocity_cap  (laminar inflow + above-cube anti-splash)
core:      warp_mpm.p2g2p(dt)        (the third-party solver, patches S2.17.PATCH.* active)
post_step: S2.17.2 cube_pushback     (catch G2P-induced cube leakage)
           S2.17.3 wall_pushback     (catch G2P-induced wall escape)
```

`MpmSolver` is **not** a drop-in replacement for `FlipSolver3D`. The
output is particle positions only (no MAC velocity field); the mesher
(M5.11) and the cache writer (I6.5) consume positions directly. Coupling
with FLIP/APIC stages is not planned.

The TOML scene config (C7.1) gets a new top-level key `[simulation].solver = "mpm" | "flip"` that the CLI bake (C7.2) inspects to pick between F3.2 and F3.7. Default is "flip" so existing scenes are not affected.

The pipeline order in F3.3 is fixed:

```
clear_grid → P2G(S2.1) → normalize(S2.2) → gravity(S2.3) → bc(S2.4)
  → divergence(S2.5) → pressure(S2.6) → grad_subtract(S2.7) → bc(S2.4)
  → G2P_advect(S2.8 + S2.9)
```

### 6.7.1 Coordinate system convention (MPM path)

The MPM solver and the marching-cubes mesher both operate in a **fixed
unit cube `[0, 1]³`**. The user, however, authors scenes in world metres
inside Blender — the Domain Empty has an arbitrary world AABB. Mapping
between those spaces happens at two ring boundaries, both of which the
addon owns:

**Input (Blender → solver) — `bake.collect_scene`:**

For every position, AABB, half-extent, velocity, and gravity that the
solver consumes, the addon applies

```
v_sim = (v_world − domain.origin) / domain.size
```

componentwise (anisotropic domains supported). `domain.origin` is the
world-space lower corner of the Domain Empty's AABB; `domain.size` is
the world-space extent. Gravity is in m/s² and divided by
`domain.size_z` so a particle covers real metres in real time inside the
solver's `[0, 1]³`. Without this division, water in a 2.32 m domain
falls 2.32× too fast (one of the visual "tells" before this layer was
written).

**Output (solver → Blender) — `cache_loader._preload_sequence`:**

The mesher writes PLY vertices in `[0, 1]³`. The addon's attach step
reverses the transform when populating each preloaded mesh datablock:

```
v_world = v_sim * domain.size + domain.origin
```

baked directly into the vertex array. The `gpufluid_cache` object then
sits at world `(0, 0, 0)` with identity scale — what you see in the
viewport is what's in the mesh data, no per-frame transform tricks.

This is symmetric: the same `(domain.origin, domain.size)` pair is used
on the way in and the way out, so the rendered fluid lines up with the
Inflow/Fluid sources the user placed in the scene to within a cell.

Solver-internal constants (`MpmDomainWalls.lo/hi`, anti-splash z, tap
geometry, etc.) are all expressed in `[0, 1]` because they only ever
see the normalised space. Material parameters (`bulk_modulus`,
`density`, `rpic_damping`, `grid_v_damping_scale`) are dimensionless or
in normalised units after the gravity / velocity scaling above.

---

## 7. Layer D4 — Domain

Geometry of the simulation domain: walls, obstacles, fluid seeders.

| ID      | Block | Status |
|---------|-------|--------|
| D4.1    | Solid wall shell (init marker boundary)                       | impl |
| D4.2    | SDF analytic primitives — RELOCATED to G1.10–G1.14 on 2026-05-17 (F3.6.A1). See §4. | moved |
| D4.3    | Mesh → SDF (triangle-soup distance)                           | impl |
| D4.3.GPU | GPU triangle ray-cast inside test — RELOCATED to schemes/mesh_marker.py 2026-05-17 (F3.6.A2). ID kept for now. | moved |
| D4.3.GPU.BVH | BVH-accelerated wp.Mesh winding inside test — RELOCATED 2026-05-17 (F3.6.A2). ID kept. | moved |
| D4.4    | Apply SDF as solid markers (mutates solver marker, stays D4)  | impl |
| D4.5    | Fluid seeders                                                 | —    |
| D4.5.1  |   Box seeder (uniform jittered)                               | impl |
| D4.5.2  |   Mesh seeder (volumetric fill)                               | planned |
| D4.6    | Animated obstacles — motion specs RELOCATED to G1.15/G1.16 (F3.6.B). What's left in D4: register/refresh hooks that mutate solver marker. | impl |
| D4.7    | Inflow / outflow regions                                      | planned |

---

## 8. Layer M5 — Meshing

### 5.5 S2.16 + S2.6.4 — Sparse-aware iteration (block-skip pressure solve)

A partial step toward priority-10 sparse FLIP. Dense storage is kept;
the pressure solve (hottest per-step kernel) is taught to early-exit
on 8³ blocks that contain no fluid cells. Memory is **not** saved — this
is a time-only optimisation. True NanoVDB-style sparse allocation across
all S2.x kernels remains future work (Sparse v2, post-v1.0).

Mechanism:

* `S2.16` — `k_mark_active_blocks` builds a `(nx/8, ny/8, nz/8)` int
  bitmask. A block is *active* if any of its 512 cells has
  `marker == 1` (fluid). Cheap: O(cells) read-only sweep with a
  single `atomic_or` per block.
* `S2.6.4` — `k3_jacobi_pressure_blocksparse` reads the bitmask first;
  if the block is dead the whole 8³ tile of threads exits before any
  divergence/stencil reads. Same numeric result as dense Jacobi
  (block-aligned skip leaves an inactive cell's pressure at zero,
  which matches the dense path's behaviour for non-fluid cells).

Auto-engages at ≥ 64³ when fill ratio < 60% (heuristic — the host-side
scan to count active blocks costs ~0.1 ms even at 256³, so the break-even
threshold is low). Forcing on/off via `pressure_block_sparse=True/False`
on `step()`.

The first-cut whitewater (W7.1/W7.2/W7.3) emits one class of particles
with single-rate gravity + drag. Production fluid renderers need three
visually distinct classes:

* **Spray** (kind=1): airborne droplets above the surface. Free-falls
  under full gravity, low drag, short lifetime (~1 s). Renders as
  fine glistening points.
* **Foam** (kind=0): bubbles + air pockets riding the surface. Near-zero
  net vertical force (buoyancy cancels gravity), high drag (sticks to
  surface flow), long lifetime (~3 s). Renders as white opaque clusters.
* **Bubble** (kind=2): air pockets *under* the surface, rising via
  buoyancy at ~0.3·g upward, medium drag, "pops" (lifetime instantly
  expires) once it reaches the surface — modelled as a density-grid
  lookup with a `pop_threshold` ≤ 0.5.

Block IDs:

| Sub-ID  | Role                                                       |
|---------|------------------------------------------------------------|
| W7.4    | Kind classifier (per-emit, density-based)                  |
| W7.5    | Per-class advection (gravity/drag/buoyancy/pop rules)      |
| W7.6    | Render-side kind sidecar (`colors/kind_NNNN.npy`)          |

The density grid used for classification is the same one M5 builds for
marching cubes — reusing it costs nothing and ensures the foam/spray/bubble
split agrees with where M5 says the surface is.

---

### 5.4 S2.11.GPU — GPU particle reseed

Drop-in replacement for the host-side `S2.11` reseed. The CPU path
sorts particles by cell and iterates a Python loop over cells with
excess particles — this becomes the per-reseed bottleneck above ~200k
particles (Python loop scales with cells_with_excess, which grows
linearly with total particle count).

Pipeline (all device-resident except a single counts D→H and an
emit positions H→D):

1. **`S2.11.GPU.COUNT`** — `k_count_particles_per_cell`: per-particle
   `atomic_add(counts, cell_id, 1)`. O(n_particles).
2. **`S2.11.GPU.RANK`** — `k_mark_keep_by_rank`: each particle does
   `rank = atomic_add(seen_count[cell_id], 1)`; if `rank >=
   max_per_cell` mark `alive=0`. This deterministically drops the
   *last-arriving* particles per cell (not random), but the choice
   is unbiased given that GPU thread order is not statistically
   correlated with particle history.
3. **Compact** (`D4.7.GPU` pattern) — inclusive prefix-sum + scatter.
4. **Emit** — host reads `counts.numpy()`, computes per-cell deficits,
   builds jittered positions, uploads, concatenates with compacted
   arrays. The host side stays O(cells), not O(particles).

Auto-engages above `RESEED_GPU_THRESHOLD = 100_000` particles; below
that the CPU path's lower setup cost wins. Per-particle attributes
(`affine_C`, `attr_color`) are compacted in lockstep — the colour
sidecar uses the same `prefix` array to scatter into a fresh buffer.

---

### 8.1 M5.4 — GPU marching cubes (wp.MarchingCubes)

Drop-in replacement for the skimage CPU path that becomes the per-frame
bottleneck at 64³+ grids (~30 ms/frame for 128³). Uses Warp's built-in
`wp.MarchingCubes`: a stateful context with pre-allocated vertex/index
buffers, called per frame via `mc.surface(density_field, threshold)`.
Returns indexed triangle soup in grid-index space; we rescale by `dx`
on copy-out. The wall-margin mask (M5.7) is reimplemented as a GPU
kernel (`k_mc_zero_walls`) so the entire path stays device-resident
until the final `verts.numpy()` / `indices.numpy()` host transfer.

Auto-engages for grids ≥ 64³ (below that the launch overhead of
`wp.MarchingCubes.surface` exceeds skimage's pure-C MC). Honest
disclosure: like `D4.3.GPU.BVH`, this uses Warp's primitive rather
than a hand-rolled MC. Tradeoff: zero LUTs to maintain, identical
topology to literature MC; cost is a dependency on Warp internals.


Particles → triangle surface mesh per frame.

| ID    | Block | Status |
|-------|-------|--------|
| M5.1  | Particle density grid scatter (isotropic, trilinear, atomic)  | impl |
| M5.2  | Density grid smoothing (box-blur N passes)                    | impl |
| M5.3  | Marching cubes (skimage, CPU)                                 | impl |
| M5.4  | Marching cubes on Warp (GPU port)                             | impl |
| M5.5  | Taubin / Laplacian mesh smoothing                             | impl |
| M5.6  | Mesh decimation                                               | impl |
| M5.7  | GPU wall-mask kernel (zero density within margin cells)       | impl |
| M5.8  | Yu-Turk anisotropic kernel surface reconstruction             | eval, deferred |
| M5.9  | Wider cubic kernel particle scatter (isotropic, opt-in)       | impl |
| M5.10 | Akinci adaptive cubic kernel (per-particle R via neighbour count) | planned |
| M5.11 | Bridson SDF surface reconstruction (signed-distance from particle cloud) | planned |

---

### 8.3 M5.9 — Wider cubic kernel particle scatter

**Discovered as B13.8 fallout (2026-05-17):** evaluating Yu-Turk on real
step30 water bake data showed the **dominant** surface-quality win came
from the wider cubic kernel, **not** from the anisotropic ellipsoid
shape. CPU eval at frame 200 of step30 (96³, 26733 particles, iso=0.2):

| Kernel                              | Mesh CC count |
|-------------------------------------|---------------|
| M5.1 trilinear (current default)    | 5942          |
| Yu-Turk forced-isotropic (k_r=1)    | 205 (29× less)|
| Yu-Turk anisotropic k_r=4 (default) | 227           |
| Yu-Turk anisotropic k_r=8           | 242           |

The isotropic→anisotropic shape change moved the needle by ≤10%; the
M5.1→isotropic kernel change moved it by 29×. So M5.9 ships the **cheap
part** (isotropic cubic scatter, ~30 lines of Warp) standalone, deferring
the SVD/covariance machinery of M5.8 until / unless a use case actually
needs the shape information.

#### Algorithm

For each particle `p_i` at world position `x_i`:

1. Determine support radius `r_world = radius_cells * dx` (default
   `radius_cells = 2.0` cells, so support ≈ 2 voxels).
2. Compute AABB: `[x_i - r_world, x_i + r_world]` in grid index space,
   clipped to `[0, n)`.
3. For each cell `c` in AABB at cell-centre `x_c = (c + 0.5) * dx`:
     `r_sq = ||x_c - x_i||² / r_world²`
     if `r_sq < 1`: `density[c] += (1 - r_sq)³` (Müller-style cubic kernel)
4. Atomic-add into the M5.2 density field. M5.4 MC runs unchanged on
   the result.

No neighbour search, no eigendecomposition, no fallback paths. The
kernel is a fixed-radius sphere with cubic falloff — uniform across
all particles. Mass per particle (integral of cubic kernel over its
3D support) is a fixed constant ≈ `0.21 * 4π/3 * r_world³`, so any iso
threshold tuned for M5.1 needs re-tuning for M5.9 (cubic kernel
concentrates mass differently than the trilinear delta).

#### Sub-blocks

| Sub-ID  | Role                                                   |
|---------|--------------------------------------------------------|
| M5.9.1  | Cubic scatter Warp kernel (per-particle AABB iter)     |
| M5.9.H  | Host wrapper integrated into `MeshExtractor` as a
           `mesh_method = "cubic"` opt-in (default stays M5.1)  |

#### Config knob

In `[output]` section of scene TOML:

```toml
mesh_method = "cubic"          # default: "trilinear" (M5.1)
cubic_radius_cells = 2.0       # support radius in cells
```

Defaults chosen so a scene that doesn't set `mesh_method` is byte-for-
byte unchanged. `mesh_method = "cubic"` swaps M5.1 for M5.9 in the
`MeshExtractor._scatter()` call.

#### Cost expectations

Per-particle: AABB is `(2*r+1)³` cells ≈ `5³ = 125` cells at default
`radius_cells = 2.0`. Each cell does one cubic-falloff evaluation +
one atomic add. For 50k particles at 96³: 6.25M atomic_add → estimated
**3-6 ms per frame** on RTX 4080 SUPER (M5.1 trilinear is ~1 ms for
the same particle count; the 5× ratio matches the AABB cell-count
ratio 125/8). 

At 192³ / 500k: ~50-100 ms per frame. Still bake-able, no longer
realtime.

#### Acceptance criteria

1. **Parity with CPU reference.** GPU output matches `cubic_isotropic_density_cpu`
   from `gpufluid/meshing/cubic_ref.py` element-wise to fp32 precision
   on 1k random particles. Test: `tests/test_m5_9_gpu_cpu_parity.py`.

2. **Demo30 surface quality.** Frame 200 of the step30 water bake
   meshed with M5.9 produces ≤ 300 connected components (down from
   ~6000 with M5.1). Test: `tests/test_m5_9_demo30_cc_count.py`.

3. **No regression on existing scenes.** All scenes that omit
   `mesh_method` produce bit-identical output to before. Test:
   `tests/test_m5_9_default_is_m5_1.py` runs an existing test scene
   with default config and asserts mesh hash unchanged.

4. **Performance budget.** Per-frame meshing time at 96³ / 50k
   particles ≤ 10 ms. Test: `tests/test_m5_9_perf_budget.py`
   (marked `slow`).

#### What this does NOT do

* Does not produce a continuous **mid-air** stream for sparse particles
  spaced > 2 cells apart — that's a fundamental kernel-reach limit
  (B14 reach = 2 cells, beyond that gaps remain). Akinci 2012
  adaptive sampling (kernel widens in sparse regions) is the next
  step for that, tracked separately as future work.
* Does not give "production water" surface smoothness — for a truly
  smooth water look you still need either Bridson SDF (next-tier
  algorithm) or aggressive `mesh_smooth_passes` (we already proved
  with `mesh_smooth_passes = 20` that Taubin smoothing alone gives
  ~80% of the way to water-looking surface).
* Does not implement Yu-Turk anisotropic shape — that's M5.8, left
  on the shelf as evaluated-and-deferred.

#### Risk

* **Iso threshold needs re-tuning.** M5.1 trilinear and M5.9 cubic
  have different integral mass per particle (~8× different) and
  different peak density. Existing iso thresholds in TOML scenes are
  tuned for M5.1 and will need to be raised by ~5-10× for M5.9. The
  default `iso_level = 0.2` should become `iso_level = 1.5-2.0` for
  M5.9. Migration note in BACKLOG.md B14.4.
* **Atomic contention.** 6× more atomic adds than M5.1. We've already
  shown this is fine at 50k particles, but at 500k particles the
  atomic-add throughput may become the meshing bottleneck. M5.9.1
  GPU bench (B14.4) will measure this.

---

### 8.4 M5.10 — Akinci adaptive cubic kernel

**Reference:** Adams, Pauly, Keiser, Guibas 2007 *Adaptively Sampled
Particle Fluids* (SIGGRAPH); Akinci/Solenthaler-line follow-ups on
SPH adaptive sampling. The specific "per-particle kernel radius from
local neighbour count" formulation we use here.

#### Why this fix

M5.9 ships a wider cubic kernel of **fixed** radius (default 2 cells).
B14.3 / B14.6 demo30 evaluation (2026-05-17) showed M5.9 gives a
visibly smooth surface on the **cube** and the **pool** — but the
**mid-air column** between emitter and cube remains broken. The
diagnostic histogram (frame 200, cubic R=2, iso=0.8) shows a
**17 cm empty band** in mid-air vertex y-positions: mesh exists near
the emitter, mesh exists near the cube, **nothing in between**. Cause:
gravity accelerates the falling particles → particle spacing in mid-air
grows from ~1 cell (top) to ~3-4 cells (bottom) → fixed R=2 cell kernel
can't bridge that.

Two pivots we tried and rejected:

* **Brute-force R=4 + iso=6** (B14.3 retrospect, 2026-05-18). Mid-air
  column appears but cube becomes a **gel-cube** — surface inflates
  everywhere, no separation between "stream" and "pool".
* **Brute-force rate=30k + R=4 + iso=6** (Option A retro 2026-05-18).
  Same gel-cube failure mode, worse — entire cube engulfed in one
  cohesive blob.

The lesson: **a single radius can't be right for both regions**. The
mid-air column needs *wider* kernel to bridge sparse particles; the
cube/pool needs *narrower* kernel to stay sharp. **Per-particle
adaptive radius** is the only way to satisfy both simultaneously.

#### Algorithm

For each particle `p_i`:

1. **Neighbour count.** Use `wp.HashGrid` (reuse the per-frame one
   built by W7.7 or build a dedicated one) to count
   `n_i = | { p_j : ||p_j - p_i|| < base_radius_world } |`.

2. **Adaptive radius.** Compute
     `R_i = base_radius_cells × (n_target / n_i)^(1/3)`
   * `n_target` is the desired neighbour count for a "well-supported"
     particle (default 30 — matches Akinci/Adams reference scenes;
     n_target tuned empirically for our 96³ regime).
   * Cube-root exponent because volume scales as R³: doubling R for
     a particle gives it 8× the support volume → 8× more neighbours
     in expectation.
   * Clamp R_i to `[R_min, R_max]` (defaults `[base_R, 4·base_R]`) so
     dense regions don't shrink below the resolution floor and sparse
     regions don't extend past sensible bounds.

3. **Per-particle scatter.** Each particle scatters its cubic kernel
   with its own `R_i` into the density grid (same kernel form as
   M5.9.1, only the radius is now per-particle).

The output is the same `density: wp.array3d(dtype=float)` that M5.2
smooths and M5.4 meshes. **M5.2 / M5.4 / M5.5 / M5.6 unchanged.** Only
M5.9.1's `k_cubic_scatter` (or its M5.10 sibling) needs the per-particle
radius input.

#### Sub-blocks

| Sub-ID   | Role                                                        |
|----------|-------------------------------------------------------------|
| M5.10.1  | `wp.HashGrid` build + per-particle neighbour-count kernel   |
| M5.10.2  | Per-particle adaptive radius kernel (cube-root scaling + clamp) |
| M5.10.3  | Cubic scatter with per-particle radius (variant of M5.9.1)  |
| M5.10.H  | Host wrapper, `mesh_method = "adaptive_cubic"` opt-in       |

All sub-blocks are device-resident; no D→H syncs introduced.

#### Config knobs

In `[output]` section of scene TOML:

```toml
mesh_method = "adaptive_cubic"   # default stays "trilinear" (M5.1)
adaptive_base_cells = 2.0        # base R, same as M5.9
adaptive_n_target = 30           # target neighbours (Akinci default)
adaptive_R_max_factor = 4.0      # clamp: R_i ≤ R_max_factor * adaptive_base_cells
adaptive_R_min_factor = 1.0      # clamp: R_i ≥ R_min_factor * adaptive_base_cells
```

Hashgrid cell size is set internally to `R_max = R_max_factor * base_R`
so neighbour queries see all potential contributors.

#### Cost expectations

For 50k particles at 96³, per frame:

* HashGrid build: ~1 ms (same pattern as W7.7).
* Neighbour count: per-particle queries hashgrid with `R_max` radius,
  yields ~50-100 neighbours per query → ~5-10M memory reads,
  **~2-4 ms**.
* Adaptive R compute: trivial per-particle math, **<0.5 ms**.
* Per-particle scatter: same atomic-add pattern as M5.9 but with
  per-particle AABB size. Average AABB now larger because some
  particles have R_i > base_R. Estimate **2-3× M5.9 cost = ~15-30 ms**.

**Total realistic budget: 20-40 ms per frame at 96³ / 50k.** Sim is
unaffected. At our 600-frame bake, M5.10 adds ~10-20 s to the total
bake compared to M5.9 — still acceptable for offline rendering.

#### Acceptance criteria

1. **Mid-air column visible.** Demo30 frame 200 baked with M5.10
   produces a continuous mid-air mesh column between emitter
   (y=0.95) and cube top (y=0.55). Quantified: the 17 cm "empty
   band" gap (B14.3 baseline) shrinks to ≤ 5 cm. Test:
   `tests/test_m5_10_demo30_midair_gap.py`.

2. **Cube + pool surface unchanged.** Demo30 frame 200's vertex
   distribution in `y < 0.55` (cube + pool) is within 20% of the
   M5.9 baseline (vertex count, surface area). Test:
   `tests/test_m5_10_no_pool_regression.py`. Acceptance: the
   adaptive kernel must NOT inflate cube/pool surfaces (the
   "gel-cube" failure mode that brute-force R=4 produced).

3. **GPU/CPU parity.** `cubic_adaptive_density_cpu` reference matches
   GPU output to fp32. Test: `tests/test_m5_10_gpu_cpu_parity.py`.

4. **Determinism.** Same input → same output (atomic-add commutes on
   floats up to fp32 noise). Test:
   `tests/test_m5_10_deterministic.py`.

5. **Default unchanged.** Scenes that omit `mesh_method` produce
   bit-identical output to M5.1. Test (extend existing
   `test_m5_9_default_is_m5_1.py`).

6. **Visual on demo30.** Test-render at frame 200 shows a visible
   smooth column from emitter to cube. Manual inspection, anchor
   PNG committed under `docs/images/m5_10_before_after/`.

#### What this does NOT do

* Does not produce *production-quality* smoothness on the surface — the
  ellipsoid bumps stay. SDF-based reconstruction (Bridson 2007) is
  still the next-tier upgrade after this; tracked separately.
* Does not solve the **flux mismatch** in demo30 (we still emit ~1 L/s
  on a 30 cm cube). Scene-design TOML pass needed in parallel; not
  blocking on M5.10.
* Does not add whitewater spray/foam — W7.x is a separate system.

#### Risk

* **Hashgrid cell-size mismatch.** If `R_max_factor` increases the
  required hashgrid cell size beyond what W7.7 uses, we cannot reuse
  W7.7's hashgrid — need our own. Solved by always building a
  dedicated hashgrid sized for `R_max` (cost: ~1 ms cold per frame).
* **Per-particle AABB variance.** Particles in sparse zones get large
  AABBs (up to ~10³ = 1000 cells). This is fine on average but creates
  load imbalance across GPU threads. Mitigated by Warp's per-particle
  parallelism — slow particles don't block fast ones, just shift the
  tail latency.
* **Adaptive R-range tuning.** `n_target`, `R_min_factor`, `R_max_factor`
  are scene-dependent. Defaults from Akinci/Adams are reasonable but
  may need per-scene tuning. Documented as known limitation; if it
  becomes a pain point, add a `--mesh-method-auto-tune` pass that
  measures the per-particle n_i distribution and suggests defaults.

---

### 8.5 M5.11 — Bridson SDF surface reconstruction

**Reference:** Bridson 2007 *Fluid Simulation for Computer Graphics*
§5.4 (surface reconstruction); Bridson 2008 SIGGRAPH course "Fluid
Simulation" notes. Adapted for per-cell signed-distance computation
on a regular grid using `wp.HashGrid` for neighbour queries.

#### Why this fix and why not another

M5.1, M5.9, M5.10 all belong to the **same algorithm class**:
*density-then-MC*. Each particle deposits "mass" into a kernel, the
sum of all particle kernels gives a density field, marching cubes
extracts an iso-contour. This class has **fundamental visual limits**:

* Surface is intrinsically **bumpy**: it's a level set of summed
  kernels, so individual particle bumps always show.
* Thin features get **chunky or vanish**: a single-particle-wide
  stream is a single kernel sphere, not a smooth thin tube.
* Reduces to a parameter chase: M5.10 adaptive vs M5.9 vs heavy
  smoothing all trade off "fragmented" vs "gel-cube" without
  delivering a clean **production-water look**.

The B14.6 / B15.x demo30 evaluations made this concrete: at 96³ and
192³, with cubic and adaptive cubic kernels, with smoothing tuned from
0 to 20 passes, the *best* result is still "water tower of bumps on
cube" — never a thin smooth laminar line.

**Bridson SDF is a different algorithm class.** Each cell stores
**`phi = distance(cell, nearest_particle) - particle_radius`**, then
MC at `level = 0`. The surface is the union of particle-spheres,
*by construction smooth*: every contour is at a fixed distance from
the nearest particle, so isolated particles paint clean spheres,
overlapping particles form clean fillets, and the resulting mesh
has the surface area of a real geometric union of spheres — not the
sum of kernel bumps.

Houdini, Mantaflow, RealFlow all use SDF reconstruction (often with
extensions like Akinci/Yu-Turk anisotropic SDF). This is the
production-tier algorithm.

#### Algorithm

For each cell `c` at world position `x_c`:

1. **Nearest particle search.** Query `wp.HashGrid` for particles
   within `search_radius_world = search_cells * dx`. Compute squared
   distance to each, track the minimum.
2. **Signed distance.** Let `d_min` be the nearest distance.
     `phi(c) = d_min - particle_radius_world`
   * Inside-particle cells get negative phi.
   * Outside-but-near-particle cells get positive phi.
   * Cells with no particle within `search_radius` get
     `phi = +search_radius` (positive sentinel — they're outside).

3. **Mean-curvature smoothing** (optional but recommended). Apply a
   few box-blur passes (M5.2 kernel reused) directly on `phi`. This
   smooths the level set without changing topology — fixes
   particle-radius discretisation artefacts at near-contact between
   sphere boundaries.

4. **Marching cubes at level 0.** MC threshold = 0 (instead of a
   positive iso). Cells with `phi < 0` are inside fluid; cells with
   `phi > 0` are outside; the mesh tracks the `phi = 0` contour.

The output is a `phi: wp.array3d(dtype=float)` field — same shape as
M5.1's density field — but the MC threshold semantic flips. M5.2 box-
blur reuses unchanged (smoothing a SDF gives a smoothed SDF). M5.4 GPU
MC reuses with threshold=0. M5.5 mesh smoothing + M5.6 decimate
unchanged.

#### Sub-blocks

| Sub-ID   | Role                                                          |
|----------|---------------------------------------------------------------|
| M5.11.1  | `wp.HashGrid` build sized for `search_radius_cells`           |
| M5.11.2  | Per-cell SDF compute kernel (nearest-particle distance)        |
| M5.11.3  | SDF smoothing (level-set Laplacian, reuses M5.2 box-blur)      |
| M5.11.H  | Host wrapper, `mesh_method = "sdf"` opt-in                    |

All device-resident; no D→H syncs introduced.

#### Config knobs

In `[output]` section of scene TOML:

```toml
mesh_method = "sdf"                     # default stays "trilinear" (M5.1)
sdf_particle_radius_cells = 1.0         # radius per particle in cells
sdf_search_radius_cells = 4.0           # how far to look for nearest particle
sdf_smooth_passes = 2                   # box-blur passes on phi
```

`iso_level` is **ignored** when `mesh_method = "sdf"` — MC threshold is
hardcoded to 0 (the SDF zero contour). Documented as migration note.

#### Cost expectations

For 50k particles at 96³, per frame:

* HashGrid build: ~1 ms (reuses W7.7 pattern).
* Per-cell SDF compute: 96³ ≈ 884k cells, each does a hashgrid query
  (~30 cells / ~5-15 particles checked) plus distance arithmetic.
  Estimated **5-15 ms**. Dominant cost.
* SDF smoothing: 2 box-blur passes on a 96³ field, ~1-2 ms.
* MC at level 0: same cost as before (~10-20 ms).

**Total realistic budget: 20-40 ms per frame at 96³ / 50k.** Roughly
matches M5.10's budget (the per-cell work is comparable). Sim time
unaffected.

At 192³ / 50k particles: per-cell scales to 7M cells → **40-80 ms** per
frame. Bake-able.

At 192³ / 500k particles: hashgrid queries return ~5× more candidates,
SDF compute may rise to **100-200 ms** per frame. Acceptable for
offline.

#### Acceptance criteria

1. **Surface smoothness.** Per-vertex roughness on a synthetic thin
   sheet of particles (200 points in single-cell-thick slab) is
   ≤ 30% of the M5.9 cubic baseline roughness. Test:
   `tests/test_m5_11_sheet_smoothness.py`. (Reusing the same
   synthetic setup as M5.10 tests.)

2. **Thin-feature preservation.** A synthetic 1-cell-wide column of
   200 particles produces **1 connected mesh component** with mean
   cross-sectional thickness ≤ 2.5 cells (i.e. the mesh is a thin
   tube, not a fat blob). Test: `tests/test_m5_11_thin_column.py`.

3. **Demo30 visual.** Re-bake step30 with `mesh_method = "sdf"` →
   anchor frames show a **smooth thin water stream from emitter to
   cube** (subjective acceptance — anchor PNG diffed against current
   M5.9 baseline). Test: manual inspection + `docs/images/m5_11_sdf/`.

4. **CPU/GPU parity.** GPU SDF output matches `sdf_density_cpu`
   reference to fp32 absolute tolerance 1e-3 on a 200-particle
   random cloud at 32³. Test: `tests/test_m5_11_gpu_cpu_parity.py`.

5. **Default unchanged.** Scenes without `mesh_method` produce
   bit-identical output to M5.1. Existing
   `test_m5_9_default_is_m5_1.py` extends to cover M5.11.

6. **Determinism.** Same input → same output (atomic-add isn't used
   here — per-cell is independent — so determinism is structural).
   Test: `tests/test_m5_11_deterministic.py`.

#### What this does NOT do

* **Flux mismatch unchanged.** demo30 still emits ~1 L/s on a 30 cm
  cube. SDF gives smooth surface but water still accumulates on cube
  faster than it drains. Scene-design pass remains separate work
  (tracked in HANDOFF, not B16).
* **No whitewater spray/foam.** W7.x is a separate system; SDF on
  the bulk surface composes with W7 layered render.
* **No adaptive radius.** All particles use the same
  `particle_radius_cells`. An adaptive-radius extension (Akinci-style
  SDF) could ship as a follow-on M5.12 if the fixed radius proves
  insufficient.
* **No mean curvature flow (proper).** We use box-blur on `phi` as a
  cheap approximation. True mean curvature flow (`∂φ/∂t = κ|∇φ|`)
  preserves the SDF property better but costs more — could ship as
  M5.11.3-alt if box-blur drifts the surface visibly.

#### Risk

* **Particle radius tuning.** Too small → discrete spheres with visible
  gaps between particles. Too large → fat blob hiding fluid structure.
  Default `sdf_particle_radius_cells = 1.0` (= 1 voxel = ~dx) is the
  Bridson book recommendation; tuned upward to 1.5-2 for sparse-fluid
  scenes if needed.
* **Hashgrid query cost at high particle count.** 500k particles +
  finer grid → more candidates per query. If perf becomes the
  bottleneck, narrow-band SDF (only compute phi for cells near
  particles, leave the rest at sentinel) cuts this 10×. Tracked as
  M5.11.X follow-on if needed.
* **Sentinel value for "no particle nearby".** Set to
  `+search_radius_world` so MC sees these cells as cleanly "outside".
  If search_radius is too small, cells genuinely outside fluid can
  end up with phi values close to 0 instead of +search_radius, causing
  spurious mesh fragments. Documented + asserted in B16 tests.



**Reference:** Yu & Turk 2013, *Reconstructing Surfaces of Particle-Based
Fluids Using Anisotropic Kernels*, ACM TOG 32(1).

Replaces the isotropic spherical kernel that M5.1 scatters into the
density grid with a per-particle *oriented ellipsoid* whose shape is
derived from the local neighbourhood geometry. This is the single biggest
surface-quality lever for particle fluids that exists today and is what
Houdini / Mantaflow / RealFlow use under the hood.

#### Why this fix and not another

M5.1's isotropic scatter draws each particle as a sphere. When the fluid
is **thick** (10+ particles per voxel) the spheres overlap into a smooth
sheet — fine. When the fluid is **thin or sparse** (a tap stream, a
splash droplet, the leading edge of a wave) every particle paints a
visible round bump and the marching-cubes surface inherits that
bumpiness: chunky "cauliflower" texture, no continuous thin features.
This is the dominant visual artefact at demo30's 96³ even with all
tuning we've done. Surface roughness is independent of resolution —
192³ produced *more* cauliflower bumps, not fewer, because each finer
particle still paints its own sphere.

Yu-Turk fixes the root cause: the kernel becomes an ellipsoid whose
*long axis* aligns with the local fluid surface direction and whose
*short axis* aligns with the surface normal. A thin sheet of particles
becomes a thin sheet of overlapping ellipsoids → smooth membrane.
A coherent column becomes a smooth tube. An isolated splash droplet
gets a fallback isotropic kernel of small radius, so it does not bloom
into a fake blob.

#### Algorithm

For each particle `p_i`:

1. **Neighbour gather (M5.8.1).** Use `wp.HashGrid` with cell size
   `h = 2·R_kernel` to enumerate `N_i = { p_j : ‖x_j - x_i‖ < R_kernel }`.
   The same hashgrid already exists from W7.7 trapped-air; we reuse it
   per frame rather than building a second.

2. **Weighted centroid (M5.8.2).** Compute
   `x̄_i = Σ_j w_ij · x_j  /  Σ_j w_ij`
   with the cubic falloff kernel
   `w_ij = (1 - (‖x_j - x_i‖/R)²)³`  if  ‖·‖ < R else 0.

3. **Weighted covariance (M5.8.2).** Compute
   `C_i = Σ_j w_ij · (x_j - x̄_i)(x_j - x̄_i)ᵀ  /  Σ_j w_ij` as a
   symmetric 3×3 matrix. Six scalars per particle stored in shared
   `wp.array(dtype=mat33)`.

4. **SVD + eigenvalue clamping (M5.8.3).** Per-particle eigendecomp
   `C_i = R_i Σ_i R_iᵀ`. Three eigenvalues `σ₁ ≥ σ₂ ≥ σ₃`. Clamp
   `σ_k ← max(σ_k, σ_max/k_r)` with `k_r = 4` (Yu-Turk §3.2 default
   anisotropy ratio cap), where `σ_max = σ₁`. Form the inverse-radius
   matrix `G_i = (1/k_n) R_i · diag(1/σ_k) · R_iᵀ` with
   `k_n = (Σ σ_k)/3` (normalisation).

   *GPU SVD note:* a fully iterative SVD on the GPU is expensive. The
   Jacobi-rotation method converges in 6-8 sweeps for a 3×3 and is
   numerically robust. Alternative: closed-form 3×3 cubic-root
   eigenvalue solver (Smith 1961) plus QR for eigenvectors — faster
   per particle but ~30 lines of arithmetic. **Decision: Jacobi
   rotations.** Worth the few extra ops for robustness; 50k particles
   at 8 sweeps is still <1 ms on the 4080S.

5. **Isolated-particle fallback (M5.8.4).** If `|N_i| < N_thresh` (Yu-Turk
   uses 25; we will too for now), skip steps 2-4 and use an isotropic
   kernel of radius `R/2`. This prevents single splash particles from
   producing pancake ellipsoids from noisy 2-3-neighbour covariances.

6. **Anisotropic density scatter (M5.8.5).** For each particle, compute
   its AABB in the density grid using its ellipsoid extents, iterate
   the enclosed cells, and atomic-add a contribution
   `ρ_cell += det(G_i)^(1/2) · P((cell_center - x̄_i)ᵀ G_i (cell_center - x̄_i))`
   where `P(r²) = max(0, (1 - r²)³)` is the same cubic kernel as step 2.

   The `det(G_i)^(1/2)` normalisation keeps the *total mass* of each
   particle equal to 1, so a thin-sheet particle paints the same total
   "ink" as a fat-sphere particle — just spread differently.

The output is the same `density: wp.array3d(dtype=float)` that M5.2
smooths and M5.4 meshes. **Existing M5.2 / M5.4 / M5.5 / M5.6 paths
work unchanged** — only the input to M5.2 changes when anisotropic
is enabled.

#### Block layout

| Sub-ID  | Role                                                                    |
|---------|-------------------------------------------------------------------------|
| M5.8.1  | Per-particle neighbour gather (reuse W7.7 hashgrid)                     |
| M5.8.2  | Weighted centroid + covariance kernel                                   |
| M5.8.3  | Jacobi-rotation eigendecomp + clamping kernel                           |
| M5.8.4  | Isolated-particle fallback selector (`|N_i| < 25` → isotropic flag)     |
| M5.8.5  | Anisotropic ellipsoid density scatter kernel                            |
| M5.8.H  | Host wrapper that wires 1→5 and replaces the M5.1 call when enabled     |

#### Configuration knobs

In `[output]` section of scene TOML:

```toml
mesh_method = "anisotropic"     # default: "isotropic" (M5.1)
ak_radius_cells = 4.0           # support radius in cells (Yu-Turk R)
ak_neighbor_min = 25            # below this -> fallback isotropic
ak_anisotropy_max = 4.0         # k_r clamp (eigenvalue ratio cap)
```

Defaults chosen to match Yu-Turk §3 reference parameters. The
`isotropic` mode keeps current behaviour bit-identical so existing
scenes are unaffected.

#### Cost expectations (honest, with uncertainty bands)

For 50k particles at 96³, per frame. Numbers below are **first-cut
estimates not measurements**; B13.9 perf test reconciles them with
reality at the end.

* Neighbour gather: hashgrid is built per frame. If W7 is already
  building one with compatible cell size, share it (~0 cost); otherwise
  build a dedicated one for M5.8 at ~1-2 ms cold. **Open: confirm
  whether W7.7 hashgrid radius matches Yu-Turk R or whether we need
  two hashgrids.** Treat the cost as ~1-2 ms either way.
* Centroid + covariance: 6 reductions per particle × 50k = 300k ops,
  **~0.3-1 ms** (rough).
* SVD: 8 Jacobi sweeps × 6 off-diagonals × trig pair per particle =
  ~2.4M trig pairs. **~1-3 ms** (rough — Warp's `wp.sin`/`wp.cos`
  speed on Ada GPUs is not something I've benchmarked).
* Anisotropic scatter: per particle iterates an AABB sized roughly
  `(2R+1)³` cells. For R=4 cells, **AABB ≈ 9³ = 729 cells worst
  case**. With 50k particles × ~500 atomic adds (avg, since most
  cells in the AABB are far from the ellipsoid centre and contribute
  near-zero — we will still pay the iteration cost) = **25M atomic
  adds**, **likely 15-40 ms**. This is the dominant cost. Two
  optimisations on the table if we blow the budget: (a) tight
  per-cell early-exit when the kernel value is below ε; (b) shrink
  AABB to the ellipsoid's actual oriented bounding box (cheaper
  iteration, more arithmetic per cell).

**Total realistic budget: 15-50 ms per frame at 96³ / 50k.** M5.1
isotropic scatter is ~1 ms in the same regime, so Yu-Turk is **15-50×
slower than M5.1** for the scatter step. M5.4 MC is currently
~10-20 ms; the anisotropic scatter becomes the new bottleneck. Net
meshing time per frame: ~30-70 ms vs current ~15-25 ms = **roughly
2-3× slower mesh stage**. Sim time is unaffected. At our typical
bake of 600 frames the added cost is **~20-50 s** — still fast
relative to the sim itself.

At 192³ / 500k the scatter scales linearly in particle count and
super-linearly in AABB size (R measured in cells, not metres, so R
stays at 4 → AABB stays ~729 cells; only particle count grows) →
**~150-400 ms per frame** scatter alone. Bake-able but no longer
realtime.

**If actual measurement (B13.9) shows >100 ms at 96³ / 50k, escalate**:
either implement optimisation (a) above as a follow-on block M5.8.6,
or accept a higher per-frame cost as the price of the surface quality.

#### Acceptance criteria

1. **Synthetic sheet test.** A 2D plane of 200 particles (10×20 grid
   in a single cell-thick slab). Isotropic mesh has ≥20% per-vertex
   distance variance to nearest particle plane. Anisotropic mesh has
   ≤5%. Test: `tests/test_m5_8_anisotropic_sheet.py`.

2. **Synthetic column test.** A column of 200 particles (1×1×200 cells
   thick). Isotropic mesh fragments into ≥20 disconnected components.
   Anisotropic produces 1 connected component. Test:
   `tests/test_m5_8_anisotropic_column.py`.

3. **Isolated-particle equivalence.** A single particle with no
   neighbours produces a sphere mesh identical to isotropic
   (radius = `R/2`, fallback path). Test:
   `tests/test_m5_8_isolated_fallback.py`.

4. **Determinism.** Same input particles → same mesh output across
   runs (atomic-add ordering varies but density-grid sum is
   commutative; eigendecomp is deterministic). Test:
   `tests/test_m5_8_determinism.py`.

5. **Visual on demo30.** The step30 water nozzle bake (2cm nozzle,
   rate=8000, 96³) re-baked with `mesh_method = "anisotropic"` shows
   a continuous mid-air column in the rendered video — not the
   fragmented cottage-cheese mid-air we currently see. Test: manual
   inspection of `out/videos/step30_water.mp4`; commit before/after
   anchor frames in `docs/images/m5_8_before_after/`.

6. **Performance ceiling.** Per-frame meshing time at 96³ / 50k particles
   ≤ 2× the M5.1 baseline (target ~4 ms vs ~1-2 ms). At 192³ / 500k
   ≤ 50 ms per frame. Test: `tests/test_m5_8_perf_budget.py` (marked
   `slow`, runs at 96³ only by default).

#### What this does NOT do

Yu-Turk improves the *surface reconstruction* only. It does **not**:
* fix the *flux mismatch* in demo30 (we are still simulating a fire
  hose into a small cube — too much water per second for the cube
  surface to drain naturally; that's a scene-design issue);
* add whitewater spray/foam (W7 is a separate system);
* speed up the pressure solve.

Yu-Turk gives us a clean baseline surface; demo30 scene design
needs separate iteration to look like a tap-pour at the right flux.

#### Risk

* **GPU SVD bugs.** Jacobi rotations on 3×3 SPSD matrices are well-studied
  but easy to get wrong (sign flips, off-diagonal indexing). The
  determinism test (criterion 4) catches drift; the synthetic sheet test
  (criterion 1) catches systematic sign errors via the surface-normal
  axis.
* **Hashgrid memory at 192³ / 500k.** wp.HashGrid for 500k particles
  with cell size ~2 cm × 96³ domain = 96³ hash buckets ≈ 900k. Fits
  easily on the 4080S; same usage pattern as W7.7.
* **Atomic contention on dense fluids.** A 96³ scatter from 50k particles
  with ellipsoid AABBs of ~64 cells = 3.2M atomic_add — high contention
  but well within Warp's per-frame budget.



Persistence: write meshes to disk, manage a cache directory.

| ID    | Block | Status |
|-------|-------|--------|
| I6.1  | PLY binary writer                                            | impl |
| I6.2  | Cache manifest (`cache.json`: frames, fps, format, version)  | planned |
| I6.3  | Particle dump (`.npy`)                                       | impl |
| I6.4  | Alembic writer                                               | planned |
| I6.5  | USD writer                                                   | planned |

### 9.1 Cache layout on disk

```
<cache_dir>/
    cache.json                # manifest (I6.2)
    mesh/frame_NNNN.ply       # surface per frame (I6.1)
    particles/frame_NNNN.npy  # raw particles per frame (I6.3, optional)
    preview/frame_NNNN.png    # quick render (optional)
```

---

## 9. Layer I6 — I/O

Persisting solver output and reading caches back. Lives between M5
(produces vertex/face arrays) and C7 (orchestrates a bake) — the
write/read primitives for the on-disk cache layout described above.

Blocks (full descriptions in [BLOCKS.md](BLOCKS.md)):

* **I6.1** — binary little-endian PLY reader (mirror of `write_ply`).
* **I6.2** — `cache.json` manifest reader.
* **I6.3** — particle dump (numpy `.npy` of positions per frame).
* **I6.1.MESH** — vectorized PLY mesh+face parser (`io.ply.read_mesh_ply`).
  Companion to I6.1's point-cloud reader. Replaces the per-face Python
  loop with `np.frombuffer` reshape on the
  `[uchar count][int32 v0][int32 v1][int32 v2]` row layout. For a
  10k-face mesh: ~50ms → <1ms. Used in the Blender bridge (A8.10)
  which loads one PLY per render frame. (Note: `I6.4` is reserved
  by B10 — Alembic writer.)
* **I6.5** — USD time-sampled mesh sequence writer (`cache.usdc`),
  the polished Blender-import path.

No design decisions live here that aren't already captured by the
block table; this section exists so the layer-declared registry check
(`tests/test_blocks_registry.py::test_check_2_layer_declared`) can find
the layer heading.

---

## 10. Layer C7 — CLI

Standalone, no Blender dependency. Drives the solver from a config file.

| ID    | Block | Status |
|-------|-------|--------|
| C7.1  | Config schema (TOML)                                         | planned |
| C7.2  | `gpufluid simulate <config>` command                         | planned |
| C7.3  | `gpufluid bench` command                                     | partial (script) |
| C7.4  | `gpufluid info` (lists devices, version)                     | planned |

---

## 11. Layer A8 — Blender addon

Lives in `addon/gpufluid_blender/` (separate from `src/gpufluid` so the
addon zip is self-contained). The addon is a thin Blender-UI wrapper:
it builds a config and invokes the CLI (C7.2) as a subprocess, then
imports the resulting mesh cache (I6.x) onto a target object.

| ID    | Block | Status |
|-------|-------|--------|
| A8.1  | Addon registration (`register/unregister`)                   | planned |
| A8.2  | `GpufluidDomain` property group on Empty                     | planned |
| A8.3  | `GpufluidFluid` property group (source/initial volume)       | planned |
| A8.4  | `GpufluidObstacle` property group                            | planned |
| A8.5  | Bake operator (export config → spawn CLI → progress UI)      | planned |
| A8.6  | Cache import (PLY sequence → MeshSequenceCache modifier)     | planned |
| A8.7  | UI panels (3D-view sidebar)                                  | planned |
| A8.8  | Helper operators (add domain, add fluid, clear cache)        | planned |
| A8.9  | Eevee perf preset — explicit samples/bloom/SSR/GTAO setup for headless renders. Without this Blender uses photo-quality defaults (~3× slower) | impl |
| A8.10 | Mesh swap-in frame loader — Blender frame-change handler that reads `cache_dir/mesh/frame_NNNN.ply` via I6.4 and rebuilds the surface object's vertex/face buffers in-place via `foreach_set` | impl |
| A8.11 | Scene builder helpers — camera/lights/material/overlay primitives for the Blender bridge. Composable with A8.5..A8.8 or callable standalone for headless renders | impl |
| A8.12 | Headless render CLI command `gpufluid render <cache> <scene> --out <png_dir>` — wraps A8.9 + A8.10 + A8.11 inside a `blender --background --python` subprocess. Used by tests and CI | impl |

---

## 11.5 Layer W7 — Whitewater simulation

Foam, spray, and bubble secondary-particle dynamics. Operates downstream
of the main FLIP solver (consumes F3's velocity field + particle state)
but writes its own particle pool that the meshing layer (M5) reads back
for surface foam compositing. Imports allowed: G1 + S2 + F3 + D4 (sits
between D4 and M5 in dependency order; see §2 diagram).

Lives in `src/gpufluid/sim/` — the folder name predates the strict
naming convention. The folder→layer mapping in §2.1 records this
exception.

| ID    | Block                                                  | Source                                         |
|-------|--------------------------------------------------------|------------------------------------------------|
| W7.1  | `WhitewaterSystem` (state container)                   | `gpufluid/sim/whitewater.py`                   |
| W7.2  | `emit_from_fluid()` — emit secondaries from main pool  | `gpufluid/sim/whitewater.py`                   |
| W7.3  | Ballistic advection                                    | `gpufluid/sim/whitewater.py`                   |
| W7.4  | `classify_kinds()` — foam / spray / bubble heuristic   | `gpufluid/sim/whitewater.py`                   |
| W7.5  | `step()` — per-class dynamics (gravity/drag/buoyancy)  | `gpufluid/sim/whitewater.py`                   |
| W7.6  | Kind sidecar I/O (`whitewater/frame_NNNN.npy`)         | `gpufluid/cli/commands.py`                     |
| W7.7  | Trapped-air potential (Ihmsen 2012)                    | `gpufluid/sim/whitewater_potentials.py`        |
| W7.7.H| Host wrapper (numpy → numpy) for W7.7                  | `gpufluid/sim/whitewater_potentials.py`        |
| W7.8  | Wave-crest potential I_wc = |∇·n̂| via standalone P2G→blur→∇→∇· pipeline (no CSF dependency) | `gpufluid/sim/whitewater_potentials.py` |
| W7.8.H| Host wrapper for W7.8                                  | `gpufluid/sim/whitewater_potentials.py`        |

### 11.5.1 W7.8 wave-crest formulation

The whitewater classifier wants high emission rates exactly where the
free surface curves outward — i.e. wave crests and breaking ridges.
Ihmsen 2012 §3.2 defines the wave-crest potential as

```
I_wc(i) = |∇·n̂(x_i)|
```

where `n̂` is the unit surface normal at particle `i`. The natural
particle-only evaluation:

1. **Scatter** unit indicator to a cell-centred grid χ via trilinear P2G.
2. **Smooth** χ → χ̃ with N box-blur passes (reuses [BLK G1.7]).
3. **Normal** n̂ = ∇χ̃ / (|∇χ̃| + ε) on cell centres.
4. **Divergence** ∇·n̂ via central differences.
5. **Sample** |∇·n̂| back per-particle via trilinear gather.

W7.8 ships the kernels for steps 1, 3, 4, 5 (step 2 reuses G1.7). The
host wrapper W7.8.H mirrors W7.7.H — numpy in, numpy out, no solver
state — so callers can run whitewater on a particle dump without
constructing a `FlipSolver3D`.

**Why not reuse S2.14.1/2/3?** Those kernels live on the solver's MAC
grid and take the `marker` field as input (encodes fluid/solid cells).
Reusing them would chain whitewater to solver state and require σ>0
(CSF only allocates its grids when surface tension is enabled). W7.8
is intentionally standalone so the wave-crest classifier works on
scenes without CSF and on offline particle dumps.

**Range:** raw |∇·n̂| has units of 1/length and is unbounded; W7.8
normalises by `1/dx` (cells per meter) and clamps to [0, 1]. Wave
crests typically saturate near 1; flat surface and interior particles
sit below 0.05.

**Inputs:** particle positions (N,3) float32, grid resolution
(nx,ny,nz), cell size `dx`, blur passes `n_blur` (default 2). No
velocities needed — wave-crest is a geometric measure.

---

## 12. Data flow (one frame)

```
                 user config (toml / addon UI)
                          │
                          ▼
                ┌──────────────────────┐
                │  build domain (D4)   │  walls + obstacles + seeders
                └─────────┬────────────┘
                          ▼
                ┌──────────────────────┐
                │ FlipSolver (F3)      │ owns u, v, w, p, marker, particles
                └─────────┬────────────┘
   per frame: ┌──────────┴──────────┐
              │ N substeps (S2.10)  │
              │  step pipeline:     │
              │   S2.1 .. S2.9      │
              └──────────┬──────────┘
                         ▼
                ┌──────────────────────┐
                │ extract mesh (M5)    │
                └─────────┬────────────┘
                          ▼
                ┌──────────────────────┐
                │ write cache (I6)     │  PLY frame_NNNN.ply
                └─────────┬────────────┘
                          ▼
                ┌──────────────────────┐
                │ Blender import (A8)  │  MeshSequenceCache on object
                └──────────────────────┘
```

---

## 13. Testing strategy

`tests/` mirrors `src/gpufluid/`. One pytest file per layer:

| File | Covers blocks | Strategy |
|------|---------------|----------|
| `tests/test_g1_primitives.py` | G1.x | Synthetic input/output of helpers (trilinear sums to 1, sample = scatter inverse on grid). |
| `tests/test_s2_schemes.py`    | S2.x | P2G + G2P round trip preserves momentum within tolerance; pressure Jacobi reduces residual; gravity adds exactly g·dt. |
| `tests/test_d4_sdf.py`        | D4.2.* | Analytic SDF correctness on known points (centre = -r; far away = far). |
| `tests/test_m5_meshing.py`    | M5.x | Particles uniformly in a cube produce a mesh with bounding box within ε of cube. |
| `tests/test_i6_io.py`         | I6.x | PLY write → read back via trimesh, verts/faces match. |
| `tests/test_f3_solver.py`     | F3.x | Smoke test: 10 steps don't blow up, no NaN, particle count stable. |
| `tests/test_integration.py`   | end-to-end | Dam-break sanity: after 0.5 s, water mass distribution shifted right of initial column. |

CI gate: all tests green; `python -m gpufluid.blocks --check` reports no
orphan blocks/code; `ruff check` clean.

### 13.1 Test conventions
- `pytest -m gpu` skipped on machines without CUDA.
- All numerical tests use fixed seeds for reproducibility.
- Tolerances stated as constants at top of each test file.
- Test functions named `test_<block_id_underscored>_<aspect>`.

---

## 14. Coding conventions

- Python 3.11, type hints everywhere.
- Warp kernels prefixed `k_` (2D) or `k3_` (3D).
- One public class per solver file.
- Docstrings: numpy style. First line one-sentence summary.
- Every public callable has a `[BLK X.Y]` comment + `@block` decorator
  (decorator may be a no-op at runtime; it just registers).
- No silent excepts. Failures raise `BlockError` with block ID.

---

## 15. Roadmap

| Milestone | Blocks delivered | Demo |
|-----------|------------------|------|
| v0.1 (current sprint) | G1, S2.1–S2.9, F3.1–F3.3, D4.1, D4.2.1–D4.2.3, D4.2.5, D4.4, D4.5.1, M5.1–M5.3, I6.1, I6.3 | Dam-break with cylinder obstacle, PLY mesh sequence, manual import to Blender |
| v0.2 | D4.3 (mesh→SDF), D4.5.2 (mesh seed), C7.1–C7.4 (full CLI), I6.2 (cache manifest) | `gpufluid simulate scene.toml` from terminal, any-mesh obstacle |
| v0.3 | A8.* (full Blender addon) | Click-Bake in Blender, cache auto-loads on target object |
| v0.4 | S2.6.3 (PCG), S2.10 (CFL substepping), M5.5 (mesh smoothing) | 256³ scenes at sane speed, less jittery surface |
| v0.5 | D4.6 (anim obstacles), D4.7 (in/outflow) | Pouring water, animated splashers |
| v1.0 | whitewater, viscosity, FF integration, USD/Alembic | Production-grade comparable to FLIP Fluids basic feature set |

---

## 16. Open decisions

These are flagged for explicit review during implementation:

- **Q1.** Cache format for v0.1 stay PLY-per-frame, or commit to USD now?
  Decision: **PLY for v0.1** (no extra deps), USD in v0.2 once `usd-core`
  install path is verified on Windows.
- **Q2.** Pressure solver default for v0.1: keep Jacobi or push PCG?
  Decision: **Jacobi** (correct, simpler, tested). PCG landed in v0.4
  (S2.6.3) but currently slower in wall-clock at all tested sizes
  (64³–128³) because alpha/beta/dot scalars round-trip CPU↔GPU each
  iteration. Per-iter convergence is 5–10× better than Jacobi. Fix:
  keep CG scalars on GPU using `wp.array(shape=1)` for alpha/beta and
  an axpy kernel that reads scalars from device arrays (v0.5).
- **Q3.** Where do animated obstacles live? Rebuild SDF every frame
  (slower, exact) or rasterize from animated MAC velocity (faster,
  approximate)? Decision: rebuild SDF per frame, deferred to v0.5.

---

*End of DESIGN.md*
