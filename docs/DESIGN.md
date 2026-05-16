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
│ D4  Domain                 (SDF, seeders, walls, inflows)    │
├──────────────────────────────────────────────────────────────┤
│ F3  Solver orchestration   (FlipSolver2D, FlipSolver3D)      │
├──────────────────────────────────────────────────────────────┤
│ S2  Numerical schemes      (P2G, pressure, G2P, advect)      │
├──────────────────────────────────────────────────────────────┤
│ G1  GPU primitives         (Warp init, grid math, helpers)   │
└──────────────────────────────────────────────────────────────┘
```

Allowed imports: layer N may import only layers `1..N-1`.

### 2.1 Folder mapping

```
src/gpufluid/
    blocks.py              # block registry + decorator (cross-cutting, no layer)
    primitives/            # G1
    schemes/               # S2  (Warp kernels split by scheme)
    solvers/               # F3
    domain/                # D4
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

### 5.3 D4.3.GPU.BVH — BVH-accelerated mesh-inside test

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

The pipeline order in F3.3 is fixed:

```
clear_grid → P2G(S2.1) → normalize(S2.2) → gravity(S2.3) → bc(S2.4)
  → divergence(S2.5) → pressure(S2.6) → grad_subtract(S2.7) → bc(S2.4)
  → G2P_advect(S2.8 + S2.9)
```

---

## 7. Layer D4 — Domain

Geometry of the simulation domain: walls, obstacles, fluid seeders.

| ID      | Block | Status |
|---------|-------|--------|
| D4.1    | Solid wall shell (init marker boundary)                       | impl |
| D4.2    | SDF analytic primitives                                       | —    |
| D4.2.1  |   Sphere                                                      | impl |
| D4.2.2  |   Box                                                         | impl |
| D4.2.3  |   Cylinder (Y-aligned)                                        | impl |
| D4.2.4  |   Plane                                                       | planned |
| D4.2.5  |   Union / intersect / subtract operators                      | impl (union) |
| D4.3    | Mesh → SDF (triangle-soup distance)                           | planned |
| D4.4    | Apply SDF as solid markers                                    | impl |
| D4.5    | Fluid seeders                                                 | —    |
| D4.5.1  |   Box seeder (uniform jittered)                               | impl |
| D4.5.2  |   Mesh seeder (volumetric fill)                               | planned |
| D4.6    | Animated obstacles (rebuild SDF per frame)                    | planned |
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
| M5.1  | Particle density grid scatter                                | impl |
| M5.2  | Density grid smoothing (box-blur N passes)                   | impl |
| M5.3  | Marching cubes (skimage, CPU)                                | impl |
| M5.4  | Marching cubes on Warp (GPU port)                            | planned |
| M5.5  | Taubin / Laplacian mesh smoothing                            | planned |
| M5.6  | Mesh decimation                                              | planned |

---

## 9. Layer I6 — I/O

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
