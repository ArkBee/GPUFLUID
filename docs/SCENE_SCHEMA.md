# Scene TOML schema

The CLI `gpufluid simulate <scene.toml>` is the engine's only external
interface. Other DCC bridges (Blender addon, future Houdini/Maya plugins)
construct a TOML file and shell out to the CLI — that's the contract.

Source of truth: [`src/gpufluid/cli/config.py`](../src/gpufluid/cli/config.py).
This document is a human-readable summary; if it diverges from `config.py`,
trust `config.py`.

## Top-level structure

```toml
[domain]      # mandatory — grid resolution
[fluid]       # one initial fluid source (legacy single-source form)
[[fluids]]    # OR an array of sources (v0.7+, each with optional color)
[[obstacle]]  # 0..N obstacles
[[inflow]]    # 0..N continuous fluid sources active for a frame range
[[outflow]]   # 0..N drain regions
[simulation]  # solver parameters
[output]      # cache directory + meshing + whitewater knobs
```

The simulator works in a positive sim-space coordinate system anchored at
`(0, 0, 0)`. The domain extends to `(nx·dx, ny·dx, nz·dx)` where `dx`
defaults to `1 / max(nx, ny, nz)` so a unit-cube domain is the default.
External bridges typically translate world coordinates by subtracting the
domain's lower-corner before writing positions into the TOML.

## `[domain]`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `resolution` | `[int, int, int]` | `[64, 64, 64]` | Cells per axis (MAC grid) |
| `dx` | `float` | `1 / max(resolution)` | Cell size in metres |

## `[fluid]` (legacy single-source) or `[[fluids]]` (multi-source)

Box source:

```toml
[fluid]                  # or [[fluids]]
type = "box"
lo = [0.10, 0.10, 0.10]
hi = [0.40, 0.40, 0.40]
ppc = 8                  # particles per cell ≈ ppc (target density)
color = [1.0, 0.1, 0.1]  # optional, S2.15 — linear RGB
```

Mesh source (fills the triangle mesh interior, GPU ray-cast):

```toml
[fluid]                  # or [[fluids]]
type = "mesh"
path = "bunny.obj"
scale = 0.5
translate = [0.1, 0.0, 0.1]
ppc = 8
color = [0.5, 0.5, 0.5]  # optional
```

Use the `[[fluids]]` array form when you have more than one source or you
need per-source colour. The legacy `[fluid]` table still works for
single-source scenes (kept for back-compat).

## `[[obstacle]]`

Each table sets `type =` to one of:

| `type` | Extra keys |
|--------|-----------|
| `sphere` | `center`, `radius` |
| `box` | `center`, `half_size` |
| `cylinder_y` | `center`, `radius`, `half_height` |
| `plane` | `point`, `normal`, optional `bbox_lo` / `bbox_hi` to clip to a finite ramp |
| `mesh` | `path`, optional `scale`, `translate`, `rotate_deg`. Auto-routes to GPU BVH ≥256 tris |

All obstacle types accept an optional `motion` sub-table for animation:

```toml
[[obstacle]]
type = "box"
center = [0.5, 0.2, 0.5]
half_size = [0.1, 0.05, 0.1]
motion = { kind = "linear", velocity = [0.0, 0.0, 0.3] }

[[obstacle]]
type = "sphere"
center = [0.5, 0.5, 0.5]
radius = 0.1
motion = { kind = "keyframes", keyframes = [[0, [0.5, 0.5, 0.5]],
                                             [30, [0.5, 0.2, 0.5]]] }
```

## `[[inflow]]` / `[[outflow]]`

Inflow seeds particles into a region every frame in its active range:

```toml
[[inflow]]
lo = [0.40, 0.85, 0.40]
hi = [0.60, 0.95, 0.60]
velocity = [0.0, -6.0, 0.0]
rate_per_sec = 50000
frame_start = 0
frame_end = 120
```

Outflow culls particles inside the region every frame:

```toml
[[outflow]]
lo = [0.0, 0.0, 0.0]
hi = [1.0, 0.05, 1.0]
frame_start = 0
frame_end = 1000000
```

## `[simulation]`

| Key | Default | Notes |
|-----|---------|-------|
| `dt` | 0.005 | Base time step (s) |
| `pressure_iters` | 60 | Inner pressure-solve iterations |
| `pressure_solver` | `"jacobi"` | `"jacobi"` \| `"gsrb"` \| `"pcg"` |
| `cfl` | `false` | If `true`, auto-substep via `step_cfl`. Forced to `true` when `surface_tension > 0` |
| `cfl_factor` | 0.5 | Particle CFL coefficient |
| `cfl_max_substeps` | 16 | Hard cap on substeps per frame |
| `flip_blend` | 0.95 | 1 = pure FLIP, 0 = pure PIC. APIC ignores this |
| `gravity` | -9.81 | World-Y gravity (m/s²) |
| `rho` | 1.0 | Density |
| `viscosity` | 0.0 | Kinematic viscosity (m²/s). Add a small amount to suppress CSF parasitic currents |
| `viscosity_iters` | 12 | Implicit viscosity inner iters |
| `surface_tension` | 0.0 | σ/ρ for S2.14 CSF. Non-zero enables capillary-CFL substepping |
| `csf_smoothing_passes` | 2 | Box-blur passes on χ before curvature |
| `transfer_mode` | `"flip"` | `"flip"` \| `"pic"` \| `"apic"` |
| `reseed` | `false` | Bound per-cell density |
| `reseed_every_n_frames` | 5 | Cadence (see HANDOFF trap #12 — don't over-tune) |
| `reseed_min_per_cell` | 4 | |
| `reseed_max_per_cell` | 16 | |
| `frames` | 100 | Total frames to bake |
| `fps` | 24 | |

## `[output]`

| Key | Default | Notes |
|-----|---------|-------|
| `cache_dir` | `"out/sim"` | Per-frame outputs land here. Path is relative to the TOML file |
| `mesh` | `true` | Write surface mesh per frame |
| `iso_level` | 0.6 | Density iso for MC. Higher = thinner mesh |
| `smooth_passes` | 2 | Pre-MC density blur |
| `mesh_smooth_passes` | 0 | Post-MC mesh smoothing |
| `mesh_smooth_method` | `"taubin"` | `"taubin"` (volume-preserving) \| `"laplacian"` |
| `decimate_ratio` | 1.0 | Keep this fraction of triangles (1.0 = no decimate) |
| `wall_margin_cells` | 0 | Strip mesh near domain walls |
| `particles` | `false` | Write per-frame `.npy` of positions (needed for colour) |
| `usd` | `false` | Write `cache.usdc` — Blender imports natively via MeshSequenceCache |
| `whitewater` | `false` | Enable foam/spray/bubble system |
| `whitewater_speed_threshold` | 4.0 | Emit gate |
| `whitewater_lifetime_sec` | 1.5 | Base lifetime |
| `whitewater_emit_per_frame_max` | 4000 | Per-frame cap |
| `whitewater_total_cap` | 80000 | Hard total cap |

## Cache layout

After bake, `cache_dir` contains:

```
<cache_dir>/
├── cache.json
├── mesh/frame_NNNN.ply          # if output.mesh
├── cache.usdc                   # if output.usd
├── particles/frame_NNNN.npy     # if output.particles
├── colors/frame_NNNN.npy        # if any source has `color`
├── whitewater/frame_NNNN.npy    # if output.whitewater (positions only)
└── whitewater_kinds/frame_NNNN.npy   # foam/spray/bubble class per particle
```

`cache.json` is the manifest: which streams exist, frame range, and the
domain bounds + dx, so consumers can rebuild the world transform.

## End-to-end smoke

`examples/smoke_addon_flow.py` constructs a scene dict the way the Blender
addon does, runs it through `addon/gpufluid_blender/config_builder.py`, and
bakes via the CLI. Use it as the canonical "does this still work?" check
when changing the TOML schema or the addon bridge.
