# Block index

Authoritative index of all blocks declared in `DESIGN.md`. Kept in sync
manually for v0.1; will become auto-generated once `gpufluid.blocks
--check` lands.

Status legend: **impl** = implemented in source; **test** = also has a
passing pytest; **plan** = declared but not yet written.

| ID       | Description                                | Layer | Source location                              | Status |
|----------|--------------------------------------------|-------|----------------------------------------------|--------|
| G1.1     | Warp init & device selection               | G1    | `gpufluid/primitives/runtime.py`             | impl,test |
| G1.2     | Array allocation helpers                   | G1    | `gpufluid/primitives/runtime.py`             | impl,test |
| G1.3     | `clamp_int` etc.                           | G1    | `gpufluid/primitives/gridmath.py`            | impl,test |
| G1.4     | Trilinear weights                          | G1    | `gpufluid/primitives/gridmath.py`            | impl,test |
| G1.5     | Trilinear sample                           | G1    | `gpufluid/primitives/gridmath.py`            | impl |
| G1.6     | Trilinear scatter w/ atomic_add            | G1    | `gpufluid/primitives/gridmath.py`            | impl |
| G1.7     | Box-blur 3D                                | G1    | `gpufluid/primitives/gridmath.py`            | impl |
| G1.8     | Cell-centre grid (host)                    | G1    | `gpufluid/domain/sdf.py`                     | impl,test |
| G1.9     | Step profiler (per-section ScopedTimer)    | G1    | `gpufluid/primitives/profiling.py`           | impl,test |
| S2.1     | P2G transfer                               | S2    | `gpufluid/solvers/solver3d.py`               | impl |
| S2.2     | Normalize faces                            | S2    | `gpufluid/solvers/solver3d.py`               | impl |
| S2.3     | Add gravity                                | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.4     | Enforce solid BC                           | S2    | `gpufluid/solvers/solver3d.py`               | impl |
| S2.5     | Divergence                                 | S2    | `gpufluid/solvers/solver3d.py`               | impl |
| S2.6.1   | Jacobi pressure iteration                  | S2    | `gpufluid/solvers/solver3d.py`               | impl |
| S2.6.2   | Gauss–Seidel red-black                     | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.6.3   | PCG                                        | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.7     | Subtract pressure gradient                 | S2    | `gpufluid/solvers/solver3d.py`               | impl |
| S2.8     | G2P + FLIP/PIC blend                       | S2    | `gpufluid/solvers/solver3d.py`               | impl |
| S2.9     | Particle advection + clamp                 | S2    | `gpufluid/solvers/solver3d.py`               | impl |
| S2.10    | CFL substep count (host helper)            | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.10.GPU | GPU vmax atomic_max reduction              | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.6.4   | Block-sparse Jacobi pressure (skip empty)  | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.16    | Active-block bitmask builder               | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.11    | Particle reseed                            | S2    | `gpufluid/sim/reseed.py`                     | impl,test |
| S2.11.GPU | GPU reseed (count + rank + compact + emit) | S2    | `gpufluid/sim/reseed.py`                     | impl,test |
| S2.11.GPU.COUNT | Atomic per-cell particle count       | S2    | `gpufluid/sim/reseed.py`                     | impl,test |
| S2.11.GPU.RANK | Per-particle rank-in-cell + alive mask | S2  | `gpufluid/sim/reseed.py`                     | impl,test |
| S2.12    | APIC transfer                              | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.13    | Viscosity (implicit Jacobi)                | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.14.1  | Build smoothed fluid indicator χ̃          | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.14.2  | Compute unit normal field n̂               | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.14.3  | Compute curvature κ                        | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.14.4  | Apply CSF impulse to MAC faces             | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.14.5  | Capillary-wave dt_max (host helper)        | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.14.6  | Force-balance: subtract mean impulse       | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.15.1  | P2G scatter of per-particle color          | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.15.2  | Normalize grid color by deposited weight   | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| S2.15.3  | G2P sample grid color back to particles    | S2    | `gpufluid/solvers/solver3d.py`               | impl,test |
| F3.1     | `FlipSolver2D`                             | F3    | `gpufluid/solvers/solver2d.py`               | impl |
| F3.2     | `FlipSolver3D`                             | F3    | `gpufluid/solvers/solver3d.py`               | impl,test |
| F3.3     | `step()` pipeline                          | F3    | `gpufluid/solvers/solver3d.py`               | impl,test |
| F3.4     | `step_cfl()`                               | F3    | `gpufluid/solvers/solver3d.py`               | impl,test |
| F3.5     | Restart / checkpoint                       | F3    | —                                            | plan |
| D4.1     | Wall shell init                            | D4    | `gpufluid/solvers/solver3d.py`               | impl |
| D4.2.1   | SDF sphere                                 | D4    | `gpufluid/domain/sdf.py`                     | impl,test |
| D4.2.2   | SDF box                                    | D4    | `gpufluid/domain/sdf.py`                     | impl,test |
| D4.2.3   | SDF cylinder Y                             | D4    | `gpufluid/domain/sdf.py`                     | impl,test |
| D4.2.4   | SDF plane                                  | D4    | `gpufluid/domain/sdf.py`                     | impl,test |
| D4.2.5   | SDF union                                  | D4    | `gpufluid/domain/sdf.py`                     | impl,test |
| D4.3     | Mesh → SDF (CPU + GPU)                     | D4    | `gpufluid/domain/mesh_sdf.py`, `mesh_sdf_gpu.py` | impl,test |
| D4.3.GPU.BVH | BVH-accelerated mesh inside test (wp.Mesh + winding query) | D4 | `gpufluid/domain/mesh_sdf_gpu.py`             | impl,test |
| D4.4     | Apply SDF as solid                         | D4    | `gpufluid/domain/sdf.py`                     | impl,test |
| D4.5.1   | Box seeder                                 | D4    | `gpufluid/solvers/solver3d.py`               | impl |
| D4.5.2   | Mesh seeder                                | D4    | `gpufluid/solvers/solver3d.py`               | impl,test |
| D4.6     | Animated obstacles                         | D4    | `gpufluid/domain/animation.py`               | impl,test |
| D4.7     | Inflow / outflow                           | D4    | `gpufluid/domain/regions.py`                 | impl,test |
| F3.5     | Restart / checkpoint                       | F3    | `gpufluid/solvers/solver3d.py`               | impl,test |
| F3.6     | Per-frame hook (anim+inflow+outflow)       | F3    | `gpufluid/solvers/solver3d.py`               | impl,test |
| M5.1     | Density scatter                            | M5    | `gpufluid/meshing/surface.py`                | impl |
| M5.2     | Density smoothing                          | M5    | `gpufluid/meshing/surface.py`                | impl |
| M5.3     | Marching cubes (skimage)                   | M5    | `gpufluid/meshing/surface.py`                | impl,test |
| M5.4     | MC on Warp (wp.MarchingCubes)              | M5    | `gpufluid/meshing/surface.py`                | impl,test |
| M5.5     | Mesh smoothing (Laplacian / Taubin)        | M5    | `gpufluid/meshing/smoothing.py`              | impl,test |
| M5.6     | Mesh decimation                            | M5    | `gpufluid/meshing/decimate.py`               | impl,test |
| I6.1     | PLY writer + reader                        | I6    | `gpufluid/io/ply.py`                         | impl,test |
| I6.2     | Cache manifest                             | I6    | `gpufluid/io/cache.py`                       | impl,test |
| I6.3     | Particle dump                              | I6    | `gpufluid/io/ply.py` (np.save helper)        | impl |
| I6.4     | Alembic writer                             | I6    | —                                            | plan |
| I6.5     | USD writer                                 | I6    | `gpufluid/io/usd_writer.py`                  | impl,test |
| C7.1     | TOML config schema                         | C7    | `gpufluid/cli/config.py`                     | impl,test |
| C7.2     | `simulate` command                         | C7    | `gpufluid/cli/commands.py`                   | impl,test |
| C7.3     | `bench` command                            | C7    | `gpufluid/cli/commands.py`                   | impl |
| C7.4     | `info` command                             | C7    | `gpufluid/cli/commands.py`                   | impl,test |
| W7.1     | Whitewater state                           | W7    | `gpufluid/sim/whitewater.py`                 | impl,test |
| W7.2     | Whitewater emit                            | W7    | `gpufluid/sim/whitewater.py`                 | impl,test |
| W7.3     | Whitewater ballistic advect                | W7    | `gpufluid/sim/whitewater.py`                 | impl,test |
| W7.4     | Kind classifier (foam/spray/bubble)        | W7    | `gpufluid/sim/whitewater.py`                 | impl,test |
| W7.5     | Per-class dynamics (gravity/drag/buoyancy) | W7    | `gpufluid/sim/whitewater.py`                 | impl,test |
| W7.6     | Kind sidecar I/O for render                | W7    | `gpufluid/cli/commands.py`                   | impl,test |
| W7.7     | Trapped-air potential (Ihmsen 2012)        | W7    | `gpufluid/sim/whitewater_potentials.py`      | impl,test |
| W7.7.H   | Host wrapper for W7.7 (numpy → numpy)      | W7    | `gpufluid/sim/whitewater_potentials.py`      | impl,test |
| A8.1     | Addon register/unregister                  | A8    | `addon/gpufluid_blender/__init__.py`         | impl |
| A8.2     | Domain property group                      | A8    | `addon/gpufluid_blender/properties.py`       | impl |
| A8.3     | Fluid source                               | A8    | `addon/gpufluid_blender/properties.py`       | impl |
| A8.4     | Obstacle source                            | A8    | `addon/gpufluid_blender/properties.py`       | impl |
| A8.5     | Bake operator                              | A8    | `addon/gpufluid_blender/operators/bake.py`   | impl,test (config_builder) |
| A8.6     | Cache import (per-frame PLY swap)          | A8    | `addon/gpufluid_blender/cache_loader.py`     | impl |
| A8.6.1   | Whitewater point-cloud import              | A8    | `addon/gpufluid_blender/cache_loader.py`     | impl |
| A8.7     | UI panels                                  | A8    | `addon/gpufluid_blender/panels.py`           | impl |
| A8.8     | Helper operators                           | A8    | `addon/gpufluid_blender/operators/helpers.py`| impl |
| A8.2.1   | Surface tension property group (B1.1)      | A8    | `addon/gpufluid_blender/properties.py`       | impl |
| A8.3.1   | Per-source colour fields (B1.2)            | A8    | `addon/gpufluid_blender/properties.py`       | impl |
| A8.2.2   | Whitewater property group (B1.5)           | A8    | `addon/gpufluid_blender/properties.py`       | impl |
| A8.7.1   | Whitewater sub-panel (B1.5)                | A8    | `addon/gpufluid_blender/panels.py`           | impl |

## Test coverage (current sprint v0.1)

| File | Blocks tested |
|------|---------------|
| `tests/test_g1_primitives.py` | G1.1, G1.2, G1.3, G1.4 |
| `tests/test_s2_schemes.py`    | S2.3, F3.3 (smoke) |
| `tests/test_d4_sdf.py`        | D4.2.1, D4.2.2, D4.2.3, D4.2.5, D4.4, G1.8 |
| `tests/test_m5_meshing.py`    | M5.3 (via end-to-end) |
| `tests/test_i6_io.py`         | I6.1 |
| `tests/test_f3_solver.py`     | F3.2, F3.3 |
