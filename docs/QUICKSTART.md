# Quickstart — gpufluid in Blender

From an empty `.blend` to a rendered lava video in **about 5 minutes**.

## 1. Install (one-time)

1. Clone the repo, then `pip install -e .` inside its `.venv` so the CLI
   resolves with all CUDA/Warp deps:

   ```
   cd E:\projects\gpu_flip\gpufluid
   .venv\Scripts\python.exe -m pip install -e .
   ```

2. Zip the addon: `cd addon && Compress-Archive gpufluid_blender gpufluid_blender.zip`
   In Blender → `Edit → Preferences → Add-ons → Install from disk…` and pick
   the zip. Enable it.

3. The first time you open `Edit → Preferences → Add-ons → gpufluid →
   Preferences`, the addon auto-detects the `.venv\Scripts\python.exe`
   path. Verify it points at the Python where `gpufluid` is installed
   (the detect step is `python -c "import gpufluid"`). Hit **Detect** if
   the field is empty.

## 2. Build a scene

In the 3D viewport sidebar (**N panel → gpufluid** tab):

1. **Add Domain** — creates a 1×1×1 m Empty cube at the origin. Scale or
   move it to fit your set (the simulator will run inside its world AABB).
2. **Add an emitter or fluid source:**
   - For a continuous waterfall / tap: add an Empty, mark it as
     **Inflow** in the sidebar. Position it inside the Domain bounds
     (the bake op warns if it's not). You can keyframe its location/scale
     — the sampler walks the F-curves and the solver spawns particles
     along the path.
   - For an initial blob of fluid: mark a mesh as **Fluid**. Its bounding
     box becomes the seeded volume.
3. *(Optional)* Add an obstacle: mark a mesh as **Obstacle**, pick
   BBOX / Sphere / Cylinder / Plane / Mesh.
4. *(Optional)* Set frame range, FPS, resolution on the Domain object.
   Defaults: 240 frames @ 60 fps, 128³ grid.

## 3. Bake

Click **Bake** in the sidebar.

- The operator pre-samples animated Inflow keyframes via F-curve eval
  (<10 ms for hundreds of frames — Blender stays responsive).
- It writes `scene.toml` into the configured cache directory, then spawns
  the CLI in a subprocess (stdout drained on a background thread; UI is
  never blocked).
- Progress shows in the workspace status bar: `gpufluid: frame N/M`.
- When the CLI exits, the addon auto-attaches the cache:
  `gpufluid_cache` mesh appears in the outliner with a frame-change
  handler swapping pre-loaded meshes per frame.

Typical bake on a 4080 SUPER: **~45 s** for 240 frames @ 128³.

**Cancel:** press **Esc** anywhere in the 3D viewport while the bake is
running. The addon terminates the subprocess and reports `aborted (user
pressed Esc)`. Closing the Blender window cancels too (`cancel()`
callback routes through the same path).

**Reentrance:** clicking Bake twice while one is running rejects the
second click with a WARNING (single subprocess per cache_dir — two
would race the .ply writes).

**Sync mode (scripts / CI):** `bpy.ops.gpufluid.bake('EXEC_DEFAULT',
sync=True, sync_timeout_sec=600)` skips the modal/timer dance and
blocks until the CLI finishes. Used by `examples/_ci_headless_bake.py`
which runs the full pipeline under `blender -b -P script.py` for
render-farm / CI validation. Same for `gpufluid.render` (default
timeout 1800 s). Set `sync_timeout_sec=0` to disable the watchdog.

**TOML overrides:** Domain → "TOML overrides" multi-line text box lets
you splice arbitrary `[section] key = value` into the emitted scene.toml
(deep-merged on top). Use this to reach knobs without UI like
`[output] mesh_method = "sdf"` or `[output] temperature_colormap = "blackbody"`.

## 4. Render

Use the standard Blender render. The `gpufluid_cache` object behaves like
any other mesh — give it a material (we ship a `LavaProd` shader with
emission noise + voronoi), set up lights/camera, hit **F12** (image) or
**Ctrl+F12** (animation).

For Eevee: click **Apply Eevee Production Preset** in the sidebar — sets
TAA samples to 16 and disables bloom/SSR/GTAO/volumetrics that hurt
per-frame fluid renders. ~230 ms/frame at 1280×720 on the same hardware.

## 5. Re-bake

Tweak the scene, hit **Clear Cache** (releases MSC + purges cache
datablocks + `rmtree`s the dir, retries on Windows mmap lock), then
**Bake** again.

The addon is idempotent: the same `gpufluid_cache` object is reused
across re-bakes — no `.001/.002/...` proliferation.

---

## Coordinate system (read this if things look weirdly placed)

The MPM solver runs internally in a unit cube **[0, 1]³**, but you author
your scene in world metres. The addon handles the conversion in two
places — knowing how it works makes scaling intuitive:

| Where you author | What the solver sees | What renders |
|---|---|---|
| Domain Empty at `(0, 0, 0.5)`, scale 2 → world AABB `[-1, +1] × [-1, +1] × [-0.5, +1.5]` | Particles in `[0.05, 0.95]³` (5% wall margin) | Mesh in `[-1, +1] × [-1, +1] × [-0.5, +1.5]` |
| WaterTap Inflow at world `(0.5, 0, 1.2)` | Spawn box at normalised `((0.5−(−1))/2, ..., (1.2−(−0.5))/2) = (0.75, 0.5, 0.85)` | Stream visually emerges from world `(0.5, 0, 1.2)` |
| Gravity `−9.81 m/s²` | Scaled to `−9.81 / dom_size_z` in normalised units | Real falling speed regardless of domain size |

Practical consequences:

- **Inflows / fluids must lie inside the Domain.** Anything that
  normalises outside `[0.05, 0.95]³` gets wall-clamped — the bake op
  emits a `WARNING` in the info bar.
- **Domain size affects nothing physically** — gravity, fall times,
  pool spread all scale correctly. Resize the Domain to fit your set.
- **The cache mesh is at world `(0, 0, 0)` with scale `(1, 1, 1)`.**
  The world placement is baked into the mesh vertices during attach
  (`v_world = v_normalised * dom_size + origin`).
