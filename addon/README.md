# gpufluid Blender addon — install & quickstart

Tested on Blender 4.5 LTS and 5.1.

## 1. Prepare a Python venv with `gpufluid` installed

The addon delegates simulation to a subprocess running the `gpufluid` CLI.
That CLI needs **NVIDIA Warp** (CUDA-only). Blender's bundled Python does
NOT have these — point the addon at a separate venv.

```powershell
# anywhere outside Blender
cd E:\projects\gpu_flip\gpufluid    # or wherever you cloned this repo
python -m venv .venv
.venv\Scripts\activate
pip install -e .
gpufluid info                         # should list your CUDA device
```

Record the full path to `.venv\Scripts\python.exe` — you'll paste it into
the addon prefs.

## 2. Build the addon zip

```powershell
cd addon
python -c "import shutil; shutil.make_archive('gpufluid_blender', 'zip', '.', 'gpufluid_blender')"
# → addon/gpufluid_blender.zip
```

## 3. Install in Blender

1. Open Blender.
2. **Edit → Preferences → Add-ons → Install from Disk…** (Blender 4.5+).
3. Pick `addon/gpufluid_blender.zip`.
4. Enable the **gpufluid** add-on.
5. Open its preferences and paste the full path of `python.exe` from step 1.

## 4. Run a simulation

1. In the 3D viewport press `N` to open the sidebar; switch to the **GpuFluid** tab.
2. Click **Add gpufluid Domain** → a 1m cube empty appears.
3. Add a fluid source: any mesh object (Add → Mesh → Cube → scale → place inside the domain). Select it, click **Mark as Fluid Source**.
4. Add obstacles the same way: any mesh inside the domain, **Mark as Obstacle**, choose its type (Bounding Box / Sphere / Cylinder Y / Mesh).
5. Select the Domain empty → in the **Domain** panel adjust `Resolution`, `Frames`, etc.
6. Click **Bake gpufluid Simulation**.

The bake runs as a background subprocess and reports `frame N/M` progress in
Blender's status bar. When done, a new mesh object `gpufluid_cache` is created
that updates per frame on timeline playback.

## 5. Troubleshooting

- **"Set a valid Python interpreter"** — fill in Addon Preferences with the
  python.exe from step 1.
- **"gpufluid bake failed — see system console"** — `Window → Toggle System
  Console` (Windows). The full CLI output is logged there.
- **Cache exists but viewport doesn't update** — scrub the timeline. The
  per-frame handler runs on `frame_change_pre`.
- **No CUDA device** — the CLI also runs CPU-only (slow). Force by editing
  scene.toml or just install on a machine with NVIDIA GPU.
