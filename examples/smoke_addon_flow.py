"""End-to-end smoke for the addon flow without launching Blender.

What this verifies:
  1. The `scene_dict` shape produced by `addon/gpufluid_blender/operators/bake.py:collect_scene`
     (after v0.8 B1.3) passes through `build_toml` (B1.4) without errors.
  2. The resulting TOML loads via `gpufluid.cli.config.load_scene`.
  3. `gpufluid.cli.commands.simulate` bakes a short cache from it
     (short = 6 frames at 32³, so this runs in seconds on a 4080).
  4. The expected output files exist after bake.

Run me with:
    .venv\\Scripts\\python examples\\smoke_addon_flow.py
"""
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_config_builder():
    spec = importlib.util.spec_from_file_location(
        "_cb", REPO / "addon" / "gpufluid_blender" / "config_builder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    cb = _load_config_builder()

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "smoke_cache"
        cache_dir.mkdir()

        # Simulates what `collect_scene` produces after a user marks one
        # Domain + two coloured Fluid sources + enables Surface Tension and
        # Whitewater in the UI.
        scene_dict = {
            "domain": {"resolution": (32, 32, 32), "dx": 0.03125,
                       "origin": [0.0, 0.0, 0.0]},
            "fluids": [
                {"kind": "box", "lo": (0.10, 0.50, 0.30),
                 "hi": (0.40, 0.70, 0.70), "ppc": 4,
                 "color": (1.0, 0.1, 0.1)},
                {"kind": "box", "lo": (0.60, 0.50, 0.30),
                 "hi": (0.90, 0.70, 0.70), "ppc": 4,
                 "color": (0.1, 0.2, 1.0)},
            ],
            "obstacles": [],
            "inflows": [],
            "outflows": [],
            "simulation": {
                "dt": 0.008, "frames": 6, "fps": 24,
                "pressure_iters": 30, "pressure_solver": "jacobi",
                "cfl": True, "cfl_factor": 0.5, "cfl_max_substeps": 8,
                "flip_blend": 0.95, "gravity": -9.81,
                "surface_tension": 0.05,
                "csf_smoothing_passes": 2,
                "reseed": False,
                "reseed_every_n_frames": 5,
                "reseed_min_per_cell": 2,
                "reseed_max_per_cell": 12,
            },
            "output": {
                "cache_dir": str(cache_dir),
                "iso_level": 0.6, "smooth_passes": 2,
                "mesh_smooth_passes": 2, "mesh_smooth_method": "taubin",
                "decimate_ratio": 1.0, "wall_margin_cells": 2,
                "usd": False,         # native USD is Blender-side; CLI smoke skips
                "particles": True,    # need for color sidecars
                "preview": False,
                # Whitewater (B1.5) — small caps to keep the smoke quick.
                "whitewater": True,
                "whitewater_speed_threshold": 3.0,
                "whitewater_lifetime_sec": 1.0,
                "whitewater_emit_per_frame_max": 500,
                "whitewater_total_cap": 5000,
            },
        }

        print(f"[smoke] cache dir: {cache_dir}")
        toml_str = cb.build_toml(scene_dict)
        toml_path = cache_dir / "scene.toml"
        toml_path.write_text(toml_str, encoding="utf-8")
        print(f"[smoke] wrote scene.toml ({len(toml_str)} bytes)")

        from gpufluid.cli.config import load_scene
        scene = load_scene(str(toml_path))
        assert scene.simulation.surface_tension == 0.05, "σ didn't round-trip"
        assert scene.output.whitewater is True, "whitewater toggle lost"
        # Confirm multi-source loaded as either `fluids` (new) or a single
        # `fluid` (legacy). load_scene must surface both colours somewhere.
        try:
            n_sources = len(scene.fluids)
        except AttributeError:
            n_sources = 1
        print(f"[smoke] scene loaded; {n_sources} fluid source(s); "
              f"sigma={scene.simulation.surface_tension}; "
              f"ww={scene.output.whitewater}")

        # Run the actual bake via the CLI exactly as the addon's bake operator
        # does (subprocess of `gpufluid simulate`).
        import subprocess
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "gpufluid.cli", "simulate", str(toml_path)],
            cwd=REPO, capture_output=True, text=True, encoding="utf-8",
            errors="replace",
        )
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"[smoke] FAIL — bake exited {proc.returncode} in {dt:.2f}s")
            print("---stdout---")
            print(proc.stdout[-2000:])
            print("---stderr---")
            print(proc.stderr[-2000:])
            return 1
        print(f"[smoke] bake done in {dt:.2f}s")

        # Verify outputs.
        mesh_dir = cache_dir / "mesh"
        ply_files = sorted(mesh_dir.glob("frame_*.ply")) if mesh_dir.exists() else []
        part_dir = cache_dir / "particles"
        part_files = sorted(part_dir.glob("frame_*.npy")) if part_dir.exists() else []
        colors_dir = cache_dir / "colors"
        color_files = sorted(colors_dir.glob("frame_*.npy")) if colors_dir.exists() else []
        ww_dir = cache_dir / "whitewater"
        ww_files = sorted(ww_dir.glob("frame_*.npy")) if ww_dir.exists() else []

        print(f"[smoke] mesh    : {len(ply_files)} frames")
        print(f"[smoke] particle: {len(part_files)} frames")
        print(f"[smoke] colors  : {len(color_files)} frames")
        print(f"[smoke] ww      : {len(ww_files)} frames")

        ok = (len(ply_files) == 6 and len(part_files) == 6
              and len(color_files) == 6 and len(ww_files) == 6)
        if ok:
            print("[smoke] OK — full addon flow runs end-to-end")
            return 0
        print("[smoke] FAIL — some outputs missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
